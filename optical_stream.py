#!/usr/bin/env python3
"""Send fixed blocks through XDMA -> QSFP3 -> QSFP4 -> XDMA.

This program needs a loaded hh_qsfp bitstream and Xilinx's XDMA Linux driver.
It owns the HydraHarp during --source hydraharp; close other HH applications.
The output file contains raw little-endian 32-bit T2 records, not PTU headers.
CLI streams log completed-block mismatches and continue; transport failures stop
the stream. A run containing mismatches exits with status 1 after completion.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from collections import deque
import csv
import ctypes
import errno
import json
import mmap
import os
from pathlib import Path
import queue
import statistics
import struct
import threading
import time
import zlib

BLOCK_SIZE = 65536
# BLOCK_SIZE = 4096
HEADER = struct.Struct("<4sIII")
MAGIC = b"HHQ1"
PAYLOAD_SIZE = BLOCK_SIZE - HEADER.size
RX_FIFO_BYTES = 128 * 1024


class CpuIdleLatency:
    """Hold an optional, system-wide CPU idle-latency request during streaming.

    None leaves the existing policy alone. Linux combines active requests, and
    closing this descriptor removes only this request. The frequency governor
    and persistent system configuration are not changed.
    """

    def __init__(self, latency_us=None):
        if latency_us is not None and not 0 <= latency_us <= 2147483647:
            raise ValueError("CPU idle latency must be between 0 and 2147483647 us")
        self.latency_us = latency_us
        self.fd = None

    def __enter__(self):
        if self.latency_us is None:
            return self
        if self.fd is not None:
            raise RuntimeError("CPU idle-latency request is already active")
        try:
            self.fd = os.open("/dev/cpu_dma_latency", os.O_RDWR | os.O_CLOEXEC)
        except PermissionError as exc:
            raise PermissionError(
                exc.errno,
                "Low latency requires read/write access to /dev/cpu_dma_latency; "
                "grant this user access, then rerun as the normal user",
                "/dev/cpu_dma_latency",
            ) from exc
        try:
            request = struct.pack("=i", self.latency_us)
            if os.write(self.fd, request) != len(request):
                raise OSError("CPU idle-latency request was not fully written")
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *_):
        if self.fd is not None:
            fd, self.fd = self.fd, None
            os.close(fd)


# Use ordinary read/write syscalls with page-aligned, writable DMA buffers.
# Keep the mmap and its exported pointer alive until the syscall completes.
_libc = ctypes.CDLL(None, use_errno=True)
for _name in ("read", "write"):
    _function = getattr(_libc, _name)
    _function.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t)
    _function.restype = ctypes.c_ssize_t


def _aligned_transfer(fd: int, size: int, data: bytes | None = None, *, timing=None):
    if size == 0:
        return b"" if data is None else None
    started = time.perf_counter() if timing is not None else 0
    calls = []
    try:
        with mmap.mmap(-1, size) as buffer:
            if data is not None:
                buffer[:] = data
            anchor = ctypes.c_char.from_buffer(buffer)
            address = ctypes.addressof(anchor)
            done = 0
            function = _libc.read if data is None else _libc.write
            if timing is not None:
                prepared = time.perf_counter()
                timing["buffer_prepare_s"] = prepared - started
            try:
                while done < size:
                    if timing is not None:
                        before = time.perf_counter()
                    count = function(fd, address + done, size - done)
                    if timing is not None:
                        calls.append((before, time.perf_counter()))
                    if count < 0:
                        error = ctypes.get_errno()
                        if error == errno.EINTR:
                            continue
                        raise OSError(error, os.strerror(error))
                    if count == 0:
                        raise EOFError(f"XDMA made no progress with {size - done} bytes missing")
                    done += count
                if timing is not None:
                    before = time.perf_counter()
                returned = buffer[:] if data is None else None
                if timing is not None:
                    timing["result_copy_s"] = time.perf_counter() - before
                return returned
            finally:
                del anchor
    finally:
        if timing is not None:
            timing["total_s"] = time.perf_counter() - started
            timing["syscall_s"] = sum(end - begin for begin, end in calls)
            timing["calls"] = calls


def _read_exact(fd: int, count: int) -> bytes:
    return _aligned_transfer(fd, count)


def _write_all(fd: int, data: bytes) -> None:
    _aligned_transfer(fd, len(data), data)


def _packet(sequence: int, payload: bytes) -> bytes:
    if len(payload) > PAYLOAD_SIZE or len(payload) % 4:
        raise ValueError("payload must be 32-bit records and fit in one block")
    header = HEADER.pack(MAGIC, sequence, len(payload), zlib.crc32(payload))
    return (header + payload).ljust(BLOCK_SIZE, b"\0")


def _benchmark_thread_pair(iterations: int) -> None:
    """Measure Python thread-pair creation/start/join with no device I/O."""
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        threads = [threading.Thread(target=lambda: None) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        samples.append(time.perf_counter() - started)

    print(f"No-I/O thread-pair benchmark ({iterations} iterations):")
    print(f"  median: {statistics.median(samples) * 1000:.3f} ms per pair")
    print(f"  mean:   {statistics.mean(samples) * 1000:.3f} ms per pair")
    print(f"  min/max: {min(samples) * 1000:.3f} / {max(samples) * 1000:.3f} ms per pair")


def _profile_report(rows, wall_s, open_s, csv_path):
    if csv_path is not None and rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with csv_path.open("x", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Per-block timings (seconds): {csv_path}")
    if not rows:
        return
    completed = [row for row in rows if row["status"] == "ok"]
    print(f"Pattern profile: {len(completed)}/{len(rows)} blocks verified; "
          f"block size={BLOCK_SIZE} bytes")
    print(f"Link/buffer/worker setup: {open_s * 1000:.3f} ms (once per run)")
    print(f"{'Stage':<36} {'mean ms':>10} {'median ms':>10} {'p95 ms':>10}")
    stages = (
        ("source_prepare_s", "Pattern generation"),
        ("packet_prepare_s", "Header/CRC/packet preparation"),
        ("workers_s", "Workers + DMA, wall time"),
        ("syscall_union_s", "  At least one DMA syscall active"),
        ("outside_syscalls_s", "  Remaining worker wall time"),
        ("verify_s", "Verification + payload extraction"),
        ("exchange_s", "Complete exchange"),
        ("reader_launch_s", "Reader launch/dispatch/ready wait *"),
        ("writer_launch_s", "Writer launch/dispatch *"),
        ("h2c_buffer_prepare_s", "H2C buffer preparation *"),
        ("c2h_buffer_prepare_s", "C2H buffer preparation *"),
        ("h2c_syscall_s", "H2C blocking write *"),
        ("c2h_syscall_s", "C2H blocking read *"),
        ("c2h_result_copy_s", "C2H copy to Python bytes *"),
        ("syscall_overlap_s", "Read/write syscall overlap *"),
    )
    for key, label in stages:
        values = sorted(row[key] * 1000 for row in rows if key in row)
        if values:
            p95 = values[min(len(values) - 1, (95 * len(values) + 99) // 100 - 1)]
            print(f"{label:<36} {statistics.mean(values):10.4f} "
                  f"{statistics.median(values):10.4f} {p95:10.4f}")
    payload_bytes = sum(row["payload_bytes"] for row in completed)
    exchange_s = sum(row["exchange_s"] for row in rows)
    print(f"Payload throughput: {payload_bytes / wall_s / 1e6:.2f} MB/s over "
          f"{wall_s:.3f} s of the loop")
    if not any(row.get("pipeline_depth", 1) > 1 for row in rows):
        print(f"Throughput over exchange calls: {payload_bytes / exchange_s / 1e6:.2f} MB/s")
    else:
        print("Pipelined block latencies overlap; do not sum them to calculate throughput.")
    print("* Overlapping diagnostics: do not add these rows. DMA syscall time "
          "includes driver/hardware waits, scheduling and Python resumption; "
          "it is not pure PCIe or optical transfer time.")


class OpticalDataMismatch(ValueError):
    """A completed transaction with corrupt returned bytes, safe to continue."""

    def __init__(self, details, returned, *, bit_errors=None, compared_bits=None,
                 header_intact=None):
        super().__init__(details)
        self.returned = returned
        self.bit_errors = bit_errors
        self.compared_bits = compared_bits
        self.header_intact = header_intact


class _BERStats:
    """Count bit differences in completed host-to-host packets, not lane idles."""

    def __init__(self):
        self.completed = self.mismatched = self.bits = self.errors = 0
        self.header_mismatches = self.transport_failures = 0

    def record(self, mismatch=None):
        self.completed += 1
        self.bits += mismatch.compared_bits if mismatch is not None else BLOCK_SIZE * 8
        if mismatch is not None:
            self.mismatched += 1
            self.errors += mismatch.bit_errors
            self.header_mismatches += int(not mismatch.header_intact)

    def report(self, requested):
        rate = f"{self.errors / self.bits:.6e}" if self.bits else "unavailable (no completed blocks)"
        return (
            f"BER summary: {self.completed}/{requested} blocks completed; "
            f"{self.completed - self.mismatched} intact; {self.mismatched} mismatched\n"
            f"Compared packet bits: {self.bits}; differing bits: {self.errors}\n"
            f"Observed BER on completed packets: {rate}\n"
            f"Header-mismatched blocks: {self.header_mismatches}; "
            f"transport failures: {self.transport_failures}; "
            f"requested blocks not compared: {requested - self.completed}\n"
            "Counts include HHQ1 headers and padding; exclude Aurora/PCIe encoding and idles.\n"
            "Incomplete transfers are excluded. Header/stream misalignment can inflate this ratio."
        )


class _LegacyOpticalLink:
    """One 64 KiB transaction at a time, with C2H posted before H2C."""

    def __init__(self, h2c: Path, c2h: Path, timeout: float, *, profile=False):
        self.h2c_path = h2c
        self.c2h_path = c2h
        self.timeout = timeout
        self.h2c_fd: int | None = None
        self.c2h_fd: int | None = None
        self.sequence = 0
        self.profile = profile
        self.last_profile = None

    def __enter__(self):
        self.c2h_fd = os.open(self.c2h_path, os.O_RDONLY)
        try:
            self.h2c_fd = os.open(self.h2c_path, os.O_WRONLY)
        except BaseException:
            os.close(self.c2h_fd)
            self.c2h_fd = None
            raise
        return self

    def __exit__(self, *_):
        if self.h2c_fd is not None:
            os.close(self.h2c_fd)
        if self.c2h_fd is not None:
            os.close(self.c2h_fd)

    def exchange(self, payload: bytes) -> bytes:
        if not self.profile:
            return self._exchange(payload)
        timing = {"block": self.sequence, "payload_bytes": len(payload)}
        started = time.perf_counter()
        try:
            returned = self._exchange(payload, timing)
            timing["status"] = "ok"
            return returned
        except BaseException as exc:
            timing["status"] = type(exc).__name__
            raise
        finally:
            timing["exchange_s"] = time.perf_counter() - started
            self.last_profile = timing

    def _exchange(self, payload: bytes, timing=None) -> bytes:
        if self.h2c_fd is None or self.c2h_fd is None:
            raise RuntimeError("XDMA devices are not open")
        if timing is not None:
            started = time.perf_counter()
        expected = _packet(self.sequence, payload)
        if timing is not None:
            timing["packet_prepare_s"] = time.perf_counter() - started
            workers_started = time.perf_counter()
        read_timing = {} if timing is not None else None
        write_timing = {} if timing is not None else None
        result: dict[str, bytes | BaseException] = {}
        ready = threading.Event()
        deadline = time.monotonic() + self.timeout

        def receiver():
            ready.set()
            try:
                result["data"] = (
                    _read_exact(self.c2h_fd, BLOCK_SIZE) if timing is None else
                    _aligned_transfer(self.c2h_fd, BLOCK_SIZE, timing=read_timing)
                )
            except BaseException as exc:
                result["error"] = exc

        reader = threading.Thread(target=receiver, daemon=True)
        reader.start()
        ready.wait()
        if timing is not None:
            reader_launched = time.perf_counter()
            timing["reader_launch_s"] = reader_launched - workers_started
        # One complete block fits in the RX FIFO even if the kernel has not
        # submitted the C2H descriptor by the time H2C starts.

        def sender():
            try:
                if timing is None:
                    _write_all(self.h2c_fd, expected)
                else:
                    _aligned_transfer(self.h2c_fd, len(expected), expected, timing=write_timing)
            except BaseException as exc:
                result["write_error"] = exc

        writer = threading.Thread(target=sender, daemon=True)
        writer.start()
        if timing is not None:
            timing["writer_launch_s"] = time.perf_counter() - reader_launched
        writer.join(max(0, deadline - time.monotonic()))
        if writer.is_alive():
            raise TimeoutError("H2C write stalled; check PCIe and both Aurora links")
        if "write_error" in result:
            raise RuntimeError("XDMA H2C write failed") from result["write_error"]
        reader.join(max(0, deadline - time.monotonic()))
        if reader.is_alive():
            raise TimeoutError("No C2H block returned; check PCIe and both Aurora links")
        if "error" in result:
            raise RuntimeError("XDMA C2H read failed") from result["error"]
        if timing is not None:
            workers_finished = time.perf_counter()
            timing["workers_s"] = workers_finished - workers_started
            intervals = sorted(read_timing["calls"] + write_timing["calls"])
            union = 0.0
            end = 0.0
            for begin, finish in intervals:
                union += max(0.0, finish - max(begin, end))
                end = max(end, finish)
            timing["syscall_union_s"] = union
            timing["outside_syscalls_s"] = max(0.0, timing["workers_s"] - union)
            for prefix, values in (("h2c", write_timing), ("c2h", read_timing)):
                for name in ("buffer_prepare_s", "syscall_s", "result_copy_s", "total_s"):
                    timing[f"{prefix}_{name}"] = values[name]
                timing[f"{prefix}_syscall_count"] = len(values["calls"])
                timing[f"{prefix}_syscall_start_s"] = values["calls"][0][0] - workers_started
                timing[f"{prefix}_syscall_end_s"] = values["calls"][-1][1] - workers_started
            timing["syscall_overlap_s"] = (
                read_timing["syscall_s"] + write_timing["syscall_s"] - union
            )
            verification_started = time.perf_counter()
        try:
            return self._verify(payload, expected, result["data"])
        finally:
            if timing is not None:
                timing["verify_s"] = time.perf_counter() - verification_started

    def _verify(self, payload, expected, received):
        if not isinstance(received, bytes) or len(received) != BLOCK_SIZE:
            raise RuntimeError("C2H returned an incomplete block")
        magic, sequence, length, checksum = HEADER.unpack_from(received)
        if received != expected:
            bad_offsets = [
                offset for offset, (sent, returned) in enumerate(zip(expected, received))
                if sent != returned
            ]
            first_bad = bad_offsets[0]
            bit_errors = sum((expected[i] ^ received[i]).bit_count() for i in bad_offsets)
            header_intact = received[:HEADER.size] == expected[:HEADER.size]
            details = (
                f"Optical data mismatch on block {self.sequence}: "
                f"{len(bad_offsets)} different bytes, first at {first_bad}, "
                f"last at {bad_offsets[-1]}; "
                f"{bit_errors} different bits; "
                f"first sent=0x{expected[first_bad]:02x}, "
                f"returned=0x{received[first_bad]:02x}, "
                f"XOR=0x{expected[first_bad] ^ received[first_bad]:02x}; "
                f"returned header magic={magic!r}, sequence={sequence}, "
                f"length={length}, CRC=0x{checksum:08x}; "
                f"expected sequence={self.sequence}, length={len(payload)}, "
                f"CRC=0x{zlib.crc32(payload):08x}"
            )
            if length <= PAYLOAD_SIZE:
                actual_payload = received[HEADER.size : HEADER.size + length]
                details += f"; returned payload CRC=0x{zlib.crc32(actual_payload):08x}"
            self.sequence += 1
            raise OpticalDataMismatch(
                details, received[HEADER.size : HEADER.size + len(payload)],
                bit_errors=bit_errors, compared_bits=BLOCK_SIZE * 8,
                header_intact=header_intact,
            )
        actual_payload = received[HEADER.size : HEADER.size + length]
        self.sequence += 1
        return actual_payload


class _DMABuffer:
    """Page-aligned memory kept alive for an entire link session."""

    def __init__(self, size):
        self.size = size
        self.memory = mmap.mmap(-1, size)
        # Touch pages once, outside the measured exchange loop.
        self.memory[:] = b"\0" * size
        self.anchor = ctypes.c_char.from_buffer(self.memory)
        self.address = ctypes.addressof(self.anchor)

    def transfer(self, fd, data=None, timing=None):
        started = time.perf_counter() if timing is not None else 0
        calls = []
        try:
            if data is not None:
                if len(data) != self.size:
                    raise ValueError("DMA buffer size does not match packet size")
                self.memory[:] = data
            if timing is not None:
                timing["buffer_prepare_s"] = time.perf_counter() - started
            function = _libc.read if data is None else _libc.write
            done = 0
            while done < self.size:
                if timing is not None:
                    before = time.perf_counter()
                count = function(fd, self.address + done, self.size - done)
                if timing is not None:
                    calls.append((before, time.perf_counter()))
                if count < 0:
                    error = ctypes.get_errno()
                    if error == errno.EINTR:
                        continue
                    raise OSError(error, os.strerror(error))
                if count == 0:
                    raise EOFError(f"XDMA made no progress with {self.size - done} bytes missing")
                done += count
            if timing is not None:
                before = time.perf_counter()
            # Return an owned copy before this slot can be reused.
            returned = self.memory[:] if data is None else None
            if timing is not None:
                timing["result_copy_s"] = time.perf_counter() - before
            return returned
        finally:
            if timing is not None:
                timing["total_s"] = time.perf_counter() - started
                timing["syscall_s"] = sum(end - begin for begin, end in calls)
                timing["calls"] = calls

    def close(self):
        self.anchor = None
        self.memory.close()


class _TransferJob:
    def __init__(self, payload, expected, slot, deadline, timing, started):
        self.payload = payload
        self.expected = expected
        self.slot = slot
        self.deadline = deadline
        self.timing = timing
        self.started = started
        self.workers_started = time.perf_counter() if timing is not None else 0
        self.read_timing = {} if timing is not None else None
        self.write_timing = {} if timing is not None else None
        self.ready = threading.Event()
        self.read_done = threading.Event()
        self.write_done = threading.Event()
        self.result = {}


class _MemoryRoundtripBenchmark(_LegacyOpticalLink):
    """DMA directly between two complete, preallocated RAM datasets.

    Credits are returned only after C2H completes a block. Thus at most
    pipeline_depth blocks can be on the FPGA/optical path even if the reader
    is descheduled. No copying, packet generation or verification is done
    in the timed loop. Each syscall still transfers one framed block.
    """

    def __init__(self, h2c, c2h, timeout, *, blocks, pipeline_depth):
        super().__init__(h2c, c2h, timeout)
        if blocks <= 0 or blocks > 0x100000000:
            raise ValueError("benchmark block count must fit the 32-bit sequence field")
        if pipeline_depth not in (1, 2) or BLOCK_SIZE * pipeline_depth > RX_FIFO_BYTES:
            raise ValueError("benchmark outstanding bytes must not exceed 128 KiB")
        if BLOCK_SIZE % mmap.PAGESIZE:
            raise ValueError("memory benchmark block size must be a multiple of the page size")
        self.blocks = blocks
        self.pipeline_depth = pipeline_depth
        self.total_bytes = blocks * BLOCK_SIZE
        self.send = self.receive = None
        self._workers = []
        self._start = threading.Event()
        self._reader_ready = threading.Event()
        self._stop = threading.Event()
        self._credits = threading.Semaphore(pipeline_depth)
        self._done = [threading.Event(), threading.Event()]
        self._errors = []
        self._finished = [None, None]
        self._completed = [0, 0]
        self._progress = 0.0

    def __enter__(self):
        try:
            self.send = _DMABuffer(self.total_bytes)
            self.receive = _DMABuffer(self.total_bytes)
            # Different data in every block exposes duplicates/reordering as
            # well as corruption. All pages and packets exist before timing.
            for sequence in range(self.blocks):
                offset = sequence * BLOCK_SIZE
                self.send.memory[offset:offset + BLOCK_SIZE] = _packet(
                    sequence, os.urandom(PAYLOAD_SIZE),
                )
            super().__enter__()
            for index, fd in enumerate((self.c2h_fd, self.h2c_fd)):
                owned_fd = os.dup(fd)
                worker = threading.Thread(
                    target=self._transfer, args=(index, owned_fd),
                    name="ram-benchmark-c2h" if index == 0 else "ram-benchmark-h2c",
                    daemon=True,
                )
                try:
                    worker.start()
                except BaseException:
                    os.close(owned_fd)
                    raise
                self._workers.append(worker)
            return self
        except BaseException:
            self.__exit__()
            raise

    def __exit__(self, exc_type=None, exc=None, tb=None):
        self._stop.set()
        self._start.set()
        self._reader_ready.set()
        self._credits.release()
        deadline = time.monotonic() + self.timeout
        for worker in self._workers:
            worker.join(max(0, deadline - time.monotonic()))
        super().__exit__()
        self.h2c_fd = self.c2h_fd = None
        # A blocked daemon retains self, buffers and its duplicated fd. Do
        # not unmap any memory while the kernel may still be writing to it.
        if not any(worker.is_alive() for worker in self._workers):
            for buffer in (self.send, self.receive):
                if buffer is not None:
                    buffer.close()
            self.send = self.receive = None
        elif exc_type is None:
            raise TimeoutError("Benchmark DMA is still blocked; buffers remain allocated")

    def _transfer(self, index, fd):
        receiving = index == 0
        buffer = self.receive if receiving else self.send
        function = _libc.read if receiving else _libc.write
        try:
            self._start.wait()
            if receiving:
                self._reader_ready.set()
            elif not self._reader_ready.wait(self.timeout):
                raise TimeoutError("Benchmark receiver did not become ready")
            for block in range(self.blocks):
                if not receiving:
                    while not self._stop.is_set():
                        if self._credits.acquire(timeout=min(0.1, self.timeout)):
                            break
                if self._stop.is_set():
                    return
                done = 0
                while done < BLOCK_SIZE:
                    count = function(fd, buffer.address + block * BLOCK_SIZE + done,
                                     BLOCK_SIZE - done)
                    if count < 0:
                        error = ctypes.get_errno()
                        if error == errno.EINTR:
                            continue
                        raise OSError(error, os.strerror(error))
                    if count == 0:
                        raise EOFError("Benchmark DMA made no progress")
                    done += count
                # Timestamp native transfer completion, before notification.
                finished = time.perf_counter()
                self._completed[index] = block + 1
                self._progress = time.monotonic()
                if block + 1 == self.blocks:
                    self._finished[index] = finished
                if receiving:
                    self._credits.release()
        except BaseException as exc:
            self._errors.append(exc)
            self._stop.set()
            self._reader_ready.set()
            self._credits.release()
        finally:
            os.close(fd)
            self._done[index].set()

    def run(self):
        if self._start.is_set():
            raise RuntimeError("Use a new benchmark for each run")
        self._progress = time.monotonic()
        started = time.perf_counter()
        self._start.set()
        while not all(event.is_set() for event in self._done):
            if self._errors:
                raise RuntimeError("RAM benchmark DMA failed") from self._errors[0]
            if time.monotonic() - self._progress > self.timeout:
                self._stop.set()
                raise TimeoutError(
                    f"RAM benchmark stalled: H2C={self._completed[1]}, "
                    f"C2H={self._completed[0]} completed blocks",
                )
            self._done[0 if not self._done[0].is_set() else 1].wait(min(0.05, self.timeout))
        if self._errors:
            raise RuntimeError("RAM benchmark DMA failed") from self._errors[0]
        if any(finish is None for finish in self._finished):
            raise RuntimeError("RAM benchmark ended before both transfers completed")
        # Excludes main-thread wakeup/join and all subsequent verification.
        elapsed = max(self._finished) - started
        verify_started = time.perf_counter()
        mismatched_blocks = mismatched_bytes = 0
        examples = []
        for block in range(self.blocks):
            offset = block * BLOCK_SIZE
            expected = self.send.memory[offset:offset + BLOCK_SIZE]
            returned = self.receive.memory[offset:offset + BLOCK_SIZE]
            if expected == returned:
                continue
            mismatched_blocks += 1
            different = [i for i, (sent, got) in enumerate(zip(expected, returned)) if sent != got]
            mismatched_bytes += len(different)
            if len(examples) < 10:
                first = different[0]
                examples.append({
                    "block": block, "different_bytes": len(different),
                    "first_packet_offset": first, "last_packet_offset": different[-1],
                    "first_sent": expected[first], "first_returned": returned[first],
                    "header_matches": expected[:HEADER.size] == returned[:HEADER.size],
                    "expected_payload_crc32": zlib.crc32(expected[HEADER.size:]),
                    "returned_payload_crc32": zlib.crc32(returned[HEADER.size:]),
                })
        payload_bytes = self.blocks * PAYLOAD_SIZE
        return {
            "status": "mismatch" if mismatched_blocks else "ok",
            "blocks": self.blocks, "block_size": BLOCK_SIZE,
            "pipeline_depth": self.pipeline_depth,
            "maximum_unreceived_bytes": BLOCK_SIZE * self.pipeline_depth,
            "packet_bytes": self.total_bytes, "payload_bytes": payload_bytes,
            "elapsed_s": elapsed, "packet_MB_s": self.total_bytes / elapsed / 1e6,
            "payload_MB_s": payload_bytes / elapsed / 1e6,
            "h2c_complete_s": self._finished[1] - started,
            "c2h_complete_s": self._finished[0] - started,
            "verification_s": time.perf_counter() - verify_started,
            "verified_blocks": self.blocks - mismatched_blocks,
            "mismatched_blocks": mismatched_blocks, "mismatched_bytes": mismatched_bytes,
            "first_mismatches": examples,
        }


def _memory_benchmark(args):
    setup_started = time.perf_counter()
    print(f"Preparing {args.pattern_blocks * BLOCK_SIZE / 2**20:.2f} MiB in each RAM buffer; "
          f"at most {args.pipeline_depth} blocks unreceived", flush=True)
    with _MemoryRoundtripBenchmark(
        args.h2c, args.c2h, args.timeout, blocks=args.pattern_blocks,
        pipeline_depth=args.pipeline_depth,
    ) as benchmark:
        setup_s = time.perf_counter() - setup_started
        with CpuIdleLatency(args.low_latency):
            print("Transferring RAM -> FPGA -> QSFP -> FPGA -> RAM; "
                  "verification follows outside the timer", flush=True)
            result = benchmark.run()
    result.update({"setup_s": setup_s, "low_latency_us": args.low_latency,
                   "h2c": str(args.h2c), "c2h": str(args.c2h)})
    print(f"RAM roundtrip: {result['packet_bytes']:,} packet bytes in "
          f"{result['elapsed_s']:.6f} s; {result['packet_MB_s']:.2f} MB/s")
    print(f"Payload throughput: {result['payload_MB_s']:.2f} MB/s (dataset counted once)")
    print(f"Post-transfer verification: {result['verified_blocks']}/{result['blocks']} blocks intact; "
          f"{result['mismatched_blocks']} mismatched blocks, "
          f"{result['mismatched_bytes']} different bytes; "
          f"verification={result['verification_s']:.3f} s excluded")
    for example in result["first_mismatches"]:
        print(f"  Block {example['block']}: {example['different_bytes']} different bytes; "
              f"first packet offset={example['first_packet_offset']}")
    if args.benchmark_json is not None:
        with args.benchmark_json.open("x") as output:
            json.dump(result, output, indent=2)
            output.write("\n")
        print(f"Results saved to: {args.benchmark_json}")
    if result["mismatched_blocks"]:
        raise SystemExit(1)


class OpticalLink(_LegacyOpticalLink):
    """Reuse aligned buffers and two workers; default to one block in flight.

    Only one caller may submit/complete jobs. A two-slot pipeline is available
    through exchange_blocks(). Its total outstanding bytes cannot exceed the
    hardware RX FIFO, which cannot apply backpressure to Aurora.
    """

    def __init__(self, h2c, c2h, timeout, *, profile=False, pipeline_depth=1):
        super().__init__(h2c, c2h, timeout, profile=profile)
        if pipeline_depth not in (1, 2) or BLOCK_SIZE * pipeline_depth > RX_FIFO_BYTES:
            raise ValueError("Use 1 or 2 slots with total outstanding bytes <= 128 KiB")
        self.pipeline_depth = pipeline_depth
        self._slots = []
        self._free = deque()
        self._jobs = []
        self._workers = []
        self._read_queue = queue.Queue()
        self._write_queue = queue.Queue()
        self._stopping = threading.Event()
        self._submitted = 0
        self._broken = False

    def __enter__(self):
        if self._workers or self._stopping.is_set():
            raise RuntimeError("Create a new OpticalLink for each session")
        super().__enter__()
        try:
            for _ in range(self.pipeline_depth):
                send = _DMABuffer(BLOCK_SIZE)
                try:
                    receive = _DMABuffer(BLOCK_SIZE)
                except BaseException:
                    send.close()
                    raise
                slot = (send, receive)
                self._slots.append(slot)
                self._free.append(slot)
            for receiving, fd in ((True, self.c2h_fd), (False, self.h2c_fd)):
                # A blocked worker owns its fd and buffer until its syscall
                # returns, even if the main thread times out and closes.
                owned_fd = os.dup(fd)
                worker = threading.Thread(
                    target=self._worker, args=(receiving, owned_fd),
                    name="qsfp-c2h" if receiving else "qsfp-h2c", daemon=True,
                )
                try:
                    worker.start()
                except BaseException:
                    os.close(owned_fd)
                    raise
                self._workers.append(worker)
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, exc_type=None, exc=None, tb=None):
        deadline = time.monotonic() + self.timeout
        if not self._broken and not self._stopping.is_set():
            # On early generator exit or a mismatch, finish already submitted
            # blocks before stopping either worker. Otherwise a posted read
            # could be left waiting for a write that shutdown cancelled.
            for job in list(self._jobs):
                for event in (job.write_done, job.read_done):
                    event.wait(max(0, deadline - time.monotonic()))
        self._stopping.set()
        for job in self._jobs:
            job.ready.set()
        self._read_queue.put(None)
        self._write_queue.put(None)
        for worker in self._workers:
            worker.join(max(0, deadline - time.monotonic()))
        super().__exit__()
        self.h2c_fd = self.c2h_fd = None
        # Never unmap memory still accessible to an in-progress kernel DMA.
        # A blocked daemon retains this object (and its mappings) until return.
        if not any(worker.is_alive() for worker in self._workers):
            for send, receive in self._slots:
                send.close()
                receive.close()
            self._slots.clear()
            self._free.clear()
        elif exc_type is None:
            raise TimeoutError("DMA workers are still blocked; their buffers remain allocated")

    def _worker(self, receiving, fd):
        jobs = self._read_queue if receiving else self._write_queue
        try:
            while True:
                job = jobs.get()
                if job is None:
                    return
                try:
                    if self._stopping.is_set():
                        raise RuntimeError("Optical link is closing")
                    if receiving:
                        job.ready.set()
                        job.result["data"] = job.slot[1].transfer(
                            fd, timing=job.read_timing,
                        )
                    else:
                        if not job.ready.wait(max(0, job.deadline - time.monotonic())):
                            raise TimeoutError("C2H worker did not become ready")
                        if self._stopping.is_set():
                            raise RuntimeError("Optical link is closing")
                        job.slot[0].transfer(fd, job.expected, job.write_timing)
                except BaseException as exc:
                    job.result["error" if receiving else "write_error"] = exc
                    self._stopping.set()
                    for pending in self._jobs:
                        pending.ready.set()
                finally:
                    (job.read_done if receiving else job.write_done).set()
        finally:
            os.close(fd)

    def _submit(self, payload):
        if self.h2c_fd is None or self._stopping.is_set() or self._broken:
            raise RuntimeError("XDMA link is closed or failed; open a new session")
        if not self._free:
            raise RuntimeError("All DMA slots are still in flight")
        started = time.perf_counter() if self.profile else 0
        expected = _packet(self._submitted, payload)
        timing = None
        if self.profile:
            timing = {"block": self._submitted, "payload_bytes": len(payload),
                      "pipeline_depth": self.pipeline_depth,
                      "packet_prepare_s": time.perf_counter() - started}
        job = _TransferJob(payload, expected, self._free.popleft(),
                           time.monotonic() + self.timeout, timing, started)
        self._submitted += 1
        self._jobs.append(job)
        self._read_queue.put(job)
        self._write_queue.put(job)
        return job

    def _complete(self, job):
        timing = job.timing
        try:
            for event, message in (
                (job.write_done, "H2C write stalled; check PCIe and both Aurora links"),
                (job.read_done, "No C2H block returned; check PCIe and both Aurora links"),
            ):
                if not event.wait(max(0, job.deadline - time.monotonic())):
                    self._broken = True
                    raise TimeoutError(message)
            for key, message in (("write_error", "XDMA H2C write failed"),
                                 ("error", "XDMA C2H read failed")):
                if key in job.result:
                    self._broken = True
                    raise RuntimeError(message) from job.result[key]
            if timing is not None:
                timing["workers_s"] = time.perf_counter() - job.workers_started
                union = end = 0.0
                for begin, finish in sorted(job.read_timing["calls"] + job.write_timing["calls"]):
                    union += max(0.0, finish - max(begin, end))
                    end = max(end, finish)
                timing["syscall_union_s"] = union
                timing["outside_syscalls_s"] = max(0.0, timing["workers_s"] - union)
                for prefix, values in (("h2c", job.write_timing), ("c2h", job.read_timing)):
                    for name in ("buffer_prepare_s", "syscall_s", "result_copy_s", "total_s"):
                        timing[f"{prefix}_{name}"] = values[name]
                    timing[f"{prefix}_syscall_count"] = len(values["calls"])
                    timing[f"{prefix}_syscall_start_s"] = values["calls"][0][0] - job.workers_started
                    timing[f"{prefix}_syscall_end_s"] = values["calls"][-1][1] - job.workers_started
                timing["syscall_overlap_s"] = (
                    job.read_timing["syscall_s"] + job.write_timing["syscall_s"] - union
                )
                verification_started = time.perf_counter()
            try:
                returned = self._verify(job.payload, job.expected, job.result["data"])
            finally:
                if timing is not None:
                    timing["verify_s"] = time.perf_counter() - verification_started
            if timing is not None:
                timing["status"] = "ok"
            return returned
        except BaseException as exc:
            if timing is not None:
                timing["status"] = type(exc).__name__
            raise
        finally:
            if timing is not None:
                timing["exchange_s"] = time.perf_counter() - job.started
                self.last_profile = timing
            if job.read_done.is_set() and job.write_done.is_set():
                self._free.append(job.slot)
            self._jobs.remove(job)

    def exchange(self, payload):
        """Preserve the controller's one-block-at-a-time API."""
        if self._jobs:
            raise RuntimeError("Do not mix exchange() with an active pipeline")
        return self._complete(self._submit(payload))

    def exchange_blocks(self, payloads, *, on_mismatch=None):
        """Return (payload, received) in order with bounded outstanding bytes.

        With on_mismatch(payload, exception), report corrupt completed blocks
        and yield their returned payloads. Without it, propagate the mismatch.
        Transport errors always propagate.
        """
        if self._jobs:
            raise RuntimeError("An exchange is already active")
        pending = deque()
        iterator = iter(payloads)
        exhausted = False
        while pending or not exhausted:
            while len(pending) < self.pipeline_depth and not exhausted:
                before = time.perf_counter() if self.profile else 0
                try:
                    payload = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                source_s = time.perf_counter() - before if self.profile else 0
                job = self._submit(payload)
                if job.timing is not None:
                    job.timing["source_prepare_s"] = source_s
                pending.append(job)
            if pending:
                job = pending.popleft()
                try:
                    returned = self._complete(job)
                except OpticalDataMismatch as exc:
                    if on_mismatch is None:
                        raise
                    on_mismatch(job.payload, exc)
                    returned = exc.returned
                yield job.payload, returned


class ModeOpticalLink(OpticalLink):
    """Use the existing hardware payload gate for a whole acquisition batch."""

    def __init__(self, h2c, c2h, timeout, *, control, mode="continuous",
                 guard_ms=15.0, status=None, profile=False, pipeline_depth=1):
        super().__init__(h2c, c2h, timeout, profile=profile, pipeline_depth=pipeline_depth)
        self.control, self.mode, self.guard_ms = control, mode, guard_ms
        self.status = status or (lambda message: None)
        self._activated = False

    def __enter__(self):
        self.control.prepare(self.mode, self.guard_ms)
        try:
            return super().__enter__()
        except BaseException:
            self.control.finish("burst")
            raise

    def _submit(self, payload):
        job = super()._submit(payload)
        if not self._activated:
            self.status(f"Enabling TX2; waiting {self.guard_ms:g} ms guard" if self.mode == "burst"
                        else "Starting continuous QSFP transfer")
            try:
                # Hardware first confirms a complete packet was preloaded.
                # The I2C enable STOP then starts the FPGA guard countdown.
                self.control.start()
            except BaseException:
                self._broken = True
                raise
            self._activated = True
            self.status("Transmitting QSFP batch")
        return job

    def __exit__(self, exc_type=None, exc=None, tb=None):
        try:
            return super().__exit__(exc_type, exc, tb)
        finally:
            # A complete return is already in host RAM before laser shutdown.
            # On transport failure force darkness even in continuous mode.
            finish_mode = "burst" if self._broken or exc_type is not None else self.mode
            try:
                self.control.finish(finish_mode)
                self.status("Laser off · Burst ready" if finish_mode == "burst"
                            else "Laser on · Continuous ready")
            except Exception as close_error:
                self.status(f"QSFP shutdown error: {close_error}")
                if exc_type is None:
                    raise


def hydraharp_blocks(
    hh, duration_ms, buffer_records=1048576, poll_ms=10,
    stop_event=None, *, started=False,
):
    """Yield raw T2 records from an already initialized HydraHarp."""
    if not started and not hh.raw.startBlock(duration_ms, buffer_records, savePTU=False):
        raise RuntimeError("HydraHarp startBlock failed")

    pending = bytearray()
    stopped = False
    stop_deadline = None
    try:
        while True:
            if stop_event is not None and stop_event.is_set() and not stopped:
                hh.raw.stopMeasure()
                stopped = True
                stop_deadline = time.monotonic() + 10
            records = hh.raw.getBlock()
            if len(records):
                pending.extend(records.tobytes())
            while len(pending) >= PAYLOAD_SIZE:
                yield bytes(pending[:PAYLOAD_SIZE])
                del pending[:PAYLOAD_SIZE]
            if hh.raw.isFinished():
                # snAPI may still hold a final block when isFinished turns true.
                last_records = hh.raw.getBlock()
                if len(last_records):
                    pending.extend(last_records.tobytes())
                while len(pending) >= PAYLOAD_SIZE:
                    yield bytes(pending[:PAYLOAD_SIZE])
                    del pending[:PAYLOAD_SIZE]
                break
            if stopped and time.monotonic() >= stop_deadline:
                raise TimeoutError("HydraHarp did not finish after stopMeasure")
            time.sleep(poll_ms / 1000)
        if pending:
            yield bytes(pending)
    finally:
        if not stopped:
            hh.raw.stopMeasure()


def capture_hh_qsfp(hh, duration_ms, output, stop_event, *, started=False, progress=None,
                    mode="continuous", guard_ms=15.0, control=None, status=None):
    """Preserve all HH data and log mismatches without stopping acquisition.

    Continuous mode transmits while recording. Burst mode records the completed
    acquisition in .source.bin, then sends that file as one optical burst. Disk
    buffering avoids unbounded RAM use. Guard timing is enforced in FPGA logic.
    """
    if mode not in ("continuous", "burst"):
        raise ValueError("QSFP mode must be continuous or burst")
    if mode == "burst" and control is None:
        raise ValueError("Burst mode requires the QSFP mode control connection")
    report = status or (lambda message: None)
    output = Path(output)
    source_path = output.with_name(output.name + ".source.bin")
    log_path = output.with_name(output.name + ".error.txt")
    count = byte_count = captured_bytes = mismatch_count = 0
    error = stop_error = None
    blocks = link = None
    available = link_entered = False
    burst_control_ready = True
    with output.open("xb") as saved, source_path.open("x+b") as source, log_path.open("x", encoding="utf-8") as log:
        def record(message):
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
            log.flush()

        def open_link():
            nonlocal link, available, link_entered, error
            try:
                args = (Path("/dev/xdma0_h2c_0"), Path("/dev/xdma0_c2h_0"), 10.0)
                link = (OpticalLink(*args) if control is None else
                        ModeOpticalLink(*args, control=control, mode=mode,
                                        guard_ms=guard_ms, status=report))
                link.__enter__()
                available = link_entered = True
            except Exception as exc:
                error = f"QSFP unavailable: {exc}"
                record(error + "; HH data remains local")
                report(error)

        def transmit(index, payload, offset):
            nonlocal count, byte_count, mismatch_count, available, error
            try:
                returned = link.exchange(payload)
            except OpticalDataMismatch as exc:
                mismatch_count += 1
                record(f"Capture block {index}, source offset {offset}: {exc}")
                saved.write(exc.returned)
            except Exception as exc:
                error = f"QSFP transport failure on capture block {index}: {exc}"
                record(error + "; disabling QSFP transfers and preserving HH locally")
                report(error)
                available = False
            else:
                saved.write(returned)
                count += 1
                byte_count += len(returned)
            if progress is not None:
                progress(count, byte_count)

        record(f"Mode: {mode}; effective guard: {guard_ms if mode == 'burst' else 0:g} ms; "
               "batch: complete HH acquisition")
        record(f"Original HH data: {source_path}; QSFP returned data: {output}")
        try:
            if mode == "continuous":
                open_link()
            else:
                # The link is prepared after capture, before the first DMA.
                try:
                    control.idle("burst")
                except Exception as exc:
                    burst_control_ready = False
                    error = f"QSFP unavailable: {exc}"
                    record(error + "; continuing HH recording locally")
                    report(error)
                else:
                    report("Laser off · Recording HH locally; transmission follows acquisition")
            blocks = hydraharp_blocks(hh, duration_ms, stop_event=stop_event, started=started)
            last_update = 0.0
            for capture_index, payload in enumerate(blocks):
                source.write(payload)
                captured_bytes += len(payload)
                if mode == "continuous" and available:
                    transmit(capture_index, payload, captured_bytes - len(payload))
                if mode == "burst" and burst_control_ready and time.monotonic() - last_update >= 0.25:
                    report(f"Laser off · Recording HH: {captured_bytes:,} T2 bytes saved locally")
                    last_update = time.monotonic()
            if mode == "burst" and captured_bytes and burst_control_ready:
                source.flush()
                source.seek(0)
                open_link()
                capture_index = 0
                while available:
                    offset = source.tell()
                    payload = source.read(PAYLOAD_SIZE)
                    if not payload:
                        break
                    transmit(capture_index, payload, offset)
                    capture_index += 1
        except Exception as exc:
            error = f"{error}; capture failure: {exc}" if error else f"Capture failure: {exc}"
            record(error)
            report(error)
        finally:
            try:
                if blocks is not None:
                    blocks.close()
                else:
                    hh.raw.stopMeasure()
            except Exception as exc:
                stop_error = str(exc)
                record(f"HydraHarp stop error: {exc}")
            try:
                if link_entered:
                    link.__exit__()
            except Exception as exc:
                error = f"{error}; QSFP close: {exc}" if error else f"QSFP close: {exc}"
                record(error)
            record(f"Verified blocks: {count}; verified T2 bytes: {byte_count}; "
                   f"mismatched blocks: {mismatch_count}; original HH bytes saved: {captured_bytes}")
    if mismatch_count:
        summary = f"{mismatch_count} mismatched QSFP blocks (capture continued)"
        error = f"{summary}; {error}" if error else summary
    if error or stop_error:
        error = f"{error or 'HydraHarp stop failed'}; log: {log_path}"
        report(f"QSFP error: {error}")
    return {"blocks": count, "bytes": byte_count, "error": error,
            "stop_error": stop_error, "path": output, "source_path": source_path,
            "error_path": log_path, "mismatches": mismatch_count,
            "captured_bytes": captured_bytes, "mode": mode, "guard_ms": guard_ms}


def _hydraharp_blocks(args):
    from picoquant_snapi import MeasMode, RefSource, create_hydraharp_api

    hh = create_hydraharp_api()
    try:
        if not hh.getDevice():
            raise RuntimeError("HydraHarp not found or already in use")
        ref_source = (
            RefSource.External_10MHZ if args.ref_source == "external" else RefSource.Internal
        )
        if not hh.initDevice(MeasMode.T2, refSrc=ref_source):
            raise RuntimeError("HydraHarp T2 initialization failed")
        if args.ini and args.ini.is_file() and not hh.loadIniConfig(str(args.ini)):
            raise RuntimeError(f"Could not load {args.ini}")
        if not hh.device.setSyncCFD(300, 10):
            raise RuntimeError("Could not set HydraHarp sync CFD")
        for channel, threshold in ((0, 300), (1, 150)):
            if channel < int(hh.deviceConfig.get("NumChans", 0)):
                if not hh.device.setInputCFD(channel, threshold, 10):
                    raise RuntimeError(f"Could not set input CFD for channel {channel}")
                if not hh.device.setInputChannelOffset(channel, 0):
                    raise RuntimeError(f"Could not set input offset for channel {channel}")
        yield from hydraharp_blocks(hh, args.duration_ms, args.buffer_records, args.poll_ms)
    finally:
        hh.closeDevice()
        # snAPI.__del__ calls exitAPI() when this object is released. Calling
        # exitAPI() here as well makes some Linux snAPI builds free the native
        # API state twice during interpreter shutdown.


def main() -> None:
    global BLOCK_SIZE, PAYLOAD_SIZE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("pattern", "hydraharp"))
    parser.add_argument("--link-mode", choices=("continuous", "burst"),
                        help="pattern test using the controller mode image and live laser/gate controls")
    parser.add_argument("--guard-ms", type=float, default=15.0,
                        help="burst payload guard in ms from TX-enable I2C STOP (default: 15)")
    parser.add_argument(
        "--pattern-blocks", type=int, default=1, metavar="N",
        help="send N numbered pattern blocks (default: 1)",
    )
    parser.add_argument(
        "--thread-overhead-benchmark",
        type=int,
        metavar="N",
        help="benchmark N no-I/O reader/writer thread pairs, then exit",
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="measure pattern exchange stages and print a summary instead of per-block lines",
    )
    parser.add_argument(
        "--profile-csv", type=Path,
        help="save per-block timings to a new CSV file (requires --profile)",
    )
    parser.add_argument("--quiet", action="store_true", help="omit per-block success lines")
    parser.add_argument("--guard-sweep", action="store_true",
                        help="run the separate FPGA optical-burst diagnostic, then restore streaming")
    parser.add_argument("--guard-us", default="1000000,100000,10000,1000,100,50,20,10,8,6,4,2",
                        help="comma-separated command-to-payload guard times in us")
    parser.add_argument("--guard-trials", type=int, default=10)
    parser.add_argument("--guard-header-trace", action="store_true",
                        help="Read native receiver header/alignment counters in 100 us windows via JTAG")
    parser.add_argument("--guard-baseline-us", type=float, default=0,
                        help="Continuously illuminated baseline observation time in microseconds")
    parser.add_argument("--guard-baseline-trials", type=int, default=20)
    parser.add_argument("--guard-dark-ms", type=float, default=100)
    parser.add_argument("--guard-timeout-ms", type=float, default=100)
    parser.add_argument("--guard-reset-mode", choices=("reset", "keep", "compare"), default="reset",
                        help="reset Aurora between bursts, keep it running, or compare both")
    parser.add_argument("--guard-test", choices=("burst", "illuminated-reset", "illuminated-rx-reset", "compare", "compare-all", "burst-restart", "compare-restart", "prbs-compare", "prbs-restart", "prbs-hold"), default="burst",
                        help="toggle lasers, reset Aurora/GTs with steady light, or compare conditions")
    parser.add_argument("--guard-prbs-quiet-us", type=float, default=100,
                        help="Native PRBS error-free qualification interval (default 100 us)")
    parser.add_argument("--guard-hold-modes", default="none,equalizer,cdr,both",
                        help="comma-separated native GTY hold modes, applied to QSFP4 RX only")
    parser.add_argument("--guard-hold-release-us", default="0,10,100,500,1000,2000,5000,10000",
                        help="hardware-timed GTY hold release delays after TX-enable STOP")
    parser.add_argument("--guard-rx-release-us", default="0,100,500,1000",
                        help="comma-separated RX restart release delays after TX-enable STOP")
    parser.add_argument("--guard-output", type=Path,
                        help="new directory for sweep plan, trial CSV, summary and Vivado log")
    parser.add_argument(
        "--ber", action="store_true",
        help="pattern-only bit-error measurement using independent random payloads; "
             "continue on mismatches and report completed-packet BER even on transport failure",
    )
    parser.add_argument(
        "--error-log", type=Path,
        help="new file for streaming mismatches/transport errors; default: a timestamped "
             "qsfp_errors_*.log in the current directory (created only on error)",
    )
    parser.add_argument(
        "--memory-benchmark", action="store_true",
        help="prebuild pattern data in aligned RAM, transfer concurrently, then verify outside "
             "the timer; use --pattern-blocks for dataset size and --pipeline-depth for credits",
    )
    parser.add_argument("--benchmark-json", type=Path,
                        help="save RAM benchmark results to a new JSON file")
    parser.add_argument(
        "--low-latency", nargs="?", const=2, default=None, type=int, metavar="US",
        help="temporarily limit CPU idle exit latency while streaming (default when enabled: "
             "2 us); requires access to /dev/cpu_dma_latency",
    )
    parser.add_argument(
        "--engine", choices=("reused", "legacy"), default="reused",
        help="reuse workers/buffers (default), or use the original per-block method",
    )
    parser.add_argument(
        "--pipeline-depth", type=int, choices=(1, 2), default=1,
        help="pattern blocks in flight (default: 1); experimental depth 2 requires the reused engine",
    )
    parser.add_argument("--h2c", type=Path, default=Path("/dev/xdma0_h2c_0"))
    parser.add_argument("--c2h", type=Path, default=Path("/dev/xdma0_c2h_0"))
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument(
        "--block-size", type=int, default=BLOCK_SIZE, metavar="BYTES",
        help=f"full XDMA/Aurora block size in bytes (default: {BLOCK_SIZE})",
    )
    parser.add_argument("--output", type=Path,
                        help="write returned HH T2 records here, including corrupt returns "
                             "identified in the error log")
    parser.add_argument("--duration-ms", type=int, default=10000)
    parser.add_argument("--poll-ms", type=int, default=10)
    parser.add_argument("--buffer-records", type=int, default=1048576)
    parser.add_argument("--ref-source", choices=("external", "internal"), default="external")
    parser.add_argument(
        "--ini", type=Path,
        default=Path(__file__).with_name("HH.ini"),
    )
    args = parser.parse_args()
    if args.link_mode:
        from qsfp_mode_control import guard_ticks
        try:
            guard_ticks(args.guard_ms)
        except ValueError as exc:
            parser.error(str(exc))
        if (args.source != "pattern" or args.engine != "reused" or args.memory_benchmark
                or args.guard_sweep or args.thread_overhead_benchmark is not None):
            parser.error("--link-mode requires --source pattern --engine reused; "
                         "omit other benchmark/sweep modes")
    if args.guard_sweep:
        if (args.source != "pattern" or args.ber or args.memory_benchmark or args.profile
                or args.profile_csv or args.output or args.error_log
                or args.thread_overhead_benchmark is not None or args.low_latency is not None
                or args.engine != "reused" or args.pipeline_depth != 1):
            parser.error("--guard-sweep requires --source pattern and cannot be combined "
                         "with DMA/HH/other benchmark modes")
        from guard_sweep import run_sweep
        try:
            run_sweep(guards_us=args.guard_us, repeats=args.guard_trials,
                      dark_ms=args.guard_dark_ms, baseline_trials=args.guard_baseline_trials,
                      block_size=args.block_size, timeout_ms=args.guard_timeout_ms,
                      output_dir=args.guard_output, reset_mode=args.guard_reset_mode, test_mode=args.guard_test, rx_release_us=args.guard_rx_release_us,
                      header_trace=args.guard_header_trace, baseline_guard_us=args.guard_baseline_us, prbs_quiet_us=args.guard_prbs_quiet_us,
                      hold_modes=args.guard_hold_modes, hold_release_us=args.guard_hold_release_us)
        except (ValueError, FileNotFoundError) as exc:
            parser.error(str(exc))
        return
    if args.ber and (args.source != "pattern" or args.memory_benchmark
                     or args.thread_overhead_benchmark is not None):
        parser.error("--ber requires --source pattern and cannot be combined with other benchmarks")
    if args.ber and args.pattern_blocks > 0x100000000:
        parser.error("--ber block count must fit the 32-bit sequence field")
    if args.error_log is not None and args.error_log.exists():
        parser.error("--error-log must name a new file")
    if args.error_log is not None and not args.error_log.parent.is_dir():
        parser.error("--error-log parent directory must exist")
    if args.error_log is not None and (args.memory_benchmark or args.thread_overhead_benchmark is not None):
        parser.error("--error-log is for streaming; use --benchmark-json for the RAM benchmark")
    if args.low_latency is not None and not 0 <= args.low_latency <= 2147483647:
        parser.error("--low-latency must be between 0 and 2147483647 us")
    if args.benchmark_json is not None and not args.memory_benchmark:
        parser.error("--benchmark-json requires --memory-benchmark")
    if args.benchmark_json is not None and args.benchmark_json.exists():
        parser.error("--benchmark-json must name a new file")
    if args.memory_benchmark:
        if args.source != "pattern" or args.engine != "reused":
            parser.error("--memory-benchmark requires --source pattern --engine reused")
        if args.profile or args.profile_csv or args.output or args.thread_overhead_benchmark is not None:
            parser.error("--memory-benchmark cannot be combined with --profile, --profile-csv, "
                         "--output or --thread-overhead-benchmark")
        if args.block_size % mmap.PAGESIZE:
            parser.error("--memory-benchmark requires --block-size to be a multiple of the page size")
        if args.pattern_blocks > 0x100000000:
            parser.error("--memory-benchmark block count must fit the 32-bit sequence field")
    if args.profile_csv is not None and not args.profile:
        parser.error("--profile-csv requires --profile")
    if args.profile and (args.source != "pattern" or args.thread_overhead_benchmark is not None):
        parser.error("--profile requires --source pattern")
    if args.profile_csv is not None and args.profile_csv.exists():
        parser.error("--profile-csv must name a new file")
    if args.thread_overhead_benchmark is not None:
        if args.low_latency is not None:
            parser.error("--low-latency is only for streaming, not the thread benchmark")
        if args.thread_overhead_benchmark <= 0:
            parser.error("--thread-overhead-benchmark must be positive")
        if args.source is not None:
            parser.error("omit --source when running --thread-overhead-benchmark")
        _benchmark_thread_pair(args.thread_overhead_benchmark)
        return
    if args.source is None:
        parser.error("--source is required unless running --thread-overhead-benchmark")
    if args.block_size <= HEADER.size or (args.block_size - HEADER.size) % 4:
        parser.error("--block-size must exceed the header size and leave a payload divisible by 4")
    if args.source == "hydraharp" and args.output is None:
        parser.error("--source hydraharp requires --output")
    if (
        args.duration_ms <= 0
        or args.poll_ms <= 0
        or args.buffer_records <= 0
        or args.timeout <= 0
        or args.pattern_blocks <= 0
    ):
        parser.error("duration, poll interval, buffer size, timeout, and pattern blocks must be positive")
    if args.source == "hydraharp" and args.pattern_blocks != 1:
        parser.error("--pattern-blocks is only for --source pattern")
    if args.pipeline_depth > 1 and (args.engine != "reused" or args.source != "pattern"):
        parser.error("--pipeline-depth 2 requires --engine reused --source pattern")
    if args.engine == "reused" and args.block_size * args.pipeline_depth > RX_FIFO_BYTES:
        parser.error("total outstanding bytes must not exceed the 128 KiB FPGA RX FIFO")

    # Keep the existing 64 KiB default, but allow smaller aligned packets to
    # distinguish per-transfer failures from failures after a cumulative byte
    # count. All stream helpers use these module-level sizes at runtime.
    BLOCK_SIZE = args.block_size
    PAYLOAD_SIZE = BLOCK_SIZE - HEADER.size
    if args.memory_benchmark:
        _memory_benchmark(args)
        return

    blocks = (
        (
            (os.urandom(PAYLOAD_SIZE) if args.ber else
             (bytes(range(256)) * ((PAYLOAD_SIZE + 255) // 256))[:PAYLOAD_SIZE])
            for _ in range(args.pattern_blocks)
        )
        if args.source == "pattern"
        else _hydraharp_blocks(args)
    )
    rows = []
    open_started = time.perf_counter()
    link_class = OpticalLink if args.engine == "reused" else _LegacyOpticalLink
    options = {"profile": args.profile}
    if args.engine == "reused":
        options["pipeline_depth"] = args.pipeline_depth
    mismatch_count = 0
    ber = _BERStats() if args.ber else None
    error_log = None
    error_path = args.error_log or Path(
        f"qsfp_errors_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.log"
    )

    def record_error(message):
        nonlocal error_log
        if error_log is None:
            error_log = error_path.open("x", encoding="utf-8")
            print(f"QSFP error log: {error_path.resolve()}", flush=True)
            error_log.write(f"Source: {args.source}; block size: {BLOCK_SIZE}; "
                            f"requested pattern blocks: {args.pattern_blocks}; BER mode: {args.ber}\n")
        error_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
        error_log.flush()

    def record_mismatch(payload, exc):
        nonlocal mismatch_count
        mismatch_count += 1
        if ber is not None:
            ber.record(exc)
        record_error(str(exc))

    with ExitStack() as cleanup:
        if args.link_mode:
            from qsfp_mode_control import QSFPModeControl
            mode_control = QSFPModeControl()
            cleanup.callback(mode_control.close)
            link_class = ModeOpticalLink
            options.update(control=mode_control, mode=args.link_mode, guard_ms=args.guard_ms)
        with CpuIdleLatency(args.low_latency), link_class(args.h2c, args.c2h, args.timeout, **options) as link:
            if args.low_latency is not None:
                print(f"Low latency enabled: CPU idle exit limit {args.low_latency} us "
                      "for this streaming session")
            open_s = time.perf_counter() - open_started
            output = args.output.open("wb") if args.output is not None else None
            loop_started = time.perf_counter()
            pipeline = (
                link.exchange_blocks(blocks, on_mismatch=record_mismatch)
                if args.pipeline_depth > 1 else None
            )
            try:
                while True:
                    if pipeline is not None:
                        previous_profile = link.last_profile
                        previous_mismatches = mismatch_count
                        started = time.perf_counter()
                        try:
                            block, returned = next(pipeline)
                        except StopIteration:
                            break
                        finally:
                            if args.profile and link.last_profile is not previous_profile:
                                rows.append(dict(link.last_profile))
                        if output is not None:
                            output.write(returned)
                        if ber is not None and mismatch_count == previous_mismatches:
                            ber.record()
                        if mismatch_count == previous_mismatches and not (args.quiet or args.profile):
                            print(f"Block {link.sequence - 1}: {len(block)} bytes returned intact; "
                                  f"completion_wait={(time.perf_counter() - started) * 1000:.3f} ms")
                        continue
                    source_started = time.perf_counter() if args.profile else 0
                    try:
                        block = next(blocks)
                    except StopIteration:
                        break
                    if args.profile:
                        source_s = time.perf_counter() - source_started
                    started = time.perf_counter()
                    matched = True
                    try:
                        returned = link.exchange(block)
                    except OpticalDataMismatch as exc:
                        matched = False
                        record_mismatch(block, exc)
                        returned = exc.returned
                    finally:
                        if args.profile and link.last_profile is not None:
                            row = dict(link.last_profile)
                            row["source_prepare_s"] = source_s
                            rows.append(row)
                    elapsed = time.perf_counter() - started
                    if ber is not None and matched:
                        ber.record()
                    if output is not None:
                        output.write(returned)
                    if matched and not (args.quiet or args.profile):
                        rate = len(block) / elapsed / 1_000_000
                        print(
                            f"Block {link.sequence - 1}: {len(block)} bytes returned intact; "
                            f"exchange={elapsed * 1000:.3f} ms, payload_rate={rate:.2f} MB/s"
                        )
            except Exception as exc:
                record_error(f"Stream stopped after {link.sequence} completed blocks: "
                             f"{type(exc).__name__}: {exc}")
                if ber is not None:
                    ber.transport_failures += 1
                    print(f"BER run stopped: {type(exc).__name__}: {exc}", flush=True)
                    raise SystemExit(1) from None
                raise
            finally:
                wall_s = time.perf_counter() - loop_started
                if ber is not None:
                    summary = ber.report(args.pattern_blocks)
                    print(summary, flush=True)
                    if error_log is not None:
                        error_log.write(summary + "\n")
                if output is not None:
                    output.close()
                if error_log is not None:
                    try:
                        error_log.write(f"Completed blocks: {link.sequence}; verified: "
                                        f"{link.sequence - mismatch_count}; mismatched: {mismatch_count}\n")
                    finally:
                        error_log.close()
                if args.profile:
                    _profile_report(rows, wall_s, open_s, args.profile_csv)
    print(f"Finished: {link.sequence} blocks; verified: {link.sequence - mismatch_count}; "
          f"mismatched: {mismatch_count}")
    if mismatch_count:
        print(f"Error log: {error_path.resolve()}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
