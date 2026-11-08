"""Convert a HydraHarp T2 PTU recording to the QCS timetag HDF5 layout."""

import argparse
from pathlib import Path

import h5py
import numpy as np

from picoquant_snapi import create_hydraharp_api


def ptu_to_h5(ptu_filename, output_filename=None, chunk_size=1_000_000, compression="gzip"):
    """Unfold one PTU file and save ``t_ps`` and ``ch`` datasets to HDF5.

    The datasets and their types deliberately match ``timetag_to_h5`` in
    QCS_delay_estimator: ``t_ps`` is int64 and ``ch`` is uint8.
    """
    ptu_path = Path(ptu_filename)
    if not ptu_path.is_file():
        raise FileNotFoundError(f"PTU file was not found: {ptu_path}")

    if output_filename is None:
        output_path = ptu_path.with_name(f"{ptu_path.stem}_HH.h5")
    else:
        output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunk_size = max(1, int(chunk_size))
    reader = None
    try:
        reader = create_hydraharp_api()
        if not reader.getFileDevice(str(ptu_path)):
            raise RuntimeError(f"snAPI could not open PTU file: {ptu_path}")

        # In T2, unfolded tags are integer ticks at the device base resolution.
        # QCS ``t_ps`` is expressed in picoseconds, so convert before storing.
        resolution_ps = float(reader.deviceConfig.get("BaseResolution", 1.0))
        resolution_ps_int = int(round(resolution_ps))
        if resolution_ps_int <= 0 or not np.isclose(resolution_ps, resolution_ps_int):
            raise ValueError(
                f"Unsupported non-integer HydraHarp base resolution: {resolution_ps} ps"
            )

        # snAPI's block reader does not signal end-of-file reliably for a
        # FileDevice. A normal Unfold measurement does, and the PTU length is
        # a safe upper bound because T2 raw records are four bytes each.
        record_capacity = max(1, ptu_path.stat().st_size // 4)
        measured = reader.unfold.measure(
            1,
            record_capacity,
            waitFinished=True,
            savePTU=False,
        )
        if not measured:
            raise RuntimeError("snAPI could not unfold the PTU file.")
        times, channels = reader.unfold.getData()
        if len(times) == 0:
            raise ValueError("PTU unfolding returned no TTTR records.")

        n_total = 0
        next_progress = chunk_size
        with h5py.File(output_path, "w") as h5:
            # Keep the existing QCS timetag_to_h5 layout and core metadata.
            h5.attrs["source_format"] = "hydraharp_ptu_t2"
            h5.attrs["timestamp_unit"] = "ps"
            h5.attrs["channel_dtype"] = "uint8"
            h5.attrs["source_ptu_filename"] = str(ptu_path)
            h5.attrs["hydraharp_base_resolution_ps"] = resolution_ps

            t_ds = h5.create_dataset(
                "t_ps",
                shape=(0,),
                maxshape=(None,),
                dtype=np.int64,
                chunks=(min(chunk_size, 1_000_000),),
                compression=compression,
            )
            ch_ds = h5.create_dataset(
                "ch",
                shape=(0,),
                maxshape=(None,),
                dtype=np.uint8,
                chunks=(min(chunk_size, 1_000_000),),
                compression=compression,
            )

            # Write in HDF5-sized chunks even though snAPI has supplied the
            # complete FileDevice measurement in memory.
            for start in range(0, len(times), chunk_size):
                stop = min(start + chunk_size, len(times))
                t_chunk = np.asarray(times[start:stop], dtype=np.int64)
                t_chunk *= resolution_ps_int
                ch_chunk = np.asarray(channels[start:stop], dtype=np.uint8)
                n_new = len(t_chunk)

                t_ds.resize(n_total + n_new, axis=0)
                ch_ds.resize(n_total + n_new, axis=0)
                t_ds[n_total : n_total + n_new] = t_chunk
                ch_ds[n_total : n_total + n_new] = ch_chunk
                n_total += n_new

                while n_total >= next_progress:
                    print(f"Unfolded {next_progress:,} records")
                    next_progress += chunk_size

        print(f"Unfolded records: {n_total:,}")
        print(f"Saved H5 file: {output_path}")
        return str(output_path)
    finally:
        if reader is not None:
            try:
                reader.closeDevice()
            except Exception:
                pass
            # The installed Linux snAPI object releases its API in __del__.


def _main():
    parser = argparse.ArgumentParser(description="Convert a HydraHarp T2 PTU file to QCS-format HDF5.")
    parser.add_argument("ptu_filename", help="Input HydraHarp .ptu file")
    parser.add_argument("output_filename", help="Output .h5 file")
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    args = parser.parse_args()
    ptu_to_h5(args.ptu_filename, args.output_filename, chunk_size=args.chunk_size)


if __name__ == "__main__":
    _main()
