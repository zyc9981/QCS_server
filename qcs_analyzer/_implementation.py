"""Implementation for QCS timetag loading, alignment, and analysis.

The functions in this module are usable from a notebook, another Python
script, or the command line.  File-selection dialogs share one configurable
starting directory; change it with :func:`set_default_file_dialog_directory`
when a different acquisition folder is needed.
"""

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tkinter import Tk, filedialog
from scipy.fft import fft, fftshift, fftfreq, ifft, ifftshift
from scipy.special import erfc, erfcx, erf

try:
    from scipy.io import loadmat
except ImportError:
    loadmat = None

try:
    from scipy.optimize import curve_fit
except ImportError:
    curve_fit = None

try:
    from Swabian import TimeTagger
except ImportError:
    try:
        import TimeTagger
    except ImportError:
        TimeTagger = None


# =============================================================================
# FILE-DIALOG CONFIGURATION
# =============================================================================
# Change DEFAULT_FILE_DIALOG_DIRECTORY to update the starting folder used by
# every open/save dialog. The public setter changes it at runtime.

# Keep the current acquisition folder in one place.  All file dialogs below
# use this value, so it is the only default path that needs updating.
DEFAULT_FILE_DIALOG_DIRECTORY = Path(
    # r"C:\Yichi Zhang\GHZ_QN\WDM_GHZ_Tx5\20260803_2timetaggers_test2"
    # r"C:\Yichi Zhang\QCS\20260816_OCXO_ADEV_test"
    # r"C:\Yichi Zhang\QCS\20260818_TwoWay_2tagger_test"
    # r"C:\Yichi Zhang\QCS\20260821_TwoWay_2tagger_TTXandHH"
    # r"C:\Yichi Zhang\QCS\20260825_TwoWay_2tagger_TTXandHH"
    # r"C:\Yichi Zhang\QCS\20260831_OneWay_2tagger_TTXandHH"
    # r"C:\Yichi Zhang\QCS\20260917_TwoWay_2taggerRb"
    r"C:\Yichi Zhang\QCS\20260924_Tx1_RB4_Characterization"
)

TIMETAG_SOURCE_FILETYPES = [
    ("Timetag files", "*.txt *.csv *.bin"),
    ("Binary timetag files", "*.bin"),
    ("Text/CSV files", "*.txt *.csv"),
    ("All files", "*.*"),
]
TIMETAGGER_TTBIN_FILETYPES = [
    ("Time Tagger files", "*.ttbin"),
    ("All files", "*.*"),
]
TIMETAG_ARRAY_FILETYPES = [
    ("Timetag arrays", "*.npz *.h5 *.hdf5 *.mat"),
    ("NPZ files", "*.npz"),
    ("HDF5 files", "*.h5 *.hdf5"),
    ("MAT files", "*.mat"),
    ("All files", "*.*"),
]
NPZ_FILETYPES = [("NPZ files", "*.npz"), ("All files", "*.*")]


class FileSelectionCancelled(RuntimeError):
    """Raised when a file dialog is closed without selecting a file."""


def set_default_file_dialog_directory(directory):
    """Set the starting folder used by all subsequent file dialogs.

    The folder does not have to exist when this function is called, which is
    useful when configuring an acquisition location before it is created.
    """
    global DEFAULT_FILE_DIALOG_DIRECTORY
    DEFAULT_FILE_DIALOG_DIRECTORY = Path(directory).expanduser()


def _open_file_dialog(title, filetypes):
    """Open a file chooser using the shared acquisition-folder default."""
    root = Tk()
    root.withdraw()
    try:
        return filedialog.askopenfilename(
            title=title,
            filetypes=filetypes,
            initialdir=str(DEFAULT_FILE_DIALOG_DIRECTORY),
        )
    finally:
        root.destroy()


def _save_file_dialog(title, filetypes, default_extension):
    """Open a save-file chooser using the shared acquisition-folder default."""
    root = Tk()
    root.withdraw()
    try:
        return filedialog.asksaveasfilename(
            title=title,
            defaultextension=default_extension,
            filetypes=filetypes,
            initialdir=str(DEFAULT_FILE_DIALOG_DIRECTORY),
        )
    finally:
        root.destroy()


def _require_selected_filename(filename, action="file"):
    """Return a selected filename or raise a clear, catchable cancellation."""
    if not filename:
        raise FileSelectionCancelled(f"No {action} selected.")
    return filename


def save_dict_npz(data_dict, output_filename=None):
    """Save a dictionary to a compressed NPZ file, prompting for its location.

    Arbitrary pickleable values are supported. Load the resulting file only
    from trusted sources with ``np.load(filename, allow_pickle=True)``.
    """
    if not isinstance(data_dict, dict):
        raise TypeError("data_dict must be a dictionary.")

    if output_filename is None:
        output_filename = _save_file_dialog("Save dictionary as NPZ", NPZ_FILETYPES, ".npz")
    output_filename = _require_selected_filename(output_filename, "save file")

    output_path = Path(output_filename)
    if output_path.suffix.lower() != ".npz":
        output_path = output_path.with_suffix(".npz")

    np.savez_compressed(output_path, data_dict=np.asarray(data_dict, dtype=object))
    return str(output_path)


def load_dict_npz(input_filename=None):
    """Load a dictionary saved by :func:`save_dict_npz`.

    Only load files from trusted sources because this function enables NumPy
    pickle support to restore arbitrary dictionary values.
    """
    if input_filename is None:
        input_filename = _open_file_dialog("Load dictionary NPZ", NPZ_FILETYPES)
    input_filename = _require_selected_filename(input_filename, "results file")

    with np.load(input_filename, allow_pickle=True) as data:
        if "data_dict" not in data:
            raise ValueError("The NPZ file was not created by save_dict_npz().")
        data_dict = data["data_dict"].item()

    if not isinstance(data_dict, dict):
        raise ValueError("The saved data_dict entry is not a dictionary.")
    return data_dict


# =============================================================================
# TIMETAG FORMAT AND STREAMING HELPERS
# =============================================================================
# Private helpers for ASCII, QuTAG binary, and HDF5 timetag data. Most users
# should call load_timetag(), timetag_to_h5(), or timetag_to_h5_fast().

def _iter_timetag_chunks(filename, header_lines=1, delimiter=",", chunk_size=1_000_000):
    """Yield t_ps/ch chunks while dropping the final parsed row, matching timetag_to_npz."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    t_buf = np.empty(chunk_size, dtype=np.float64)
    ch_buf = np.empty(chunk_size, dtype=np.int64)
    n_buf = 0
    pending = None

    def parse_line(line, line_no):
        line = line.strip()
        if not line:
            return False, line_no, None, None, "empty line"

        try:
            if delimiter is None:
                parts = line.split()
            else:
                parts = next(csv.reader([line], delimiter=delimiter))
        except csv.Error as exc:
            return False, line_no, None, None, f"CSV parse error: {exc}"

        if len(parts) < 2:
            return False, line_no, None, None, "fewer than two columns"

        try:
            t_value = float(parts[0])
            ch_value = float(parts[1])
        except ValueError:
            return False, line_no, None, None, "non-numeric value"

        if not np.isfinite(t_value) or not np.isfinite(ch_value):
            return False, line_no, None, None, "non-finite value"

        return True, line_no, t_value, int(ch_value), None

    with open(filename, "r", newline="") as f:
        for line_no, line in enumerate(f, start=1):
            if line_no <= header_lines:
                continue

            current = parse_line(line, line_no)

            if pending is not None:
                ok, old_line_no, old_t, old_ch, error = pending
                if not ok:
                    raise ValueError(
                        f"Invalid row before the final dropped row at line {old_line_no}: {error}"
                    )

                t_buf[n_buf] = old_t
                ch_buf[n_buf] = old_ch
                n_buf += 1

                if n_buf == chunk_size:
                    yield t_buf.copy(), ch_buf.copy()
                    n_buf = 0

            pending = current

    if n_buf:
        yield t_buf[:n_buf].copy(), ch_buf[:n_buf].copy()


def _estimate_fixed_width_line_count(filename):
    """Estimate total lines from file size and first-line byte length."""
    file_size = Path(filename).stat().st_size

    with open(filename, "rb") as f:
        first_line = f.readline()

    if not first_line:
        return 0

    return file_size // len(first_line)


QUTAG_BINARY_HEADER_BYTES = 40
QUTAG_BINARY_RECORD_DTYPE = np.dtype([("t_ps", "<i8"), ("ch", "<i2")])
QUTAG_BINARY_RECORD_BYTES = QUTAG_BINARY_RECORD_DTYPE.itemsize


def _normalize_timetag_source_format(filename, source_format=None):
    """Return 'ascii' or 'binary' for a timetag source file."""
    if source_format is None or str(source_format).lower() == "auto":
        suffix = Path(filename).suffix.lower()
        if suffix in (".bin", ".qutagbin"):
            return "binary"
        return "ascii"

    source_format = str(source_format).lower()
    aliases = {
        "txt": "ascii",
        "text": "ascii",
        "csv": "ascii",
        "ascii": "ascii",
        "bin": "binary",
        "binary": "binary",
    }
    if source_format not in aliases:
        raise ValueError("source_format must be 'auto', 'ascii', or 'binary'.")
    return aliases[source_format]


def _estimate_qutag_binary_record_count(
    filename,
    header_bytes=QUTAG_BINARY_HEADER_BYTES,
    allow_trailing_partial_record=True,
):
    """Return the number of complete QuTAG binary records after the file header."""
    file_size = Path(filename).stat().st_size
    data_bytes = file_size - int(header_bytes)

    if data_bytes < 0:
        raise ValueError(
            f"Binary timetag file is smaller than the {header_bytes}-byte header."
        )

    record_count, trailing_bytes = divmod(data_bytes, QUTAG_BINARY_RECORD_BYTES)
    if trailing_bytes and not allow_trailing_partial_record:
        raise ValueError(
            "Binary timetag file has "
            f"{trailing_bytes} trailing byte(s), not a complete "
            f"{QUTAG_BINARY_RECORD_BYTES}-byte record."
        )

    if trailing_bytes:
        print(
            "Warning: ignoring "
            f"{trailing_bytes} trailing byte(s) after complete binary records."
        )

    return record_count


def _iter_qutag_binary_chunks(
    filename,
    chunk_size=1_000_000,
    header_bytes=QUTAG_BINARY_HEADER_BYTES,
    allow_trailing_partial_record=True,
    total_records=None,
):
    """Yield t_ps/ch chunks from QuTAG FILEFORMAT_BINARY files."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if total_records is None:
        total_records = _estimate_qutag_binary_record_count(
            filename,
            header_bytes=header_bytes,
            allow_trailing_partial_record=allow_trailing_partial_record,
        )
    remaining = total_records

    with open(filename, "rb") as f:
        f.seek(int(header_bytes))

        while remaining > 0:
            count = min(int(chunk_size), remaining)
            records = np.fromfile(f, dtype=QUTAG_BINARY_RECORD_DTYPE, count=count)
            if len(records) == 0:
                break

            t_chunk = records["t_ps"].astype(np.int64, copy=True)
            ch_chunk = records["ch"].astype(np.int64, copy=True)

            yield t_chunk, ch_chunk
            remaining -= len(records)


def _normalize_timetagger_physical_channels(channels):
    """Map Time Tagger X edge channels +/-1 through +/-4 to inputs 1 through 4.

    Swabian uses negative channel IDs for falling edges.  QCS analysis treats
    each physical Time Tagger X input as one channel, so this intentionally
    discards edge direction: for example, both ``-3`` and ``3`` become ``3``.
    Other channel IDs are left unchanged.
    """
    channels = np.asarray(channels, dtype=np.int64)
    physical_channels = np.abs(channels)
    negative_input_mask = (channels < 0) & np.isin(physical_channels, (1, 2, 3, 4))
    if not np.any(negative_input_mask):
        return channels

    normalized_channels = channels.copy()
    normalized_channels[negative_input_mask] = physical_channels[negative_input_mask]
    return normalized_channels


def _iter_timetagger_ttbin_chunks(filename, chunk_size=1_000_000):
    """Yield Time Tagger X chunks with physical inputs +/-1 through +/-4 normalized."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if TimeTagger is None:
        raise ImportError(
            "The Swabian Time Tagger Python package is required to read .ttbin files."
        )

    try:
        reader = TimeTagger.FileReader(str(filename))
    except Exception as exc:
        raise ValueError(
            "Could not open the Time Tagger X .ttbin stream. Select the "
            "FileWriter sequence header (for example, 'measurement.ttbin') "
            "after recording has been stopped."
        ) from exc

    while reader.hasData():
        buffer = reader.getData(int(chunk_size))
        timestamps = np.asarray(buffer.getTimestamps(), dtype=np.int64)
        channels = _normalize_timetagger_physical_channels(buffer.getChannels())
        if len(timestamps) != len(channels):
            raise ValueError("Time Tagger X FileReader returned mismatched timestamps and channels.")
        if len(timestamps):
            yield timestamps, channels


def _timetagger_sequence_header_filename(filename):
    """Prefer a FileWriter sequence header when a numbered block was selected."""
    path = Path(filename)
    parts = path.stem.rsplit(".", 1)
    if len(parts) == 2 and parts[1].isdigit():
        header_path = path.with_name(f"{parts[0]}{path.suffix}")
        if header_path.is_file():
            return header_path
    return path


def _matching_hydraharp_h5_filename(filename):
    """Return the HH H5 name produced alongside a QuTAG timestamp file."""
    source_path = Path(filename)
    return source_path.with_name(f"{source_path.stem}_HH.h5")


def _matching_hydraharp_qsfp_filename(filename):
    """Find the matching HydraHarp binary recording."""
    source_path = Path(filename)
    return source_path.with_name(f"{source_path.stem}.internet.bin")


def _iter_hydraharp_qsfp_chunks(
    filename, chunk_size, channel_offset=0, base_resolution_ps=1,
):
    """Decode headerless little-endian HydraHarp V2 T2 records in chunks.

    Sync is channel 0 and regular inputs start at 1, matching snAPI.
    Overflow records extend timestamps; overflow and marker records are
    excluded from the physical-input stream. Format reference:
    https://picoquant.github.io/snAPI/tttr%20data%20format.html
    """
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    resolution = float(base_resolution_ps)
    if not np.isfinite(resolution) or resolution <= 0 or not resolution.is_integer():
        raise ValueError("hydraharp_base_resolution_ps must be a positive integer in ps.")
    if Path(filename).stat().st_size % 4:
        raise ValueError(f"HydraHarp QSFP file ends with an incomplete 32-bit T2 record: {filename}")

    overflow_correction = 0
    with open(filename, "rb") as stream:
        while True:
            records = np.fromfile(stream, dtype="<u4", count=int(chunk_size))
            if not len(records):
                break
            special = (records >> 31) != 0
            channels = (records >> 25) & 0x3F
            timetags = (records & 0x1FFFFFF).astype(np.int64)
            overflow = special & (channels == 63)
            overflow_counts = np.where(overflow, np.maximum(timetags, 1), 0)
            corrections = np.cumsum(overflow_counts, dtype=np.int64) * (1 << 25)
            corrections += overflow_correction
            overflow_correction = int(corrections[-1])
            events = ~special | (channels == 0)
            if not np.any(events):
                continue
            times = (timetags + corrections)[events]
            if times.max() > np.iinfo(np.int64).max // int(resolution):
                raise ValueError("HydraHarp timestamps exceed the int64 picosecond range.")
            times *= int(resolution)
            event_channels = channels[events].astype(np.int64)
            event_channels += (~special[events]).astype(np.int64)
            event_channels += int(channel_offset)
            yield times, event_channels


def _iter_h5_timetag_chunks(
    filename,
    chunk_size,
    channel_offset=0,
    time_offset_ps=0,
):
    """Yield QCS-layout H5 timetag chunks with optional channel/time offsets."""
    with h5py.File(filename, "r") as h5:
        if "t_ps" not in h5 or "ch" not in h5:
            raise ValueError(f"H5 timetag file is missing t_ps and/or ch: {filename}")

        times = h5["t_ps"]
        channels = h5["ch"]
        if len(times) != len(channels):
            raise ValueError(f"H5 t_ps/ch lengths differ: {filename}")

        for start in range(0, len(times), chunk_size):
            stop = min(start + chunk_size, len(times))
            time_chunk = np.asarray(times[start:stop])
            channel_chunk = np.asarray(channels[start:stop], dtype=np.int64)
            if time_offset_ps:
                time_chunk = time_chunk + time_offset_ps
            if channel_offset:
                channel_chunk += int(channel_offset)
            yield time_chunk, channel_chunk


def _initial_t_ps_for_channel(chunk_iter, channel, source_name):
    """Return the earliest timestamp for a channel in a timetag chunk stream."""
    initial_t_ps = None
    for times, channels in chunk_iter:
        matches = times[channels == int(channel)]
        if len(matches):
            candidate = np.min(matches)
            if initial_t_ps is None or candidate < initial_t_ps:
                initial_t_ps = candidate

    if initial_t_ps is None:
        raise ValueError(
            f"No event found for {source_name} trigger channel {int(channel)}."
        )
    return initial_t_ps


def _channel_timestamps(chunk_iter, channel, source_name):
    """Return all timestamps for one channel in chronological order."""
    timestamp_chunks = []
    for times, channels in chunk_iter:
        matches = times[channels == int(channel)]
        if len(matches):
            timestamp_chunks.append(np.asarray(matches))

    if not timestamp_chunks:
        raise ValueError(f"No event found for {source_name} trigger channel {int(channel)}.")
    return np.sort(np.concatenate(timestamp_chunks))


def _offset_timetag_chunks(chunk_iter, time_offset_ps=0):
    """Yield chunks with a constant timestamp offset applied."""
    for times, channels in chunk_iter:
        if time_offset_ps:
            times = times + time_offset_ps
        yield times, channels


def _merge_sorted_timetag_chunks(left_chunks, right_chunks, chunk_size):
    """Merge two time-sorted timetag streams without loading either fully."""
    left_iter = iter(left_chunks)
    right_iter = iter(right_chunks)
    left_t = left_ch = right_t = right_ch = None
    left_index = right_index = 0
    left_done = right_done = False

    def load_left():
        nonlocal left_t, left_ch, left_index, left_done
        while left_t is None or left_index >= len(left_t):
            try:
                left_t, left_ch = next(left_iter)
            except StopIteration:
                left_done = True
                left_t = left_ch = None
                return
            left_index = 0

    def load_right():
        nonlocal right_t, right_ch, right_index, right_done
        while right_t is None or right_index >= len(right_t):
            try:
                right_t, right_ch = next(right_iter)
            except StopIteration:
                right_done = True
                right_t = right_ch = None
                return
            right_index = 0

    while True:
        load_left()
        load_right()
        if left_done and right_done:
            return

        out_t = []
        out_ch = []
        remaining = int(chunk_size)
        while remaining:
            load_left()
            load_right()
            if left_done and right_done:
                break

            if left_done:
                take = min(remaining, len(right_t) - right_index)
                out_t.append(right_t[right_index : right_index + take])
                out_ch.append(right_ch[right_index : right_index + take])
                right_index += take
            elif right_done:
                take = min(remaining, len(left_t) - left_index)
                out_t.append(left_t[left_index : left_index + take])
                out_ch.append(left_ch[left_index : left_index + take])
                left_index += take
            elif left_t[left_index] <= right_t[right_index]:
                stop = np.searchsorted(
                    left_t,
                    right_t[right_index],
                    side="right",
                    sorter=None,
                )
                take = min(remaining, max(1, int(stop) - left_index))
                out_t.append(left_t[left_index : left_index + take])
                out_ch.append(left_ch[left_index : left_index + take])
                left_index += take
            else:
                stop = np.searchsorted(
                    right_t,
                    left_t[left_index],
                    side="left",
                    sorter=None,
                )
                take = min(remaining, max(1, int(stop) - right_index))
                out_t.append(right_t[right_index : right_index + take])
                out_ch.append(right_ch[right_index : right_index + take])
                right_index += take
            remaining -= take

        if out_t:
            yield np.concatenate(out_t), np.concatenate(out_ch)


def _sort_h5_timetags(t_dataset, ch_dataset):
    """Sort a combined timetag H5 by t_ps and keep its channels paired."""
    times = np.asarray(t_dataset)
    channels = np.asarray(ch_dataset)
    order = np.argsort(times, kind="stable")
    if np.array_equal(order, np.arange(len(times))):
        return False

    t_dataset[:] = times[order]
    ch_dataset[:] = channels[order]
    return True


# =============================================================================
# TIME TAGGER X / QUTAG + HYDRAHARP FILE CONVERSION
# =============================================================================
# timetag_to_h5_qutagHH(), timetag_to_h5(), and timetag_to_h5_TTXonly() are
# chunked and memory-efficient.
# timetag_to_h5_fast() remains the legacy QuTAG in-memory converter.

def timetag_to_h5_qutagHH(
    filename=None,
    header_lines=1,
    delimiter=",",
    output_filename=None,
    chunk_size=1_000_000,
    compression="gzip",
    progress_every=1_000_000,
    source_format="auto",
    binary_header_bytes=QUTAG_BINARY_HEADER_BYTES,
    allow_trailing_partial_binary_record=True,
    hydraharp_filename=None,
    hydraharp_channel_offset=5,
    ch_trig_qutag=3,
    ch_trig_hh=7,
    sort_by_timestamp=True,
):
    """Merge QuTAG and HydraHarp timetags, aligned on their trigger channels."""
    if filename is None:
        filename = _open_file_dialog("Select timetag file", TIMETAG_SOURCE_FILETYPES)
    filename = _require_selected_filename(filename, "timetag file")

    if output_filename is None:
        output_filename = str(Path(filename).with_suffix(".h5"))

    if hydraharp_filename is None:
        hydraharp_filename = _matching_hydraharp_h5_filename(filename)
    hydraharp_path = Path(hydraharp_filename)
    if not hydraharp_path.is_file():
        raise FileNotFoundError(
            "Matching HydraHarp H5 file was not found: "
            f"{hydraharp_path}"
        )

    source_format = _normalize_timetag_source_format(filename, source_format)
    if source_format == "binary":
        total_items = _estimate_qutag_binary_record_count(
            filename,
            header_bytes=binary_header_bytes,
            allow_trailing_partial_record=allow_trailing_partial_binary_record,
        )
        total_label = "records"
        t_dtype = np.int64
        def make_qutag_chunks():
            return _iter_qutag_binary_chunks(
                filename,
                chunk_size=chunk_size,
                header_bytes=binary_header_bytes,
                allow_trailing_partial_record=allow_trailing_partial_binary_record,
                total_records=total_items,
            )
    else:
        total_items = _estimate_fixed_width_line_count(filename)
        total_label = "lines"
        t_dtype = np.float64
        def make_qutag_chunks():
            return _iter_timetag_chunks(
                filename,
                header_lines=header_lines,
                delimiter=delimiter,
                chunk_size=chunk_size,
            )

    print(f"Input format: {source_format}")
    print(f"Estimated total {total_label}: {total_items:,}")
    with h5py.File(hydraharp_path, "r") as hh_h5:
        if "t_ps" not in hh_h5 or "ch" not in hh_h5:
            raise ValueError(f"HydraHarp H5 is missing t_ps and/or ch: {hydraharp_path}")
        if len(hh_h5["t_ps"]) != len(hh_h5["ch"]):
            raise ValueError(f"HydraHarp H5 t_ps/ch lengths differ: {hydraharp_path}")
        hydraharp_items = len(hh_h5["t_ps"])
    if hydraharp_items == 0:
        raise ValueError(f"HydraHarp H5 contains no timetag records: {hydraharp_path}")
    print(f"HydraHarp records: {hydraharp_items:,} (channel offset +{hydraharp_channel_offset})")

    qutag_trigger_timestamps = _channel_timestamps(
        make_qutag_chunks(),
        ch_trig_qutag,
        "QuTAG",
    )
    hydraharp_trigger_timestamps = _channel_timestamps(
        _iter_h5_timetag_chunks(
            hydraharp_path,
            chunk_size=chunk_size,
            channel_offset=hydraharp_channel_offset,
        ),
        ch_trig_hh,
        "HydraHarp",
    )

    # Rebase each tagger to its own first trigger, making both triggers 0 ps.
    qutag_initial_trigger_t_ps = qutag_trigger_timestamps[0]
    hydraharp_initial_trigger_t_ps = hydraharp_trigger_timestamps[0]
    qutag_time_offset_ps = -qutag_initial_trigger_t_ps
    hydraharp_time_offset_ps = -hydraharp_initial_trigger_t_ps
    print(
        "Trigger alignment (first triggers rebased to zero): "
        f"QuTAG ch {ch_trig_qutag} at {qutag_initial_trigger_t_ps} ps; "
        f"HydraHarp ch {ch_trig_hh} at {hydraharp_initial_trigger_t_ps} ps; "
        f"QuTAG offset {qutag_time_offset_ps} ps; "
        f"HydraHarp offset {hydraharp_time_offset_ps} ps"
    )

    merged_chunks = _merge_sorted_timetag_chunks(
        _offset_timetag_chunks(
            make_qutag_chunks(),
            time_offset_ps=qutag_time_offset_ps,
        ),
        _offset_timetag_chunks(
            _iter_h5_timetag_chunks(
                hydraharp_path,
                chunk_size=chunk_size,
                channel_offset=hydraharp_channel_offset,
            ),
            time_offset_ps=hydraharp_time_offset_ps,
        ),
        chunk_size=chunk_size,
    )
    ch_info = np.iinfo(np.uint8)

    n_total = 0
    negative_timestamp_events_removed = 0
    next_progress = progress_every

    with h5py.File(output_filename, "w") as h5:
        h5.attrs["source_format"] = f"{source_format}_plus_hydraharp_h5"
        h5.attrs["timestamp_unit"] = "ps"
        h5.attrs["channel_dtype"] = "uint8"
        h5.attrs["qutag_source_filename"] = str(Path(filename))
        h5.attrs["hydraharp_source_filename"] = str(hydraharp_path)
        h5.attrs["hydraharp_channel_offset"] = int(hydraharp_channel_offset)
        h5.attrs["qutag_trigger_channel"] = int(ch_trig_qutag)
        h5.attrs["hydraharp_trigger_channel"] = int(ch_trig_hh)
        h5.attrs["qutag_trigger_event_count"] = len(qutag_trigger_timestamps)
        h5.attrs["hydraharp_trigger_event_count"] = len(hydraharp_trigger_timestamps)
        h5.attrs["qutag_first_observed_trigger_t_ps"] = qutag_trigger_timestamps[0]
        h5.attrs["hydraharp_first_observed_trigger_t_ps"] = hydraharp_trigger_timestamps[0]
        h5.attrs["qutag_initial_trigger_t_ps"] = qutag_initial_trigger_t_ps
        h5.attrs["hydraharp_initial_trigger_t_ps"] = hydraharp_initial_trigger_t_ps
        h5.attrs["alignment_reference"] = "each_tagger_first_trigger_at_zero"
        h5.attrs["qutag_time_offset_ps"] = qutag_time_offset_ps
        h5.attrs["hydraharp_time_offset_ps"] = hydraharp_time_offset_ps
        if source_format == "binary":
            h5.attrs["qutag_binary_header_bytes"] = int(binary_header_bytes)

        t_ds = h5.create_dataset(
            "t_ps",
            shape=(0,),
            maxshape=(None,),
            dtype=t_dtype,
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

        for t_chunk, ch_chunk in merged_chunks:
            nonnegative_mask = t_chunk >= 0
            if not np.all(nonnegative_mask):
                negative_timestamp_events_removed += int(
                    len(t_chunk) - np.count_nonzero(nonnegative_mask)
                )
                t_chunk = t_chunk[nonnegative_mask]
                ch_chunk = ch_chunk[nonnegative_mask]

            n_new = len(t_chunk)
            if n_new and (ch_chunk.min() < ch_info.min or ch_chunk.max() > ch_info.max):
                raise ValueError(
                    f"Channel values {ch_chunk.min()}..{ch_chunk.max()} do not fit in "
                    "uint8."
                )
            ch_chunk = ch_chunk.astype(np.uint8, copy=False)
            t_ds.resize(n_total + n_new, axis=0)
            ch_ds.resize(n_total + n_new, axis=0)
            t_ds[n_total : n_total + n_new] = t_chunk
            ch_ds[n_total : n_total + n_new] = ch_chunk
            n_total += n_new

            processed_items = n_total
            while progress_every and processed_items >= next_progress:
                print(
                    f"Merged records {next_progress:,}/"
                    f"{total_items + hydraharp_items:,}"
                )
                next_progress += progress_every

        h5.attrs["negative_timestamp_events_removed"] = negative_timestamp_events_removed

        if sort_by_timestamp:
            was_reordered = _sort_h5_timetags(t_ds, ch_ds)
            h5.attrs["timestamps_sorted"] = True
            h5.attrs["timestamp_sort_order"] = "stable ascending t_ps"
            if was_reordered:
                print("Sorted combined t_ps and applied the same order to ch.")

    if n_total == 0:
        raise ValueError("File parse returned empty data. Check input format and file contents.")

    print(f"Merged records: {n_total:,}/{total_items + hydraharp_items:,}")
    if negative_timestamp_events_removed:
        print(f"Removed {negative_timestamp_events_removed:,} event(s) with t_ps < 0.")
    print(f"Saved H5 file: {output_filename}")
    print(f"Rows saved: {n_total}")
    return output_filename


def timetag_to_h5(
    filename=None,
    output_filename=None,
    chunk_size=1_000_000,
    compression="gzip",
    progress_every=1_000_000,
    hydraharp_filename=None,
    hydraharp_channel_offset=5,
    ch_trig_time_tagger=3,
    ch_trig_hh=7,
    sort_by_timestamp=True,
):
    """Merge a Time Tagger X ``.ttbin`` stream with its HydraHarp H5 stream.

    Pass the FileWriter sequence header (``filename.ttbin``), not one of its
    numbered data blocks. The FileReader then includes ``filename.1.ttbin``,
    ``filename.2.ttbin``, and any later blocks automatically. The output H5
    retains the established ``t_ps`` (int64, ps) and ``ch`` (uint8) layout.
    Time Tagger X channels ``-1`` through ``-4`` are normalized to physical
    channels ``1`` through ``4`` before merging.
    """
    if filename is None:
        filename = _open_file_dialog(
            "Select Time Tagger X .ttbin file", TIMETAGGER_TTBIN_FILETYPES
        )
    filename = _require_selected_filename(filename, "Time Tagger X .ttbin file")

    selected_time_tagger_path = Path(filename)
    if selected_time_tagger_path.suffix.lower() != ".ttbin":
        raise ValueError("Time Tagger X input must be a FileWriter .ttbin file.")
    time_tagger_path = _timetagger_sequence_header_filename(selected_time_tagger_path)
    if not time_tagger_path.is_file():
        raise FileNotFoundError(f"Time Tagger X .ttbin file was not found: {time_tagger_path}")
    if time_tagger_path != selected_time_tagger_path:
        print(f"Using FileWriter sequence header: {time_tagger_path}")

    if output_filename is None:
        output_filename = str(time_tagger_path.with_suffix(".h5"))

    if hydraharp_filename is None:
        hydraharp_filename = _matching_hydraharp_h5_filename(time_tagger_path)
    hydraharp_path = Path(hydraharp_filename)
    if not hydraharp_path.is_file():
        raise FileNotFoundError(
            "Matching HydraHarp H5 file was not found: "
            f"{hydraharp_path}"
        )

    with h5py.File(hydraharp_path, "r") as hh_h5:
        if "t_ps" not in hh_h5 or "ch" not in hh_h5:
            raise ValueError(f"HydraHarp H5 is missing t_ps and/or ch: {hydraharp_path}")
        if len(hh_h5["t_ps"]) != len(hh_h5["ch"]):
            raise ValueError(f"HydraHarp H5 t_ps/ch lengths differ: {hydraharp_path}")
        hydraharp_items = len(hh_h5["t_ps"])
    if hydraharp_items == 0:
        raise ValueError(f"HydraHarp H5 contains no timetag records: {hydraharp_path}")

    print(f"Time Tagger X input: {time_tagger_path}")
    print(f"HydraHarp records: {hydraharp_items:,} (channel offset +{hydraharp_channel_offset})")

    time_tagger_trigger_timestamps = _channel_timestamps(
        _iter_timetagger_ttbin_chunks(time_tagger_path, chunk_size=chunk_size),
        ch_trig_time_tagger,
        "Time Tagger X",
    )
    hydraharp_trigger_timestamps = _channel_timestamps(
        _iter_h5_timetag_chunks(
            hydraharp_path,
            chunk_size=chunk_size,
            channel_offset=hydraharp_channel_offset,
        ),
        ch_trig_hh,
        "HydraHarp",
    )

    # Rebase each tagger to its own first trigger, making both triggers 0 ps.
    time_tagger_initial_trigger_t_ps = time_tagger_trigger_timestamps[0]
    hydraharp_initial_trigger_t_ps = hydraharp_trigger_timestamps[0]
    time_tagger_time_offset_ps = -time_tagger_initial_trigger_t_ps
    hydraharp_time_offset_ps = -hydraharp_initial_trigger_t_ps
    print(
        "Trigger alignment (first triggers rebased to zero): "
        f"Time Tagger X ch {ch_trig_time_tagger} at {time_tagger_initial_trigger_t_ps} ps; "
        f"HydraHarp ch {ch_trig_hh} at {hydraharp_initial_trigger_t_ps} ps; "
        f"Time Tagger X offset {time_tagger_time_offset_ps} ps; "
        f"HydraHarp offset {hydraharp_time_offset_ps} ps"
    )

    merged_chunks = _merge_sorted_timetag_chunks(
        _offset_timetag_chunks(
            _iter_timetagger_ttbin_chunks(time_tagger_path, chunk_size=chunk_size),
            time_offset_ps=time_tagger_time_offset_ps,
        ),
        _offset_timetag_chunks(
            _iter_h5_timetag_chunks(
                hydraharp_path,
                chunk_size=chunk_size,
                channel_offset=hydraharp_channel_offset,
            ),
            time_offset_ps=hydraharp_time_offset_ps,
        ),
        chunk_size=chunk_size,
    )
    ch_info = np.iinfo(np.uint8)

    n_total = 0
    negative_timestamp_events_removed = 0
    next_progress = progress_every

    with h5py.File(output_filename, "w") as h5:
        h5.attrs["source_format"] = "timetagger_x_ttbin_plus_hydraharp_h5"
        h5.attrs["timestamp_unit"] = "ps"
        h5.attrs["channel_dtype"] = "uint8"
        h5.attrs["time_tagger_x_source_filename"] = str(time_tagger_path)
        h5.attrs["time_tagger_x_edge_channels_normalized"] = "abs(channel) for inputs 1 through 4"
        h5.attrs["hydraharp_source_filename"] = str(hydraharp_path)
        h5.attrs["hydraharp_channel_offset"] = int(hydraharp_channel_offset)
        h5.attrs["time_tagger_x_trigger_channel"] = int(ch_trig_time_tagger)
        h5.attrs["hydraharp_trigger_channel"] = int(ch_trig_hh)
        h5.attrs["time_tagger_x_trigger_event_count"] = len(time_tagger_trigger_timestamps)
        h5.attrs["hydraharp_trigger_event_count"] = len(hydraharp_trigger_timestamps)
        h5.attrs["time_tagger_x_first_observed_trigger_t_ps"] = time_tagger_trigger_timestamps[0]
        h5.attrs["hydraharp_first_observed_trigger_t_ps"] = hydraharp_trigger_timestamps[0]
        h5.attrs["time_tagger_x_initial_trigger_t_ps"] = time_tagger_initial_trigger_t_ps
        h5.attrs["hydraharp_initial_trigger_t_ps"] = hydraharp_initial_trigger_t_ps
        h5.attrs["alignment_reference"] = "each_tagger_first_trigger_at_zero"
        h5.attrs["time_tagger_x_time_offset_ps"] = time_tagger_time_offset_ps
        h5.attrs["hydraharp_time_offset_ps"] = hydraharp_time_offset_ps

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

        for t_chunk, ch_chunk in merged_chunks:
            nonnegative_mask = t_chunk >= 0
            if not np.all(nonnegative_mask):
                negative_timestamp_events_removed += int(
                    len(t_chunk) - np.count_nonzero(nonnegative_mask)
                )
                t_chunk = t_chunk[nonnegative_mask]
                ch_chunk = ch_chunk[nonnegative_mask]

            n_new = len(t_chunk)
            if n_new and (ch_chunk.min() < ch_info.min or ch_chunk.max() > ch_info.max):
                raise ValueError(
                    f"Channel values {ch_chunk.min()}..{ch_chunk.max()} do not fit in "
                    "uint8."
                )
            ch_chunk = ch_chunk.astype(np.uint8, copy=False)
            t_ds.resize(n_total + n_new, axis=0)
            ch_ds.resize(n_total + n_new, axis=0)
            t_ds[n_total : n_total + n_new] = t_chunk
            ch_ds[n_total : n_total + n_new] = ch_chunk
            n_total += n_new

            while progress_every and n_total >= next_progress:
                print(f"Merged records: {next_progress:,}")
                next_progress += progress_every

        h5.attrs["negative_timestamp_events_removed"] = negative_timestamp_events_removed

        if sort_by_timestamp:
            was_reordered = _sort_h5_timetags(t_ds, ch_ds)
            h5.attrs["timestamps_sorted"] = True
            h5.attrs["timestamp_sort_order"] = "stable ascending t_ps"
            if was_reordered:
                print("Sorted combined t_ps and applied the same order to ch.")

    if n_total == 0:
        raise ValueError("Time Tagger X/HydraHarp merge returned no timetag records.")

    print(f"Merged records: {n_total:,}")
    if negative_timestamp_events_removed:
        print(f"Removed {negative_timestamp_events_removed:,} event(s) with t_ps < 0.")
    print(f"Saved H5 file: {output_filename}")
    print(f"Rows saved: {n_total}")
    return output_filename


def timetag_to_h5_sep(
    filename=None,
    output_filename=None,
    chunk_size=1_000_000,
    compression="gzip",
    progress_every=1_000_000,
    hydraharp_filename=None,
    hydraharp_channel_offset=5,
    ch_trig_time_tagger=3,
    ch_trig_hh=7,
    hydraharp_base_resolution_ps=1,
):
    """Save Time Tagger X and HydraHarp events as separate HDF5 streams.

    ``t1_ps``/``ch1`` contain Time Tagger X events and ``t2_ps``/``ch2``
    contain HydraHarp events decoded from the matching ``.internet.bin`` file. Raw input must
    use HydraHarp V2 T2 format; its base resolution defaults to 1 ps and can
    be set with ``hydraharp_base_resolution_ps``. Overflow and marker records
    are excluded. Channels are normalized/offset exactly as in
    :func:`timetag_to_h5`, and each source's first trigger is rebased to zero.
    Events remain in their source order; the two streams are never merged.
    Use ``load_timetag_sep`` and ``truncate_timetag_sep`` for this layout;
    ``load_timetag`` and ``truncate_timetag`` expect ``t_ps``/``ch`` datasets.
    """
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    if filename is None:
        filename = _open_file_dialog(
            "Select Time Tagger X .ttbin file", TIMETAGGER_TTBIN_FILETYPES
        )
    filename = _require_selected_filename(filename, "Time Tagger X .ttbin file")

    selected_time_tagger_path = Path(filename)
    if selected_time_tagger_path.suffix.lower() != ".ttbin":
        raise ValueError("Time Tagger X input must be a FileWriter .ttbin file.")
    time_tagger_path = _timetagger_sequence_header_filename(selected_time_tagger_path)
    if not time_tagger_path.is_file():
        raise FileNotFoundError(f"Time Tagger X .ttbin file was not found: {time_tagger_path}")
    if time_tagger_path != selected_time_tagger_path:
        print(f"Using FileWriter sequence header: {time_tagger_path}")

    if hydraharp_filename is None:
        hydraharp_filename = _matching_hydraharp_qsfp_filename(time_tagger_path)
    hydraharp_path = Path(hydraharp_filename)
    if not hydraharp_path.is_file():
        raise FileNotFoundError(
            f"Matching HydraHarp .internet.bin file was not found: {hydraharp_path}"
        )

    if output_filename is None:
        output_filename = time_tagger_path.with_name(f"{time_tagger_path.stem}_sep.h5")
    output_path = Path(output_filename)
    if output_path.resolve() == hydraharp_path.resolve():
        raise ValueError("Output filename must differ from the HydraHarp input file.")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")

    # Find each first trigger without retaining every trigger timestamp.
    time_tagger_initial_trigger_t_ps = _initial_t_ps_for_channel(
        _iter_timetagger_ttbin_chunks(time_tagger_path, chunk_size=chunk_size),
        ch_trig_time_tagger,
        "Time Tagger X",
    )
    hydraharp_initial_trigger_t_ps = _initial_t_ps_for_channel(
        _iter_hydraharp_qsfp_chunks(
            hydraharp_path,
            chunk_size=chunk_size,
            channel_offset=hydraharp_channel_offset,
            base_resolution_ps=hydraharp_base_resolution_ps,
        ),
        ch_trig_hh,
        "HydraHarp",
    )
    time_tagger_time_offset_ps = -time_tagger_initial_trigger_t_ps
    hydraharp_time_offset_ps = -hydraharp_initial_trigger_t_ps

    print(
        "Trigger alignment (first triggers rebased to zero): "
        f"Time Tagger X ch {ch_trig_time_tagger} at {time_tagger_initial_trigger_t_ps} ps; "
        f"HydraHarp ch {ch_trig_hh} at {hydraharp_initial_trigger_t_ps} ps"
    )

    def write_stream(h5, time_name, channel_name, chunks):
        dataset_chunk_size = min(int(chunk_size), 1_000_000)
        time_dataset = h5.create_dataset(
            time_name,
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=(dataset_chunk_size,),
            compression=compression,
        )
        channel_dataset = h5.create_dataset(
            channel_name,
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint8,
            chunks=(dataset_chunk_size,),
            compression=compression,
        )
        channel_limits = np.iinfo(np.uint8)
        count = 0
        removed = 0
        next_progress = progress_every
        for times, channels in chunks:
            nonnegative = times >= 0
            if not np.all(nonnegative):
                removed += len(times) - int(np.count_nonzero(nonnegative))
                times = times[nonnegative]
                channels = channels[nonnegative]
            if not len(times):
                continue
            if channels.min() < channel_limits.min or channels.max() > channel_limits.max:
                raise ValueError(
                    f"Channel values {channels.min()}..{channels.max()} do not fit in uint8."
                )
            new_count = count + len(times)
            time_dataset.resize(new_count, axis=0)
            channel_dataset.resize(new_count, axis=0)
            time_dataset[count:new_count] = times
            channel_dataset[count:new_count] = channels.astype(np.uint8, copy=False)
            count = new_count
            while progress_every and count >= next_progress:
                print(f"{time_name} records: {next_progress:,}")
                next_progress += progress_every
        return count, removed

    with h5py.File(output_path, "x") as h5:
        h5.attrs["source_format"] = "timetagger_x_ttbin_plus_hydraharp_qsfp_t2_separate"
        h5.attrs["hydraharp_base_resolution_ps"] = int(hydraharp_base_resolution_ps)
        h5.attrs["timestamp_unit"] = "ps"
        h5.attrs["channel_dtype"] = "uint8"
        h5.attrs["time_tagger_x_source_filename"] = str(time_tagger_path)
        h5.attrs["time_tagger_x_edge_channels_normalized"] = "abs(channel) for inputs 1 through 4"
        h5.attrs["hydraharp_source_filename"] = str(hydraharp_path)
        h5.attrs["hydraharp_channel_offset"] = int(hydraharp_channel_offset)
        h5.attrs["time_tagger_x_trigger_channel"] = int(ch_trig_time_tagger)
        h5.attrs["hydraharp_trigger_channel"] = int(ch_trig_hh)
        h5.attrs["time_tagger_x_initial_trigger_t_ps"] = time_tagger_initial_trigger_t_ps
        h5.attrs["hydraharp_initial_trigger_t_ps"] = hydraharp_initial_trigger_t_ps
        h5.attrs["alignment_reference"] = "each_tagger_first_trigger_at_zero"
        h5.attrs["time_tagger_x_time_offset_ps"] = time_tagger_time_offset_ps
        h5.attrs["hydraharp_time_offset_ps"] = hydraharp_time_offset_ps

        count1, removed1 = write_stream(
            h5,
            "t1_ps",
            "ch1",
            _offset_timetag_chunks(
                _iter_timetagger_ttbin_chunks(time_tagger_path, chunk_size=chunk_size),
                time_offset_ps=time_tagger_time_offset_ps,
            ),
        )
        count2, removed2 = write_stream(
            h5,
            "t2_ps",
            "ch2",
            _offset_timetag_chunks(
                _iter_hydraharp_qsfp_chunks(
                    hydraharp_path,
                    chunk_size=chunk_size,
                    channel_offset=hydraharp_channel_offset,
                    base_resolution_ps=hydraharp_base_resolution_ps,
                ),
                time_offset_ps=hydraharp_time_offset_ps,
            ),
        )
        h5.attrs["time_tagger_x_negative_timestamp_events_removed"] = removed1
        h5.attrs["hydraharp_negative_timestamp_events_removed"] = removed2
        h5.attrs["time_tagger_x_record_count"] = count1
        h5.attrs["hydraharp_record_count"] = count2

    if count1 == 0 or count2 == 0:
        raise ValueError("At least one tagger stream has no nonnegative timetag records.")
    print(f"Saved separate-stream H5 file: {output_path}")
    print(f"Time Tagger X records: {count1:,}; HydraHarp records: {count2:,}")
    if removed1 or removed2:
        print(f"Removed negative timestamps: Time Tagger X {removed1:,}; HydraHarp {removed2:,}.")
    return str(output_path)


def timetag_to_h5_TTXonly(
    filename=None,
    output_filename=None,
    chunk_size=1_000_000,
    compression="gzip",
    progress_every=1_000_000,
    sort_by_timestamp=True,
):
    """Convert one Time Tagger X FileWriter ``.ttbin`` stream to HDF5.

    No HydraHarp file is selected, read, or merged. Timestamps are retained
    exactly as stored by the Time Tagger X FileWriter (they are not rebased to
    a trigger). Pass the sequence header (``filename.ttbin``), not a numbered
    data block; FileReader automatically includes all ``.1.ttbin``,
    ``.2.ttbin``, and later blocks.

    The output uses the standard ``t_ps`` (int64, ps) and ``ch`` (uint8)
    datasets. Its default name is ``<recording>_TTXonly.h5`` to avoid
    overwriting the combined Time Tagger X + HydraHarp conversion. Time Tagger
    X channels ``-1`` through ``-4`` are normalized to physical channels
    ``1`` through ``4``.
    """
    if filename is None:
        filename = _open_file_dialog(
            "Select Time Tagger X .ttbin file", TIMETAGGER_TTBIN_FILETYPES
        )
    filename = _require_selected_filename(filename, "Time Tagger X .ttbin file")

    selected_time_tagger_path = Path(filename)
    if selected_time_tagger_path.suffix.lower() != ".ttbin":
        raise ValueError("Time Tagger X input must be a FileWriter .ttbin file.")
    time_tagger_path = _timetagger_sequence_header_filename(selected_time_tagger_path)
    if not time_tagger_path.is_file():
        raise FileNotFoundError(f"Time Tagger X .ttbin file was not found: {time_tagger_path}")
    if time_tagger_path != selected_time_tagger_path:
        print(f"Using FileWriter sequence header: {time_tagger_path}")

    if output_filename is None:
        output_filename = str(
            time_tagger_path.with_name(f"{time_tagger_path.stem}.h5")
        )

    print(f"Time Tagger X input: {time_tagger_path}")
    ch_info = np.iinfo(np.uint8)
    n_total = 0
    next_progress = progress_every

    with h5py.File(output_filename, "w") as h5:
        h5.attrs["source_format"] = "timetagger_x_ttbin"
        h5.attrs["timestamp_unit"] = "ps"
        h5.attrs["channel_dtype"] = "uint8"
        h5.attrs["time_tagger_x_source_filename"] = str(time_tagger_path)
        h5.attrs["time_tagger_x_edge_channels_normalized"] = "abs(channel) for inputs 1 through 4"
        h5.attrs["timestamps_rebased"] = False
        h5.attrs["hydraharp_merged"] = False

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

        for t_chunk, ch_chunk in _iter_timetagger_ttbin_chunks(
            time_tagger_path, chunk_size=chunk_size
        ):
            if ch_chunk.min() < ch_info.min or ch_chunk.max() > ch_info.max:
                raise ValueError(
                    f"Channel values {ch_chunk.min()}..{ch_chunk.max()} do not fit in uint8."
                )
            ch_chunk = ch_chunk.astype(np.uint8, copy=False)
            n_new = len(t_chunk)
            t_ds.resize(n_total + n_new, axis=0)
            ch_ds.resize(n_total + n_new, axis=0)
            t_ds[n_total : n_total + n_new] = t_chunk
            ch_ds[n_total : n_total + n_new] = ch_chunk
            n_total += n_new

            while progress_every and n_total >= next_progress:
                print(f"Converted records: {next_progress:,}")
                next_progress += progress_every

        if sort_by_timestamp:
            was_reordered = _sort_h5_timetags(t_ds, ch_ds)
            h5.attrs["timestamps_sorted"] = True
            h5.attrs["timestamp_sort_order"] = "stable ascending t_ps"
            if was_reordered:
                print("Sorted t_ps and applied the same order to ch.")
        else:
            h5.attrs["timestamps_sorted"] = False

    if n_total == 0:
        raise ValueError("Time Tagger X FileWriter stream contains no timetag records.")

    print(f"Converted records: {n_total:,}")
    print(f"Saved H5 file: {output_filename}")
    print(f"Rows saved: {n_total}")
    return output_filename


def timetag_to_h5_divider(
    filename=None,
    divider=10,
    output_filename=None,
    chunk_size=1_000_000,
    compression="gzip",
    progress_every=1_000_000,
    sort_by_timestamp=True,
    hydraharp_base_resolution_ps=1,
):
    """Save one tag out of every ``divider`` tags per channel to a new H5.

    Read a Time Tagger X .ttbin sequence, a raw HydraHarp .internet.bin file,
    or a standard H5/HDF5/NPZ/MAT file containing t_ps and ch.
    Each channel independently keeps its first tag,
    then tags 1+divider, 1+2*divider, etc., in input order. Counts continue
    across chunks. Kept timestamps are unchanged; no second tagger is required.

    HydraHarp input must be headerless little-endian V2 T2 data. Overflow
    records extend timestamps, markers are discarded, and sync is channel 0
    with physical inputs starting at 1 (no channel offset or time rebasing).
    Set hydraharp_base_resolution_ps to the recording's resolution in integer
    picoseconds; the default is 1 ps. Sync tags are divided independently too.

    The default output is <recording>_div<divider>.h5. Existing outputs are
    never overwritten. H5 and raw inputs are read in chunks; NPZ/MAT inputs
    use the existing in-memory loader. Final sorting loads the retained data;
    disable it for chronological inputs to limit memory use. Output retains
    input numeric dtypes (raw inputs use int64 timestamps and uint8 channels).
    """
    for name, value, minimum in (
        ("divider", divider, 1),
        ("chunk_size", chunk_size, 1),
        ("progress_every", progress_every, 0),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    divider = int(divider)
    if filename is None:
        filename = _open_file_dialog(
            "Select timetag file to divide",
            [
                ("Timetag files", "*.ttbin *.internet.bin *.h5 *.hdf5 *.npz *.mat"),
                ("HydraHarp raw files", "*.internet.bin"),
                ("All files", "*.*"),
            ],
        )
    filename = _require_selected_filename(filename, "timetag file")
    source = Path(filename)
    suffix = source.suffix.lower()
    is_hydraharp = source.name.lower().endswith(".internet.bin")
    if suffix == ".ttbin":
        source = _timetagger_sequence_header_filename(source)
    if not source.is_file():
        raise FileNotFoundError(f"Timetag file was not found: {source}")

    if suffix == ".ttbin":
        t_dtype, ch_dtype = np.dtype(np.int64), np.dtype(np.uint8)
        chunks = _iter_timetagger_ttbin_chunks(source, chunk_size=chunk_size)
    elif is_hydraharp:
        t_dtype, ch_dtype = np.dtype(np.int64), np.dtype(np.uint8)
        chunks = _iter_hydraharp_qsfp_chunks(
            source, chunk_size=chunk_size,
            base_resolution_ps=hydraharp_base_resolution_ps,
        )
    elif suffix in (".h5", ".hdf5"):
        with h5py.File(source, "r") as h5:
            if "t_ps" not in h5 or "ch" not in h5:
                raise ValueError("Input H5 must contain t_ps and ch datasets.")
            if h5["t_ps"].ndim != 1 or h5["ch"].ndim != 1:
                raise ValueError("t_ps and ch must be one-dimensional.")
            if len(h5["t_ps"]) != len(h5["ch"]):
                raise ValueError("t_ps and ch must have the same length.")
            t_dtype, ch_dtype = h5["t_ps"].dtype, h5["ch"].dtype

        def h5_chunks():
            with h5py.File(source, "r") as h5:
                for start in range(0, len(h5["t_ps"]), chunk_size):
                    yield h5["t_ps"][start:start + chunk_size], h5["ch"][start:start + chunk_size]

        chunks = h5_chunks()
    elif suffix in (".npz", ".mat"):
        t_ps, ch = _load_timetag_arrays(source)
        if len(t_ps) != len(ch):
            raise ValueError("t_ps and ch must have the same length.")
        t_dtype, ch_dtype = t_ps.dtype, ch.dtype
        chunks = (
            (t_ps[start:start + chunk_size], ch[start:start + chunk_size])
            for start in range(0, len(t_ps), chunk_size)
        )
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    if not (np.issubdtype(t_dtype, np.integer) or np.issubdtype(t_dtype, np.floating)):
        raise ValueError("Timestamps must have a real numeric dtype.")
    if not np.issubdtype(ch_dtype, np.integer):
        raise ValueError("Channels must have an integer dtype.")
    output_path = (
        Path(output_filename) if output_filename is not None
        else source.with_name(f"{source.stem}_div{divider}.h5")
    )
    channel_counts = {}
    n_input = n_saved = 0
    next_progress = progress_every
    with h5py.File(output_path, "x") as h5:
        h5.attrs["source_filename"] = str(source)
        h5.attrs["timestamp_unit"] = "ps"
        h5.attrs["channel_record_divider"] = divider
        h5.attrs["divider_first_tag_retained"] = True
        if is_hydraharp:
            h5.attrs["source_format"] = "hydraharp_v2_t2"
            h5.attrs["hydraharp_base_resolution_ps"] = hydraharp_base_resolution_ps
            h5.attrs["hydraharp_channel_offset"] = 0
            h5.attrs["timestamps_rebased"] = False
        t_ds = h5.create_dataset(
            "t_ps", shape=(0,), maxshape=(None,), dtype=t_dtype,
            chunks=(min(chunk_size, 1_000_000),), compression=compression,
        )
        ch_ds = h5.create_dataset(
            "ch", shape=(0,), maxshape=(None,), dtype=ch_dtype,
            chunks=(min(chunk_size, 1_000_000),), compression=compression,
        )
        for t_chunk, ch_chunk in chunks:
            if suffix == ".ttbin" and len(ch_chunk):
                if ch_chunk.min() < 0 or ch_chunk.max() > 255:
                    raise ValueError("Time Tagger channel values must fit in uint8.")
            n_input += len(t_chunk)
            keep = np.zeros(len(ch_chunk), dtype=bool)
            for channel in np.unique(ch_chunk):
                indices = np.flatnonzero(ch_chunk == channel)
                count = channel_counts.get(int(channel), 0)
                keep[indices[(-count) % divider::divider]] = True
                channel_counts[int(channel)] = count + len(indices)
            n_new = int(np.count_nonzero(keep))
            t_ds.resize(n_saved + n_new, axis=0)
            ch_ds.resize(n_saved + n_new, axis=0)
            t_ds[n_saved:n_saved + n_new] = t_chunk[keep]
            ch_ds[n_saved:n_saved + n_new] = ch_chunk[keep]
            n_saved += n_new
            while progress_every and n_input >= next_progress:
                print(f"Processed input records: {next_progress:,}; saved: {n_saved:,}")
                next_progress += progress_every
        if sort_by_timestamp:
            _sort_h5_timetags(t_ds, ch_ds)
            h5.attrs["timestamp_sort_order"] = "stable ascending t_ps"
        h5.attrs["timestamps_sorted"] = bool(sort_by_timestamp)
        h5.attrs["input_records"] = n_input
        h5.attrs["saved_records"] = n_saved

    print(f"Divider: {divider} per channel; kept {n_saved:,} of {n_input:,} tags.")
    print(f"Saved H5 file: {output_path}")
    return str(output_path)


def timetag_to_h5_qutag(
    filename=None,
    header_lines=1,
    delimiter=",",
    output_filename=None,
    chunk_size=1_000_000,
    compression="gzip",
    progress_every=1_000_000,
    source_format="auto",
    binary_header_bytes=QUTAG_BINARY_HEADER_BYTES,
    allow_trailing_partial_binary_record=True,
    skip=10,
    sort_by_timestamp=True,
):
    """Convert QuTAG timetags to QCS-layout H5 with per-channel thinning.

    ``skip`` retains the first event and every ``skip``-th event thereafter
    for each channel independently.  For example, ``skip=10`` retains one
    event for every ten events on each channel.
    """
    if filename is None:
        filename = _open_file_dialog("Select timetag file", TIMETAG_SOURCE_FILETYPES)
    filename = _require_selected_filename(filename, "timetag file")
    if isinstance(skip, bool) or not isinstance(skip, (int, np.integer)) or skip < 1:
        raise ValueError("skip must be a positive integer.")
    skip = int(skip)

    if output_filename is None:
        output_filename = str(Path(filename).with_suffix(".h5"))

    source_format = _normalize_timetag_source_format(filename, source_format)
    if source_format == "binary":
        total_items = _estimate_qutag_binary_record_count(
            filename,
            header_bytes=binary_header_bytes,
            allow_trailing_partial_record=allow_trailing_partial_binary_record,
        )
        total_label = "records"
        t_dtype = np.int64
        def make_qutag_chunks():
            return _iter_qutag_binary_chunks(
                filename,
                chunk_size=chunk_size,
                header_bytes=binary_header_bytes,
                allow_trailing_partial_record=allow_trailing_partial_binary_record,
                total_records=total_items,
            )
    else:
        total_items = _estimate_fixed_width_line_count(filename)
        total_label = "lines"
        t_dtype = np.float64
        def make_qutag_chunks():
            return _iter_timetag_chunks(
                filename,
                header_lines=header_lines,
                delimiter=delimiter,
                chunk_size=chunk_size,
            )

    print(f"Input format: {source_format}")
    print(f"Estimated total {total_label}: {total_items:,}")
    ch_info = np.iinfo(np.uint8)

    channel_record_counts = np.zeros(ch_info.max + 1, dtype=np.int64)
    n_input = 0
    n_total = 0
    next_progress = progress_every

    with h5py.File(output_filename, "w") as h5:
        h5.attrs["source_format"] = source_format
        h5.attrs["timestamp_unit"] = "ps"
        h5.attrs["channel_dtype"] = "uint8"
        h5.attrs["qutag_source_filename"] = str(Path(filename))
        h5.attrs["channel_record_skip"] = skip
        if source_format == "binary":
            h5.attrs["qutag_binary_header_bytes"] = int(binary_header_bytes)

        t_ds = h5.create_dataset(
            "t_ps",
            shape=(0,),
            maxshape=(None,),
            dtype=t_dtype,
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

        for t_chunk, ch_chunk in make_qutag_chunks():
            n_input += len(t_chunk)
            if len(t_chunk) and (ch_chunk.min() < ch_info.min or ch_chunk.max() > ch_info.max):
                raise ValueError(
                    f"Channel values {ch_chunk.min()}..{ch_chunk.max()} do not fit in "
                    "uint8."
                )
            ch_chunk = ch_chunk.astype(np.uint8, copy=False)

            retained_event_mask = np.zeros(len(ch_chunk), dtype=bool)
            for channel in np.unique(ch_chunk):
                channel_indices = np.flatnonzero(ch_chunk == channel)
                first_retained_index = (-channel_record_counts[channel]) % skip
                retained_event_mask[channel_indices[first_retained_index::skip]] = True
                channel_record_counts[channel] += len(channel_indices)

            t_chunk = t_chunk[retained_event_mask]
            ch_chunk = ch_chunk[retained_event_mask]
            n_new = len(t_chunk)
            t_ds.resize(n_total + n_new, axis=0)
            ch_ds.resize(n_total + n_new, axis=0)
            t_ds[n_total : n_total + n_new] = t_chunk
            ch_ds[n_total : n_total + n_new] = ch_chunk
            n_total += n_new

            processed_items = n_input
            while progress_every and processed_items >= next_progress:
                print(
                    f"Processed input records {next_progress:,}/{total_items:,}"
                )
                next_progress += progress_every

        if sort_by_timestamp:
            was_reordered = _sort_h5_timetags(t_ds, ch_ds)
            h5.attrs["timestamps_sorted"] = True
            h5.attrs["timestamp_sort_order"] = "stable ascending t_ps"
            if was_reordered:
                print("Sorted QuTAG t_ps and applied the same order to ch.")

    if n_total == 0:
        raise ValueError("File parse returned empty data. Check input format and file contents.")

    print(f"Processed input records: {n_input:,}/{total_items:,}")
    print(f"Saved H5 file: {output_filename}")
    print(f"Rows saved: {n_total}")
    return output_filename


def _load_timetag_arrays(filename):
    """Load t_ps and ch arrays from .npz or MATLAB .mat files."""
    filename_path = Path(filename)

    if filename_path.suffix.lower() == ".npz":
        with np.load(filename) as f:
            print("Variables in file:", list(f.keys()))
            t_ps = np.array(f["t_ps"]).squeeze()
            ch = np.array(f["ch"]).squeeze()
        return t_ps.ravel(), ch.ravel()

    if filename_path.suffix.lower() in (".h5", ".hdf5"):
        with h5py.File(filename, "r") as f:
            print("Variables in file:", list(f.keys()))
            t_ps = np.array(f["t_ps"]).squeeze()
            ch = np.array(f["ch"]).squeeze()
        return t_ps.ravel(), ch.ravel()

    if filename_path.suffix.lower() != ".mat":
        raise ValueError(f"Unsupported file type: {filename_path.suffix}")

    try:
        with h5py.File(filename, "r") as f:
            print("Variables in file:", list(f.keys()))
            t_ps = np.array(f["t_ps"]).squeeze()
            ch = np.array(f["ch"]).squeeze()
    except OSError:
        if loadmat is None:
            raise ImportError(
                "This MAT file is not HDF5/v7.3. Install scipy to read older MAT files."
            )
        f = loadmat(filename)
        print("Variables in file:", [key for key in f.keys() if not key.startswith("__")])
        t_ps = np.array(f["t_ps"]).squeeze()
        ch = np.array(f["ch"]).squeeze()

    return t_ps.ravel(), ch.ravel()


# =============================================================================
# TIMETAG LOADING AND CLOCK ALIGNMENT
# =============================================================================
# load_timetag() reads an existing file. The next two functions inspect a
# channel delay and correct QuTAG/HydraHarp clock alignment.

def timetag_channel_time_difference(
    t_ps,
    ch,
    ch_a=2,
    ch_b=7,
    show=True,
    bin_s=0.1,
):
    """Plot channel delay and optional clock-frequency skew.

    The timestamps for each channel are sorted chronologically and paired by
    event order. If the channels contain different numbers of events, the
    longer one is truncated to the shorter one before calculating
    ``t_ps[ch_a] - t_ps[ch_b]``. All valid pairs within each ``bin_s``
    recording-time interval are averaged into one upper-subplot time-offset point.
    Those same pairs are fitted to ``tau0 + u0*t + 0.5*u1*t**2``, where ``t``
    is measured from the mean time within that bin. The lower two subplots show
    ``u0`` (clock-frequency skew, us/s) and ``u1`` (skew rate, ns/s²).
    """
    t_ps = np.asarray(t_ps).ravel()
    ch = np.asarray(ch).ravel()
    if len(t_ps) != len(ch):
        raise ValueError("t_ps and ch must contain the same number of records.")
    try:
        bin_s = float(bin_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("bin_s must be zero or a positive duration in seconds.") from exc
    if not np.isfinite(bin_s) or bin_s < 0:
        raise ValueError("bin_s must be zero or a positive duration in seconds.")

    ch_a = int(ch_a)
    ch_b = int(ch_b)
    t_a = np.sort(t_ps[ch == ch_a])
    t_b = np.sort(t_ps[ch == ch_b])
    if len(t_a) == 0 or len(t_b) == 0:
        raise ValueError(
            f"Both channels must contain events; ch {ch_a}: {len(t_a)}, "
            f"ch {ch_b}: {len(t_b)}."
        )

    pair_count = min(len(t_a), len(t_b))
    t_a = t_a[:pair_count]
    t_b = t_b[:pair_count]
    difference_ps = t_a - t_b

    if len(t_a) != len(t_b):  # Defensive: both slices should always match.
        raise RuntimeError("Could not create equal-length channel timestamp arrays.")

    recording_time_s = t_b.astype(np.float64) * 1e-12
    figure, (axis, skew_axis, skew_rate_axis) = plt.subplots(
        3, 1, sharex=True, figsize=(12, 9)
    )
    raw_plot_mode = bin_s == 0
    if raw_plot_mode:
        axis.set_ylabel("Timestamp difference (raw pairs, ps)")
    else:
        axis.set_ylabel(f"Mean timestamp difference ({bin_s:g} s bins, ps)")
    axis.grid(True, alpha=0.3)
    clock_skew_us_per_s = []
    clock_skew_rate_us_per_s2 = []
    skew_time_s = []
    binned_time_s = []
    binned_time_difference_ps = []
    binned_pair_counts = []
    valid_pairs = np.isfinite(recording_time_s) & np.isfinite(difference_ps)
    valid_indices = np.flatnonzero(valid_pairs)
    if raw_plot_mode:
        # Preserve every valid chronological pair exactly as calculated above.
        # The existing binned_* attributes expose the plotted values in either
        # mode; in raw mode every point is a single pair.
        binned_time_s = recording_time_s[valid_indices].astype(float)
        binned_time_difference_ps = difference_ps[valid_indices].astype(np.float64)
        binned_pair_counts = np.ones(len(valid_indices), dtype=np.int64)
    elif len(valid_indices):
        valid_time_s = recording_time_s[valid_indices]
        relative_bin_position = (valid_time_s - valid_time_s[0]) / bin_s
        # Advance by one floating-point step so a sample exactly on a bin edge
        # belongs to the following bin, without moving genuinely earlier data.
        bin_numbers = np.floor(
            np.nextafter(relative_bin_position, np.inf)
        ).astype(np.int64)

        for bin_number in np.unique(bin_numbers):
            bin_indices = valid_indices[bin_numbers == bin_number]
            bin_time_s = recording_time_s[bin_indices]
            bin_delay_us = difference_ps[bin_indices].astype(np.float64) * 1e-6
            binned_time_s.append(np.mean(bin_time_s))
            binned_time_difference_ps.append(
                np.mean(difference_ps[bin_indices].astype(np.float64))
            )
            binned_pair_counts.append(len(bin_indices))

            if len(bin_indices) < 3:
                continue

            centered_time_s = bin_time_s - np.mean(bin_time_s)
            # Coefficients map directly to tau0 + u0*t + 0.5*u1*t**2.
            design_matrix = np.column_stack(
                (np.ones(len(centered_time_s)), centered_time_s, 0.5 * centered_time_s**2)
            )
            if np.linalg.matrix_rank(design_matrix) < 3:
                continue

            tau0_us, u0_us_per_s, u1_us_per_s2 = np.linalg.lstsq(
                design_matrix, bin_delay_us, rcond=None
            )[0]
            clock_skew_us_per_s.append(u0_us_per_s)
            clock_skew_rate_us_per_s2.append(u1_us_per_s2)
            skew_time_s.append(np.mean(bin_time_s))

    clock_skew_us_per_s = np.asarray(clock_skew_us_per_s, dtype=float)
    clock_skew_rate_us_per_s2 = np.asarray(clock_skew_rate_us_per_s2, dtype=float)
    clock_skew_rate_ns_per_s2 = clock_skew_rate_us_per_s2 * 1e3
    skew_time_s = np.asarray(skew_time_s, dtype=float)
    binned_time_s = np.asarray(binned_time_s, dtype=float)
    binned_time_difference_ps = np.asarray(binned_time_difference_ps, dtype=float)
    binned_pair_counts = np.asarray(binned_pair_counts, dtype=np.int64)
    if len(binned_time_s):
        axis.plot(
            binned_time_s,
            binned_time_difference_ps,
            "o-",
            color="tab:blue",
            markersize=3,
            alpha=0.8,
        )
    if len(clock_skew_us_per_s):
        skew_axis.plot(
            skew_time_s,
            clock_skew_us_per_s,
            "o-",
            color="tab:red",
            markersize=3,
            alpha=0.7,
        )
    if raw_plot_mode:
        skew_axis.set_ylabel("u0: not calculated in raw mode", color="tab:red")
    else:
        skew_axis.set_ylabel(
            f"u0: clock-frequency skew ({bin_s:g} s bins, us/s)", color="tab:red"
        )
    skew_axis.tick_params(axis="y", labelcolor="tab:red")
    skew_axis.grid(True, alpha=0.3)
    if len(clock_skew_rate_ns_per_s2):
        skew_rate_axis.plot(
            skew_time_s,
            clock_skew_rate_ns_per_s2,
            "o-",
            color="tab:green",
            markersize=3,
            alpha=0.7,
        )
    skew_rate_axis.set_xlabel("Recording Time (s)")
    if raw_plot_mode:
        skew_rate_axis.set_ylabel("u1: not calculated in raw mode", color="tab:green")
    else:
        skew_rate_axis.set_ylabel(
        f"u1: clock-skew rate ({bin_s:g} s bins, ns/s²)", color="tab:green"
        )
    skew_rate_axis.tick_params(axis="y", labelcolor="tab:green")
    skew_rate_axis.grid(True, alpha=0.3)
    # Make the derived data accessible without changing the existing return tuple.
    axis.clock_skew_axis = skew_axis
    axis.clock_skew_rate_axis = skew_rate_axis
    axis.clock_skew_us_per_s = clock_skew_us_per_s
    axis.clock_skew_rate_us_per_s2 = clock_skew_rate_us_per_s2
    axis.clock_skew_rate_ns_per_s2 = clock_skew_rate_ns_per_s2
    axis.clock_skew_time_s = skew_time_s
    axis.binned_time_s = binned_time_s
    axis.binned_time_difference_ps = binned_time_difference_ps
    axis.binned_pair_counts = binned_pair_counts
    axis.bin_s = bin_s
    figure.tight_layout()
    if show:
        plt.show()

    return difference_ps, t_a, t_b, figure, axis


def timetag_correction(
    t_ps,
    ch,
    ch_qutag=2,
    ch_hh=7,
    qutag_channel_max=4,
    correction_mode="per_trigger",
    return_parameters=False,
):
    """Correct QuTAG clock offset and elapsed-time scale against HydraHarp.

    The first trigger from each tagger is used as the common time origin;
    trigger events are then paired by chronological order. By default, every
    adjacent trigger pair defines a local linear
    mapping (``correction_mode="per_trigger"``), so each matched trigger is
    aligned exactly. Set ``correction_mode="linear_fit"`` for a single
    least-squares linear fit across all trigger pairs, or
    ``correction_mode="global"`` to determine one scale from only the first
    and last paired triggers. Set ``correction_mode="first"`` to align only
    the first trigger from each tagger without changing the clock slope. The
    correction is applied to QuTAG
    channels from 0 through ``qutag_channel_max``; HydraHarp channels are
    unchanged. The shorter trigger array determines how many trigger pairs
    are used. In ``global`` mode, the returned ``t_ps`` and ``ch`` arrays are
    limited to the corrected time interval shared by both taggers. The other
    correction modes retain all supplied events.
    ``per_trigger``, ``linear_fit``, and ``global`` require at least two
    paired trigger events; ``first`` requires only one per tagger.
    """
    t_ps = np.asarray(t_ps).ravel()
    ch = np.asarray(ch).ravel()
    if len(t_ps) != len(ch):
        raise ValueError("t_ps and ch must contain the same number of records.")

    ch_qutag = int(ch_qutag)
    ch_hh = int(ch_hh)
    correction_mode = str(correction_mode).lower()
    if correction_mode not in ("per_trigger", "linear_fit", "global", "first"):
        raise ValueError(
            "correction_mode must be 'per_trigger', 'linear_fit', 'global', or 'first'."
        )

    if correction_mode == "first":
        qutag_trigger_timestamps = t_ps[ch == ch_qutag]
        hydraharp_trigger_timestamps = t_ps[ch == ch_hh]
        if len(qutag_trigger_timestamps):
            qutag_trigger_timestamps = np.array(
                [np.min(qutag_trigger_timestamps)], dtype=np.float64
            )
        if len(hydraharp_trigger_timestamps):
            hydraharp_trigger_timestamps = np.array(
                [np.min(hydraharp_trigger_timestamps)], dtype=np.float64
            )
    else:
        qutag_trigger_timestamps = np.sort(t_ps[ch == ch_qutag].astype(np.float64))
        hydraharp_trigger_timestamps = np.sort(t_ps[ch == ch_hh].astype(np.float64))
    if len(qutag_trigger_timestamps) == 0 or len(hydraharp_trigger_timestamps) == 0:
        raise ValueError(
            f"Both trigger channels must contain events; QuTAG ch {ch_qutag}: "
            f"{len(qutag_trigger_timestamps)}, HydraHarp ch {ch_hh}: "
            f"{len(hydraharp_trigger_timestamps)}."
        )

    # Always align the first observed trigger from each tagger.  Do not apply
    # the former temporary extra-trigger skip workaround here.
    qutag_trigger_skip_count = 0
    hydraharp_trigger_skip_count = 0
    pair_count = min(len(qutag_trigger_timestamps), len(hydraharp_trigger_timestamps))
    if pair_count < 1:
        raise ValueError("At least one paired trigger event is required for timebase correction.")
    if correction_mode != "first" and pair_count < 2:
        raise ValueError(
            "At least two paired trigger events are required for per_trigger, "
            "linear_fit, or global correction."
        )

    qutag_trigger_pairs = qutag_trigger_timestamps[:pair_count]
    hydraharp_trigger_pairs = hydraharp_trigger_timestamps[:pair_count]
    qutag_reference_t_ps = qutag_trigger_pairs[0]
    hydraharp_reference_t_ps = hydraharp_trigger_pairs[0]
    if correction_mode == "first":
        qutag_elapsed_ps = np.array([0.0])
        hydraharp_elapsed_ps = np.array([0.0])
        local_time_scales = np.array([], dtype=float)
        overall_time_scale = 1.0
        trigger_pairs_used = 1
    else:
        qutag_elapsed_ps = qutag_trigger_pairs - qutag_reference_t_ps
        hydraharp_elapsed_ps = hydraharp_trigger_pairs - hydraharp_reference_t_ps
        qutag_interval_ps = np.diff(qutag_trigger_pairs)
        hydraharp_interval_ps = np.diff(hydraharp_trigger_pairs)
        if np.any(qutag_interval_ps <= 0):
            raise ValueError("QuTAG trigger timestamps do not span elapsed time.")
        local_time_scales = hydraharp_interval_ps / qutag_interval_ps
        overall_time_scale = float(
            hydraharp_elapsed_ps[-1] / qutag_elapsed_ps[-1]
        )
        trigger_pairs_used = pair_count
    corrected_t_ps = t_ps.astype(np.float64, copy=True)
    qutag_event_mask = (ch >= 0) & (ch <= int(qutag_channel_max))
    qutag_event_timestamps = corrected_t_ps[qutag_event_mask]
    applied_time_scale = overall_time_scale
    if correction_mode == "first":
        corrected_t_ps[qutag_event_mask] = (
            qutag_event_timestamps - qutag_reference_t_ps + hydraharp_reference_t_ps
        )
    elif correction_mode == "per_trigger":
        # Use the scale for the trigger interval containing each event. Values
        # before/after the trigger range use the first/last interval to
        # extrapolate linearly.
        interval_indices = np.searchsorted(
            qutag_trigger_pairs,
            qutag_event_timestamps,
            side="right",
        ) - 1
        interval_indices = np.clip(interval_indices, 0, pair_count - 2)
        corrected_t_ps[qutag_event_mask] = (
            hydraharp_trigger_pairs[interval_indices]
            + (qutag_event_timestamps - qutag_trigger_pairs[interval_indices])
            * local_time_scales[interval_indices]
        )
    elif correction_mode == "linear_fit":
        elapsed_norm = np.dot(qutag_elapsed_ps, qutag_elapsed_ps)
        if elapsed_norm == 0:
            raise ValueError("QuTAG trigger timestamps do not span elapsed time.")
        time_scale = float(np.dot(qutag_elapsed_ps, hydraharp_elapsed_ps) / elapsed_norm)
        applied_time_scale = time_scale
        corrected_t_ps[qutag_event_mask] = (
            (qutag_event_timestamps - qutag_reference_t_ps) * time_scale
            + hydraharp_reference_t_ps
        )
    elif correction_mode == "global":
        # Deliberately use only the first pair for the common origin and the
        # last pair for one elapsed-time scale; intermediate pairs are ignored.
        corrected_t_ps[qutag_event_mask] = (
            (qutag_event_timestamps - qutag_reference_t_ps) * overall_time_scale
            + hydraharp_reference_t_ps
        )

    # Express both taggers relative to their aligned first trigger.
    corrected_t_ps -= hydraharp_reference_t_ps
    hydraharp_event_mask = ~qutag_event_mask
    qutag_duration_ps = float(np.max(corrected_t_ps[qutag_event_mask]))
    hydraharp_duration_ps = float(np.max(corrected_t_ps[hydraharp_event_mask]))
    shared_duration_ps = min(qutag_duration_ps, hydraharp_duration_ps)
    if correction_mode == "global":
        # A global mapping uses the first and last matched trigger pair.  Do
        # not extrapolate either tagger past the common corrected recording
        # interval; doing so can leave unmatched reference pulses at the tail.
        shared_interval_mask = (
            (corrected_t_ps >= 0.0) & (corrected_t_ps <= shared_duration_ps)
        )
        removed_outside_shared_interval = int(
            len(corrected_t_ps) - np.count_nonzero(shared_interval_mask)
        )
        corrected_t_ps = corrected_t_ps[shared_interval_mask]
        corrected_ch = ch[shared_interval_mask].copy()
    else:
        removed_outside_shared_interval = 0
        corrected_ch = ch.copy()
    # chronological_order = np.argsort(corrected_t_ps, kind="stable")
    # corrected_t_ps = corrected_t_ps[chronological_order]
    # corrected_ch = corrected_ch[chronological_order]

    correction = {
        "correction_mode": correction_mode,
        "time_scale": applied_time_scale,
        "local_time_scales": local_time_scales,
        "qutag_reference_t_ps": qutag_reference_t_ps,
        "hydraharp_reference_t_ps": hydraharp_reference_t_ps,
        "paired_trigger_count": trigger_pairs_used,
        "qutag_trigger_skip_count": qutag_trigger_skip_count,
        "hydraharp_trigger_skip_count": hydraharp_trigger_skip_count,
        "corrected_qutag_channel_max": int(qutag_channel_max),
        "shared_duration_ps": shared_duration_ps,
        "events_removed_outside_shared_interval": removed_outside_shared_interval,
        "timestamps_clipped_to_shared_interval": False,
    }
    print(
        "Timebase correction: "
        f"{trigger_pairs_used:,} trigger pairs used, mode={correction_mode}, "
        f"applied QuTAG scale={applied_time_scale:.12g}, "
        f"reference {qutag_reference_t_ps:g} ps -> {hydraharp_reference_t_ps:g} ps; "
        + (
            f"removed {removed_outside_shared_interval:,} events outside the "
            "shared interval."
            if correction_mode == "global"
            else "kept all input events."
        )
    )

    if return_parameters:
        return corrected_t_ps, corrected_ch, correction
    return corrected_t_ps, corrected_ch


def timetag_correction_sep(
    t1_ps,
    ch1,
    t2_ps,
    ch2,
    ch_qutag=3,
    ch_hh=7,
    qutag_channel_max=4,
    correction_mode="per_trigger",
    return_parameters=False,
):
    """Correct two separate tagger streams without merging their events.

    The mapping is determined from trigger channels in both streams. It is
    applied to Time Tagger X events in ``t1_ps`` with channel IDs from 0
    through ``qutag_channel_max``; HydraHarp timestamps in ``t2_ps`` retain
    their scale. Both streams are referred to the first paired trigger.
    ``global`` mode removes events outside their shared corrected interval.

    Return ``(corrected_t1_ps, ch1, corrected_t2_ps, ch2)`` and, when
    ``return_parameters`` is true, append the correction dictionary.
    """
    t1_ps = np.asarray(t1_ps).ravel()
    ch1 = np.asarray(ch1).ravel()
    t2_ps = np.asarray(t2_ps).ravel()
    ch2 = np.asarray(ch2).ravel()
    if len(t1_ps) != len(ch1) or len(t2_ps) != len(ch2):
        raise ValueError("Each timestamp array must match its channel array in length.")

    ch_qutag = int(ch_qutag)
    ch_hh = int(ch_hh)
    qutag_channel_max = int(qutag_channel_max)
    correction_mode = str(correction_mode).lower()
    if correction_mode not in ("per_trigger", "linear_fit", "global", "first"):
        raise ValueError(
            "correction_mode must be 'per_trigger', 'linear_fit', 'global', or 'first'."
        )

    if correction_mode == "first":
        qutag_triggers = t1_ps[ch1 == ch_qutag]
        hydraharp_triggers = t2_ps[ch2 == ch_hh]
        if len(qutag_triggers):
            qutag_triggers = np.array([np.min(qutag_triggers)], dtype=np.float64)
        if len(hydraharp_triggers):
            hydraharp_triggers = np.array([np.min(hydraharp_triggers)], dtype=np.float64)
    else:
        qutag_triggers = np.sort(t1_ps[ch1 == ch_qutag].astype(np.float64))
        hydraharp_triggers = np.sort(t2_ps[ch2 == ch_hh].astype(np.float64))
    if len(qutag_triggers) == 0 or len(hydraharp_triggers) == 0:
        raise ValueError(
            f"Both trigger channels must contain events; Time Tagger X ch {ch_qutag}: "
            f"{len(qutag_triggers)}, HydraHarp ch {ch_hh}: {len(hydraharp_triggers)}."
        )

    pair_count = min(len(qutag_triggers), len(hydraharp_triggers))
    if correction_mode != "first" and pair_count < 2:
        raise ValueError(
            "At least two paired trigger events are required for per_trigger, "
            "linear_fit, or global correction."
        )
    qutag_trigger_pairs = qutag_triggers[:pair_count]
    hydraharp_trigger_pairs = hydraharp_triggers[:pair_count]
    qutag_reference_t_ps = qutag_trigger_pairs[0]
    hydraharp_reference_t_ps = hydraharp_trigger_pairs[0]

    if correction_mode == "first":
        local_time_scales = np.array([], dtype=float)
        overall_time_scale = 1.0
    else:
        qutag_elapsed_ps = qutag_trigger_pairs - qutag_reference_t_ps
        hydraharp_elapsed_ps = hydraharp_trigger_pairs - hydraharp_reference_t_ps
        qutag_interval_ps = np.diff(qutag_trigger_pairs)
        hydraharp_interval_ps = np.diff(hydraharp_trigger_pairs)
        if np.any(qutag_interval_ps <= 0):
            raise ValueError("Time Tagger X trigger timestamps do not span elapsed time.")
        local_time_scales = hydraharp_interval_ps / qutag_interval_ps
        overall_time_scale = float(hydraharp_elapsed_ps[-1] / qutag_elapsed_ps[-1])

    corrected_t1_ps = t1_ps.astype(np.float64, copy=True)
    corrected_t2_ps = t2_ps.astype(np.float64, copy=True)
    qutag_event_mask = (ch1 >= 0) & (ch1 <= qutag_channel_max)
    if not np.any(qutag_event_mask):
        raise ValueError("No Time Tagger X events match qutag_channel_max.")
    qutag_event_timestamps = corrected_t1_ps[qutag_event_mask]
    applied_time_scale = overall_time_scale

    if correction_mode == "first":
        corrected_t1_ps[qutag_event_mask] = (
            qutag_event_timestamps - qutag_reference_t_ps + hydraharp_reference_t_ps
        )
    elif correction_mode == "per_trigger":
        interval_indices = np.searchsorted(
            qutag_trigger_pairs, qutag_event_timestamps, side="right"
        ) - 1
        interval_indices = np.clip(interval_indices, 0, pair_count - 2)
        corrected_t1_ps[qutag_event_mask] = (
            hydraharp_trigger_pairs[interval_indices]
            + (qutag_event_timestamps - qutag_trigger_pairs[interval_indices])
            * local_time_scales[interval_indices]
        )
    elif correction_mode == "linear_fit":
        elapsed_norm = np.dot(qutag_elapsed_ps, qutag_elapsed_ps)
        if elapsed_norm == 0:
            raise ValueError("Time Tagger X trigger timestamps do not span elapsed time.")
        time_scale = float(np.dot(qutag_elapsed_ps, hydraharp_elapsed_ps) / elapsed_norm)
        applied_time_scale = time_scale
        corrected_t1_ps[qutag_event_mask] = (
            (qutag_event_timestamps - qutag_reference_t_ps) * time_scale
            + hydraharp_reference_t_ps
        )
    else:  # global
        corrected_t1_ps[qutag_event_mask] = (
            (qutag_event_timestamps - qutag_reference_t_ps) * overall_time_scale
            + hydraharp_reference_t_ps
        )

    corrected_t1_ps -= hydraharp_reference_t_ps
    corrected_t2_ps -= hydraharp_reference_t_ps
    shared_duration_ps = min(
        float(np.max(corrected_t1_ps[qutag_event_mask])),
        float(np.max(corrected_t2_ps)),
    )
    if correction_mode == "global":
        keep1 = (corrected_t1_ps >= 0.0) & (corrected_t1_ps <= shared_duration_ps)
        keep2 = (corrected_t2_ps >= 0.0) & (corrected_t2_ps <= shared_duration_ps)
        removed_outside_shared_interval = (
            len(corrected_t1_ps) - int(np.count_nonzero(keep1))
            + len(corrected_t2_ps) - int(np.count_nonzero(keep2))
        )
        corrected_t1_ps = corrected_t1_ps[keep1]
        corrected_ch1 = ch1[keep1].copy()
        corrected_t2_ps = corrected_t2_ps[keep2]
        corrected_ch2 = ch2[keep2].copy()
    else:
        removed_outside_shared_interval = 0
        corrected_ch1 = ch1.copy()
        corrected_ch2 = ch2.copy()

    correction = {
        "correction_mode": correction_mode,
        "time_scale": applied_time_scale,
        "local_time_scales": local_time_scales,
        "qutag_reference_t_ps": qutag_reference_t_ps,
        "hydraharp_reference_t_ps": hydraharp_reference_t_ps,
        "paired_trigger_count": pair_count,
        "qutag_trigger_skip_count": 0,
        "hydraharp_trigger_skip_count": 0,
        "corrected_qutag_channel_max": qutag_channel_max,
        "shared_duration_ps": shared_duration_ps,
        "events_removed_outside_shared_interval": removed_outside_shared_interval,
        "timestamps_clipped_to_shared_interval": False,
    }
    print(
        "Timebase correction: "
        f"{pair_count:,} trigger pairs used, mode={correction_mode}, "
        f"applied Time Tagger X scale={applied_time_scale:.12g}, "
        f"reference {qutag_reference_t_ps:g} ps -> {hydraharp_reference_t_ps:g} ps; "
        + (
            f"removed {removed_outside_shared_interval:,} events outside the shared interval."
            if correction_mode == "global"
            else "kept all input events."
        )
    )
    if return_parameters:
        return corrected_t1_ps, corrected_ch1, corrected_t2_ps, corrected_ch2, correction
    return corrected_t1_ps, corrected_ch1, corrected_t2_ps, corrected_ch2


def load_timetag(filename=None, return_filename=False):
    """Load t_ps and ch arrays from a timetag .npz, .h5/.hdf5, or .mat file.

    If filename is None, open a file browser. By default this returns
    (t_ps, ch). Set return_filename=True to also return the selected filename.
    """
    if filename is None:
        filename = _open_file_dialog("Select timetag file", TIMETAG_ARRAY_FILETYPES)
    filename = _require_selected_filename(filename, "timetag file")

    t_ps, ch = _load_timetag_arrays(filename)

    if return_filename:
        return t_ps, ch, filename

    return t_ps, ch


def load_timetag_sep(filename=None):
    """Load separate Time Tagger X and HydraHarp streams from an HDF5 file.

    Return ``(t1_ps, ch1, t2_ps, ch2, filename)``. The arrays are loaded in
    full; ``t1_ps``/``ch1`` are the first tagger and ``t2_ps``/``ch2`` the
    second tagger, as written by :func:`timetag_to_h5_sep`.
    """
    if filename is None:
        filename = _open_file_dialog(
            "Select separate-stream timetag H5 file",
            [("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")],
        )
    filename = _require_selected_filename(filename, "separate-stream timetag H5 file")

    with h5py.File(filename, "r") as h5:
        required = ("t1_ps", "ch1", "t2_ps", "ch2")
        missing = [name for name in required if name not in h5]
        if missing:
            raise ValueError(
                "Separate-stream H5 file is missing dataset(s): " + ", ".join(missing)
            )
        if any(h5[name].ndim != 1 for name in required):
            raise ValueError("Separate-stream timetag datasets must be one-dimensional.")
        if len(h5["t1_ps"]) != len(h5["ch1"]):
            raise ValueError("t1_ps and ch1 must contain the same number of records.")
        if len(h5["t2_ps"]) != len(h5["ch2"]):
            raise ValueError("t2_ps and ch2 must contain the same number of records.")

        t1_ps = np.array(h5["t1_ps"])
        ch1 = np.array(h5["ch1"])
        t2_ps = np.array(h5["t2_ps"])
        ch2 = np.array(h5["ch2"])

    return t1_ps, ch1, t2_ps, ch2, filename


# =============================================================================
# TIMETAG TRUNCATION
# =============================================================================

def truncate_timetag(
    t_truncate,
    filename=None,
    chunk_size=1_000_000,
    compression="gzip",
):
    """Save timetags through ``t_truncate`` seconds to a new HDF5 file.

    ``t_truncate`` is expressed in seconds, while the source ``t_ps`` values
    are in picoseconds. The output is saved next to the source as
    ``<source>_truncated_<t_truncate>s.h5``. HDF5 input must be chronological;
    it is copied in chunks and reading stops at the cutoff. Other formats use
    the same full-array loading path as :func:`load_timetag`.
    """
    try:
        t_truncate = float(t_truncate)
    except (TypeError, ValueError) as exc:
        raise ValueError("t_truncate must be a positive number of seconds.") from exc
    if not np.isfinite(t_truncate) or t_truncate <= 0:
        raise ValueError("t_truncate must be a positive number of seconds.")
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    if filename is None:
        filename = _open_file_dialog(
            "Select timetag file to truncate", TIMETAG_ARRAY_FILETYPES
        )
    filename = _require_selected_filename(filename, "timetag file")

    cutoff_ps = t_truncate * 1e12
    source_path = Path(filename)
    cutoff_label = format(t_truncate, ".12g")
    output_filename = source_path.with_name(
        f"{source_path.stem}_truncated_{cutoff_label}s.h5"
    )
    if output_filename.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing truncated file: {output_filename}"
        )

    def create_output_datasets(output_h5, t_dtype, ch_dtype):
        output_h5.attrs["source_filename"] = str(source_path)
        output_h5.attrs["t_truncate_s"] = t_truncate
        output_h5.attrs["t_truncate_ps"] = cutoff_ps
        output_chunk_size = min(int(chunk_size), 1_000_000)
        return (
            output_h5.create_dataset(
                "t_ps",
                shape=(0,),
                maxshape=(None,),
                dtype=t_dtype,
                chunks=(output_chunk_size,),
                compression=compression,
            ),
            output_h5.create_dataset(
                "ch",
                shape=(0,),
                maxshape=(None,),
                dtype=ch_dtype,
                chunks=(output_chunk_size,),
                compression=compression,
            ),
        )

    def append_chunk(t_dataset, ch_dataset, time_chunk, channel_chunk):
        n_old = len(t_dataset)
        n_new = n_old + len(time_chunk)
        t_dataset.resize(n_new, axis=0)
        ch_dataset.resize(n_new, axis=0)
        t_dataset[n_old:n_new] = time_chunk
        ch_dataset[n_old:n_new] = channel_chunk

    if source_path.suffix.lower() in (".h5", ".hdf5"):
        with h5py.File(source_path, "r") as source_h5:
            if "t_ps" not in source_h5 or "ch" not in source_h5:
                raise ValueError(f"HDF5 timetag file is missing t_ps and/or ch: {source_path}")

            source_t_ps = source_h5["t_ps"]
            source_ch = source_h5["ch"]
            if source_t_ps.ndim != 1 or source_ch.ndim != 1:
                raise ValueError("Chunked truncation requires one-dimensional t_ps and ch datasets.")
            if len(source_t_ps) != len(source_ch):
                raise ValueError("t_ps and ch must contain the same number of records.")

            with h5py.File(output_filename, "w") as output_h5:
                output_t_ps, output_ch = create_output_datasets(
                    output_h5, source_t_ps.dtype, source_ch.dtype
                )
                for start in range(0, len(source_t_ps), int(chunk_size)):
                    stop = min(start + int(chunk_size), len(source_t_ps))
                    time_chunk = np.asarray(source_t_ps[start:stop])
                    channel_chunk = np.asarray(source_ch[start:stop])
                    keep_count = np.searchsorted(time_chunk, cutoff_ps, side="right")
                    if keep_count:
                        append_chunk(
                            output_t_ps,
                            output_ch,
                            time_chunk[:keep_count],
                            channel_chunk[:keep_count],
                        )
                    if keep_count < len(time_chunk):
                        break
    else:
        t_ps, ch = _load_timetag_arrays(source_path)
        if len(t_ps) != len(ch):
            raise ValueError("t_ps and ch must contain the same number of records.")
        keep = t_ps <= cutoff_ps
        with h5py.File(output_filename, "w") as output_h5:
            output_t_ps, output_ch = create_output_datasets(
                output_h5, t_ps.dtype, ch.dtype
            )
            append_chunk(output_t_ps, output_ch, t_ps[keep], ch[keep])

    print(f"Saved truncated H5 file: {output_filename}")
    print(f"Kept tags through {t_truncate:g} s ({cutoff_ps:g} ps).")
    return str(output_filename)


# Kept for notebooks/scripts that used the original misspelled public name.
trancate_timetag = truncate_timetag


def truncate_timetag_sep(
    t_truncate,
    filename=None,
    chunk_size=1_000_000,
    compression="gzip",
):
    """Save both separate HDF5 streams through ``t_truncate`` seconds.

    Input must contain chronological ``t1_ps``/``ch1`` and ``t2_ps``/``ch2``
    datasets, as written by :func:`timetag_to_h5_sep`. Each stream is copied
    in chunks, keeping timestamps <= ``t_truncate * 1e12`` without rebasing
    them. The cutoff is measured from timestamp zero, not the first event.
    The output preserves the separate layout, dtypes, and attributes and is
    saved beside the source as ``<source>_truncated_<t_truncate>s.h5``.
    """
    try:
        t_truncate = float(t_truncate)
    except (TypeError, ValueError) as exc:
        raise ValueError("t_truncate must be a positive number of seconds.") from exc
    if not np.isfinite(t_truncate) or t_truncate <= 0:
        raise ValueError("t_truncate must be a positive number of seconds.")
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    if filename is None:
        filename = _open_file_dialog(
            "Select separate-stream timetag H5 file to truncate",
            [("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")],
        )
    filename = _require_selected_filename(filename, "separate-stream timetag H5 file")
    source_path = Path(filename)
    cutoff_ps = t_truncate * 1e12
    cutoff_label = format(t_truncate, ".12g")
    output_filename = source_path.with_name(
        f"{source_path.stem}_truncated_{cutoff_label}s.h5"
    )
    if output_filename.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing truncated file: {output_filename}"
        )

    kept_counts = []
    with h5py.File(source_path, "r") as source_h5:
        required = ("t1_ps", "ch1", "t2_ps", "ch2")
        missing = [name for name in required if name not in source_h5]
        if missing:
            raise ValueError(
                "Separate-stream H5 file is missing dataset(s): " + ", ".join(missing)
            )
        if any(source_h5[name].ndim != 1 for name in required):
            raise ValueError("Separate-stream timetag datasets must be one-dimensional.")
        for time_name, channel_name in (("t1_ps", "ch1"), ("t2_ps", "ch2")):
            if len(source_h5[time_name]) != len(source_h5[channel_name]):
                raise ValueError(
                    f"{time_name} and {channel_name} must contain the same number of records."
                )

        with h5py.File(output_filename, "x") as output_h5:
            output_h5.attrs.update(source_h5.attrs)
            output_h5.attrs["source_filename"] = str(source_path)
            output_h5.attrs["t_truncate_s"] = t_truncate
            output_h5.attrs["t_truncate_ps"] = cutoff_ps
            for time_name, channel_name in (("t1_ps", "ch1"), ("t2_ps", "ch2")):
                source_time = source_h5[time_name]
                source_channel = source_h5[channel_name]
                for name in (time_name, channel_name):
                    dataset = output_h5.create_dataset(
                        name,
                        shape=(0,),
                        maxshape=(None,),
                        dtype=source_h5[name].dtype,
                        chunks=(min(int(chunk_size), 1_000_000),),
                        compression=compression,
                    )
                    dataset.attrs.update(source_h5[name].attrs)
                output_time = output_h5[time_name]
                output_channel = output_h5[channel_name]
                for start in range(0, len(source_time), int(chunk_size)):
                    time_chunk = np.asarray(source_time[start:start + int(chunk_size)])
                    keep_count = int(np.searchsorted(time_chunk, cutoff_ps, side="right"))
                    if keep_count:
                        n_old = len(output_time)
                        n_new = n_old + keep_count
                        output_time.resize(n_new, axis=0)
                        output_channel.resize(n_new, axis=0)
                        output_time[n_old:n_new] = time_chunk[:keep_count]
                        output_channel[n_old:n_new] = source_channel[start:start + keep_count]
                    if keep_count < len(time_chunk):
                        break
                kept_counts.append(len(output_time))

    print(f"Saved truncated separate-stream H5 file: {output_filename}")
    print(
        f"Kept tags through {t_truncate:g} s ({cutoff_ps:g} ps): "
        f"stream 1: {kept_counts[0]:,}, stream 2: {kept_counts[1]:,}."
    )
    return str(output_filename)


# =============================================================================
# PHOTON COUNTING AND FIXED-WINDOW G2 PROCESSING
# =============================================================================
# g2_process() produces a coincidence histogram and summary metrics for one
# acquisition window. The time-dependent routines below repeat it over time.

def plot_photon_count(t_ps, ch, bin_width_s, channel=None, show=True):
    """Plot photon counts per time bin for all events or one channel.

    Parameters
    ----------
    t_ps, ch : array-like
        Matching timetag timestamps (ps) and channel numbers.
    bin_width_s : float
        Width of each count bin in seconds.
    channel : int or None, optional
        Channel to count. When None, events from all channels are counted.
    show : bool, optional
        Show the matplotlib figure immediately.

    Returns
    -------
    figure, axis, bin_centers_s, counts
        The plot objects and the binned count data. Time is measured from the
        first timetag in the supplied arrays.
    """
    t_ps = np.asarray(t_ps).ravel()
    ch = np.asarray(ch).ravel()
    if len(t_ps) != len(ch):
        raise ValueError("t_ps and ch must contain the same number of records.")
    if len(t_ps) == 0:
        raise ValueError("t_ps and ch must not be empty.")

    try:
        bin_width_s = float(bin_width_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("bin_width_s must be a positive number of seconds.") from exc
    if not np.isfinite(bin_width_s) or bin_width_s <= 0:
        raise ValueError("bin_width_s must be a positive number of seconds.")

    if channel is None:
        selected_t_ps = t_ps
        channel_label = "all channels"
    else:
        channel = int(channel)
        selected_t_ps = t_ps[ch == channel]
        if len(selected_t_ps) == 0:
            raise ValueError(f"No events found for channel {channel}.")
        channel_label = f"channel {channel}"

    time_origin_ps = np.min(t_ps)
    recording_duration_ps = np.max(t_ps) - time_origin_ps
    bin_width_ps = bin_width_s * 1e12
    bin_count = max(1, int(np.ceil(recording_duration_ps / bin_width_ps)))
    if bin_count > 10_000_000:
        raise ValueError(
            "bin_width_s would create more than 10,000,000 bins; increase the bin width."
        )

    edges_ps = np.arange(bin_count + 1, dtype=np.float64) * bin_width_ps
    selected_elapsed_ps = selected_t_ps.astype(np.float64) - time_origin_ps
    counts, _ = np.histogram(selected_elapsed_ps, bins=edges_ps)
    bin_centers_s = (edges_ps[:-1] + edges_ps[1:]) * 0.5e-12

    figure, axis = plt.subplots(figsize=(12, 4.5))
    axis.step(bin_centers_s, counts, where="mid")
    axis.set_xlabel("Elapsed time (s)")
    axis.set_ylabel("Photon count per bin")
    axis.set_title(f"Photon count vs. time ({channel_label}, {bin_width_s:g} s bins)")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    if show:
        plt.show()

    return figure, axis, bin_centers_s, counts


def _make_edges(t_lo, t_hi, time_bin):
    """Match MATLAB colon-style edges: t_lo:time_bin:t_hi."""
    if time_bin <= 0:
        raise ValueError("time_bin must be positive.")

    edges = np.arange(t_lo, t_hi + time_bin * 0.5, time_bin, dtype=float)

    if len(edges) < 2:
        raise ValueError("Window too small for given bin width. Increase span or decrease bin.")

    return edges


def g2_process(
    t_ps,
    ch,
    t_center=6.25,
    t_span=40,
    t_stop=None,
    bg_A=500,
    bg_B=740,
    freq=50e6,
    time_bin=1.0,
    chA=1,
    chB=2,
    sig_half=5,
    filename=None,
):
    """Process timetag arrays and return coincidence data and metrics.

    t_stop is an optional stop time in seconds, measured from the first timetag.
    """
    t_ps = np.asarray(t_ps).ravel()
    ch = np.asarray(ch).ravel()

    if len(t_ps) == 0 or len(ch) == 0:
        raise ValueError("t_ps and ch must not be empty.")
    if len(t_ps) != len(ch):
        raise ValueError("t_ps and ch must have the same length.")

    # ---------------- Convert ps -> ns and start at 0 ----------------
    t_ns = (t_ps - t_ps[0]) / 1000.0

    # ---------------- Optional stop time ----------------
    T_file_ns = t_ns[-1] - t_ns[0]
    T_file_s = T_file_ns * 1e-9

    t_stop_requested_s = None

    if t_stop is None:
        T_acq_s = T_file_s
        T_acq_ns = T_file_ns
    else:
        t_stop = float(t_stop)
        if not np.isfinite(t_stop) or t_stop <= 0:
            raise ValueError("t_stop must be a positive number of seconds.")

        t_stop_requested_s = t_stop
        T_acq_s = min(T_file_s, t_stop)
        T_acq_ns = T_acq_s * 1e9
        process_mask = t_ns <= T_acq_ns
        t_ns = t_ns[process_mask]
        ch = ch[process_mask]

        if len(t_ns) == 0:
            raise ValueError("No timetags remain before t_stop.")

    cntA_total = np.sum(ch == chA)
    cntB_total = np.sum(ch == chB)

    print("--- File Summary ---")
    if filename is not None:
        print(f"File: {filename}")
    else:
        print("File: loaded timetag arrays")
    print(f"File acquisition time: {T_file_s:.6f} s")
    if t_stop_requested_s is not None:
        print(f"Processed time window: 0.000000 s to {T_acq_s:.6f} s")
    else:
        print(f"Processed time window: full file ({T_acq_s:.6f} s)")
    print(
        f"Total counts on channel A (chA={chA}): {cntA_total} "
        f"-> {cntA_total / T_acq_s:.3f} cps"
    )
    print(
        f"Total counts on channel B (chB={chB}): {cntB_total} "
        f"-> {cntB_total / T_acq_s:.3f} cps"
    )

    # ---------------- Build histogram bins ----------------
    t_lo = t_center - t_span
    t_hi = t_center + t_span

    edges = np.arange(t_lo, t_hi + time_bin, time_bin)

    if len(edges) < 2:
        raise ValueError("Window too small for given bin width. Increase span or decrease bin.")

    centers = edges[:-1] + time_bin / 2
    H = np.zeros(len(centers), dtype=np.int64)

    # ---------------- Extract events for chA and chB ----------------
    tA = t_ns[ch == chA]
    tB = t_ns[ch == chB]

    # If your file is not time ordered, uncomment:
    # tA = np.sort(tA)
    # tB = np.sort(tB)

    # ---------------- Coincidence counting, two-pointer method ----------------
    numA_used = 0
    numPairs = 0

    t_col = 1e9     # ns, equivalent to 1 second
    k_col = 1
    num_SPA = 0
    SPA = []

    j_start = 0     # Python uses zero-based indexing

    for ti in tA:
        numA_used += 1

        if ti >= T_acq_ns:
            break

        # Single photon count within certain periods
        if ti >= k_col * t_col:
            SPA.append(num_SPA)
            num_SPA = 0
            k_col += 1

        num_SPA += 1

        # Move j_start so that tB[j_start] - ti >= t_lo
        while j_start < len(tB) and (tB[j_start] - ti) < t_lo:
            j_start += 1

        # Count forward from j_start while dt <= t_hi
        j = j_start
        while j < len(tB):
            dt = tB[j] - ti

            if dt > t_hi:
                break

            kbin = int(np.floor((dt - t_lo) / time_bin))

            if 0 <= kbin < len(H):
                H[kbin] += 1
                numPairs += 1

            j += 1

    # ---------------- Optional: background / CAR style metrics ----------------
    sig_mask = (centers >= (t_center - sig_half)) & (centers <= (t_center + sig_half))
    bg_mask = ~sig_mask

    bg = np.mean(H[bg_mask])
    bg = 0.0
    cmax = np.max(H)

    CC = np.sum(H[sig_mask]) - bg * np.sum(sig_mask)   # background-subtracted coincidences
    # CC = np.sum(H[sig_mask])   # background-subtracted coincidences
    CAR = (cmax - bg) / max(bg, np.finfo(float).eps)

    CCps = CC / T_acq_s
    CCppulse = CCps / freq

    cntAps = cntA_total / T_acq_s - bg_A
    cntBps = cntB_total / T_acq_s - bg_B

    CC_raw = cntAps * cntBps / CCps
    etaA = cntAps / CC_raw
    etaB = cntBps / CC_raw

    print("--- Counting Summary ---")
    print(f"Detected CC (per second): {CCps:.6f}")
    print(f"Detected CC (per pulse): {CCppulse:.8f}")
    print(f"Raw CC (per second): {CC_raw:.6f}")
    print(f"Ch A efficiency: {etaA:.6f}")
    print(f"Ch B efficiency: {etaB:.6f}")
    # print(f"CAR: {CAR:.6f}")

    return {
        "filename": filename,
        "centers": centers,
        "histogram": H,
        "single_photon_counts": SPA,
        "T_acq_s": T_acq_s,
        "T_file_s": T_file_s,
        "t_stop": t_stop_requested_s,
        "time_bin": time_bin,
        "chA": chA,
        "chB": chB,
        "cntA_total": cntA_total,
        "cntB_total": cntB_total,
        "CC": CC,
        "CCps": CCps,
        "CCppulse": CCppulse,
        "CC_raw": CC_raw,
        "etaA": etaA,
        "etaB": etaB,
        "CAR": CAR,
        "numA_used": numA_used,
        "numPairs": numPairs,
    }


def g2_process_sep(
    t1_ps,
    t2_ps,
    t_center=6.25,
    t_span=40,
    t_stop=None,
    bg_A=500,
    bg_B=740,
    freq=50e6,
    time_bin=1.0,
    sig_half=5,
    filename=None,
):
    """Count coincidences between two channel-specific timestamp arrays.

    Inputs are picoseconds; histogram settings are nanoseconds. ``t_stop`` is
    measured in seconds from the earliest timestamp in either array.
    """
    # Import here because qcs_kernal imports this module. This reuses its
    # compiled pair counter without introducing a module-level import cycle.
    from .qcs_kernal import g2_histogram_numba

    t1_ps = np.asarray(t1_ps).ravel()
    t2_ps = np.asarray(t2_ps).ravel()
    if len(t1_ps) == 0 or len(t2_ps) == 0:
        raise ValueError("t1_ps and t2_ps must not be empty.")
    if not np.all(np.isfinite(t1_ps)) or not np.all(np.isfinite(t2_ps)):
        raise ValueError("t1_ps and t2_ps must contain only finite timestamps.")

    # The Numba two-pointer counter requires each channel to be chronological.
    if np.any(t1_ps[1:] < t1_ps[:-1]):
        t1_ps = np.sort(t1_ps)
    if np.any(t2_ps[1:] < t2_ps[:-1]):
        t2_ps = np.sort(t2_ps)

    time_origin_ps = min(t1_ps[0], t2_ps[0])
    T_file_s = (max(t1_ps[-1], t2_ps[-1]) - time_origin_ps) * 1e-12
    if not np.isfinite(T_file_s) or T_file_s <= 0:
        raise ValueError("The two timestamp arrays must span a positive duration.")

    t_stop_requested_s = None
    if t_stop is None:
        T_acq_s = T_file_s
    else:
        t_stop = float(t_stop)
        if not np.isfinite(t_stop) or t_stop <= 0:
            raise ValueError("t_stop must be a positive number of seconds.")
        t_stop_requested_s = t_stop
        T_acq_s = min(T_file_s, t_stop)

    # Use the same time origin for both channels so their delays are retained.
    T_acq_ps = T_acq_s * 1e12
    tA = np.asarray(t1_ps - time_origin_ps, dtype=np.float64)
    tB = np.asarray(t2_ps - time_origin_ps, dtype=np.float64)
    if t_stop_requested_s is not None:
        tA = tA[tA <= T_acq_ps]
        tB = tB[tB <= T_acq_ps]
    if len(tA) == 0 or len(tB) == 0:
        raise ValueError("Both channels need timetags inside the processed window.")

    cntA_total = len(tA)
    cntB_total = len(tB)

    print("--- File Summary ---")
    if filename is not None:
        print(f"File: {filename}")
    else:
        print("File: loaded timetag arrays")
    print(f"File acquisition time: {T_file_s:.6f} s")
    if t_stop_requested_s is not None:
        print(f"Processed time window: 0.000000 s to {T_acq_s:.6f} s")
    else:
        print(f"Processed time window: full file ({T_acq_s:.6f} s)")
    print(
        f"Total counts on channel A: {cntA_total} "
        f"-> {cntA_total / T_acq_s:.3f} cps"
    )
    print(
        f"Total counts on channel B: {cntB_total} "
        f"-> {cntB_total / T_acq_s:.3f} cps"
    )

    # ---------------- Build histogram bins ----------------
    t_lo = t_center - t_span
    t_hi = t_center + t_span

    edges = np.arange(t_lo, t_hi + time_bin, time_bin)

    if len(edges) < 2:
        raise ValueError("Window too small for given bin width. Increase span or decrease bin.")

    centers = edges[:-1] + time_bin / 2

    # Reuse the exact compiled pair-counting method used by qcs_kernal.
    # g2_histogram_numba expects timestamps and edges in picoseconds.
    histogram_result = g2_histogram_numba(
        tA,
        tB,
        edges * 1e3,
        Ta_s=T_acq_s,
        chA=None,
        chB=None,
        time_bin_ps_init=time_bin * 1e3,
    )
    H = histogram_result["histogram_raw"]
    numA_used = len(tA)
    numPairs = int(np.sum(H))

    # Channel-A counts in each complete one-second interval.
    full_seconds = int(np.floor(T_acq_s))
    second_edges_ps = np.arange(full_seconds + 1, dtype=np.float64) * 1e12
    SPA = np.diff(np.searchsorted(tA, second_edges_ps, side="left")).tolist()

    # ---------------- Optional: background / CAR style metrics ----------------
    sig_mask = (centers >= (t_center - sig_half)) & (centers <= (t_center + sig_half))
    bg_mask = ~sig_mask

    bg = np.mean(H[bg_mask])
    bg = 0.0
    cmax = np.max(H)

    CC = np.sum(H[sig_mask]) - bg * np.sum(sig_mask)   # background-subtracted coincidences
    # CC = np.sum(H[sig_mask])   # background-subtracted coincidences
    CAR = (cmax - bg) / max(bg, np.finfo(float).eps)

    CCps = CC / T_acq_s
    CCppulse = CCps / freq

    cntAps = cntA_total / T_acq_s - bg_A
    cntBps = cntB_total / T_acq_s - bg_B

    CC_raw = cntAps * cntBps / CCps
    etaA = cntAps / CC_raw
    etaB = cntBps / CC_raw

    print("--- Counting Summary ---")
    print(f"Detected CC (per second): {CCps:.6f}")
    print(f"Detected CC (per pulse): {CCppulse:.8f}")
    print(f"Raw CC (per second): {CC_raw:.6f}")
    print(f"Ch A efficiency: {etaA:.6f}")
    print(f"Ch B efficiency: {etaB:.6f}")
    # print(f"CAR: {CAR:.6f}")

    return {
        "filename": filename,
        "centers": centers,
        "histogram": H,
        "single_photon_counts": SPA,
        "T_acq_s": T_acq_s,
        "T_file_s": T_file_s,
        "t_stop": t_stop_requested_s,
        "time_bin": time_bin,
        "cntA_total": cntA_total,
        "cntB_total": cntB_total,
        "CC": CC,
        "CCps": CCps,
        "CCppulse": CCppulse,
        "CC_raw": CC_raw,
        "etaA": etaA,
        "etaB": etaB,
        "CAR": CAR,
        "numA_used": numA_used,
        "numPairs": numPairs,
    }


# =============================================================================
# RESULT FILES (.NPZ)
# =============================================================================
# Save/load helpers for dictionaries returned by g2_process() and
# g2_process_time_dependent().

def save_g2_results(results, output_filename=None):
    """Save a g2_process results dictionary to a compressed .npz file."""
    if not isinstance(results, dict):
        raise TypeError("results must be the dictionary returned by g2_process().")

    time_bin = results.get("time_bin")
    if time_bin is None:
        centers = np.asarray(results.get("centers", []), dtype=float)
        if len(centers) >= 2:
            time_bin = float(np.median(np.diff(centers)))

    time_bin_text = ""
    if time_bin is not None:
        time_bin_text = f"_timebin_{float(time_bin):g}ns".replace(".", "p")

    acq_time = results.get("T_acq_s")
    acq_time_text = ""
    if acq_time is not None:
        acq_time_text = f"_{float(acq_time):g}s".replace(".", "p")

    if output_filename is None:
        source_filename = results.get("filename")
        if source_filename:
            output_path = Path(source_filename).with_name(
                f"{Path(source_filename).stem}_g2_results{time_bin_text}{acq_time_text}.npz"
            )
        else:
            output_path = Path(f"g2_results{time_bin_text}{acq_time_text}.npz")
    else:
        output_path = Path(output_filename)
        if output_path.suffix.lower() != ".npz":
            output_path = output_path.with_suffix(".npz")
        if time_bin_text and time_bin_text not in output_path.stem:
            output_path = output_path.with_name(f"{output_path.stem}{time_bin_text}{output_path.suffix}")

    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {}
    for key, value in results.items():
        if value is None:
            payload[key] = np.array(np.nan)
        else:
            payload[key] = np.asarray(value)

    np.savez_compressed(output_path, **payload)
    print(f"Saved g2 results: {output_path}")
    return str(output_path)


def save_time_dependent_results(time_dependent_results, output_filename=None):
    """Save time-dependent results to ``output_filename`` or a file chooser."""
    if not isinstance(time_dependent_results, dict):
        raise TypeError("time_dependent_results must be the dictionary returned by g2_process_time_dependent().")

    if output_filename is None:
        output_filename = _save_file_dialog(
            "Save time-dependent g2 results", NPZ_FILETYPES, ".npz"
        )
    filename = _require_selected_filename(output_filename, "save file")

    output_path = Path(filename)
    if output_path.suffix.lower() != ".npz":
        output_path = output_path.with_suffix(".npz")
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "t_stop": np.asarray(time_dependent_results.get("t_stop", [])),
        "center": np.asarray(time_dependent_results.get("center", [])),
        "center_uncertainty": np.asarray(
            time_dependent_results.get("center_uncertainty", [])
        ),
        "fwhm": np.asarray(time_dependent_results.get("fwhm", [])),
        "coincidences": np.asarray(time_dependent_results.get("coincidences", [])),
    }

    g2_results = time_dependent_results.get("g2_results", [])
    payload["t_start"] = np.asarray([
        np.nan if result is None else result.get("t_start", np.nan)
        for result in g2_results
    ])
    payload["CCps"] = np.asarray([
        np.nan if result is None else result.get("CCps", np.nan)
        for result in g2_results
    ])
    payload["numPairs"] = np.asarray([
        np.nan if result is None else result.get("numPairs", np.nan)
        for result in g2_results
    ])

    np.savez_compressed(output_path, **payload)
    print(f"Saved time-dependent g2 results: {output_path}")
    return str(output_path)


def load_time_dependent_results(filename=None):
    """Load saved time-dependent results and show their summary figure."""
    if filename is None:
        filename = _open_file_dialog("Load time-dependent g2 results", NPZ_FILETYPES)
    filename = _require_selected_filename(filename, "results file")

    time_dependent_results = {}
    with np.load(filename, allow_pickle=False) as data:
        for key in data.files:
            time_dependent_results[key] = data[key].copy()

    t_stop = time_dependent_results.get("t_stop", np.array([]))
    center = time_dependent_results.get("center", np.array([]))
    expected_shape = np.asarray(center).shape

    def load_series_or_nan(key):
        values = time_dependent_results.get(key)
        if values is None:
            print(f"Warning: {Path(filename).name} does not contain '{key}'.")
            return np.full(expected_shape, np.nan, dtype=float)
        values = np.asarray(values)
        if values.shape != expected_shape:
            print(
                f"Warning: '{key}' shape {values.shape} does not match center "
                f"shape {expected_shape}; using NaN values."
            )
            return np.full(expected_shape, np.nan, dtype=float)
        return values

    center_uncertainty = load_series_or_nan("center_uncertainty")
    num_pairs = load_series_or_nan("numPairs")
    # Always expose these arrays to callers, including for legacy files that
    # predate saved fit uncertainties.
    time_dependent_results["center_uncertainty"] = center_uncertainty
    time_dependent_results["numPairs"] = num_pairs

    fig, axes = plt.subplots(3, 1, sharex=True)
    axes[0].plot(t_stop, center, "o-")
    axes[0].set_ylabel("Center (ns)")
    axes[0].grid(True)

    axes[1].plot(t_stop, center_uncertainty, "o-")
    axes[1].set_ylabel("Center uncertainty (ns)")
    axes[1].grid(True)

    axes[2].plot(t_stop, num_pairs, "o-")
    axes[2].set_xlabel("t_stop (s)")
    axes[2].set_ylabel("Photon pairs")
    axes[2].grid(True)

    fig.tight_layout()
    plt.show()

    time_dependent_results["figure"] = fig
    time_dependent_results["axes"] = axes
    return time_dependent_results


def load_g2_results(filename=None):
    """Load a results dictionary saved by save_g2_results()."""
    if filename is None:
        filename = _open_file_dialog("Select stored g2 results file", NPZ_FILETYPES)
    filename = _require_selected_filename(filename, "results file")

    results = {}

    with np.load(filename, allow_pickle=False) as data:
        for key in data.files:
            value = data[key]

            if value.shape == ():
                value = value.item()
                if key == "t_stop" and isinstance(value, float) and np.isnan(value):
                    value = None
            else:
                value = value.copy()

            results[key] = value

    return results


def _gaussian_with_offset(x, height, center, sigma, offset):
    return offset + height * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _lorentzian_with_offset(x, height, center, gamma, offset):
    return offset + height * gamma**2 / ((x - center) ** 2 + gamma**2)


def _expo_with_offset(x, height, center, gamma, offset):
    return offset + height * np.exp(-np.abs(x - center) / gamma)


def _expogaussconv_with_offset(x, height, center, gamma, sigma, offset):
    """Symmetric exponential-Gaussian convolution with a constant offset."""
    distance = np.abs(x - center)
    root_two_sigma = np.sqrt(2.0) * sigma
    z1 = (sigma**2 / gamma - distance) / root_two_sigma
    z2 = (sigma**2 / gamma + distance) / root_two_sigma

    # erfcx keeps exp(distance / gamma) * erfc(z2) finite in the tails.
    left_term = np.exp(-distance / gamma) * erfc(z1)
    right_exponent = -0.5 * (distance / sigma) ** 2 - 0.5 * (sigma / gamma) ** 2
    right_term = np.exp(right_exponent) * erfcx(z2)
    return offset + height * (left_term + right_term)


def _g2calib_1(x, height, center, sigma, offset):
    """Symmetric exponential-Gaussian convolution with a constant offset."""
    # gamma = 0.702672
    gamma = 702.67227
    distance = np.abs(x - center)
    root_two_sigma = np.sqrt(2.0) * sigma
    z1 = (sigma**2 / gamma - distance) / root_two_sigma
    z2 = (sigma**2 / gamma + distance) / root_two_sigma

    # erfcx keeps exp(distance / gamma) * erfc(z2) finite in the tails.
    left_term = np.exp(-distance / gamma) * erfc(z1)
    right_exponent = -0.5 * (distance / sigma) ** 2 - 0.5 * (sigma / gamma) ** 2
    right_term = np.exp(right_exponent) * erfcx(z2)
    return offset + height * (left_term + right_term)


# _GAMMA = 732.22054
# _SIGMAG = 71.134089
_GAMMA = 61.676
_SIGMAG = 33.417
_A = 1.0 / (np.sqrt(2.0) * _SIGMAG)
_B = 1.0 / _GAMMA
_COEF1 = _GAMMA * np.exp(-_SIGMAG**2 / _GAMMA**2)
_COEF2 = _GAMMA * np.exp(-_SIGMAG**2 / _GAMMA**2 / 2)
_OFFSET = _B / 2 / _A

_EXP_ERFC_SLOPE = _B / _A


def _exp_zb_erfc_za(z):
    """Numerically stable evaluation of exp(z*_B) * erfc(z*_A)."""
    z = np.asarray(z, dtype=float)
    q = z * _A
    result = np.empty_like(q)

    # For q >= 0:
    # exp(q*c) erfc(q) = exp(-(q-c/2)^2 + c^2/4) erfcx(q)
    positive = q >= 0
    c = _EXP_ERFC_SLOPE
    result[positive] = (
        np.exp(-(q[positive] - c / 2.0) ** 2 + c**2 / 4.0)
        * erfcx(q[positive])
    )

    # Here z*_B <= 0, so the ordinary expression cannot overflow.
    result[~positive] = np.exp(z[~positive] * _B) * erfc(q[~positive])

    return result


def _g2calib_2_Hfunc(d):
    z1 = _SIGMAG ** 2 / _GAMMA - d
    z2 = _SIGMAG ** 2 / _GAMMA + d

    term1 = -_COEF1 * _exp_zb_erfc_za(z1)
    term2 = -_COEF2 * erf(z1 * _A - _OFFSET)
    term3 = _COEF1 * _exp_zb_erfc_za(z2)
    term4 = _COEF2 * erf(z2 * _A - _OFFSET)

    return term1 + term2 + term3 + term4


# def _g2calib_2_Hfunc(d):
#     # gamma = 702.67227
#     # sigmaG = 78.800111
#     # a = 1/(np.sqrt(2) * sigmaG)
#     # b = 1/gamma

#     z1 = _SIGMAG ** 2 / _GAMMA - d
#     z2 = _SIGMAG ** 2 / _GAMMA + d

#     term1 = -_COEF1 * np.exp(z1 * _B) * erfc(z1 * _A)
#     term2 = -_COEF2 * erf(z1 * _A - _OFFSET)
#     term3 = _COEF1 * np.exp(z2 * _B) * erfc(z2 * _A)
#     term4 = _COEF2 * erf(z2 * _A - _OFFSET)

#     # term1 = -gamma * np.exp(-sigmaG**2 / gamma**2) * np.exp(z1 * b) * erfc(z1 * a)
#     # term2 = -gamma * np.exp(-sigmaG**2 / gamma**2 / 2) * erf(z1 * a - b / 2 / a)
#     # term3 = gamma * np.exp(-sigmaG**2 / gamma**2) * np.exp(z2 * b) * erfc(z2 * a)
#     # term4 = gamma * np.exp(-sigmaG**2 / gamma**2 / 2) * erf(z2 * a - b / 2 / a)

#     # term1 = gamma * np.exp(-sigmaG**2 / gamma**2 / 2) * erf(d * a)
#     # term2 = -gamma * np.exp(-d * b) * erfc((sigmaG / gamma - d / sigmaG) / np.sqrt(2))
#     # term3 = gamma * np.exp(-sigmaG**2 / gamma**2 / 2) * erf(d * a)
#     # term4 = gamma * np.exp(d * b) * erfc((sigmaG / gamma + d / sigmaG) / np.sqrt(2))

#     return term1 + term2 + term3 + term4


def _g2calib_2(x, height, center, sigma, offset):
    d = x - center
    if sigma > 0.001:
        term = (_g2calib_2_Hfunc(d + sigma / 2) - _g2calib_2_Hfunc(d - sigma / 2)) / sigma
    else:
        term = (_g2calib_2_Hfunc(d + 0.001 / 2) - _g2calib_2_Hfunc(d - 0.001 / 2)) / 0.001

    return offset + height*term

    # distance = np.abs(x - center)
    # root_two_sigma = np.sqrt(2.0) * sigmaG
    # z1 = (sigmaG**2 / gamma - distance) / root_two_sigma
    # z2 = (sigmaG**2 / gamma + distance) / root_two_sigma

    # # erfcx keeps exp(distance / gamma) * erfc(z2) finite in the tails.
    # left_term = np.exp(-distance / gamma) * erfc(z1)
    # right_exponent = -0.5 * (distance / sigmaG) ** 2 - 0.5 * (sigmaG / gamma) ** 2
    # right_term = np.exp(right_exponent) * erfcx(z2)
    # return offset + height * (left_term + right_term)


FIT_CURVES = {
    "gaussian": {
        "function": _gaussian_with_offset,
        "parameters": ("height", "center", "sigma", "offset"),
        "scale_parameter": "sigma",
        "positive_parameters": ("sigma",),
        "fwhm_factor": 2.0 * np.sqrt(2.0 * np.log(2.0)),
        "label": "Gaussian",
    },
    "lorentzian": {
        "function": _lorentzian_with_offset,
        "parameters": ("height", "center", "gamma", "offset"),
        "scale_parameter": "gamma",
        "positive_parameters": ("gamma",),
        "fwhm_factor": 2.0,
        "label": "Lorentzian",
    },
    "expo": {
        "function": _expo_with_offset,
        "parameters": ("height", "center", "gamma", "offset"),
        "scale_parameter": "gamma",
        "positive_parameters": ("gamma",),
        "fwhm_factor": 2.0 * np.log(2.0),
        "label": "Exponential",
    },
    "expogaussconv": {
        "function": _expogaussconv_with_offset,
        "parameters": ("height", "center", "gamma", "sigma", "offset"),
        "scale_parameter": "gamma",
        "positive_parameters": ("gamma", "sigma"),
        "fwhm_factor": 2.0 * np.log(2.0),
        "label": "Exponential-Gaussian Convolution",
    },
    "g2calib_1": {
        "function": _g2calib_1,
        "parameters": ("height", "center", "sigma", "offset"),
        "scale_parameter": "sigma",
        "positive_parameters": ("sigma", ),
        "fwhm_factor": 2.0 * np.sqrt(2.0 * np.log(2.0)),
        "label": "g2_calib_No_1",
    },
    "g2calib_2": {
        "function": _g2calib_2,
        "parameters": ("height", "center", "sigma", "offset"),
        "scale_parameter": "sigma",
        "positive_parameters": ("sigma", ),
        "fwhm_factor": 2.0 * np.sqrt(2.0 * np.log(2.0)),
        "label": "g2_calib_No_2",
    },
}
FIT_CURVE_ALIASES = {"exp": "expo", "exponential": "expo"}


def fit_g2(results, curve, parameters, *, fit_window=None, use_rate=True, show=True):
    """Fit a named peak curve to g2 results using curve-specific parameters.

    Parameters
    ----------
    results : dict
        Dictionary returned by :func:`g2_process`.
    curve : {"gaussian", "lorentzian", "expo", "expogaussconv"}
        Curve family. ``"exp"`` and ``"exponential"`` are aliases for the
        symmetric exponential peak ``offset + height * exp(-abs(x-center)/gamma)``.
    parameters : dict
        Initial parameter values. Gaussian requires ``height``, ``center``, and
        ``sigma``; Lorentzian and exponential require ``height``, ``center``,
        and ``gamma``; ``expogaussconv`` requires ``height``, ``center``,
        ``gamma``, and ``sigma``. ``offset`` is optional and defaults to the
        minimum data value. The offset remains a free fitted parameter.
    fit_window : float, optional
        Width of the fitting region, centred on the supplied ``center``.
    use_rate : bool, optional
        Fit coincidence rate instead of raw histogram counts.
    show : bool, optional
        Create and display a data/fit figure. When False, the returned
        ``x_fit`` and ``y_fit`` are None; display-curve sampling is skipped.
    """
    if curve_fit is None:
        raise ImportError("scipy is required for g2 fitting. Install scipy to use fit_g2().")

    curve_key = str(curve).strip().lower()
    curve_key = FIT_CURVE_ALIASES.get(curve_key, curve_key)
    if curve_key not in FIT_CURVES:
        available = ", ".join(sorted(FIT_CURVES))
        raise ValueError(f"Unknown curve '{curve}'. Choose one of: {available}.")
    definition = FIT_CURVES[curve_key]
    parameter_names = definition["parameters"]

    try:
        initial = dict(parameters)
    except (TypeError, ValueError) as exc:
        raise TypeError("parameters must be a dictionary of initial fit values.") from exc
    unknown = set(initial).difference(parameter_names)
    if unknown:
        raise ValueError(
            f"Unsupported parameter(s) for {curve_key}: {sorted(unknown)}. "
            f"Expected: {list(parameter_names)}."
        )
    required = set(parameter_names).difference({"offset"})
    missing = required.difference(initial)
    if missing:
        raise ValueError(
            f"Missing initial parameter(s) for {curve_key}: {sorted(missing)}."
        )

    centers = np.asarray(results["centers"], dtype=float)
    histogram = np.asarray(results["histogram"], dtype=float)
    if centers.shape != histogram.shape:
        raise ValueError("results['centers'] and results['histogram'] must have matching shapes.")
    if use_rate:
        acquisition_time_s = float(results["T_acq_s"])
        if acquisition_time_s <= 0:
            raise ValueError("results['T_acq_s'] must be positive when use_rate=True.")
        y_data = histogram / acquisition_time_s
        y_label = "Coincidence rate (per second)"
    else:
        y_data = histogram
        y_label = "Counts"

    initial.setdefault("offset", float(np.min(y_data)))
    try:
        p0 = np.asarray([float(initial[name]) for name in parameter_names], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("All initial fit parameters must be finite numbers.") from exc
    if not np.all(np.isfinite(p0)):
        raise ValueError("All initial fit parameters must be finite numbers.")

    scale_index = parameter_names.index(definition["scale_parameter"])
    positive_parameter_indices = [
        parameter_names.index(name) for name in definition["positive_parameters"]
    ]
    for index in positive_parameter_indices:
        if p0[index] <= 0:
            raise ValueError(f"{parameter_names[index]} must be positive.")

    fit_mask = np.isfinite(centers) & np.isfinite(y_data)
    if fit_window is not None:
        fit_half_width = float(fit_window) / 2.0
        if not np.isfinite(fit_half_width) or fit_half_width <= 0:
            raise ValueError("fit_window must be a positive finite width.")
        fit_mask &= np.abs(centers - initial["center"]) <= fit_half_width
    x_fit_data = centers[fit_mask]
    y_fit_data = y_data[fit_mask]
    if len(x_fit_data) < len(parameter_names):
        raise ValueError(
            f"Need at least {len(parameter_names)} data points for {definition['label']} fitting."
        )

    lower_bounds = np.full(len(parameter_names), -np.inf)
    upper_bounds = np.full(len(parameter_names), np.inf)
    lower_bounds[positive_parameter_indices] = np.finfo(float).eps
    try:
        popt, pcov = curve_fit(
            definition["function"],
            x_fit_data,
            y_fit_data,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),

            # method="trf",
            # ftol=1e-12,
            # xtol=1e-12,
            # gtol=1e-12,
            # x_scale="jac",
            # max_nfev=10000,
        )
        fit_succeeded = True
        fit_message = None
    except RuntimeError as exc:
        # Optimizer non-convergence: retain the requested curve and continue.
        popt = p0.copy()
        pcov = np.full((len(parameter_names), len(parameter_names)), np.nan)
        fit_succeeded = False
        fit_message = str(exc)
        print(
            f"Warning: {definition['label']} fit did not converge; "
            "plotting the input parameters instead."
        )
    with np.errstate(invalid="ignore"):
        uncertainties = np.sqrt(np.diag(pcov))
    fitted_parameters = dict(zip(parameter_names, popt))
    parameter_uncertainties = dict(zip(parameter_names, uncertainties))
    fwhm_factor = definition["fwhm_factor"]
    fwhm = fwhm_factor * abs(fitted_parameters[definition["scale_parameter"]])
    fwhm_uncertainty = fwhm_factor * parameter_uncertainties[definition["scale_parameter"]]

    fit_results = {
        "curve": curve_key,
        "parameters": fitted_parameters,
        "parameter_uncertainties": parameter_uncertainties,
        "fwhm": fwhm,
        "fwhm_uncertainty": fwhm_uncertainty,
        "covariance": pcov,
        "success": fit_succeeded,
        "message": fit_message,
        "used_initial_parameters": not fit_succeeded,
        "x_fit": None,
        "y_fit": None,
        **fitted_parameters,
        **{f"{name}_uncertainty": value for name, value in parameter_uncertainties.items()},
    }

    status = "Fit" if fit_succeeded else "Input Parameters (fit did not converge)"
    # print(f"--- {definition['label']} {status} ---")
    # for name in parameter_names:
    #     print(f"{name.capitalize()}: {fitted_parameters[name]:.6g}")
    # print(f"FWHM: {fwhm:.6g}")

    if show:
        # Sample the display curve only when a plot is requested.
        x_fit = np.arange(
            float(np.min(x_fit_data)),
            float(np.max(x_fit_data)),
            0.01,
        )
        y_fit = definition["function"](x_fit, *popt)
        fit_results["x_fit"] = x_fit
        fit_results["y_fit"] = y_fit
        fig, ax = plt.subplots()
        ax.plot(centers, y_data, ".", label="Data")
        fit_label = (
            f"{definition['label']} fit"
            if fit_succeeded
            else f"{definition['label']} input curve (fit failed)"
        )
        ax.plot(x_fit, y_fit, "-", linewidth=2, label=fit_label)
        if "chA" in results and "chB" in results:
            delay_label = f"Time Delay t{results['chB']} - t{results['chA']} (ns)"
        else:
            delay_label = "Time Delay t2 - t1 (ns)"
        ax.set_xlabel(delay_label)
        ax.set_ylabel(y_label)
        ax.grid(True)
        ax.legend(["Data", "Fitting"])
        fit_results["figure"] = fig
        fit_results["axis"] = ax
        plt.show()

    return fit_results


# Clearer alternate name for callers that want the curve-selection emphasis.
fit_g2_curve = fit_g2


# =============================================================================
# PEAK FITTING
# =============================================================================
# fit_g2() is the generic public interface. The original functions below are
# retained as compatibility implementations until their wrappers are defined.

def fit_g2_gaussian(
    results,
    sigma,
    height,
    center,
    offset=None,
    fit_window=None,
    use_rate=True,
    show=True,
):
    """Compatibility wrapper for :func:`fit_g2` with ``curve='gaussian'``."""
    parameters = {"height": height, "center": center, "sigma": sigma}
    if offset is not None:
        parameters["offset"] = offset
    return fit_g2(
        results,
        curve="gaussian",
        parameters=parameters,
        fit_window=fit_window,
        use_rate=use_rate,
        show=show,
    )


def fit_g2_lorentzian(
    results,
    gamma,
    height,
    center,
    offset=None,
    fit_window=None,
    use_rate=True,
    show=True,
):
    """Compatibility wrapper for :func:`fit_g2` with ``curve='lorentzian'``."""
    parameters = {"height": height, "center": center, "gamma": gamma}
    if offset is not None:
        parameters["offset"] = offset
    return fit_g2(
        results,
        curve="lorentzian",
        parameters=parameters,
        fit_window=fit_window,
        use_rate=use_rate,
        show=show,
    )


def fit_g2_exponential(
    results,
    gamma,
    height,
    center,
    offset=None,
    fit_window=None,
    use_rate=True,
    show=True,
):
    """Fit a symmetric exponential peak using ``curve='expo'``."""
    parameters = {"height": height, "center": center, "gamma": gamma}
    if offset is not None:
        parameters["offset"] = offset
    return fit_g2(
        results,
        curve="expo",
        parameters=parameters,
        fit_window=fit_window,
        use_rate=use_rate,
        show=show,
    )


# Short alias matching the ``curve='expo'`` selector.
fit_g2_expo = fit_g2_exponential


# =============================================================================
# TIME-DEPENDENT G2 ANALYSIS
# =============================================================================
# Process progressively longer acquisitions and track the fitted coincidence
# peak centre, width, and count rate over time.

def g2_process_tstop_dependent(
    t_ps,
    ch,
    t_stop_start,
    t_stop_stop,
    t_stop_step,
    lorentzian_gamma,
    lorentzian_height,
    lorentzian_center,
    lorentzian_offset=None,
    fit_window=None,
    fit_use_rate=True,
    update_fit_guess=False,
    continue_on_error=True,
    show=True,
    t_center=6.25,
    t_span=40,
    bg_A=500,
    bg_B=740,
    freq=50e6,
    time_bin=1.0,
    chA=1,
    chB=2,
    sig_half=5,
    filename=None,
):
    """Run one cumulative g2 pass and Lorentzian-fit at t_stop checkpoints.

    t_stop_start, t_stop_stop, and t_stop_step are in seconds. Each t_stop value
    is checked during the same two-photon processing loop. When the channel-A
    photon time reaches a checkpoint, the current accumulated histogram is fit
    with a Lorentzian and the fitted center/FWHM are stored.
    """
    t_ps = np.asarray(t_ps).ravel()
    ch = np.asarray(ch).ravel()

    if len(t_ps) == 0 or len(ch) == 0:
        raise ValueError("t_ps and ch must not be empty.")
    if len(t_ps) != len(ch):
        raise ValueError("t_ps and ch must have the same length.")

    t_stop_start = float(t_stop_start)
    t_stop_stop = float(t_stop_stop)
    t_stop_step = float(t_stop_step)

    if not np.isfinite(t_stop_start) or not np.isfinite(t_stop_stop) or not np.isfinite(t_stop_step):
        raise ValueError("t_stop_start, t_stop_stop, and t_stop_step must be finite numbers.")
    if t_stop_start <= 0 or t_stop_stop <= 0:
        raise ValueError("t_stop_start and t_stop_stop must be positive seconds.")
    if t_stop_step <= 0:
        raise ValueError("t_stop_step must be positive seconds.")
    if t_stop_stop < t_stop_start:
        raise ValueError("t_stop_stop must be greater than or equal to t_stop_start.")

    t_ns = (t_ps - t_ps[0]) / 1000.0
    T_file_ns = t_ns[-1] - t_ns[0]
    T_file_s = T_file_ns * 1e-9

    t_stop_values = np.arange(t_stop_start, t_stop_stop + t_stop_step * 0.5, t_stop_step, dtype=float)
    t_stop_values = t_stop_values[t_stop_values <= t_stop_stop + t_stop_step * 1e-9]
    if len(t_stop_values) == 0:
        raise ValueError("No t_stop values were generated. Check start, stop, and step.")

    center_values = np.full(t_stop_values.shape, np.nan, dtype=float)
    fwhm_values = np.full(t_stop_values.shape, np.nan, dtype=float)
    center_uncertainty_values = np.full(t_stop_values.shape, np.nan, dtype=float)
    fwhm_uncertainty_values = np.full(t_stop_values.shape, np.nan, dtype=float)
    coincidence_values = np.full(t_stop_values.shape, np.nan, dtype=float)
    fit_results = [None] * len(t_stop_values)
    g2_results = [None] * len(t_stop_values)
    errors = [None] * len(t_stop_values)

    t_lo = t_center - t_span
    t_hi = t_center + t_span
    edges = np.arange(t_lo, t_hi + time_bin, time_bin)

    if len(edges) < 2:
        raise ValueError("Window too small for given bin width. Increase span or decrease bin.")

    centers = edges[:-1] + time_bin / 2
    H = np.zeros(len(centers), dtype=np.int64)

    tA = t_ns[ch == chA]
    tB = t_ns[ch == chB]

    numA_used = 0
    numPairs = 0

    t_col = 1e9
    k_col = 1
    num_SPA = 0
    SPA = []

    j_start = 0
    checkpoint_index = 0
    window_start_s = 0.0
    window_start_ns = 0.0
    window_start_s = 0.0
    window_start_ns = 0.0

    gamma_guess = float(lorentzian_gamma)
    height_guess = float(lorentzian_height)
    center_guess = float(lorentzian_center)
    offset_guess = lorentzian_offset
    if offset_guess is not None:
        offset_guess = float(offset_guess)

    def build_checkpoint_results(t_stop):
        t_stop = min(T_file_s, float(t_stop))
        t_stop_ns = t_stop * 1e9
        T_acq_s = t_stop - window_start_s

        if T_acq_s <= 0:
            raise ValueError("Checkpoint time window is empty.")

        cntA_total = int(np.sum((tA >= window_start_ns) & (tA < t_stop_ns)))
        cntB_total = int(np.sum((tB >= window_start_ns) & (tB < t_stop_ns)))

        sig_mask = (centers >= (t_center - sig_half)) & (centers <= (t_center + sig_half))
        bg = 0.0
        cmax = np.max(H) if len(H) else 0.0
        CC = np.sum(H[sig_mask]) - bg * np.sum(sig_mask)
        CAR = (cmax - bg) / max(bg, np.finfo(float).eps)
        CCps = CC / T_acq_s
        CCppulse = CCps / freq
        cntAps = cntA_total / T_acq_s - bg_A
        cntBps = cntB_total / T_acq_s - bg_B
        CC_raw = cntAps * cntBps / CCps if CCps != 0 else np.nan
        etaA = cntAps / CC_raw if CC_raw != 0 else np.nan
        etaB = cntBps / CC_raw if CC_raw != 0 else np.nan

        return {
            "filename": filename,
            "centers": centers.copy(),
            "histogram": H.copy(),
            "single_photon_counts": list(SPA),
            "T_acq_s": T_acq_s,
            "T_file_s": T_file_s,
            "t_start": window_start_s,
            "t_stop": float(t_stop),
            "time_bin": time_bin,
            "chA": chA,
            "chB": chB,
            "cntA_total": cntA_total,
            "cntB_total": cntB_total,
            "CC": CC,
            "CCps": CCps,
            "CCppulse": CCppulse,
            "CC_raw": CC_raw,
            "etaA": etaA,
            "etaB": etaB,
            "CAR": CAR,
            "numA_used": numA_used,
            "numPairs": numPairs,
        }

    def fit_checkpoint(index):
        nonlocal H, numA_used, numPairs, k_col, num_SPA, SPA, window_start_s, window_start_ns
        nonlocal gamma_guess, height_guess, center_guess, offset_guess

        t_stop = float(t_stop_values[index])
        print(f"--- Time-dependent g2 checkpoint: t_stop = {t_stop:.6g} s ---")

        results = build_checkpoint_results(t_stop)
        coincidence_values[index] = results["CC"]
        try:
            fit = fit_g2_lorentzian(
                results,
                gamma=gamma_guess,
                height=height_guess,
                center=center_guess,
                offset=offset_guess,
                fit_window=fit_window,
                use_rate=fit_use_rate,
                show=False,
            )

            center_values[index] = fit["center"]
            fwhm_values[index] = fit["fwhm"]
            center_uncertainty_values[index] = fit["center_uncertainty"]
            fwhm_uncertainty_values[index] = fit["fwhm_uncertainty"]

            if update_fit_guess:
                gamma_guess = float(fit["gamma"])
                height_guess = float(fit["height"])
                center_guess = float(fit["center"])
                offset_guess = float(fit["offset"])

        except Exception as exc:
            if not continue_on_error:
                raise

            print(f"Time-dependent g2 fit failed at t_stop={t_stop:.6g} s: {exc}")
            fit = None
            exc_value = exc
        else:
            exc_value = None

        g2_results[index] = results
        fit_results[index] = fit
        errors[index] = exc_value

        H.fill(0)
        numA_used = 0
        numPairs = 0
        num_SPA = 0
        SPA = []
        window_start_s = min(T_file_s, t_stop)
        window_start_ns = window_start_s * 1e9
        k_col = int(np.floor(window_start_s)) + 1

    for ti in tA:
        while checkpoint_index < len(t_stop_values) and ti >= t_stop_values[checkpoint_index] * 1e9:
            fit_checkpoint(checkpoint_index)
            checkpoint_index += 1

        if ti >= t_stop_values[-1] * 1e9:
            break

        numA_used += 1

        if ti >= k_col * t_col:
            SPA.append(num_SPA)
            num_SPA = 0
            k_col += 1

        num_SPA += 1

        while j_start < len(tB) and (tB[j_start] - ti) < t_lo:
            j_start += 1

        j = j_start
        while j < len(tB):
            dt = tB[j] - ti

            if dt > t_hi:
                break

            kbin = int(np.floor((dt - t_lo) / time_bin))

            if 0 <= kbin < len(H):
                H[kbin] += 1
                numPairs += 1

            j += 1

    while checkpoint_index < len(t_stop_values) and t_stop_values[checkpoint_index] <= T_file_s:
        fit_checkpoint(checkpoint_index)
        checkpoint_index += 1

    if checkpoint_index < len(t_stop_values):
        print(
            "Time-dependent g2 stopped before some t_stop values because the file "
            f"duration is {T_file_s:.6g} s."
        )

    fig, axes = plt.subplots(5, 1, sharex=True)
    axes[0].plot(t_stop_values, center_values, "o-")
    axes[0].set_ylabel("Center (ns)")
    axes[0].grid(True)

    axes[1].plot(t_stop_values, fwhm_values, "o-")
    axes[1].set_ylabel("FWHM (ns)")
    axes[1].grid(True)

    axes[2].plot(t_stop_values, center_uncertainty_values, "o-")
    axes[2].set_ylabel("Center uncertainty (ns)")
    axes[2].grid(True)

    axes[3].plot(t_stop_values, fwhm_uncertainty_values, "o-")
    axes[3].set_ylabel("FWHM uncertainty (ns)")
    axes[3].grid(True)

    axes[4].plot(t_stop_values, coincidence_values, "o-")
    axes[4].set_xlabel("t_stop (s)")
    axes[4].set_ylabel("Coincidences")
    axes[4].grid(True)

    fig.tight_layout()

    if show:
        plt.show()

    return {
        "t_stop": t_stop_values,
        "center": center_values,
        "fwhm": fwhm_values,
        "center_uncertainty": center_uncertainty_values,
        "fwhm_uncertainty": fwhm_uncertainty_values,
        "coincidences": coincidence_values,
        "g2_results": g2_results,
        "fit_results": fit_results,
        "errors": errors,
        "figure": fig,
        "axes": axes,
        "uncertainty_figure": fig,
        "uncertainty_axes": axes[2:4],
    }


def g2_process_time_dependent(
    t_ps,
    ch,
    t_stop_start,
    t_stop_stop,
    t_stop_step,
    fit_curve="lorentzian",
    fit_parameters=None,
    fit_window=None,
    fit_use_rate=True,
    update_fit_parameters=False,
    fit_center_from_peak=True,
    adaptive_t_center=False,
    display_values=None,
    continue_on_error=True,
    show=True,
    t_center=6.25,
    t_span=40,
    bg_A=500,
    bg_B=740,
    freq=50e6,
    time_bin=1.0,
    chA=1,
    chB=2,
    sig_half=5,
    show_fit=False,
):
    """Run one g2 pass and fit interval histograms at t_stop checkpoints.

    t_stop_start, t_stop_stop, and t_stop_step are in seconds. Each t_stop value
    is checked during the same two-photon processing loop. When the channel-A
    photon time reaches a checkpoint, the current interval histogram is fit
    using ``fit_curve`` and ``fit_parameters``. The fitted center/FWHM are
    stored, and the histogram is reset. ``fit_center_from_peak=True`` (the
    default) initializes the center at the current histogram maximum; otherwise
    supply ``center`` in ``fit_parameters``. When ``adaptive_t_center`` is True,
    each successful fit centers the next histogram window at its fitted center,
    rounded to 0.1 ns.

    ``fit_parameters`` contains curve-specific initial values, e.g.
    ``{"height": 15, "gamma": 0.7}`` for a Lorentzian, or
    ``{"height": 15, "gamma": 0.7, "sigma": 0.2}`` for
    ``fit_curve="expogaussconv"``. ``offset`` is optional. Set
    ``update_fit_parameters=True`` to use a successful fit as the next
    checkpoint's starting guess.

    ``display_values`` selects and orders the final trend panels. Choose from
    ``"center"``, ``"fwhm"``, ``"center_uncertainty"``,
    ``"fwhm_uncertainty"``, and ``"coincidences"``. By default, all five are
    displayed.
    """
    t_ps = np.asarray(t_ps).ravel()
    ch = np.asarray(ch).ravel()

    if len(t_ps) == 0 or len(ch) == 0:
        raise ValueError("t_ps and ch must not be empty.")
    if len(t_ps) != len(ch):
        raise ValueError("t_ps and ch must have the same length.")

    t_stop_start = float(t_stop_start)
    t_stop_stop = float(t_stop_stop)
    t_stop_step = float(t_stop_step)

    if not np.isfinite(t_stop_start) or not np.isfinite(t_stop_stop) or not np.isfinite(t_stop_step):
        raise ValueError("t_stop_start, t_stop_stop, and t_stop_step must be finite numbers.")
    if t_stop_start <= 0 or t_stop_stop <= 0:
        raise ValueError("t_stop_start and t_stop_stop must be positive seconds.")
    if t_stop_step <= 0:
        raise ValueError("t_stop_step must be positive seconds.")
    if t_stop_stop < t_stop_start:
        raise ValueError("t_stop_stop must be greater than or equal to t_stop_start.")

    fit_curve = FIT_CURVE_ALIASES.get(str(fit_curve).strip().lower(), str(fit_curve).strip().lower())
    if fit_curve not in FIT_CURVES:
        available = ", ".join(sorted(FIT_CURVES))
        raise ValueError(f"Unknown fit_curve '{fit_curve}'. Choose one of: {available}.")
    try:
        fit_parameter_guess = {} if fit_parameters is None else dict(fit_parameters)
    except (TypeError, ValueError) as exc:
        raise TypeError("fit_parameters must be a dictionary of initial fit values.") from exc

    fit_parameter_names = FIT_CURVES[fit_curve]["parameters"]
    unknown_parameters = set(fit_parameter_guess).difference(fit_parameter_names)
    if unknown_parameters:
        raise ValueError(
            f"Unsupported fit_parameters for {fit_curve}: {sorted(unknown_parameters)}. "
            f"Expected: {list(fit_parameter_names)}."
        )
    required_parameters = set(fit_parameter_names).difference({"center", "offset"})
    missing_parameters = required_parameters.difference(fit_parameter_guess)
    if missing_parameters:
        raise ValueError(
            f"Missing fit_parameters for {fit_curve}: {sorted(missing_parameters)}."
        )
    if not fit_center_from_peak and "center" not in fit_parameter_guess:
        raise ValueError("fit_parameters must include 'center' when fit_center_from_peak=False.")

    t_ns = (t_ps - t_ps[0]) / 1000.0
    T_file_ns = t_ns[-1] - t_ns[0]
    T_file_s = T_file_ns * 1e-9

    t_stop_values = np.arange(t_stop_start, t_stop_stop + t_stop_step * 0.5, t_stop_step, dtype=float)
    t_stop_values = t_stop_values[t_stop_values <= t_stop_stop + t_stop_step * 1e-9]
    if len(t_stop_values) == 0:
        raise ValueError("No t_stop values were generated. Check start, stop, and step.")

    center_values = np.full(t_stop_values.shape, np.nan, dtype=float)
    fwhm_values = np.full(t_stop_values.shape, np.nan, dtype=float)
    center_uncertainty_values = np.full(t_stop_values.shape, np.nan, dtype=float)
    fwhm_uncertainty_values = np.full(t_stop_values.shape, np.nan, dtype=float)
    coincidence_values = np.full(t_stop_values.shape, np.nan, dtype=float)
    fit_results = [None] * len(t_stop_values)
    g2_results = [None] * len(t_stop_values)
    errors = [None] * len(t_stop_values)

    t_lo = t_center - t_span
    t_hi = t_center + t_span
    edges = np.arange(t_lo, t_hi + time_bin, time_bin)

    if len(edges) < 2:
        raise ValueError("Window too small for given bin width. Increase span or decrease bin.")

    centers = edges[:-1] + time_bin / 2
    H = np.zeros(len(centers), dtype=np.int64)

    tA = t_ns[ch == chA]
    tB = t_ns[ch == chB]

    numA_used = 0
    numPairs = 0

    t_col = 1e9
    k_col = 1
    num_SPA = 0
    SPA = []

    j_start = 0
    checkpoint_index = 0
    window_start_s = 0.0
    window_start_ns = 0.0

    def build_checkpoint_results(t_stop):
        t_stop = min(T_file_s, float(t_stop))
        t_stop_ns = t_stop * 1e9
        T_acq_s = t_stop - window_start_s

        if T_acq_s <= 0:
            raise ValueError("Checkpoint time window is empty.")

        cntA_total = int(np.sum((tA >= window_start_ns) & (tA < t_stop_ns)))
        cntB_total = int(np.sum((tB >= window_start_ns) & (tB < t_stop_ns)))

        sig_mask = (centers >= (t_center - sig_half)) & (centers <= (t_center + sig_half))
        bg = 0.0
        cmax = np.max(H) if len(H) else 0.0
        CC = np.sum(H[sig_mask]) - bg * np.sum(sig_mask)
        CAR = (cmax - bg) / max(bg, np.finfo(float).eps)
        CCps = CC / T_acq_s
        CCppulse = CCps / freq
        cntAps = cntA_total / T_acq_s - bg_A
        cntBps = cntB_total / T_acq_s - bg_B
        CC_raw = cntAps * cntBps / CCps if CCps != 0 else np.nan
        etaA = cntAps / CC_raw if CC_raw != 0 else np.nan
        etaB = cntBps / CC_raw if CC_raw != 0 else np.nan

        return {
            "centers": centers.copy(),
            "histogram": H.copy(),
            "single_photon_counts": list(SPA),
            "T_acq_s": T_acq_s,
            "T_file_s": T_file_s,
            "t_start": window_start_s,
            "t_stop": float(t_stop),
            "time_bin": time_bin,
            "chA": chA,
            "chB": chB,
            "cntA_total": cntA_total,
            "cntB_total": cntB_total,
            "CC": CC,
            "CCps": CCps,
            "CCppulse": CCppulse,
            "CC_raw": CC_raw,
            "etaA": etaA,
            "etaB": etaB,
            "CAR": CAR,
            "numA_used": numA_used,
            "numPairs": numPairs,
        }

    def fit_checkpoint(index):
        nonlocal H, numA_used, numPairs, k_col, num_SPA, SPA, window_start_s, window_start_ns
        nonlocal fit_parameter_guess
        nonlocal t_center, t_lo, t_hi, edges, centers

        t_stop = float(t_stop_values[index])
        print(f"--- Time-dependent g2 checkpoint: t_stop = {t_stop:.6g} s ---")

        results = build_checkpoint_results(t_stop)
        coincidence_values[index] = results["CC"]
        try:
            checkpoint_parameters = fit_parameter_guess.copy()
            if fit_center_from_peak or "center" not in checkpoint_parameters:
                checkpoint_parameters["center"] = float(
                    results["centers"][np.argmax(results["histogram"])]
                )
            fit = fit_g2(
                results,
                curve=fit_curve,
                parameters=checkpoint_parameters,
                fit_window=fit_window,
                use_rate=fit_use_rate,
                show=show_fit,
            )

            center_values[index] = fit["center"]
            fwhm_values[index] = fit["fwhm"]
            center_uncertainty_values[index] = fit["center_uncertainty"]
            fwhm_uncertainty_values[index] = fit["fwhm_uncertainty"]

            if update_fit_parameters and fit["success"]:
                fit_parameter_guess.update(fit["parameters"])

        except Exception as exc:
            if not continue_on_error:
                raise

            print(f"Time-dependent g2 fit failed at t_stop={t_stop:.6g} s: {exc}")
            fit = None
            exc_value = exc
        else:
            exc_value = None

        g2_results[index] = results
        fit_results[index] = fit
        errors[index] = exc_value

        if adaptive_t_center and fit is not None:
            t_center = round(float(fit["center"]), 1)
            t_lo = t_center - t_span
            t_hi = t_center + t_span
            edges = np.arange(t_lo, t_hi + time_bin, time_bin)
            centers = edges[:-1] + time_bin / 2
            H = np.zeros(len(centers), dtype=np.int64)
        else:
            H.fill(0)
        numA_used = 0
        numPairs = 0
        num_SPA = 0
        SPA = []
        window_start_s = min(T_file_s, t_stop)
        window_start_ns = window_start_s * 1e9
        k_col = int(np.floor(window_start_s)) + 1

    for ti in tA:
        while checkpoint_index < len(t_stop_values) and ti >= t_stop_values[checkpoint_index] * 1e9:
            fit_checkpoint(checkpoint_index)
            checkpoint_index += 1

        if ti >= t_stop_values[-1] * 1e9:
            break

        numA_used += 1

        if ti >= k_col * t_col:
            SPA.append(num_SPA)
            num_SPA = 0
            k_col += 1

        num_SPA += 1

        while j_start < len(tB) and (tB[j_start] - ti) < t_lo:
            j_start += 1

        j = j_start
        while j < len(tB):
            dt = tB[j] - ti

            if dt > t_hi:
                break

            kbin = int(np.floor((dt - t_lo) / time_bin))

            if 0 <= kbin < len(H):
                H[kbin] += 1
                numPairs += 1

            j += 1

    while checkpoint_index < len(t_stop_values) and t_stop_values[checkpoint_index] <= T_file_s:
        fit_checkpoint(checkpoint_index)
        checkpoint_index += 1

    if checkpoint_index < len(t_stop_values):
        print(
            "Time-dependent g2 stopped before some t_stop values because the file "
            f"duration is {T_file_s:.6g} s."
        )

    available_display_values = {
        "center": (center_values, "Center (ns)"),
        "fwhm": (fwhm_values, "FWHM (ns)"),
        "center_uncertainty": (center_uncertainty_values, "Center uncertainty (ns)"),
        "fwhm_uncertainty": (fwhm_uncertainty_values, "FWHM uncertainty (ns)"),
        "coincidences": (coincidence_values, "Coincidences"),
    }
    if display_values is None:
        display_values = tuple(available_display_values)
    elif isinstance(display_values, str):
        display_values = (display_values,)
    else:
        display_values = tuple(display_values)

    if not display_values:
        raise ValueError("display_values must contain at least one metric.")
    invalid_display_values = set(display_values) - set(available_display_values)
    if invalid_display_values:
        raise ValueError(
            "Unknown display value(s): "
            f"{sorted(invalid_display_values)}. Choose from "
            f"{list(available_display_values)}."
        )
    if len(set(display_values)) != len(display_values):
        raise ValueError("display_values must not contain duplicate metrics.")

    fig, axes = plt.subplots(len(display_values), 1, sharex=True, squeeze=False)
    axes = axes[:, 0]
    for axis, display_value in zip(axes, display_values):
        values, ylabel = available_display_values[display_value]
        axis.plot(t_stop_values, values, "o-")
        axis.set_ylabel(ylabel)
        axis.grid(True)

    axes[-1].set_xlabel("t_stop (s)")

    fig.tight_layout()

    if show:
        plt.show()

    return {
        "fit_curve": fit_curve,
        "fit_parameters": fit_parameter_guess.copy(),
        "t_stop": t_stop_values,
        "center": center_values,
        "fwhm": fwhm_values,
        "center_uncertainty": center_uncertainty_values,
        "fwhm_uncertainty": fwhm_uncertainty_values,
        "coincidences": coincidence_values,
        "g2_results": g2_results,
        "fit_results": fit_results,
        "errors": errors,
        "display_values": display_values,
        "figure": fig,
        "axes": axes,
        "uncertainty_figure": fig,
        "uncertainty_axes": np.asarray(
            [
                axes[index]
                for index, value in enumerate(display_values)
                if value in ("center_uncertainty", "fwhm_uncertainty")
            ],
            dtype=object,
        ),
    }


# =============================================================================
# PLOTTING AND HIGH-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

def g2_plot(results, show=True):
    """Plot single photon counts and coincidence histogram from processed results."""
    SPA = results["single_photon_counts"]
    centers = results["centers"]
    H = results["histogram"]
    T_acq_s = results["T_acq_s"]
    # chA = results["chA"]
    # chB = results["chB"]

    # ---------------- Plot single photon count trend ----------------
    plt.figure()
    plt.plot(SPA)
    plt.ylim(bottom=0)
    plt.xlabel("Time bin index")
    plt.ylabel("Counts per bin")
    plt.title(f"Single photon counts on channel 1")
    plt.grid(True)

    # ---------------- Plot coincidence histogram ----------------
    plt.figure()
    plt.plot(centers, H / T_acq_s, linewidth=1.5)
    plt.xlabel(f"Time Delay t2 - t1 (ns)")
    plt.ylabel("Coincidence rate (per second)")
    plt.grid(True)

    ax = plt.gca()
    ax.spines["top"].set_linewidth(1)
    ax.spines["right"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.spines["left"].set_linewidth(1)
    ax.tick_params(labelsize=14)

    plt.ylim(0, 1.1 * np.max(H / T_acq_s + np.finfo(float).eps))

    if show:
        plt.show()


def process_qcs_delay(*args, **kwargs):
    """Alias for g2_process()."""
    return g2_process(*args, **kwargs)


def plot_qcs_delay_results(*args, **kwargs):
    """Backward-compatible name for g2_plot()."""
    return g2_plot(*args, **kwargs)


def estimate_qcs_delay(t_ps, ch, show_plots=True, **kwargs):
    """Process loaded timetag arrays, optionally plot the results, and return metrics."""
    results = g2_process(t_ps, ch, **kwargs)

    if show_plots:
        g2_plot(results)

    return results


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def _build_argument_parser():
    """Create the small, safe command-line interface for common workflows."""
    parser = argparse.ArgumentParser(
        description="Load, convert, truncate, and inspect QCS timetag files."
    )
    parser.add_argument(
        "--directory",
        metavar="PATH",
        help="starting folder for any file dialog used by this command",
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    inspect_parser = commands.add_parser(
        "inspect", help="load a timetag file and print a short summary"
    )
    inspect_parser.add_argument("filename", nargs="?", help=".npz, .h5, .hdf5, or .mat file")

    convert_parser = commands.add_parser(
        "convert", help="merge Time Tagger X .ttbin and matching HydraHarp timetags into H5"
    )
    convert_parser.add_argument("filename", nargs="?", help="Time Tagger X .ttbin file")
    convert_parser.add_argument(
        "--fast", action="store_true", help="use the legacy QuTAG in-memory conversion"
    )

    truncate_parser = commands.add_parser(
        "truncate", help="write an H5 copy ending at a specified time"
    )
    truncate_parser.add_argument("seconds", type=float, help="duration to retain, in seconds")
    truncate_parser.add_argument("filename", nargs="?", help="input timetag file")

    load_time_parser = commands.add_parser(
        "load-time", help="open and plot stored time-dependent g2 results"
    )
    load_time_parser.add_argument("filename", nargs="?", help="stored .npz results file")

    load_g2_parser = commands.add_parser(
        "load-g2", help="load stored g2 results"
    )
    load_g2_parser.add_argument("filename", nargs="?", help="stored .npz results file")
    return parser


def main(argv=None):
    """Run a common workflow without embedding experiment-specific settings."""
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    if args.directory:
        set_default_file_dialog_directory(args.directory)

    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "inspect":
            t_ps, ch, filename = load_timetag(args.filename, return_filename=True)
            duration_s = (np.max(t_ps) - np.min(t_ps)) * 1e-12 if len(t_ps) else 0.0
            print(f"File: {filename}")
            print(f"Events: {len(t_ps):,}")
            print(f"Channels: {np.unique(ch).tolist()}")
            print(f"Duration: {duration_s:.6g} s")
        elif args.command == "convert":
            converter = timetag_to_h5_fast if args.fast else timetag_to_h5
            converter(args.filename)
        elif args.command == "truncate":
            truncate_timetag(args.seconds, args.filename)
        elif args.command == "load-time":
            load_time_dependent_results(args.filename)
        elif args.command == "load-g2":
            load_g2_results(args.filename)
    except FileSelectionCancelled as exc:
        print(exc)
        return 1
    return 0


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # import numpy as np
    # from qcs_analyzer._implementation import _g2calib_2, _g2calib_1, _g2calib_2_Hfunc
    # x = np.linspace(-10000, 10000, 10000)
    # y = _g2calib_2(
    #     x,
    #     height=1.0,
    #     center=0.0,
    #     # sigma=1000.0,
    #     sigma=0.0,
    #     offset=0.0,
    # )
    # plt.plot(x, y)
    # plt.xlabel("Delay (ps)")
    # plt.ylabel(r"$g^{(2)}$")
    # plt.grid(True)
    # plt.show()

    raise SystemExit(main())
