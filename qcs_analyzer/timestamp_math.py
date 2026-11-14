"""Integer picosecond arithmetic shared by live acquisition and alignment."""

from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
import operator

import numpy as np

INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1


def integer_scalar(value, name):
    """Require a real integer, without truncating floats or decimal strings."""
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f'{name} must be an integer') from exc


def milliseconds_to_picoseconds(value):
    """Round decimal milliseconds to the nearest integer ps (ties to even)."""
    try:
        milliseconds = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError('latency_ms must be a finite nonnegative number') from exc
    if not milliseconds.is_finite() or milliseconds < 0:
        raise ValueError('latency_ms must be a finite nonnegative number')
    with localcontext() as context:
        context.prec = max(28, len(milliseconds.as_tuple().digits) + 10)
        result = int((milliseconds * 1_000_000_000).to_integral_value(rounding=ROUND_HALF_EVEN))
    if result > INT64_MAX:
        raise OverflowError('latency_ms exceeds the int64 picosecond range')
    return result


def int64_timestamps(values, name='timestamps'):
    """Reject floating timestamps and unsigned values that would wrap on cast."""
    array = np.asarray(values)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f'{name} must be a one-dimensional array of integer picoseconds')
    if np.issubdtype(array.dtype, np.unsignedinteger) and array.size:
        if int(array.max()) > INT64_MAX:
            raise OverflowError(f'{name} exceeds the signed int64 picosecond range')
    return array.astype(np.int64, copy=False)


def integer_resolution_ps(value):
    """Accept integral resolution metadata without a potentially lossy float cast."""
    if isinstance(value, (int, np.integer)):
        result = int(value)
    else:
        try:
            resolution = Decimal(str(value))
        except InvalidOperation as exc:
            raise ValueError('HH base resolution must be a positive integer in ps') from exc
        if not resolution.is_finite() or resolution != resolution.to_integral_value():
            raise ValueError('HH base resolution must be a positive integer in ps')
        result = int(resolution)
    if not 0 < result <= INT64_MAX:
        raise ValueError('HH base resolution must be a positive int64 integer in ps')
    return result


def shift_picoseconds(values, shift_ps, *, out=None):
    """Shift integer tags with checked int64 arithmetic, optionally into a buffer."""
    array = int64_timestamps(values)
    shift = integer_scalar(shift_ps, 'picosecond shift')
    if not INT64_MIN <= shift <= INT64_MAX:
        raise OverflowError('Picosecond shift exceeds the int64 range')
    if array.size:
        if int(array.min()) + shift < INT64_MIN or int(array.max()) + shift > INT64_MAX:
            raise OverflowError('Shifted timestamps exceed the int64 picosecond range')
    return np.add(array, np.int64(shift), out=out)


def midpoint_elapsed_picoseconds(first_ps, last_ps, state):
    """Subtract exact doubled midpoints before converting the short delta to float."""
    twice_midpoint = integer_scalar(first_ps, 'first timestamp') + integer_scalar(
        last_ps, 'last timestamp')
    previous_twice = state.get('T_lastmid_twice_ps')
    if previous_twice is None:
        # Accept legacy/manual state with no companion field. Existing floats
        # cannot regain already lost bits; subsequent intervals use exact ints.
        previous = Decimal(str(state['T_lastmid_ps'])) * 2
        if not previous.is_finite() or previous != previous.to_integral_value():
            raise ValueError('T_lastmid_ps must be a finite multiple of half a picosecond')
        previous_twice = int(previous)
    else:
        previous_twice = integer_scalar(previous_twice, 'T_lastmid_twice_ps')
    elapsed_twice = twice_midpoint - previous_twice
    if elapsed_twice <= 0:
        raise ValueError('Live interval midpoint did not advance')
    if elapsed_twice > 1 << 53:
        raise ValueError('Interval midpoint gap is too large for exact half-picosecond conversion')
    return twice_midpoint, elapsed_twice / 2.0
