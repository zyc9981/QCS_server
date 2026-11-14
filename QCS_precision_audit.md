# QCS timestamp precision audit

Scope: the active live path in `QCS_algorithm.py`, HH T2 decoding in
`qcs_analyzer/_implementation.py`, global initialization in
`qcs_analyzer/global_freq_analysis.py`, `fit_g2`, the GUI latency input, and
server clock-command serialization. Other offline routines in `qcs_kernal.py`
are not called by the current live analyzer and are outside this audit.

## Findings and changes

| Stage | Units and representation | Result |
|---|---|---|
| TTX SDK input | Integer ps | Validate before conversion; reject floats and unsigned values outside signed int64 rather than truncate or wrap. |
| HH decoding | Integer ticks multiplied by integer ps/tick | Preserve integer overflow state and resolution; check overflow before NumPy additions/multiplications can wrap. |
| Unix capture times | Integer microseconds | Subtract the two epoch values as Python integers, then multiply the difference by 1,000,000. Never form an absolute Unix timestamp in ps. Reject float metadata instead of silently truncating it. |
| GUI latency | Decimal ms, then integer ps | Forward the entry text to the controller. Convert once with decimal arithmetic, rounding to nearest ps, ties to even. Quantization is at most half a ps. |
| Preallocation/rebasing | Signed int64 ps | Check shift bounds, subtract the calibrated origin using integer arithmetic, write into int64 channel buffers. |
| Initialization | Integer ps until rebased | Subtract a common origin as integers, then use float64 for fractional skew and FFT calculations. Reject unsupported spans instead of silently losing whole-ps bits. |
| Live histograms | Integer event times, float delay edges | Numba subtracts event times as int64 before comparison/binning with float edges. Bounds protect pair subtraction from signed overflow. Histogram counts are int64. |
| Fits | Float64 ps, count/rate values | Floating point is intentional: the fitted center can be fractional ps. The tested RT center near 58,968,630 ps has float spacing around 0.0000000075 ps. |
| Live midpoint | Exact integer twice-midpoint | Sum Python integers, store `T_lastmid_twice_ps`, subtract consecutive twice-midpoints before converting the short interval delta to float. Avoid NumPy sum overflow and subtraction of rounded large float midpoints. |
| `u0` | Float ps/ps | Difference of fitted centers divided by midpoint separation in ps. Multiply by 1e12 for ps/s. |
| Live command | Float ps/s | Send `-live_correction_gain * u0 * 1e12` through JSON. The server does not format this into a rounded display string before sending. |
| Plots/logs | Seconds/ms/ps for presentation | Display `T_lastmid_ps` can be approximate at extreme times, but the next skew calculation uses its exact companion. Event-relative seconds are now formatted from integer ns directly. |

The original midpoint expression `(t1_ps[0] + t1_ps[-1]) / 2` was vulnerable
to int64 addition overflow at extreme values and loss of half-ps resolution on
long runs. These are now guarded against. They were not large enough to explain
the observed hundreds of ps of offset variation over a 632-second run.

## Exact correction equations

```text
latency_ps = round_decimal(latency_ms * 1,000,000,000)
offset_ps = (HH_start_us - TTX_start_us) * 1,000,000 - latency_ps

initial_TTX = TTX_tags - first_TTX_tag
initial_HH  = HH_tags - first_HH_tag + offset_ps

calibrated_offset_ps = offset_ps + round(global_additive_residual_ps)
live_TTX = TTX_tags - first_TTX_tag
live_HH  = HH_tags - (first_HH_tag - calibrated_offset_ps)
```

All tag corrections above are checked integer operations. Rounding the global
fractional residual to integer ps is deliberate and introduces a fixed error
of at most half a ps. It does not create an interval-to-interval oscillation.

## Floating-point limits and remaining qualifications

- Signed int64 holds approximately 106.75 days of ps. Out-of-range input,
  shifts, decoded values, and pair differences now fail explicitly.
- Float64 represents all whole integers through 2^53, corresponding to about
  9,007 seconds in ps. Global alignment rejects larger relative spans before
  conversion. Fractional skew operations remain approximate: at 2 seconds,
  float spacing is approximately 0.000244 ps. This is much smaller than the
  current live 20 ps histogram bins.
- The live histogram avoids converting large session timestamps to float;
  its timestamp differences stay integer. As a comparison, converting a
  632-second timestamp to float first would have 0.125 ps spacing.
- Fractional midpoint intervals are converted only after exact integer
  subtraction. Gaps above the supported exact half-ps range are rejected.
- Host start times are intentionally recorded in microseconds. Their 1 us
  recording resolution, NTP accuracy and SDK start timing are measurement
  limits, not loss from picosecond arithmetic. These host times are used at
  initialization, not repeatedly subtracted during live tracking.
- The fit and FFT are numerical estimators, not exact physical measurements.
  Histogram binning, optimizer convergence, model mismatch, counting noise and
  hardware instability remain possible error sources. This audit does not
  establish the cause of the observed non-RT oscillation.

## Verification

The unittest suite passes 18 tests, including synthetic regressions for decimal
latency entered through the GUI, Unix-us-to-ps correction, large common origins,
large HH overflow state, resolution scaling, signed/unsigned overflow rejection,
preallocated integer rebasing, histogram translation invariance, global
alignment translation invariance near int64 limits, and consecutive exact
midpoint/skew updates. Existing clock-result and GUI tests were updated for the
current separated-buffer interface and checkbox.

A separate noiseless `g2calib_2` fit check recovered a 13.25 ps peak at both a
zero origin and a 58,968,630 ps origin to better than 0.000001 ps. This tests
basic numerical conditioning at the current RT offset; it does not bound the
fit error with real photon data.

No hardware acquisition or Windows clock-driver validation was performed.
