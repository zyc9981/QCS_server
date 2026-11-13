# Analysis of qcs_live_tracking_20261112T173157.289121Z.csv

The log contains 37 fitted intervals over 36.00 s, with 35 applied adjustments, one initial skipped adjustment, and one final command stopped during shutdown.

## Measurements

- Dominant tau oscillation: approximately 5.7 s. Only about six cycles were recorded; this period is approximate.
- Tau range: -1876.43 to -1779.67 ps (96.76 ps peak-to-peak).
- Residual u0 range after the first measurement-only sample: -42.54 to 51.53 ps/s.
- Acquisition spacing: 0.9999 s, standard deviation 0.0070 s; no interval-index gaps.
- Acquisition completion to analyzer return: mean 0.302 s, range 0.228 to 0.423 s.
- Acquisition completion to observed adjustment completion: mean 0.432 s, range 0.331 to 0.575 s.
- Analyzer return to observed adjustment completion: mean 0.128 s.
- Node-reported clock-write time: mean 0.040 s; network, command queueing, and server polling account for additional elapsed time.
- Applied increment agrees with negative measured u0 within 0.171 ps/s.
- Logged u0 agrees with finite differences of tau over T_lastmid within 1.71e-13 ps/s.

## Interpretation

The strongest explanation is delayed closed-loop ringing with unity correction gain. The analyzer measures phase from a completed interval; its finite difference estimates skew over two consecutive phase points. The node is changed while the following acquisition is already running. Therefore the next fitted result still partly reflects older clock settings, but another full correction is issued from it.

The clock-command sign and units match the requested rule. Command quantization is much smaller than the observed skew swings. The first anomalous u0 (-418.65 ps/s) was skipped and did not drive the node. Acquisition and acknowledgment timing do not show a five-to-six-second scheduling cycle or a growing queue backlog.

An illustrative model with one-second, uniformly weighted phase windows and actuator steps at the mean observed acknowledgment time predicts a 6.1 s oscillation at gain 1. This is close to the observed 5.7 s period. The model has damped poles, so it supports ringing; it does not establish unstable hardware or self-sustained oscillation. Fit noise and additional unmeasured latency can excite or sustain that response.

The clock acknowledgment is the server's observation of completion, rather than a hardware-timestamped actuator event. Arrival-time windows do not record the HH event-time boundaries. No histogram, fit uncertainty, or actual clock readback is in this log. Therefore fit noise, device response delay, and changes in the physical time offset cannot be independently separated here.

## Suggested next experiment

Introduce a configurable feedback gain and try sending `-0.3 * u0` while keeping the acquisition interval fixed. In the same illustrative model the oscillatory pole magnitude drops from about 0.92 to 0.56 per interval. This is a diagnostic experiment, not a measured hardware stability guarantee. A short interval with live adjustments disabled provides a comparison for fitting noise and uncontrolled drift. Log source timestamp ranges, fit center uncertainty, and command-send/receive timestamps to identify the remaining delay.

Application code was not changed by this analysis.
