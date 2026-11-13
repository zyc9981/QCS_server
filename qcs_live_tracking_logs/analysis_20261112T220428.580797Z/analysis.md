# Four-second live tracking analysis

Source: /home/zyc/QCS/QCS_server/qcs_live_tracking_logs/qcs_live_tracking_20261112T220428.580797Z.csv

The log contains 20 analysis points over 76.00 s, with approximately four-second spacing. The command/request ratio confirms gain 0.3 in this file, independent of the current source-code default. There are 18 applied increments and one normal below-resolution zero adjustment, plus the first measurement-only point. All acquisition checks pass, there are no missing interval indices, and no command in this log is failed or stopped.

| Metric | 1 s, gain 0.3 | 4 s, gain 0.3 |
|---|---:|---:|
| Analysis points | 50 | 20 |
| Record duration | 49.01 s | 76.00 s |
| Acquisition end to analyzer return | 0.212 s | 0.442 s |
| Analyzer return to acknowledgment | 0.122 s | 0.115 s |
| Acquisition end to acknowledgment | 0.335 s | 0.555 s |
| Acquisition-end-to-acknowledgment / interval | 33.5% | 13.9% |
| Analyzer-return-to-acknowledgment / interval | 12.2% | 2.9% |
| Tau standard deviation | 23.51 ps | 27.68 ps |
| Tau peak-to-peak | 99.34 ps | 103.31 ps |
| Successive tau-change standard deviation | 13.75 ps | 32.90 ps |
| u0 standard deviation, first point excluded | 13.61 ps/s | 8.20 ps/s |
| Approximate strongest tau period | 15.7 s | 25.1 s |

## Interpretation

The user's expectation about relative computational/communication latency is confirmed: the post-acquisition acknowledgment delay is a smaller fraction of a four-second interval. It does not follow that the feedback measurement is instantaneous. Tau is fitted from the completed acquisition window, and u0 is calculated from two consecutive fitted offsets. With approximately uniform interval weighting, its effective measurement time is around one acquisition interval before the current interval finishes. The approximate measurement age at acknowledgment is therefore T plus the acquisition-end-to-acknowledgment delay: about 1.34 s for the one-second run versus 4.55 s here. These are timing-model approximations, not hardware timestamps; actual clock application precedes the observed acknowledgment by an unknown amount.

Longer acquisition also means fewer feedback updates. Reducing post-processing latency as a fraction of the interval does not remove acquisition averaging, finite-difference lag, or the slower controller update rate. This record does not show an improvement in phase scatter: tau ranges from -3476.28 to -3372.97 ps, approximately 103.3 ps peak-to-peak. Its broad strongest period is roughly 25 s, approximately six updates per cycle. Only three such cycles were recorded, so the period estimate is approximate and does not establish a steady periodic hardware oscillation.

The smaller u0 scatter should not be interpreted by itself as better phase stability. The same offset change divided by four seconds produces a rate four times smaller than when divided by one second; longer acquisition also averages faster variation. In this dataset the successive offset jumps are larger even though the rate estimate is smaller.

At four-second sampling, periods shorter than eight seconds cannot be resolved uniquely. The earlier approximately six-second oscillation cannot be directly ruled out from this record; high-frequency phase variations can be averaged down or aliased. Consequently the persistent phase variation weakens an explanation based solely on post-processing/network latency, but it cannot isolate feedback, device response, physical drift, and fit noise.

The sign, gain, and units are internally consistent. Actual versus requested delta differs by at most 0.172 ps/s. Logged u0 matches the finite difference of tau over T_lastmid to numerical precision. The initial u0 of -344.11 ps/s was skipped as intended. Mean tau differs across runs because initialization provides a different constant reference; this is not a direct clock-accuracy comparison.

## Suggested next diagnostic

Keep the four-second interval and gain fixed for a longer run, then compare against a record at the same acquisition interval with live clock updates disabled while analysis continues. A similar phase variation without feedback would point toward input data, fitting, or physical phase changes; a substantial dependence on feedback would support controller/clock-response dynamics. Record fit-center uncertainty and source event-time ranges to separate fitting noise from actual offset motion. Hardware actuation or node timestamps would refine the delay estimate beyond the acknowledgment time.

Application code was not modified by this analysis.
