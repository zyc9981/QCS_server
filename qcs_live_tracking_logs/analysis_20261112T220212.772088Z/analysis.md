# Diagnostic gain 0.3 compared with gain 1.0

Source: /home/zyc/QCS/QCS_server/qcs_live_tracking_logs/qcs_live_tracking_20261112T220212.772088Z.csv

The new log contains 50 analysis results over 49.01 s, with no missing interval indices. All acquisition entries report checks passed. There are 46 applied adjustments, two successful below-resolution zero changes (intervals 13 and 16), and one final stopped command. Both zero changes were followed by further analysis and clock updates. Every logged request equals -0.3 times measured u0.

| Metric | Gain 1.0 | Gain 0.3 |
|---|---:|---:|
| Duration | 36.00 s | 49.01 s |
| u0 standard deviation, excluding first measurement-only value | 24.28 ps/s | 13.61 ps/s |
| Tau standard deviation, whole run | 24.71 ps | 23.51 ps |
| Tau standard deviation, after 10 s | 26.69 ps | 22.72 ps |
| Tau peak-to-peak, whole run | 96.76 ps | 99.34 ps |
| Approximate strongest tau period | 5.7 s | 15.7 s |
| Fraction of tau spectral power in 5-7 s band | 57.0% | 8.8% |
| Mean acquisition-end to analyzer return | 0.302 s | 0.212 s |
| Mean acquisition-end to adjustment acknowledgment | 0.432 s | 0.335 s |
| RMS applied clock increment | 23.81 ps/s | 4.12 ps/s |

## Interpretation

The rapid skew swings decreased: u0 standard deviation fell by 43.9%. The earlier approximately six-second mode is much weaker. However, tau scatter fell only 4.9% over the complete runs, or 14.9% after excluding the first 10 s. The lower gain trades rapid correction for slower response; appreciable slow offset variation remains.

The new tau record has a broad strongest period near 15-16 s (about three cycles in this record). The largest later excursion runs from -1984.95 ps at about 29 s to -1898.42 ps at about 33 s, then back to -1977.94 ps around 40 s. Consequently this run is not evidence that offset oscillation has disappeared. u0 also retains shorter-period fluctuations; taking the derivative of tau emphasizes those components, so its strongest spectral period can differ from tau's.

The full-gain and low-gain runs started with different tau offsets and residual skews. Absolute tau baselines should not be interpreted as accuracy changes. Processing and acknowledgment delays also decreased in the new run, so this was not a strictly controlled gain-only experiment. The log still lacks histograms and fit uncertainties, preventing separation of fit noise, physical phase changes, and feedback dynamics.

The gain, command direction, and unit conversion are consistent with the intended implementation. Requested versus applied correction differs by at most 0.172 ps/s, consistent with divider rounding. Logged u0 reproduces the finite difference of tau over T_lastmid to numerical precision. No command failed in the new log; the final command is labeled stopped.

## Next diagnostic

Keep gain 0.3 fixed and obtain a longer record (roughly 3-5 min) to resolve the slower variation. For a comparison, collect a short record with live clock adjustments disabled while preserving analysis and plotting. If similar tau variation persists without feedback, investigate fitting and physical timing changes; if the oscillation depends strongly on correction gain, controller dynamics remain the main explanation. Add fit uncertainty or selected histogram diagnostics before increasing gain again.

This analysis did not modify application code.
