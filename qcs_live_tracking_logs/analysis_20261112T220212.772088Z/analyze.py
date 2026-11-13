"""Compare the diagnostic 0.3-gain log with the earlier full-gain run."""
from collections import Counter
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import periodogram


directory = Path(__file__).resolve().parent
logs = directory.parent


def load(stamp, label):
    source = logs / f'qcs_live_tracking_20261112T{stamp}.csv'
    rows = list(csv.DictReader(source.open()))
    fitted = [r for r in rows if r['event'] == 'analysis_finished']
    acquisitions = {int(r['interval']): r for r in rows if r['event'] == 'acquisition_finished'}
    commands = [r for r in rows if r['event'] == 'clock_adjustment_finished']
    completed = [r for r in commands if r['clock_status'] in ('applied', 'unchanged (below resolution)')]
    seq = np.array([int(r['interval']) for r in fitted])
    t = np.array([float(acquisitions[s]['relative_s']) for s in seq])
    tau = np.array([float(r['tau_ps']) for r in fitted])
    u0 = np.array([float(r['u0_ps_per_s']) for r in fitted])
    midpoint = np.array([
        float(r['T_lastmid_ms']) * 1e-3 if r.get('T_lastmid_ms')
        else float(r['T_lastmid_ps']) * 1e-12 for r in fitted])
    processing = np.array([float(r['relative_s']) - t[i] for i, r in enumerate(fitted)])
    ack_t = np.array([float(acquisitions[int(r['interval'])]['relative_s']) for r in completed])
    ack_delay = np.array([float(r['relative_s']) - ack_t[i] for i, r in enumerate(completed)])
    actual = np.array([float(r['applied_correction_ps_per_s']) for r in completed])
    requested = np.array([float(r['correction_ps_per_s']) for r in completed])
    gains = np.array([-float(r['correction_ps_per_s']) / float(r['u0_ps_per_s'])
                      for r in commands if float(r['u0_ps_per_s']) != 0])
    f, power = periodogram(tau, fs=1 / np.median(np.diff(t)), nfft=16384, detrend='linear')
    f_u, power_u = periodogram(u0[1:], fs=1 / np.median(np.diff(t)), nfft=16384, detrend='linear')
    band = (f >= 1 / 7) & (f <= 1 / 5)
    after10 = t >= 10
    metrics = {
        'source': str(source), 'samples': len(fitted), 'duration_s': float(t[-1]),
        'command_statuses': dict(Counter(r['clock_status'] for r in commands)),
        'acquisition_statuses': dict(Counter(r['acquisition_status'] for r in acquisitions.values())),
        'gain_min': float(gains.min()), 'gain_max': float(gains.max()),
        'tau_mean_ps': float(tau.mean()), 'tau_std_ps': float(tau.std()),
        'tau_peak_to_peak_ps': float(np.ptp(tau)),
        'tau_std_after_10_s_ps': float(tau[after10].std()),
        'tau_peak_to_peak_after_10_s_ps': float(np.ptp(tau[after10])),
        'u0_std_excluding_first_ps_per_s': float(u0[1:].std()),
        'u0_std_after_10_s_ps_per_s': float(u0[after10].std()),
        'u0_min_excluding_first_ps_per_s': float(u0[1:].min()),
        'u0_max_excluding_first_ps_per_s': float(u0[1:].max()),
        'first_u0_skipped_ps_per_s': float(u0[0]),
        'tau_dominant_period_s': float(1 / f[np.argmax(power)]),
        'u0_dominant_period_s': float(1 / f_u[np.argmax(power_u)]),
        'fraction_tau_power_in_5_to_7_s_band': float(np.trapezoid(power[band], f[band]) / np.trapezoid(power, f)),
        'mean_acquisition_interval_s': float(np.mean(np.diff(t))),
        'std_acquisition_interval_s': float(np.std(np.diff(t))),
        'analysis_delay_mean_s': float(processing.mean()),
        'ack_delay_mean_s': float(ack_delay.mean()),
        'ack_delay_min_s': float(ack_delay.min()), 'ack_delay_max_s': float(ack_delay.max()),
        'max_command_quantization_error_ps_per_s': float(np.max(abs(actual - requested))),
        'applied_delta_rms_ps_per_s': float(np.sqrt(np.mean(actual ** 2))),
        'cumulative_applied_live_delta_ps_per_s': float(actual.sum()),
        'max_u0_derivative_disagreement_ps_per_s': float(np.max(abs(np.diff(tau) / np.diff(midpoint) - u0[1:]))),
        'interval_index_gaps': int(np.count_nonzero(np.diff(seq) != 1)),
    }
    return dict(label=label, metrics=metrics, seq=seq, t=t, tau=tau, u0=u0,
                processing=processing, ack_t=ack_t, ack_delay=ack_delay,
                frequency=f, power=power,
                no_change_seq=[int(r['interval']) for r in completed if r['clock_status'] == 'unchanged (below resolution)'])


old = load('173157.289121Z', 'Gain 1.0')
new = load('220212.772088Z', 'Gain 0.3')
before, after = old['metrics'], new['metrics']
statistics = {'gain_1_0': before, 'gain_0_3': after}
(directory / 'statistics.json').write_text(json.dumps(statistics, indent=2) + '\n')
with (directory / 'comparison.csv').open('w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['metric', 'gain_1_0', 'gain_0_3'])
    for key in before:
        if isinstance(before[key], (int, float)):
            writer.writerow([key, before[key], after[key]])

plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .22})
fig, axes = plt.subplots(4, 1, figsize=(11, 12), constrained_layout=True)
for run, color, alpha in ((old, '0.50', .72), (new, 'tab:blue', 1.0)):
    axes[0].plot(run['t'], run['tau'] - np.median(run['tau']), '.-', color=color, alpha=alpha, label=run['label'])
    axes[1].plot(run['t'][1:], run['u0'][1:], '.-', color=color, alpha=alpha, label=run['label'])
    mask = (run['frequency'] >= 1 / 25) & (run['frequency'] <= 1 / 2)
    axes[2].plot(1 / run['frequency'][mask], run['power'][mask], color=color, alpha=alpha, label=run['label'])
    axes[3].plot(run['t'], run['processing'], '--', color=color, alpha=alpha, label=f"{run['label']}: analyzer done")
    axes[3].plot(run['ack_t'], run['ack_delay'], '.-', color=color, alpha=alpha, label=f"{run['label']}: adjustment acknowledged")
zero = np.isin(new['seq'], new['no_change_seq'])
axes[0].plot(new['t'][zero], (new['tau'] - np.median(new['tau']))[zero], 'D',
             ms=7, color='tab:green', label='Successful zero adjustment')
axes[0].set_title('Diagnostic feedback gain comparison\nEach run has an independent initialized offset')
axes[0].set_ylabel('tau minus run median (ps)')
axes[0].legend(loc='upper right', fontsize=9)
axes[1].set_ylabel('Measured u0 (ps/s)')
axes[1].set_title('First measurement-only u0 excluded in both runs')
axes[1].axhline(0, color='0.4', lw=.7)
axes[1].legend(loc='upper left')
axes[2].axvspan(5, 7, color='tab:orange', alpha=.14, label='Original 5-7 s oscillation band')
axes[2].set_xlabel('Period (s); approximate because records are short')
axes[2].set_ylabel('tau spectral power (ps²/Hz)')
axes[2].set_xlim(2, 25)
axes[2].legend(loc='upper right', fontsize=9)
axes[3].set_ylabel('Delay after acquisition end (s)')
axes[3].set_ylim(0, .65)
axes[3].legend(loc='upper right', fontsize=8, ncol=2)
for ax in (axes[0], axes[1], axes[3]):
    ax.set_xlim(-.5, new['t'][-1] + .5)
    ax.set_xlabel('Elapsed time since first live acquisition completion (s)')
fig.savefig(directory / 'gain_comparison.png', dpi=170)
fig.savefig(directory / 'gain_comparison.svg')
plt.close(fig)

u_reduction = 100 * (1 - after['u0_std_excluding_first_ps_per_s'] / before['u0_std_excluding_first_ps_per_s'])
tau_reduction = 100 * (1 - after['tau_std_ps'] / before['tau_std_ps'])
settled_tau_reduction = 100 * (1 - after['tau_std_after_10_s_ps'] / before['tau_std_after_10_s_ps'])
report = f'''# Diagnostic gain 0.3 compared with gain 1.0

Source: {after['source']}

The new log contains 50 analysis results over 49.01 s, with no missing interval indices. All acquisition entries report checks passed. There are 46 applied adjustments, two successful below-resolution zero changes (intervals 13 and 16), and one final stopped command. Both zero changes were followed by further analysis and clock updates. Every logged request equals -0.3 times measured u0.

| Metric | Gain 1.0 | Gain 0.3 |
|---|---:|---:|
| Duration | 36.00 s | 49.01 s |
| u0 standard deviation, excluding first measurement-only value | {before['u0_std_excluding_first_ps_per_s']:.2f} ps/s | {after['u0_std_excluding_first_ps_per_s']:.2f} ps/s |
| Tau standard deviation, whole run | {before['tau_std_ps']:.2f} ps | {after['tau_std_ps']:.2f} ps |
| Tau standard deviation, after 10 s | {before['tau_std_after_10_s_ps']:.2f} ps | {after['tau_std_after_10_s_ps']:.2f} ps |
| Tau peak-to-peak, whole run | {before['tau_peak_to_peak_ps']:.2f} ps | {after['tau_peak_to_peak_ps']:.2f} ps |
| Approximate strongest tau period | {before['tau_dominant_period_s']:.1f} s | {after['tau_dominant_period_s']:.1f} s |
| Fraction of tau spectral power in 5-7 s band | {100 * before['fraction_tau_power_in_5_to_7_s_band']:.1f}% | {100 * after['fraction_tau_power_in_5_to_7_s_band']:.1f}% |
| Mean acquisition-end to analyzer return | {before['analysis_delay_mean_s']:.3f} s | {after['analysis_delay_mean_s']:.3f} s |
| Mean acquisition-end to adjustment acknowledgment | {before['ack_delay_mean_s']:.3f} s | {after['ack_delay_mean_s']:.3f} s |
| RMS applied clock increment | {before['applied_delta_rms_ps_per_s']:.2f} ps/s | {after['applied_delta_rms_ps_per_s']:.2f} ps/s |

## Interpretation

The rapid skew swings decreased: u0 standard deviation fell by {u_reduction:.1f}%. The earlier approximately six-second mode is much weaker. However, tau scatter fell only {tau_reduction:.1f}% over the complete runs, or {settled_tau_reduction:.1f}% after excluding the first 10 s. The lower gain trades rapid correction for slower response; appreciable slow offset variation remains.

The new tau record has a broad strongest period near 15-16 s (about three cycles in this record). The largest later excursion runs from -1984.95 ps at about 29 s to -1898.42 ps at about 33 s, then back to -1977.94 ps around 40 s. Consequently this run is not evidence that offset oscillation has disappeared. u0 also retains shorter-period fluctuations; taking the derivative of tau emphasizes those components, so its strongest spectral period can differ from tau's.

The full-gain and low-gain runs started with different tau offsets and residual skews. Absolute tau baselines should not be interpreted as accuracy changes. Processing and acknowledgment delays also decreased in the new run, so this was not a strictly controlled gain-only experiment. The log still lacks histograms and fit uncertainties, preventing separation of fit noise, physical phase changes, and feedback dynamics.

The gain, command direction, and unit conversion are consistent with the intended implementation. Requested versus applied correction differs by at most {after['max_command_quantization_error_ps_per_s']:.3f} ps/s, consistent with divider rounding. Logged u0 reproduces the finite difference of tau over T_lastmid to numerical precision. No command failed in the new log; the final command is labeled stopped.

## Next diagnostic

Keep gain 0.3 fixed and obtain a longer record (roughly 3-5 min) to resolve the slower variation. For a comparison, collect a short record with live clock adjustments disabled while preserving analysis and plotting. If similar tau variation persists without feedback, investigate fitting and physical timing changes; if the oscillation depends strongly on correction gain, controller dynamics remain the main explanation. Add fit uncertainty or selected histogram diagnostics before increasing gain again.

This analysis did not modify application code.
'''
(directory / 'analysis.md').write_text(report)
print(json.dumps(statistics, indent=2))
print('ARTIFACTS:', directory)
