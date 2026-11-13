"""Compare four-second tracking against earlier one-second runs."""
from collections import Counter
import csv
import json
from pathlib import Path
import re

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
    analysis = [r for r in rows if r['event'] == 'analysis_finished']
    acquisitions = {int(r['interval']): r for r in rows if r['event'] == 'acquisition_finished'}
    commands = [r for r in rows if r['event'] == 'clock_adjustment_finished']
    completed = [r for r in commands if r['clock_status'] in ('applied', 'unchanged (below resolution)')]
    seq = np.array([int(r['interval']) for r in analysis])
    t = np.array([float(acquisitions[k]['relative_s']) for k in seq])
    tau = np.array([float(r['tau_ps']) for r in analysis])
    u = np.array([float(r['u0_ps_per_s']) for r in analysis])
    midpoint = np.array([float(r['T_lastmid_ms']) * 1e-3 if r.get('T_lastmid_ms')
                         else float(r['T_lastmid_ps']) * 1e-12 for r in analysis])
    T = float(np.median(np.diff(t)))
    gain = np.array([-float(r['correction_ps_per_s']) / float(r['u0_ps_per_s'])
                     for r in commands if float(r['u0_ps_per_s']) != 0])
    processing = np.array([float(r['relative_s']) - t[i] for i, r in enumerate(analysis)])
    ack_t = np.array([float(acquisitions[int(r['interval'])]['relative_s']) for r in completed])
    ack = np.array([float(r['relative_s']) - ack_t[i] for i, r in enumerate(completed)])
    analysis_by_seq = {int(r['interval']): r for r in analysis}
    after_analysis = np.array([float(r['relative_s']) - float(analysis_by_seq[int(r['interval'])]['relative_s'])
                               for r in completed])
    applied = np.array([float(r['applied_correction_ps_per_s']) for r in completed])
    requested = np.array([float(r['correction_ps_per_s']) for r in completed])
    write_time = [float(match[1]) for r in completed
                  if (match := re.search(r'completed in ([\d.]+) s', r['details']))]
    f, power = periodogram(tau, fs=1 / T, detrend='linear', nfft=16384)
    fu, pu = periodogram(u[1:], fs=1 / T, detrend='linear', nfft=16384)
    stats = {
        'source': str(source), 'label': label,
        'points': len(analysis), 'duration_s': float(t[-1]),
        'gain_min': float(gain.min()), 'gain_max': float(gain.max()),
        'interval_mean_s': float(np.mean(np.diff(t))), 'interval_median_s': T,
        'interval_std_s': float(np.std(np.diff(t))),
        'interval_index_gaps': int(np.count_nonzero(np.diff(seq) != 1)),
        'tau_mean_ps': float(tau.mean()), 'tau_std_ps': float(tau.std()),
        'tau_min_ps': float(tau.min()), 'tau_max_ps': float(tau.max()),
        'tau_peak_to_peak_ps': float(np.ptp(tau)),
        'successive_tau_change_std_ps': float(np.std(np.diff(tau))),
        'u0_std_excluding_first_ps_per_s': float(u[1:].std()),
        'u0_min_excluding_first_ps_per_s': float(u[1:].min()),
        'u0_max_excluding_first_ps_per_s': float(u[1:].max()),
        'first_skipped_u0_ps_per_s': float(u[0]),
        'tau_dominant_period_s': float(1 / f[np.argmax(power)]),
        'u0_dominant_period_s': float(1 / fu[np.argmax(pu)]),
        'analysis_delay_mean_s': float(processing.mean()),
        'ack_delay_mean_s': float(ack.mean()),
        'ack_delay_min_s': float(ack.min()), 'ack_delay_max_s': float(ack.max()),
        'ack_delay_fraction_of_interval': float(ack.mean() / T),
        'analysis_to_ack_mean_s': float(after_analysis.mean()),
        'analysis_to_ack_fraction_of_interval': float(after_analysis.mean() / T),
        'mean_reported_hardware_write_s': float(np.mean(write_time)),
        'approximate_skew_estimate_age_at_ack_s': float(T + ack.mean()),
        'nyquist_min_resolvable_period_s': 2 * T,
        'command_statuses': dict(Counter(r['clock_status'] for r in commands)),
        'acquisition_statuses': dict(Counter(r['acquisition_status'] for r in acquisitions.values())),
        'max_command_rounding_error_ps_per_s': float(max(abs(applied - requested))),
        'net_applied_live_increment_ps_per_s': float(sum(applied)),
        'rms_applied_increment_ps_per_s': float(np.sqrt(np.mean(applied ** 2))),
        'max_u0_finite_difference_error_ps_per_s': float(max(abs(np.diff(tau) / np.diff(midpoint) - u[1:]))),
    }
    return dict(stats=stats, t=t, seq=seq, tau=tau, u=u, T=T,
                processing=processing, ack_t=ack_t, ack=ack, f=f, power=power)


full = load('173157.289121Z', '1 s, gain 1.0')
one = load('220212.772088Z', '1 s, gain 0.3')
four = load('220428.580797Z', '4 s, gain 0.3')
runs = (full, one, four)
stats = {key: run['stats'] for key, run in zip(('one_second_gain_1', 'one_second_gain_03', 'four_second_gain_03'), runs)}
(directory / 'statistics.json').write_text(json.dumps(stats, indent=2) + '\n')
with (directory / 'comparison.csv').open('w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['metric'] + [run['stats']['label'] for run in runs])
    for key in four['stats']:
        if isinstance(four['stats'][key], (int, float)):
            writer.writerow([key] + [run['stats'][key] for run in runs])

plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .22})
fig, axes = plt.subplots(4, 1, figsize=(11, 12), constrained_layout=True)
for run, color, alpha in ((one, '0.55', .75), (four, 'tab:blue', 1)):
    label = run['stats']['label']
    axes[0].plot(run['t'], run['tau'] - np.median(run['tau']), '.-', color=color, alpha=alpha, label=label)
    axes[1].plot(run['t'][1:], run['u'][1:], '.-', color=color, alpha=alpha, label=label)
    axes[2].plot(run['t'], run['processing'] / run['T'], '--', color=color, alpha=alpha, label=f'{label}: analyzer done')
    axes[2].plot(run['ack_t'], run['ack'] / run['T'], '.-', color=color, alpha=alpha, label=f'{label}: acknowledgment')
    mask = (run['f'] >= 1 / 60) & (run['f'] > 0)
    axes[3].plot(1 / run['f'][mask], run['power'][mask], color=color, alpha=alpha, label=label)
axes[0].set_title('Four-second acquisition diagnostic\nPhase baselines removed separately; both runs use gain 0.3')
axes[0].set_ylabel('tau minus run median (ps)')
axes[0].legend(loc='upper right')
axes[1].set_ylabel('Measured u0 (ps/s)')
axes[1].set_title('u0 is divided by the midpoint spacing: approximately 1 s versus 4 s')
axes[1].axhline(0, color='0.4', lw=.7)
axes[1].legend(loc='upper right')
axes[2].set_ylabel('Delay / acquisition interval')
axes[2].legend(loc='upper right', fontsize=8, ncol=2)
axes[3].axvspan(0, 8, color='tab:orange', alpha=.12, label='Periods below 8 s unresolved with 4 s sampling')
axes[3].set_xlim(0, 60)
axes[3].set_xlabel('Period (s); broad estimates from short records')
axes[3].set_ylabel('tau spectral power (ps²/Hz)')
axes[3].legend(loc='upper right', fontsize=8)
for ax in axes[:3]:
    ax.set_xlim(-1, 78)
    ax.set_xlabel('Elapsed time since first live acquisition completion (s)')
fig.savefig(directory / 'four_second_comparison.png', dpi=170)
fig.savefig(directory / 'four_second_comparison.svg')
plt.close(fig)

a, b = one['stats'], four['stats']
report = f'''# Four-second live tracking analysis

Source: {b['source']}

The log contains 20 analysis points over 76.00 s, with approximately four-second spacing. The command/request ratio confirms gain 0.3 in this file, independent of the current source-code default. There are 18 applied increments and one normal below-resolution zero adjustment, plus the first measurement-only point. All acquisition checks pass, there are no missing interval indices, and no command in this log is failed or stopped.

| Metric | 1 s, gain 0.3 | 4 s, gain 0.3 |
|---|---:|---:|
| Analysis points | {a['points']} | {b['points']} |
| Record duration | {a['duration_s']:.2f} s | {b['duration_s']:.2f} s |
| Acquisition end to analyzer return | {a['analysis_delay_mean_s']:.3f} s | {b['analysis_delay_mean_s']:.3f} s |
| Analyzer return to acknowledgment | {a['analysis_to_ack_mean_s']:.3f} s | {b['analysis_to_ack_mean_s']:.3f} s |
| Acquisition end to acknowledgment | {a['ack_delay_mean_s']:.3f} s | {b['ack_delay_mean_s']:.3f} s |
| Acquisition-end-to-acknowledgment / interval | {100 * a['ack_delay_fraction_of_interval']:.1f}% | {100 * b['ack_delay_fraction_of_interval']:.1f}% |
| Analyzer-return-to-acknowledgment / interval | {100 * a['analysis_to_ack_fraction_of_interval']:.1f}% | {100 * b['analysis_to_ack_fraction_of_interval']:.1f}% |
| Tau standard deviation | {a['tau_std_ps']:.2f} ps | {b['tau_std_ps']:.2f} ps |
| Tau peak-to-peak | {a['tau_peak_to_peak_ps']:.2f} ps | {b['tau_peak_to_peak_ps']:.2f} ps |
| Successive tau-change standard deviation | {a['successive_tau_change_std_ps']:.2f} ps | {b['successive_tau_change_std_ps']:.2f} ps |
| u0 standard deviation, first point excluded | {a['u0_std_excluding_first_ps_per_s']:.2f} ps/s | {b['u0_std_excluding_first_ps_per_s']:.2f} ps/s |
| Approximate strongest tau period | {a['tau_dominant_period_s']:.1f} s | {b['tau_dominant_period_s']:.1f} s |

## Interpretation

The user's expectation about relative computational/communication latency is confirmed: the post-acquisition acknowledgment delay is a smaller fraction of a four-second interval. It does not follow that the feedback measurement is instantaneous. Tau is fitted from the completed acquisition window, and u0 is calculated from two consecutive fitted offsets. With approximately uniform interval weighting, its effective measurement time is around one acquisition interval before the current interval finishes. The approximate measurement age at acknowledgment is therefore T plus the acquisition-end-to-acknowledgment delay: about {a['approximate_skew_estimate_age_at_ack_s']:.2f} s for the one-second run versus {b['approximate_skew_estimate_age_at_ack_s']:.2f} s here. These are timing-model approximations, not hardware timestamps; actual clock application precedes the observed acknowledgment by an unknown amount.

Longer acquisition also means fewer feedback updates. Reducing post-processing latency as a fraction of the interval does not remove acquisition averaging, finite-difference lag, or the slower controller update rate. This record does not show an improvement in phase scatter: tau ranges from {b['tau_min_ps']:.2f} to {b['tau_max_ps']:.2f} ps, approximately {b['tau_peak_to_peak_ps']:.1f} ps peak-to-peak. Its broad strongest period is roughly 25 s, approximately six updates per cycle. Only three such cycles were recorded, so the period estimate is approximate and does not establish a steady periodic hardware oscillation.

The smaller u0 scatter should not be interpreted by itself as better phase stability. The same offset change divided by four seconds produces a rate four times smaller than when divided by one second; longer acquisition also averages faster variation. In this dataset the successive offset jumps are larger even though the rate estimate is smaller.

At four-second sampling, periods shorter than eight seconds cannot be resolved uniquely. The earlier approximately six-second oscillation cannot be directly ruled out from this record; high-frequency phase variations can be averaged down or aliased. Consequently the persistent phase variation weakens an explanation based solely on post-processing/network latency, but it cannot isolate feedback, device response, physical drift, and fit noise.

The sign, gain, and units are internally consistent. Actual versus requested delta differs by at most {b['max_command_rounding_error_ps_per_s']:.3f} ps/s. Logged u0 matches the finite difference of tau over T_lastmid to numerical precision. The initial u0 of {b['first_skipped_u0_ps_per_s']:.2f} ps/s was skipped as intended. Mean tau differs across runs because initialization provides a different constant reference; this is not a direct clock-accuracy comparison.

## Suggested next diagnostic

Keep the four-second interval and gain fixed for a longer run, then compare against a record at the same acquisition interval with live clock updates disabled while analysis continues. A similar phase variation without feedback would point toward input data, fitting, or physical phase changes; a substantial dependence on feedback would support controller/clock-response dynamics. Record fit-center uncertainty and source event-time ranges to separate fitting noise from actual offset motion. Hardware actuation or node timestamps would refine the delay estimate beyond the acknowledgment time.

Application code was not modified by this analysis.
'''
(directory / 'analysis.md').write_text(report)
print(json.dumps(stats, indent=2))
print('ARTIFACTS:', directory)
