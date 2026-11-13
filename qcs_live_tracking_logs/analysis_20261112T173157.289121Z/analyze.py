from pathlib import Path
import csv
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import periodogram


root = Path('/home/zyc/QCS/QCS_server')
source = root / 'qcs_live_tracking_logs/qcs_live_tracking_20261112T173157.289121Z.csv'
output = source.parent / ('analysis_' + source.stem.removeprefix('qcs_live_tracking_'))
output.mkdir(exist_ok=True)
rows = list(csv.DictReader(source.open()))
analysis = [row for row in rows if row['event'] == 'analysis_finished']
acquisitions = {int(row['interval']): row for row in rows if row['event'] == 'acquisition_finished'}
commands = {int(row['interval']): row for row in rows if row['event'] == 'clock_adjustment_finished'}
applied = {seq: row for seq, row in commands.items() if row['clock_status'] == 'applied'}
sequence = np.array([int(row['interval']) for row in analysis])
end = np.array([float(acquisitions[seq]['relative_s']) for seq in sequence])
analysis_end = np.array([float(row['relative_s']) for row in analysis])
mid = np.array([float(row['T_lastmid_ps']) * 1e-12 for row in analysis])
tau = np.array([float(row['tau_ps']) for row in analysis])
u = np.array([float(row['u0_ps_per_s']) for row in analysis])
command_index = np.array([i for i, seq in enumerate(sequence) if seq in applied])
ack = np.array([float(applied[sequence[i]]['relative_s']) for i in command_index])
requested = np.array([float(applied[sequence[i]]['correction_ps_per_s']) for i in command_index])
actual = np.array([float(applied[sequence[i]]['applied_correction_ps_per_s']) for i in command_index])
period = np.median(np.diff(end))


def summary(values):
    return {name: float(fn(values)) for name, fn in (
        ('min', np.min), ('mean', np.mean), ('median', np.median),
        ('max', np.max), ('std', np.std),
    )}


frequency, power = periodogram(tau, fs=1 / period, nfft=16384, detrend='linear')
tau_peak_period = float(1 / frequency[np.argmax(power)])
frequency_u, power_u = periodogram(u[1:], fs=1 / period, nfft=16384, detrend='linear')
u_peak_period = float(1 / frequency_u[np.argmax(power_u)])
delay = ack - end[command_index]
processing = analysis_end - end
round_trip = ack - analysis_end[command_index]
hardware_write_s = np.array([
    float(re.search(r'completed in ([\d.]+) s', applied[sequence[i]]['details']).group(1))
    for i in command_index
])

# Illustrative sampled feedback model: phase estimates average one interval,
# the clock change occurs d intervals after the interval ends, and a full
# negative residual is added to the clock. This is a derived idealized model,
# not an identified hardware transfer function.
d = float(np.mean(delay) / period)
weights = np.array([(1 - d) ** 2 / 2, .5 + d - d ** 2, d ** 2 / 2])
model = {}
for gain in (1.0, .3, .2):
    roots = np.roots([1, -(1 - gain * weights[0]), gain * weights[1], gain * weights[2]])
    pair = next((r for r in roots if r.imag > 0), None)
    model[str(gain)] = {
        'roots': [[float(r.real), float(r.imag)] for r in roots],
        'oscillation_period_s': float(2 * np.pi / np.angle(pair) * period) if pair is not None else None,
        'dominant_decay_per_interval': float(max(abs(r) for r in roots)),
    }

stats = {
    'source': str(source),
    'samples': len(analysis),
    'duration_s': float(end[-1] - end[0]),
    'applied_commands': len(applied),
    'terminal_stopped_commands': sum(row['clock_status'] == 'stopped' for row in commands.values()),
    'tau_ps': summary(tau),
    'u0_ps_per_s_excluding_first': summary(u[1:]),
    'first_u0_ps_per_s_skipped': float(u[0]),
    'tau_period_estimate_s': tau_peak_period,
    'u0_period_estimate_s': u_peak_period,
    'acquisition_spacing_s': summary(np.diff(end)),
    'midpoint_spacing_s': summary(np.diff(mid)),
    'acquisition_to_analysis_finished_s': summary(processing),
    'acquisition_to_adjustment_ack_s': summary(delay),
    'analysis_finished_to_adjustment_ack_s': summary(round_trip),
    'node_reported_write_s': summary(hardware_write_s),
    'applied_plus_measured_u0_error_ps_per_s': summary(actual + u[command_index]),
    'max_finite_difference_disagreement_ps_per_s': float(np.max(abs(np.diff(tau) / np.diff(mid) - u[1:]))),
    'tau_std_first_18_s_ps': float(np.std(tau[end < 18])),
    'tau_std_last_18_s_ps': float(np.std(tau[end >= 18])),
    'illustrative_feedback_model': model,
}
(output / 'statistics.json').write_text(json.dumps(stats, indent=2) + '\n')

plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .23})
fig, axes = plt.subplots(4, 1, figsize=(11, 12), constrained_layout=True)
ax = axes[0]
ax.plot(end, tau, '.-', color='tab:blue', label='Fitted tau')
ax.set_ylabel('tau (ps)')
ax.set_title('QCS live tracking: oscillation and feedback timing\n37 samples, approximately 1 s intervals')
ax.text(.02, .08, f'Observed period approximately {tau_peak_period:.1f} s; peak-to-peak {np.ptp(tau):.1f} ps', transform=ax.transAxes)
ax.legend(loc='upper right')

ax = axes[1]
ax.plot(end[1:], u[1:], '.-', label='Measured residual u0', color='tab:orange')
ax.plot(ack, actual, 'x--', label='Applied incremental correction, at acknowledgment', color='tab:green', alpha=.8)
ax.axhline(0, color='0.4', lw=.7)
ax.set_ylabel('Skew / change (ps/s)')
ax.legend(loc='upper left', ncol=2, fontsize=9)
ax.text(.02, .06, f'First measurement-only u0 ({u[0]:.1f} ps/s) excluded from this panel', transform=ax.transAxes, fontsize=9)

ax = axes[2]
ax.plot(end, processing, '.-', label='Acquisition end to analysis finished')
ax.plot(end[command_index], delay, '.-', label='Acquisition end to adjustment acknowledgment')
ax.axhline(period, color='0.4', linestyle=':', label='Next acquisition ends at approximately +1 s')
ax.set_ylabel('Delay (s)')
ax.set_ylim(0, 1.1)
ax.legend(loc='center left', fontsize=9)

ax = axes[3]
selection = (end >= 0) & (end < 6)
for i in np.flatnonzero(selection):
    y = i
    start = end[i] - period if i == 0 else end[i - 1]
    ax.barh(y, end[i] - start, left=start, height=.45, color='0.8', label='Acquisition interval' if i == 0 else None)
    ax.plot([end[i], analysis_end[i]], [y, y], color='tab:blue', linewidth=5, label='Buffer ready to analyzer return' if i == 0 else None)
    if sequence[i] in applied:
        finish = float(applied[sequence[i]]['relative_s'])
        ax.plot([analysis_end[i], finish], [y, y], color='tab:orange', linewidth=5, label='Analyzer return to acknowledgment' if i == 1 else None)
ax.set_yticks(np.flatnonzero(selection), [f'Interval {seq}' for seq in sequence[selection]])
ax.invert_yaxis()
ax.set_xlim(-1.1, 7)
ax.set_xlabel('Server elapsed time from first live acquisition completion (s)')
ax.set_title('Early intervals: clock updates occur while the following acquisition is running')
ax.legend(loc='upper right', fontsize=8)
for ax in axes[:3]:
    ax.set_xlim(-.5, end[-1] + .7)
    ax.set_xlabel('Server elapsed time from first live acquisition completion (s)')
fig.savefig(output / 'oscillation_analysis.png', dpi=170)
fig.savefig(output / 'oscillation_analysis.svg')
plt.close(fig)

report = f'''# Analysis of {source.name}

The log contains {len(analysis)} fitted intervals over {end[-1]:.2f} s, with 35 applied adjustments, one initial skipped adjustment, and one final command stopped during shutdown.

## Measurements

- Dominant tau oscillation: approximately {tau_peak_period:.1f} s. Only about six cycles were recorded; this period is approximate.
- Tau range: {tau.min():.2f} to {tau.max():.2f} ps ({np.ptp(tau):.2f} ps peak-to-peak).
- Residual u0 range after the first measurement-only sample: {u[1:].min():.2f} to {u[1:].max():.2f} ps/s.
- Acquisition spacing: {np.mean(np.diff(end)):.4f} s, standard deviation {np.std(np.diff(end)):.4f} s; no interval-index gaps.
- Acquisition completion to analyzer return: mean {np.mean(processing):.3f} s, range {min(processing):.3f} to {max(processing):.3f} s.
- Acquisition completion to observed adjustment completion: mean {np.mean(delay):.3f} s, range {min(delay):.3f} to {max(delay):.3f} s.
- Analyzer return to observed adjustment completion: mean {np.mean(round_trip):.3f} s.
- Node-reported clock-write time: mean {np.mean(hardware_write_s):.3f} s; network, command queueing, and server polling account for additional elapsed time.
- Applied increment agrees with negative measured u0 within {np.max(abs(actual + u[command_index])):.3f} ps/s.
- Logged u0 agrees with finite differences of tau over T_lastmid within {stats['max_finite_difference_disagreement_ps_per_s']:.3g} ps/s.

## Interpretation

The strongest explanation is delayed closed-loop ringing with unity correction gain. The analyzer measures phase from a completed interval; its finite difference estimates skew over two consecutive phase points. The node is changed while the following acquisition is already running. Therefore the next fitted result still partly reflects older clock settings, but another full correction is issued from it.

The clock-command sign and units match the requested rule. Command quantization is much smaller than the observed skew swings. The first anomalous u0 ({u[0]:.2f} ps/s) was skipped and did not drive the node. Acquisition and acknowledgment timing do not show a five-to-six-second scheduling cycle or a growing queue backlog.

An illustrative model with one-second, uniformly weighted phase windows and actuator steps at the mean observed acknowledgment time predicts a {model['1.0']['oscillation_period_s']:.1f} s oscillation at gain 1. This is close to the observed {tau_peak_period:.1f} s period. The model has damped poles, so it supports ringing; it does not establish unstable hardware or self-sustained oscillation. Fit noise and additional unmeasured latency can excite or sustain that response.

The clock acknowledgment is the server's observation of completion, rather than a hardware-timestamped actuator event. Arrival-time windows do not record the HH event-time boundaries. No histogram, fit uncertainty, or actual clock readback is in this log. Therefore fit noise, device response delay, and changes in the physical time offset cannot be independently separated here.

## Suggested next experiment

Introduce a configurable feedback gain and try sending `-0.3 * u0` while keeping the acquisition interval fixed. In the same illustrative model the oscillatory pole magnitude drops from about {model['1.0']['dominant_decay_per_interval']:.2f} to {model['0.3']['dominant_decay_per_interval']:.2f} per interval. This is a diagnostic experiment, not a measured hardware stability guarantee. A short interval with live adjustments disabled provides a comparison for fitting noise and uncontrolled drift. Log source timestamp ranges, fit center uncertainty, and command-send/receive timestamps to identify the remaining delay.

Application code was not changed by this analysis.
'''
(output / 'analysis.md').write_text(report)
print(json.dumps(stats, indent=2))
print('ARTIFACTS', output)
