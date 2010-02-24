#! /usr/bin/python
#
# Baseline calibration for a single baseline.
#
# Ludwig Schwardt
# 25 January 2010
#

import numpy as np
import optparse
import matplotlib.pyplot as plt

import scape
import katpoint
import time

# Parse command-line options and arguments
parser = optparse.OptionParser(usage="%prog [options] <data file>")
(opts, args) = parser.parse_args()

# Load data set
d = scape.DataSet(args[0], baseline='A1A2')
# Discard 'slew' scans and channels outside the Fringe Finder band
d = d.select(labelkeep='scan', freqkeep=range(100, 420), copy=True)
time_origin = np.min([scan.timestamps.min() for scan in d.scans])
# Since the phase slope is sampled, the derived delay exhibits aliasing with a period of 1 / channel_bandwidth
# The maximum delay that can be reliably represented is therefore +- 0.5 / channel_bandwidth
max_delay = 1e-6 / d.bandwidths[0] / 2
# Maximum standard deviation of delay occurs when delay samples are uniformly distributed between +- max_delay
# In this case, delay is completely random / unknown, and its estimate cannot be trusted
# The division by sqrt(N-1) converts the per-channel standard deviation to a per-snapshot deviation
max_sigma_delay = np.abs(2 * max_delay) / np.sqrt(12) / np.sqrt(len(d.freqs) - 1)

# Iterate through all scans
group_delay_per_scan, sigma_delay_per_scan, augmented_targetdir_per_scan = [], [], []
raw_delay_per_scan, timestamps = [], []
for compscan in d.compscans:
    for scan in compscan.scans:
        # Group delay is proportional to phase slope across the band - estimate this as
        # the phase difference between consecutive frequency channels calculated via np.diff.
        # Pick antenna 1 as reference antenna -> correlation product XY* means we
        # actually measure phase(antenna1) - phase(antenna2), therefore flip the sign.
        # Also divide by channel frequency difference, which correctly handles gaps in frequency coverage.
        phase_diff_per_MHz = np.diff(-np.angle(scan.pol('HH')), axis=1) / np.abs(np.diff(d.freqs))
        # Convert to a delay in seconds
        delay = phase_diff_per_MHz / 1e6 / (-2.0 * np.pi)
        # Raw delay is calculated as an intermediate step for display only - wrap this to primary interval
        raw_delay_per_scan.append(scape.stats.angle_wrap(delay / max_delay, period=1.0).transpose())
        timestamps.append(scan.timestamps - time_origin)
        # Obtain robust periodic statistics for *per-channel* phase difference
        delay_stats = scape.stats.periodic_mu_sigma(delay, axis=1, period=2 * max_delay)
        group_delay_per_scan.append(delay_stats.mu)
        # The estimated mean group delay is the average of N-1 per-channel differences. Since this is less
        # variable than the per-channel data itself, we have to divide the data sigma by sqrt(N-1).
        sigma_delay_per_scan.append(delay_stats.sigma / np.sqrt(len(d.freqs) - 1))
        # Obtain unit vectors pointing from antenna 1 to target for each timestamp in scan
        az, el = compscan.target.azel(scan.timestamps, d.antenna)
        targetdir = np.array(katpoint.azel_to_enu(az, el))
        # Invert sign of target vector, as positive dot product with baseline implies negative delay / advance
        # Augment target vector with a 1, as this allows fitting of constant (receiver) delay
        augmented_targetdir_per_scan.append(np.vstack((-targetdir, np.ones(len(el)))))
# Concatenate per-scan arrays into a single array for data set
group_delay = np.hstack(group_delay_per_scan)
sigma_delay = np.hstack(sigma_delay_per_scan)
augmented_targetdir = np.hstack(augmented_targetdir_per_scan)
# Sanitise the uncertainties (can't be too certain...)
sigma_delay[sigma_delay < 1e-5 * max_sigma_delay] = 1e-5 * max_sigma_delay

# Construct design matrix, containing weighted basis functions
A = augmented_targetdir / sigma_delay
# Measurement vector, containing weighted observed delays
b = group_delay / sigma_delay
# Solve linear least-squares problem using SVD (see NRinC, 2nd ed, Eq. 15.4.17)
U, s, Vt = np.linalg.svd(A.transpose(), full_matrices=False)
augmented_baseline = np.dot(Vt.T, np.dot(U.T, b) / s)
# Also obtain standard errors of parameters (see NRinC, 2nd ed, Eq. 15.4.19)
sigma_augmented_baseline = np.sqrt(np.sum((Vt.T / s[np.newaxis, :]) ** 2, axis=1))

# Convert to useful output (baseline ENU offsets in metres, and receiver delay in seconds)
baseline = augmented_baseline[:3] * katpoint.lightspeed
sigma_baseline = sigma_augmented_baseline[:3] * katpoint.lightspeed
receiver_delay = augmented_baseline[3]
sigma_receiver_delay = sigma_augmented_baseline[3]
# The speed of light in the fibres is lower than in vacuum
cable_lightspeed = katpoint.lightspeed / 1.4
cable_length_diff = augmented_baseline[3] * cable_lightspeed
sigma_cable_length_diff = sigma_augmented_baseline[3] * cable_lightspeed

# Stop the fringes (make a copy of the data first)
d2 = d.select(copy=True)
fitted_delay_per_scan = []
for n, scan in enumerate(d2.scans):
    # Store fitted delay and other delays with corresponding timestamps, to allow compacted plot
    fitted_delay = np.dot(augmented_baseline, augmented_targetdir_per_scan[n])
    fitted_delay_per_scan.append(np.vstack((timestamps[n], fitted_delay)).transpose())
    group_delay_per_scan[n] = np.vstack((timestamps[n], group_delay_per_scan[n])).transpose()
    sigma_delay_per_scan[n] = np.vstack((timestamps[n], sigma_delay_per_scan[n])).transpose()
    # Stop the fringes (remember that HH phase is antenna1 - antenna2, need to *add* fitted delay to fix it)
    scan.data[:,:,0] *= np.exp(2j * np.pi * np.outer(fitted_delay, d2.freqs * 1e6))
old_baseline = d.antenna.baseline_toward(d.antenna2)
old_cable_length_diff = 13.3
old_receiver_delay = old_cable_length_diff / cable_lightspeed
labels = [str(n) for n in xrange(len(d2.scans))]

# Scale delays to be in units of ADC sample periods, and normalise sigma delay
adc_samplerate = 800e6
for n in xrange(len(d.scans)):
    group_delay_per_scan[n][:, 1] *= adc_samplerate
    fitted_delay_per_scan[n][:, 1] *= adc_samplerate
    sigma_delay_per_scan[n][:, 1] /= max_sigma_delay

# Produce output plots and results
print "   Baseline (m), old,      stdev"
print "E: %.3f,       %.3f,   %g" % (baseline[0], old_baseline[0], sigma_baseline[0])
print "N: %.3f,       %.3f,   %g" % (baseline[1], old_baseline[1], sigma_baseline[1])
print "U: %.3f,       %.3f,   %g" % (baseline[2], old_baseline[2], sigma_baseline[2])
print "Receiver delay (ns): %.3f, %.3f, %g" % (receiver_delay * 1e9, old_receiver_delay * 1e9, sigma_receiver_delay * 1e9)
print "Cable length difference (m): %.3f, %.3f, %g" % (cable_length_diff, old_cable_length_diff, sigma_cable_length_diff)
print
print "Sources from good to bad"
print "------------------------"
print "(rated on stdev of group delay normalised to max = %g ns):" % (max_sigma_delay * 1e9,)
targets = np.array([compscan.target for scan in compscan.scans for compscan in d.compscans])
sigma_delays = np.array([sigma_delay[:, 1].mean() for sigma_delay in sigma_delay_per_scan])
decreasing_performance = np.argsort(sigma_delays)
for target, sigma_delay in zip(targets[decreasing_performance], sigma_delays[decreasing_performance]):
    print "%12s = %.5g" % (target.name, sigma_delay)

plt.figure(1)
plt.clf()
scape.plot_fringes(d)
plt.title('Fringes before stopping')

plt.figure(2)
plt.clf()
scape.plots_basic.plot_compacted_images(raw_delay_per_scan, timestamps, labels,
                                        ylim=(d.freqs[0], d.freqs[-1]), clim=(-0.5, 0.5))
plt.xlabel('Time (s), since %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_origin)))
plt.ylabel('Frequency (MHz)')
plt.title('Raw per-channel delay estimates, as a fraction of maximum delay')

plt.figure(3)
plt.clf()
ax = plt.subplot(211)
scape.plots_basic.plot_compacted_line_segments(group_delay_per_scan, labels)
scape.plots_basic.plot_compacted_line_segments(fitted_delay_per_scan, color='r')
ax.set_xticklabels([])
ylim_max = np.max(np.abs(ax.get_ylim()))
ax.set_ylim(-max_delay * adc_samplerate, max_delay * adc_samplerate)
plt.ylabel('Delay (ADC samples)')
plt.title('Group delay')
ax = plt.subplot(212)
scape.plots_basic.plot_compacted_line_segments(sigma_delay_per_scan)
ax.set_ylim(0, 1)
plt.xlabel('Time (s), since %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_origin)))
plt.ylabel('Normalised sigma')
plt.title('Standard deviation of group delay\n(fraction of maximum = %.3g ADC samples)' % (max_sigma_delay * adc_samplerate,))
plt.subplots_adjust(hspace=0.25)

plt.figure(4)
plt.clf()
scape.plot_fringes(d2)
plt.title('Fringes after stopping')

# plt.figure(5)
# plt.clf()
# ax = plt.subplot(311)
# scape.plots_basic.plot_compacted_line_segments(group_delay_per_scan, labels)
# scape.plots_basic.plot_compacted_line_segments(fitted_delay_per_scan, color='r')
# ylim_max = np.max(np.abs(ax.get_ylim()))
# ax.set_ylim(-1.1 * ylim_max, 1.1 * ylim_max)
# left, right, bottom, top = 870, 1140, -1.8e-7, -1.2e-7
# ax.add_patch(plt.Rectangle((left, bottom), right - left, top - bottom, fill=False, ec='g', lw=2))
# plt.ylabel('Delay (seconds)')
# plt.title('Group delay')
# ax = plt.subplot(312)
# scape.plots_basic.plot_compacted_line_segments(group_delay_per_scan, labels)
# scape.plots_basic.plot_compacted_line_segments(fitted_delay_per_scan, color='r')
# ax.set_xlim(left, right)
# ax.set_ylim(bottom, top)
# left, right, bottom, top = 1000, 1066, -1.67364e-7, -1.66716e-7
# ax.add_patch(plt.Rectangle((left, bottom), right - left, top - bottom, fill=False, ec='g', lw=2))
# plt.ylabel('Delay (seconds)')
# ax = plt.subplot(313)
# scape.plots_basic.plot_compacted_line_segments(group_delay_per_scan, labels)
# scape.plots_basic.plot_compacted_line_segments(fitted_delay_per_scan, color='r')
# ax.set_xlim(left, right)
# ax.set_ylim(bottom, top)
# plt.xlabel('Time (s), since %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_origin)))
# plt.ylabel('Delay (seconds)')

plt.show()
