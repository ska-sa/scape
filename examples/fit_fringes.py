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

# Parse command-line options and arguments
parser = optparse.OptionParser(usage="%prog [options] <data file>")
(opts, args) = parser.parse_args()

# Load data set
d = scape.DataSet(args[0])
# Discard 'slew' scans and channels outside the Fringe Finder band
d = d.select(labelkeep='scan', freqkeep=range(100, 420), copy=True)

# Iterate through all scans
group_delay, sigma_delay, targetdir = [], [], []
for compscan in d.compscans:
    for scan in compscan.scans:
        # Group delay is proportional to phase slope across the band - start with rads / MHz
        # Pick antenna 1 as reference antenna -> correlation product XY* means we
        # actually measure phase(antenna1) - phase(antenna2), therefore flip the sign
        phase_slope_per_MHz = np.diff(-np.angle(scan.pol('HH')), axis=1) / np.abs(np.diff(d.freqs))
        # Convert to delay in seconds
        delay = phase_slope_per_MHz / 1e6 / (-2.0 * np.pi)
        # The maximum delay that can be represented in the sampled phase slope is 1 / channel_bandwidth
        # Thereafter, it repeats periodically -> focus on primary range of +- 0.5 / channel_bandwidth
        # Obtain robust periodic statistics for delay based on this argument
        delay_stats = scape.stats.periodic_mu_sigma(delay, axis=1, period=1e-6 / d.bandwidths[0])
        group_delay.append(delay_stats.mu)
        sigma_delay.append(delay_stats.sigma)
        # Obtain vectors pointing from antenna 1 to target for each scan
        az, el = compscan.target.azel(scan.timestamps, d.antenna)
        targetdir.append(katpoint.azel_to_enu(az, el))
# Concatenate arrays into a single array
group_delay, sigma_delay, targetdir = np.hstack(group_delay), np.hstack(sigma_delay), np.hstack(targetdir)
# Augment target vector with a 1, as this allows fitting of constant (receiver) delay
# Also invert sign of target vector, as positive dot product with baseline implies negative delay / advance
augmented_targetdir = np.vstack((-targetdir, np.ones(targetdir.shape[1])))

# Construct design matrix, containing weighted basis functions
A = augmented_targetdir / sigma_delay
# Measurement vector, containing weighted observed delays
b = group_delay / sigma_delay
# Solve linear least-squares problem using SVD (see NRinC, 2nd ed, Eq. 15.4.17)
U, s, Vt = np.linalg.svd(A.transpose(), full_matrices=False)
augmented_baseline = np.dot(Vt.T, np.dot(U.T, b) / s)
# Also obtain standard errors of parameters (see NRinC, 2nd ed, Eq. 15.4.19)
sigma_augmented_baseline = np.sum((Vt.T / s[np.newaxis, :]) ** 2, axis=1)

# Convert to useful output (baseline ENU offsets in metres, and receiver delay in seconds)
baseline = augmented_baseline[:3] * katpoint.lightspeed
sigma_baseline = sigma_augmented_baseline[:3] * katpoint.lightspeed
receiver_delay = augmented_baseline[3]
sigma_receiver_delay = sigma_augmented_baseline[3]

# Stop the fringes (make a copy of the data first)
d2 = d.select(copy=True)
for compscan in d2.compscans:
    for scan in compscan.scans:
        az, el = compscan.target.azel(scan.timestamps, d2.antenna)
        targdir = np.array(katpoint.azel_to_enu(az, el))
        augmented_targdir = np.vstack((-targdir, np.ones(targdir.shape[1])))
        scan.data[:,:,0] *= np.exp(2j * np.pi * np.outer(np.dot(augmented_baseline, augmented_targdir), d2.freqs * 1e6))
old_baseline = d.antenna.baseline_toward(d.antenna2)
old_receiver_delay = 5.0808482980582519e-07 - 4.4597519527992932e-07

# Produce output plots and results
print "   Baseline (m), old,      stdev"
print "E: %.3f,       %.3f,   %g" % (baseline[0], old_baseline[0], sigma_baseline[0])
print "N: %.3f,       %.3f,   %g" % (baseline[1], old_baseline[1], sigma_baseline[1])
print "U: %.3f,       %.3f,   %g" % (baseline[2], old_baseline[2], sigma_baseline[2])
print "Receiver delay (ns): %.3f, %.3f, %g" % (receiver_delay * 1e9, old_receiver_delay * 1e9, sigma_receiver_delay * 1e9)

plt.figure(1)
plt.clf()
scape.plot_fringes(d)
plt.title('Fringes before stopping')

plt.figure(2)
plt.clf()
ax = plt.subplot(211)
plt.plot(group_delay)
plt.plot(np.dot(augmented_baseline, augmented_targetdir), 'r')
ax.set_xticklabels([])
plt.ylabel('Delay (seconds)')
plt.title('Group delay')
plt.subplot(212)
plt.plot(sigma_delay)
plt.xlabel('Time (seconds)')
plt.ylabel('Delay (seconds)')
plt.title('Standard deviation of group delay')

plt.figure(3)
plt.clf()
scape.plot_fringes(d2)
plt.title('Fringes after stopping')

plt.show()
