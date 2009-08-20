#
# Example script that uses scape to determine gain stability. It loads the
# specified data sets and extracts valid noise diode firings to estimate the
# receiver gains and relative phase as a function of frequency, time, pointing,
# etc.
#
# Ludwig Schwardt
# 19 August 2009
#

import os.path
import glob
import logging
import sys
import optparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import scape
import katpoint

# Parse command-line options and arguments
parser = optparse.OptionParser(usage="%prog [options] <directories or files>",
                               description="This processes one or more datasets (FITS or HDF5) and extracts \
                                            receiver gains and phases from them. By default all datasets in \
                                            the current directory and all subdirectories are processed.")

(options, args) = parser.parse_args()
if len(args) < 1:
    args = ['.']

# Set up logging: logging everything (DEBUG & above)
logger = logging.root
logger.setLevel(logging.DEBUG)

# Find all data sets (HDF5 or FITS) mentioned, and add them to datasets list
datasets = []
def walk_callback(arg, directory, files):
    datasets.extend([os.path.join(directory, f) for f in files if f.endswith('.h5') or f.endswith('_0000.fits')])
for arg in args:
    if os.path.isdir(arg):
        os.path.walk(arg, walk_callback, None)
    else:
        datasets.extend(glob.glob(arg))
if len(datasets) == 0:
    logger.error('No data sets (HDF5 or XDM FITS) found')
    sys.exit(1)

# Iterate over data sets and collect data
timestamps, gain_xx, gain_yy, phase_yx, temperature, azim, elev, freqs = [], [], [], [], [], [], [], None
for filename in datasets:
    # Load next data set
    logger.info("Loading dataset '%s'" % (filename,))
    d = scape.DataSet(filename)
    # Estimate receiver gains and phases based on valid noise diode jumps in data set
    t, gx, gy, phi = scape.gaincal.estimate_gain(d)
    temp = np.ma.masked_array(zeros(t.shape), np.tile(True, t.shape))
    az, el = np.tile(np.nan, t.shape), np.tile(np.nan, t.shape)
    freqs = d.freqs
    if len(t) > 0:
        timestamps.append(t)
        # Mask out RFI-flagged channels
        mask = np.tile(False, len(freqs))
        mask[d.rfi_channels] = True
        gain_xx.append(np.ma.masked_array(gx, np.tile(mask, (gx.shape[0], 1))))
        gain_yy.append(np.ma.masked_array(gy, np.tile(mask, (gy.shape[0], 1))))
        phase_yx.append(np.ma.masked_array(phi, np.tile(mask, (phi.shape[0], 1))))
        # Find closest temperature measurement to noise diode jump and store it, as well as pointing
        temp_time = np.zeros(t.shape)
        for scan in d.scans:
            for scan_temp, scan_time in zip(scan.environment['temperature'], scan.environment['timestamp']):
                select = np.abs(scan_time - t) < np.abs(temp_time - t)
                temp_time[select] = scan_time
                temp[select] = scan_temp
                temp.mask &= ~select
            for n, tt in enumerate(t):
                try:
                    jump_index = scan.timestamps.tolist().index(tt)
                except ValueError:
                    continue
                az[n] = scan.pointing['az'][jump_index]
                el[n] = scan.pointing['el'][jump_index]
        temperature.append(temp)
        azim.append(az)
        elev.append(el)
timestamps = np.hstack(timestamps)
gain_xx = np.ma.vstack(gain_xx)
gain_yy = np.ma.vstack(gain_yy)
phase_yx = np.ma.vstack(phase_yx)
temperature = np.ma.hstack(temperature)
azim = np.hstack(azim)
elev = np.hstack(elev)

# Average gains and phase across band (ignoring RFI-flagged channels)
gain_xx_band = gain_xx.mean(axis=1)
gain_yy_band = gain_yy.mean(axis=1)
phase_yx_band = katpoint.rad2deg(scape.stats.periodic_mu_sigma(phase_yx, axis=1).mu)

# Calculate time ticks and Sun positions
hours_since_start = (timestamps - timestamps.min()) / 3600.
time_grid = np.linspace(timestamps.min(), timestamps.max(), 400)
sun = katpoint.construct_target('Sun, special', antenna=d.antenna)
sun_el = katpoint.rad2deg(sun.azel(time_grid)[1])
dist_to_sun = np.array([sun.separation(katpoint.construct_azel_target(az, el), t)
                        for az, el, t in zip(azim, elev, timestamps)])

# Gains and phase as a function of frequency
plt.figure(1)
plt.clf()
ax1 = plt.subplot(311)
plt.plot(freqs, gain_yy.transpose())
plt.xticks([])
plt.ylabel('Gain (linear)')
plt.title('Y (H) power gain')
ax2 = plt.subplot(312)
plt.plot(freqs, gain_xx.transpose())
plt.xticks([])
plt.ylabel('Gain (linear)')
plt.title('X (V) power gain')
ax3 = plt.subplot(313)
plt.plot(freqs, katpoint.rad2deg(phase_yx.transpose()))
plt.xlabel('Frequency (MHz)')
plt.ylabel('Phase (degrees)')
plt.title('Phase of Y relative to X')
ylim = [min(ax1.get_ylim()[0], ax2.get_ylim()[0]), max(ax1.get_ylim()[1], ax2.get_ylim()[1])]
ax1.set_ylim(ylim)
ax2.set_ylim(ylim)
plt.savefig('gain_stability_freq.png')

# Gains and phase as a function of time
plt.figure(2)
plt.clf()
ax1 = plt.subplot(411)
plt.plot(hours_since_start, gain_yy_band, 'o')
plt.xticks([])
plt.ylabel('Gain (linear)')
plt.title('Y (H) power gain averaged over band')
ax2 = plt.subplot(412)
plt.plot(hours_since_start, gain_xx_band, 'o')
plt.xticks([])
plt.ylabel('Gain (linear)')
plt.title('X (V) power gain averaged over band')
ax3 = plt.subplot(413)
plt.plot(hours_since_start, phase_yx_band, 'o')
plt.xticks([])
plt.ylabel('Phase (deg)')
plt.title('Phase of Y relative to X averaged over band')
ax4 = plt.subplot(414)
plt.plot((time_grid - timestamps.min()) / 3600., sun_el, 'r')
plt.plot([hours_since_start.min(), hours_since_start.max()], [0., 0.], 'k')
plt.ylabel('Sun elevation (deg)')
ax5 = plt.twinx(ax4)
plt.plot(hours_since_start, temperature, 'bo')
plt.xlabel('Time (hours since %s)' % katpoint.Timestamp(timestamps.min()).local())
plt.ylabel('Temperature (deg C)')
plt.title('Elevation of Sun and ambient temperature')
y1_mean, y2_mean = np.mean(ax1.get_ylim()), np.mean(ax2.get_ylim())
yrange = max(ax1.get_ylim()[1] - ax1.get_ylim()[0], ax2.get_ylim()[1] - ax2.get_ylim()[0])
ax1.set_ylim([y1_mean - 0.5 * yrange, y1_mean + 0.5 * yrange])
ax2.set_ylim([y2_mean - 0.5 * yrange, y2_mean + 0.5 * yrange])
plt.savefig('gain_stability_time.png')

# Gains and phase as a function of azimuth angle
plt.figure(3)
plt.clf()
ax1 = plt.subplot(311)
plt.plot(katpoint.rad2deg(azim), gain_yy_band, 'o')
plt.xticks([])
plt.ylabel('Gain (linear)')
plt.title('Y (H) power gain averaged over band')
ax2 = plt.subplot(312)
plt.plot(katpoint.rad2deg(azim), gain_xx_band, 'o')
plt.xticks([])
plt.ylabel('Gain (linear)')
plt.title('X (V) power gain averaged over band')
ax3 = plt.subplot(313)
plt.plot(katpoint.rad2deg(azim), phase_yx_band, 'o')
plt.xlabel('Azimuth angle (deg)')
plt.ylabel('Phase (deg)')
plt.title('Phase of Y relative to X averaged over band')
y1_mean, y2_mean = np.mean(ax1.get_ylim()), np.mean(ax2.get_ylim())
yrange = max(ax1.get_ylim()[1] - ax1.get_ylim()[0], ax2.get_ylim()[1] - ax2.get_ylim()[0])
ax1.set_ylim([y1_mean - 0.5 * yrange, y1_mean + 0.5 * yrange])
ax2.set_ylim([y2_mean - 0.5 * yrange, y2_mean + 0.5 * yrange])
plt.savefig('gain_stability_azim.png')

# Gains and phase as a function of elevation angle
plt.figure(4)
plt.clf()
ax1 = plt.subplot(311)
plt.plot(katpoint.rad2deg(elev), gain_yy_band, 'o')
plt.xticks([])
plt.ylabel('Gain (linear)')
plt.title('Y (H) power gain averaged over band')
ax2 = plt.subplot(312)
plt.plot(katpoint.rad2deg(elev), gain_xx_band, 'o')
plt.xticks([])
plt.ylabel('Gain (linear)')
plt.title('X (V) power gain averaged over band')
ax3 = plt.subplot(313)
plt.plot(katpoint.rad2deg(elev), phase_yx_band, 'o')
plt.xlabel('Elevation angle (deg)')
plt.ylabel('Phase (deg)')
plt.title('Phase of Y relative to X averaged over band')
y1_mean, y2_mean = np.mean(ax1.get_ylim()), np.mean(ax2.get_ylim())
yrange = max(ax1.get_ylim()[1] - ax1.get_ylim()[0], ax2.get_ylim()[1] - ax2.get_ylim()[0])
ax1.set_ylim([y1_mean - 0.5 * yrange, y1_mean + 0.5 * yrange])
ax2.set_ylim([y2_mean - 0.5 * yrange, y2_mean + 0.5 * yrange])
plt.savefig('gain_stability_elev.png')

# Gains and phase as a function of angular distance from the Sun
plt.figure(5)
plt.clf()
ax1 = plt.subplot(311)
plt.plot(katpoint.rad2deg(dist_to_sun), gain_yy_band, 'o')
plt.xticks([])
plt.ylabel('Gain (linear)')
plt.title('Y (H) power gain averaged over band')
ax2 = plt.subplot(312)
plt.plot(katpoint.rad2deg(dist_to_sun), gain_xx_band, 'o')
plt.xticks([])
plt.ylabel('Gain (linear)')
plt.title('X (V) power gain averaged over band')
ax3 = plt.subplot(313)
plt.plot(katpoint.rad2deg(dist_to_sun), phase_yx_band, 'o')
plt.xlabel('Angular separation from Sun (deg)')
plt.ylabel('Phase (deg)')
plt.title('Phase of Y relative to X averaged over band')
y1_mean, y2_mean = np.mean(ax1.get_ylim()), np.mean(ax2.get_ylim())
yrange = max(ax1.get_ylim()[1] - ax1.get_ylim()[0], ax2.get_ylim()[1] - ax2.get_ylim()[0])
ax1.set_ylim([y1_mean - 0.5 * yrange, y1_mean + 0.5 * yrange])
ax2.set_ylim([y2_mean - 0.5 * yrange, y2_mean + 0.5 * yrange])
plt.savefig('gain_stability_sun.png')
