#!/usr/bin/python
#
# Example script that uses scape to reduce data consisting of scans across
# multiple point sources. This can be used to determine gain curves, tipping
# curves and pointing models. The user can interactively observe reduction
# results and discard bad data. The end product is a file containing pointing,
# fitted beam parameters, baseline height and weather measurements, etc.
#
# Ludwig Schwardt
# 13 July 2009
#

import os.path
import sys
import logging
import optparse
import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

import scape
import katpoint

# Parse command-line options and arguments
parser = optparse.OptionParser(usage="%prog [options] <directories or files>",
                               description="This processes one or more datasets (FITS or HDF5) and extracts \
                                            fitted beam parameters from them. It runs interactively by default, \
                                            which allows the user to inspect results and discard bad scans. \
                                            By default all datasets in the current directory and all \
                                            subdirectories are processed.")
parser.set_defaults(catfilename='source_list.csv', pmfilename='pointing_model.csv', outfilebase='point_source_scans')
parser.add_option("-b", "--batch", dest="batch", action="store_true",
                  help="True if processing is to be done in batch mode without user interaction")
parser.add_option("-c", "--catalogue", dest="catfilename", type="string",
                  help="Name of optional source catalogue file used to override XDM FITS targets")
parser.add_option("-p", "--pointing_model", dest="pmfilename", type="string",
                  help="Name of optional file containing pointing model parameters in degrees (needed for XDM)")
parser.add_option("-o", "--output", dest="outfilebase", type="string",
                  help="Base name of output files (*.csv for output data and *.log for messages)")

(options, args) = parser.parse_args()
if len(args) < 1:
    args = ['.']

# Set up logging: logging everything (DEBUG & above), both to console and file
logger = logging.root
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(options.outfilebase + '.log', 'w')
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(fh)

# Load catalogue used to convert ACSM targets to katpoint ones
try:
    cat = katpoint.Catalogue(file(options.catfilename), add_specials=False)
except IOError:
    cat = None
# Load old pointing model parameters (useful if it is not in data file, like on XDM)
try:
    pm = katpoint.deg2rad(np.loadtxt(options.pmfilename, delimiter=','))
    # These scale factors are unitless, and should not be converted to radians
    pm[8] = katpoint.rad2deg(pm[8])
    pm[11] = katpoint.rad2deg(pm[11])
    logger.debug("Loaded %d-parameter pointing model from '%s'" % (len(pm), options.pmfilename))
except IOError:
    pm = None

# Find all data sets (HDF5 or FITS) mentioned, and add them to datasets
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
# Index to step through data sets as the buttons are pressed
index = 0
output_data = []
antenna = None

def next_load_reduce_plot(fig=None):
    """Load next data set, reduce the data, update the plots in given figure and store output data."""
    # If end of list is reached, save output data to file and exit
    global index, antenna
    if index >= len(datasets):
        f = file(options.outfilebase + '.csv', 'w')
        f.write('# antenna = %s\n' % antenna.description)
        f.write('dataset, target, timestamp_ut, azimuth, elevation, delta_azimuth, delta_elevation, data_unit, ' +
                'beam_height_I, beam_width_I, baseline_height_I, refined_I, beam_height_XX, beam_width_XX, ' +
                'baseline_height_XX, refined_XX, beam_height_YY, beam_width_YY, baseline_height_YY, refined_YY, ' +
                'frequency, flux, temperature, pressure, humidity, wind_speed, wind_direction\n')
        f.writelines([(('%s, %s, %s, %.7f, %.7f, %.7f, %.7f, %s, %.7f, %.7f, %.7f, %d, %.7f, %.7f, %.7f, %d, %.7f, ' +
                        '%.7f, %.7f, %d, %.7f, %.4f, %.2f, %.2f, %.2f, %.2f, %.2f\n') % p) for p in output_data if p])
        f.close()
        sys.exit(0)

    # Load next data set
    filename = datasets[index]
    index += 1
    logger.info("Loading dataset '%s'" % (filename,))
    d = scape.DataSet(filename, catalogue=cat, pointing_model=pm)
    antenna = d.antenna
    if filename.endswith('.fits'):
        dirs = filename.split(os.path.sep)
        if len(dirs) > 2:
            name = '%s_%s' % tuple(dirs[-3:-1])
        else:
            name = '%s' % (dirs[0],)
    else:
        name = os.path.splitext(os.path.basename(filename))[0]

    # Standard reduction
    d.remove_rfi_channels()
    d.convert_power_to_temperature()
    d = d.select(labelkeep='scan')
    # Save original channel frequencies before averaging
    channel_freqs = d.freqs
    d.average()

    # Handle missing data gracefully
    if len(d.compscans) == 0:
        logger.warning('No scan data found, skipping data set')
        output_data.append(None)
        if not options.batch:
            ax1.clear()
            ax1.set_title("%s - no scan data found" % name, size='medium')
            ax2.clear()
            plt.draw()
        return
    # Only use the first compound scan in the data set (this should be expanded later)
    compscan = d.compscans[0]

    # First fit XX and YY data, and extract beam and baseline heights and refined scan count
    d.fit_beams_and_baselines(pol='XX', circular_beam=False)
    beam_height_XX = compscan.beam.height if compscan.beam else np.nan
    beam_width_XX = katpoint.rad2deg(np.mean(compscan.beam.width)) if compscan.beam else np.nan
    baseline_height_XX = compscan.baseline_height()
    if baseline_height_XX is None:
        baseline_height_XX = np.nan
    refined_XX = compscan.beam.refined if compscan.beam else 0

    d.fit_beams_and_baselines(pol='YY', circular_beam=False)
    beam_height_YY = compscan.beam.height if compscan.beam else np.nan
    beam_width_YY = katpoint.rad2deg(np.mean(compscan.beam.width)) if compscan.beam else np.nan
    baseline_height_YY = compscan.baseline_height()
    if baseline_height_YY is None:
        baseline_height_YY = np.nan
    refined_YY = compscan.beam.refined if compscan.beam else 0

    # Now fit Stokes I, as this will be used for pointing and plots as well
    d.fit_beams_and_baselines(pol='I')
    # Calculate beam and baseline height and refined scan count
    beam_height_I = compscan.beam.height if compscan.beam else np.nan
    beam_width_I = katpoint.rad2deg(compscan.beam.width) if compscan.beam else np.nan
    baseline_height_I = compscan.baseline_height()
    if baseline_height_I is None:
        baseline_height_I = np.nan
    refined_I = compscan.beam.refined if compscan.beam else 0
    # Calculate average target flux over entire band
    flux_spectrum = [compscan.target.flux_density(freq) for freq in channel_freqs]
    average_flux = np.mean([flux for flux in flux_spectrum if flux])

    # Obtain middle timestamp of compound scan, where all pointing calculations are done
    middle_time = np.median([scan.timestamps for scan in compscan.scans], axis=None)
    # Obtain average environmental data
    temperature = np.mean([scan.environment['temperature'] for scan in d.scans]) \
                  if scan.environment.dtype.fields.has_key('temperature') else np.nan
    pressure = np.mean([scan.environment['pressure'] for scan in d.scans]) \
               if scan.environment.dtype.fields.has_key('pressure') else np.nan
    humidity = np.mean([scan.environment['humidity'] for scan in d.scans]) \
               if scan.environment.dtype.fields.has_key('humidity') else np.nan
    wind_speed = np.hstack([scan.environment['wind_speed'] for scan in d.scans]) \
                 if scan.environment.dtype.fields.has_key('wind_speed') else np.nan
    wind_direction = katpoint.deg2rad(np.hstack([scan.environment['wind_direction'] for scan in d.scans])) \
                     if scan.environment.dtype.fields.has_key('wind_direction') else np.nan
    wind_n, wind_e = np.mean(wind_speed * np.cos(wind_direction)), np.mean(wind_speed * np.sin(wind_direction))
    wind_speed, wind_direction = np.sqrt(wind_n ** 2 + wind_e ** 2), katpoint.rad2deg(np.arctan2(wind_e, wind_n))

    # Calculate pointing offset
    # Start with requested (az, el) coordinates, as they apply at the middle time for a moving target
    requested_azel = compscan.target.azel(middle_time)
    # Correct for refraction, which becomes the requested value at input of pointing model
    rc = katpoint.RefractionCorrection()
    requested_azel = [requested_azel[0], rc.apply(requested_azel[1], temperature, pressure, humidity)]
    requested_azel = katpoint.rad2deg(np.array(requested_azel))
    if compscan.beam:
        # Fitted beam center is in (x, y) coordinates, in projection centred on target
        beam_center_xy = compscan.beam.center
        # Convert this offset back to spherical (az, el) coordinates
        beam_center_azel = compscan.target.plane_to_sphere(beam_center_xy[0], beam_center_xy[1], middle_time)
        # Now correct the measured (az, el) for refraction and then apply the old pointing model
        # to get a "raw" measured (az, el) at the output of the pointing model
        beam_center_azel = [beam_center_azel[0], rc.apply(beam_center_azel[1], temperature, pressure, humidity)]
        beam_center_azel = d.pointing_model.apply(*beam_center_azel)
        beam_center_azel = katpoint.rad2deg(np.array(beam_center_azel))
        # Make sure the offset is a small angle around 0 degrees
        offset_azel = scape.stats.angle_wrap(beam_center_azel - requested_azel, 360.)
    else:
        offset_azel = np.array([np.nan, np.nan])

    # Display compound scan
    if not options.batch:
        (ax1, ax2), info = fig.axes[:2], fig.texts[0]
        ax1.clear()
        scape.plot_compound_scan_in_time(compscan, ax=ax1)
        ax1.set_title("%s '%s'\nazel=(%.1f, %.1f) deg, offset=(%.1f, %.1f) arcmin" %
                      (name, compscan.target.name, requested_azel[0], requested_azel[1],
                       60. * offset_azel[0], 60. * offset_azel[1]), size='medium')
        ax1.set_ylabel('Total power (%s)' % d.data_unit)
        ax2.clear()
        scape.plot_compound_scan_on_target(compscan, ax=ax2)
        if compscan.beam:
            info.set_text("Beamwidth = %.1f' (expected %.1f')\nBeam height = %.1f %s\nBaseline height = %.1f %s" %
                          (60. * katpoint.rad2deg(compscan.beam.width),
                           60. * katpoint.rad2deg(compscan.beam.expected_width),
                           beam_height_I, d.data_unit, baseline_height_I, d.data_unit))
        else:
            info.set_text("No beam\nBaseline height = %.2f %s" % (baseline_height_I, d.data_unit))
        plt.draw()

    # If beam is marked as invalid, discard scan only if in batch mode (otherwise discard button has to do it)
    if not compscan.beam or (options.batch and not compscan.beam.is_valid):
        output_data.append(None)
    else:
        output_data.append((name, compscan.target.name, katpoint.Timestamp(middle_time),
                            requested_azel[0], requested_azel[1], offset_azel[0], offset_azel[1],
                            d.data_unit, beam_height_I, beam_width_I, baseline_height_I, refined_I,
                            beam_height_XX, beam_width_XX, baseline_height_XX, refined_XX, beam_height_YY,
                            beam_width_YY, baseline_height_YY, refined_YY, d.freqs.mean(), average_flux,
                            temperature, pressure, humidity, wind_speed, wind_direction))

### BATCH MODE ###

# This will cycle through all data sets and stop when done
if options.batch:
    while True:
        next_load_reduce_plot()

### INTERACTIVE MODE ###

# Set up figure with buttons
plt.ion()
fig = plt.figure(1)
plt.clf()
plt.subplot(211)
plt.subplot(212)
plt.subplots_adjust(bottom=0.2)
plt.figtext(0.05, 0.05, '', va='bottom', ha='left')
# Create buttons and their callbacks
keep_button = widgets.Button(plt.axes([0.48, 0.05, 0.1, 0.075]), 'Keep')
def keep_callback(event):
    next_load_reduce_plot(fig)
keep_button.on_clicked(keep_callback)
discard_button = widgets.Button(plt.axes([0.59, 0.05, 0.1, 0.075]), 'Discard')
def discard_callback(event):
    output_data[-1] = None
    next_load_reduce_plot(fig)
discard_button.on_clicked(discard_callback)
back_button = widgets.Button(plt.axes([0.7, 0.05, 0.1, 0.075]), 'Back')
def back_callback(event):
    global index
    if index > 1:
        index -= 2
        output_data.pop()
        output_data.pop()
    next_load_reduce_plot(fig)
back_button.on_clicked(back_callback)
done_button = widgets.Button(plt.axes([0.81, 0.05, 0.1, 0.075]), 'Done')
def done_callback(event):
    global index
    index = len(datasets)
    next_load_reduce_plot(fig)
done_button.on_clicked(done_callback)

# Start off the processing
next_load_reduce_plot(fig)
plt.show()
