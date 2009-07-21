#
# Example script that uses scape to reduce pointing data. The user can interactively
# observe reduction results and discard bad data. The end product is a file
# containing pointing offsets.
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
                                            pointing offsets from them. It runs interactively by default, \
                                            which allows the user to inspect results and discard bad scans. \
                                            By default all datasets in the current directory and all \
                                            subdirectories are processed.")
parser.set_defaults(catfilename='source_list.csv')
parser.add_option("-b", "--batch", dest="batch", action="store_true",
                  help="True if processing is to be done in batch mode without user interaction")
parser.add_option("-c", "--catalogue", dest="catfilename", type="string",
                  help="Name of optional source catalogue file used to override XDM FITS targets")
(options, args) = parser.parse_args()
if len(args) < 1:
    args = ['.']

# Global message logging level
logging.root.setLevel(logging.DEBUG)
# Load catalogue used to convert ACSM targets to katpoint ones
cat = katpoint.Catalogue(add_specials=False)
try:
    cat.add(file(options.catfilename))
except IOError:
    cat = None

# Find all data sets (HDF5 or FITS) mentioned, and add them to datasets
datasets = []
def walk_callback(arg, directory, files):
    datasets.extend([os.path.join(directory, f) for f in files if f.endswith('.h5') or f.endswith('_0000.fits')])
for arg in args:
    if os.path.isdir(arg):
        os.path.walk(arg, walk_callback, None)
    else:
        datasets.extend(glob.glob(arg))
# Use iterator to step through data sets instead as the buttons are pressed
dataset_iter = iter(datasets)
pointing_offsets = []

def load_reduce_plot(ax1=None, ax2=None):
    """Load next data set, reduce the data and update the plot in given axes."""
    # Get next file name - if no more, stop and save pointing offsets to file and exit
    try:
        filename = dataset_iter.next()
    except StopIteration:
        f = file('pointing_offsets.csv', 'w')
        f.write('dataset, azimuth, elevation, delta.azimuth, delta.elevation\n')
        f.writelines([('%s, %.7f, %.7f, %.7f, %.7f\n' % p) for p in pointing_offsets if p])
        f.close()
        sys.exit(0)
    
    # Load data set
    print "Loading dataset '%s'" % (filename,)
    d = scape.DataSet(filename, catalogue=cat)
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
    d.average()
    d = d.select(labelkeep='scan')
    d.fit_beams_and_baselines()
    
    # Calculate pointing offset
    compscan = d.compscans[0]
    middle_time = np.median([scan.timestamps for scan in compscan.scans], axis=None)
    requested_azel = compscan.target.azel(middle_time)
    requested_azel = katpoint.rad2deg(np.array(requested_azel))
    if compscan.beam:
        beam_center_xy = compscan.beam.center
        beam_center_azel = compscan.target.plane_to_sphere(beam_center_xy[0], beam_center_xy[1], middle_time)
        beam_center_azel = katpoint.rad2deg(np.array(beam_center_azel))
        offset_azel = scape.stats.angle_wrap(beam_center_azel - requested_azel, 360.)
    else:
        offset_azel = np.array([np.nan, np.nan])
    
    # Display compound scan
    if not options.batch:
        ax1.clear()
        scape.plot_compound_scan_in_time(compscan, ax=ax1)
        ax1.set_title("%s '%s'\nazel=(%.1f, %.1f) deg, offset=(%.1f, %.1f) arcmin" %
                      (name, compscan.target.name, requested_azel[0], requested_azel[1],
                       60. * offset_azel[0], 60. * offset_azel[1]), size='medium')
        ax1.set_ylabel('Total power (%s)' % d.data_unit)
        ax2.clear()
        scape.plot_compound_scan_on_target(compscan, ax=ax2)
        if compscan.beam:
            ax2.text(0, -0.2, 'Expected beamwidth = %.1f arcmin\nFitted beamwidth = %.1f arcmin' % 
                              (60. * katpoint.rad2deg(compscan.beam.expected_width),
                               60. * katpoint.rad2deg(compscan.beam.width)),
                     ha='left', va='top', transform=ax2.transAxes)
        plt.draw()
    
    if compscan.beam and compscan.beam.is_valid:
        pointing_offsets.append((name, requested_azel[0], requested_azel[1], offset_azel[0], offset_azel[1]))
    else:
        pointing_offsets.append(None)
    
# This will cycle through all data sets and stop when done
if options.batch:
    while True:
        load_reduce_plot()

# Set up figure with buttons
plt.ion()
plt.figure(1)
plt.clf()
ax1, ax2 = plt.subplot(211), plt.subplot(212)
plt.subplots_adjust(bottom=0.175)
# Create buttons and their callbacks
keep_button = widgets.Button(plt.axes([0.7, 0.05, 0.1, 0.075]), 'Keep')
def keep_callback(event):
    load_reduce_plot(ax1, ax2)
keep_button.on_clicked(keep_callback)
discard_button = widgets.Button(plt.axes([0.81, 0.05, 0.1, 0.075]), 'Discard')
def discard_callback(event):
    pointing_offsets.pop()
    load_reduce_plot(ax1, ax2)
discard_button.on_clicked(discard_callback)

# Start off the processing
load_reduce_plot(ax1, ax2)
plt.show()
