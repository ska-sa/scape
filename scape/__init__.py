"""The Single-dish Continuum Analysis PackagE."""

import logging
import sys

# Set up basic logging if it hasn't been done yet, in order to display error messages and such
logger = logging.getLogger("scape")
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format="%(levelname)s: %(message)s")

# Most operations are directed through the data set
from .dataset import DataSet

# Canned plots (ignored if no matplotlib is found)
try:
    from .plots import plot_waterfall, plot_compound_scan_in_time, plot_compound_scan_on_target, \
                       plot_data_set_in_mount_space, plot_measured_beam_pattern
except ImportError:
    logger.warn('Plotting routines could not be imported (probably because matplotlib is not installed)')
