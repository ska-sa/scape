"""The Single-dish Continuum Analysis PackagE."""

import logging
import sys

# Setup library logger, and suppress spurious logger messages via a null handler
class _NullHandler(logging.Handler):
    def emit(self, record):
        pass
logger = logging.getLogger("scape")
logger.addHandler(_NullHandler())
# Set up basic logging if it hasn't been done yet, in order to display error messages and such
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format="%(levelname)s: %(message)s")

# Most operations are directed through the data set
from .dataset import DataSet
from .compoundscan import CorrelatorConfig, CompoundScan
from .scan import Scan

# Check if matplotlib is present, otherwise skip plotting routines
try:
    import matplotlib as mpl
except ImportError:
    logger.warn('Matplotlib was not found - plotting will be disabled')
else:
    from .plots_canned import plot_spectrum, plot_waterfall, plot_spectrogram, plot_fringes, \
                              plot_compound_scan_in_time, plot_compound_scan_on_target, \
                              plot_data_set_in_mount_space, plot_measured_beam_pattern
