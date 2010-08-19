"""The Single-dish Continuum Analysis PackagE."""

import logging as _logging
import sys as _sys

# Setup library logger, and suppress spurious logger messages via a null handler
class _NullHandler(_logging.Handler):
    def emit(self, record):
        pass
logger = _logging.getLogger("scape")
logger.addHandler(_NullHandler())
# Set up basic logging if it hasn't been done yet, in order to display error messages and such
_logging.basicConfig(level=_logging.DEBUG, stream=_sys.stdout, format="%(levelname)s: %(message)s")

# Most operations are directed through the data set
from .dataset import DataSet
from .compoundscan import CorrelatorConfig, CompoundScan
from .scan import Scan

# Check if matplotlib is present, otherwise skip plotting routines
try:
    import matplotlib as _mpl
except ImportError:
    logger.warn('Matplotlib was not found - plotting will be disabled')
else:
    from .plots_canned import plot_xyz, extract_xyz_data, extract_scan_data, \
                              plot_spectrum, plot_waterfall, plot_spectrogram, plot_fringes, \
                              plot_compound_scan_in_time, plot_compound_scan_on_target, \
                              plot_data_set_in_mount_space, plot_measured_beam_pattern

try:
    import pkg_resources as _pkg_resources
    dist = _pkg_resources.get_distribution("scape")
    # ver needs to be a list since tuples in Python <= 2.5 don't have
    # a .index method.
    ver = list(dist.parsed_version)
    __version__ = "r%d" % int(ver[ver.index("*r") + 1])
except (ImportError, _pkg_resources.DistributionNotFound, ValueError, IndexError, TypeError):
    __version__ = "unknown"
