"""The Single-dish Continuum Analysis PackagE."""

import logging
import sys


# Setup library logger, and suppress spurious logger messages via a null handler
class _NullHandler(logging.Handler):
    def emit(self, record):
        pass
_logger = logging.getLogger("scape")
_logger.addHandler(_NullHandler())
# Set up basic logging if it hasn't been done yet, in order to display error messages and such
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format="%(levelname)s: %(message)s")

# Most operations are directed through the data set
from .dataset import DataSet  # noqa: E402,F401
from .compoundscan import CorrelatorConfig, CompoundScan  # noqa: F401
from .scan import Scan  # noqa: F401


def _module_found(name, message):
    """Check whether module *name* imports, otherwise log a warning *message*."""
    try:
        __import__(name)
        return True
    except ImportError:
        _logger.warn(message)
        return False

# Check if matplotlib is present, otherwise skip plotting routines
if _module_found('matplotlib', 'Matplotlib was not found - plotting will be disabled'):
    from .plots_canned import (plot_xyz, extract_xyz_data, extract_scan_data,  # noqa: F401
                               plot_spectrum, plot_waterfall, plot_spectrogram, plot_fringes,  # noqa: F401
                               plot_compound_scan_in_time, plot_compound_scan_on_target,  # noqa: F401
                               plot_data_set_in_mount_space, plot_measured_beam_pattern)  # noqa: F401

# Check if pyfits is present, otherwise skip FITS creation routines
if _module_found('pyfits', 'PyFITS was not found - FITS creation will be disabled'):
    from .plots_basic import save_fits_image  # noqa: F401

# BEGIN VERSION CHECK
# Get package version when locally imported from repo or via -e develop install
try:
    import katversion as _katversion
except ImportError:
    import time as _time
    __version__ = "0.0+unknown.{}".format(_time.strftime('%Y%m%d%H%M'))
else:
    __version__ = _katversion.get_version(__path__[0])  # noqa: F821
# END VERSION CHECK

# Clean up module namespace to make it easier to spot the useful parts
del logging, sys, _NullHandler, _module_found
