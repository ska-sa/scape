"""The Single-dish Continuum Analysis PackagE."""

import logging as _logging


# Setup library logger and add a print-like handler used when no logging is configured
class _NoConfigFilter(_logging.Filter):
    """Filter which only allows event if top-level logging is not configured."""

    def filter(self, record):
        return 1 if not _logging.root.handlers else 0


_no_config_handler = _logging.StreamHandler()
_no_config_handler.setFormatter(_logging.Formatter(_logging.BASIC_FORMAT))
_no_config_handler.addFilter(_NoConfigFilter())
logger = _logging.getLogger(__name__)
logger.addHandler(_no_config_handler)
logger.setLevel(_logging.DEBUG)


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
        logger.warn(message)
        return False


# Check if matplotlib is present, otherwise skip plotting routines
if _module_found('matplotlib', 'Matplotlib was not found - plotting will be disabled'):
    from .plots_canned import (plot_xyz, extract_xyz_data,  # noqa: F401
                               extract_scan_data,  # noqa: F401
                               plot_spectrum, plot_waterfall,  # noqa: F401
                               plot_spectrogram, plot_fringes,  # noqa: F401
                               plot_compound_scan_in_time,  # noqa: F401
                               plot_compound_scan_on_target,  # noqa: F401
                               plot_data_set_in_mount_space,  # noqa: F401
                               plot_measured_beam_pattern)  # noqa: F401

# Check if astropy.io.fits is present, otherwise skip FITS creation routines
if _module_found('astropy.io.fits',
                 'astropy.io.fits was not found - FITS creation will be disabled'):
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

