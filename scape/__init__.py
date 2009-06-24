"""The Single-dish Continuum Analysis PackagE."""

# Most operations are directed through the data set
from .dataset import DataSet

# Canned plots
from .plots import plot_waterfall, plot_compound_scan_in_time, plot_compound_scan_on_target, \
                   plot_data_set_in_mount_space, plot_measured_beam_pattern
