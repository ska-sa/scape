"""Container for the data of a single scan.

A *scan* is the *lowest-level object normally used by an observer,* in
accordance with the ALMA Science Data Model. This is normally done on a single
source, and includes a complete raster map of a source, a cross-hair pointing
scan, a focus scan, a gain curve scan, a holography scan, etc. It contains
several *subscans* and forms part of an overall *experiment*.

This module provides the :class:`Scan` class, which encapsulates all data
and actions related to a single scan of a point source, or a single scan at a
certain pointing. All actions requiring more than one scan are
grouped together in :class:`DataSet` instead.

Functionality: beam/baseline fitting, instant mount coords, ...

"""

class Scan(object):
    """Container for the data of a single scan."""
    def __init__(self, subscanlist):
        self.subscans = subscanlist
