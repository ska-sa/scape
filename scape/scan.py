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

import numpy as np

import coord

class SpectralConfig(object):
    """Container for spectral configuration of correlator.
    
    Parameters
    ----------
    freqs : real array-like, shape (*F*,)
        Sequence of channel/band centre frequencies, in Hz
    bandwidths : real array-like, shape (*F*,)
        Sequence of channel/band bandwidths, in Hz
    rfi_channels : list of ints
        RFI-flagged channel indices
    channels_per_band : List of lists of ints
        List of lists of channel indices (one list per band), indicating which
        channels belong to each band
    dump_rate : float
        Correlator dump rate, in Hz
    
    """
    def __init__(self, freqs, bandwidths, rfi_channels, channels_per_band, dump_rate):
        # Keep as doubles to prevent precision issues
        self.freqs = np.asarray(freqs, dtype='double')
        self.bandwidths = np.asarray(bandwidths, dtype='double')
        self.rfi_channels = rfi_channels
        self.channels_per_band = channels_per_band
        self.dump_rate = dump_rate
    
    def select(self, freqkeep=None):
        """Select a subset of frequency channels/bands.
        
        Parameters
        ----------
        freqkeep : sequence of bools or ints, optional
            Sequence of indicators of which frequency channels/bands to keep
            (either integer indices or booleans that are True for the values to
            be kept). The default is None, which keeps all channels/bands.
        
        Returns
        -------
        spectral : :class:`SpectralConfig` object
            Spectral configuration object with subset of channels/bands
        
        """
        if freqkeep is None:
            return self
        elif np.asarray(freqkeep).dtype == 'bool':
            freqkeep = np.asarray(freqkeep).nonzero()[0]
        return SpectralConfig(self.freqs[freqkeep], self.bandwidths[freqkeep], self.rfi_channels,
                              [self.channels_per_band[n] for n in xrange(len(self.channels_per_band))
                               if n in freqkeep], self.dump_rate)

class Scan(object):
    """Container for the data of a single scan.
    
    Parameters
    ----------
    subscanlist : list of :class:`subscan.SubScan` objects
        List of subscan objects
    target : string
        Name of the target of this scan
    
    """
    def __init__(self, subscanlist, target):
        self.subscans = subscanlist
        # Interpret source name string and return relevant object
        self.target = coord.construct_source(target)
        # self.target_coords = coord.sphere_to_plane(self.target, self.antenna,
        #                                            pointing['az'], pointing['el'], timestamps)
