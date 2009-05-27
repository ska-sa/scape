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

from .coord import construct_source

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
        return SpectralConfig(self.freqs[freqkeep], self.bandwidths[freqkeep],
                              [freqkeep.index(n) for n in (set(self.rfi_channels) & set(freqkeep))],
                              [[freqkeep.index(n) for n in (set(chanlist) & set(freqkeep))] 
                               for chanlist in self.channels_per_band],
                              self.dump_rate)
    
    def merge(self):
        """Merge frequency channels into bands.
        
        This applies the :attr:`channels_per_band` mapping to the rest of the
        frequency data. Each band centre frequency is the mean of the
        corresponding channel group's frequencies, while the band bandwidth is
        the sum of the corresponding channel group's bandwidths. Any band
        containing an RFI-flagged channel is RFI-flagged too, and the
        channels_per_band mapping becomes one-to-one after the merge.
         
        """
        # Each band centre frequency is the mean of the corresponding channel centre frequencies
        self.freqs = np.array([self.freqs[chans].mean() for chans in self.channels_per_band], dtype='double')
        # Each band bandwidth is the sum of the corresponding channel bandwidths
        self.bandwidths = np.array([self.bandwidths[chans].sum() for chans in self.channels_per_band],
                                   dtype='double')
        # If the band contains *any* RFI-flagged channel, it is RFI-flagged too
        self.rfi_channels = np.array([(len(set(chans) & set(self.rfi_channels)) > 0) 
                                      for chans in self.channels_per_band], dtype='bool').nonzero()[0].tolist()
        self.channels_per_band = np.arange(len(self.freqs))[:, np.newaxis].tolist()
        return self

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
        self.target = construct_source(target)
        # self.target_coords = coord.sphere_to_plane(self.target, self.antenna,
        #                                            pointing['az'], pointing['el'], timestamps)
