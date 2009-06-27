"""Container for the data of a compound scan.

A *compound scan* is the *lowest-level object normally used by an observer,*
which corresponds to the *scan* of the ALMA Science Data Model. This is normally
done on a single source. Examples include a complete raster map of a source, a
cross-hair pointing scan, a focus scan, a gain curve scan, a holography scan,
etc. It contains one or more *scans* and forms part of an overall *experiment*.

This module provides the :class:`CompoundScan` class, which encapsulates all
data and actions related to a compound scan of a point source, or a compund scan
at a certain pointing. All actions requiring more than one compound scan are
grouped together in :class:`DataSet` instead.

Functionality: beam/baseline fitting, instant mount coords, ...

"""

import numpy as np

import katpoint

#--------------------------------------------------------------------------------------------------
#--- CLASS :  CorrelatorConfig
#--------------------------------------------------------------------------------------------------

class CorrelatorConfig(object):
    """Container for spectral configuration of correlator.
    
    This is a convenience container for all the items related to the correlator
    configuration, such as channel centre frequencies and bandwidths. It
    simplifies the copying of these bits of data, while they are usually also
    found together in use.
    
    Parameters
    ----------
    freqs : real array-like, shape (*F*,)
        Sequence of channel/band centre frequencies, in Hz
    bandwidths : real array-like, shape (*F*,)
        Sequence of channel/band bandwidths, in Hz
    rfi_channels : list of ints
        RFI-flagged channel indices
    dump_rate : float
        Correlator dump rate, in Hz
    
    Notes
    -----
    This class should ideally be grouped with :class:`dataset.DataSet`, as that
    is where it is stored in the data set hierarchy. The problem is that the
    file readers also need to instantiate this class, which will lead to
    circular imports if this class is stored in the :mod:`dataset` module. If
    the functionality of this class grows, it might be useful to move it to
    its own module.
    
    """
    def __init__(self, freqs, bandwidths, rfi_channels, dump_rate):
        # Keep as doubles to prevent precision issues
        self.freqs = np.asarray(freqs, dtype='double')
        self.bandwidths = np.asarray(bandwidths, dtype='double')
        self.rfi_channels = rfi_channels
        self.dump_rate = dump_rate
    
    def __eq__(self, other):
        """Equality comparison operator."""
        return np.all(self.freqs == other.freqs) and np.all(self.bandwidths == other.bandwidths) and \
               np.all(self.rfi_channels == other.rfi_channels) and (self.dump_rate == other.dump_rate)
    
    def __ne__(self, other):
        """Inequality comparison operator."""
        return not self.__eq__(other)
    
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
        corrconf : :class:`CorrelatorConfig` object
            Correlator configuration object with subset of channels/bands
        
        """
        if freqkeep is None:
            return self
        elif np.asarray(freqkeep).dtype == 'bool':
            freqkeep = np.asarray(freqkeep).nonzero()[0]
        return CorrelatorConfig(self.freqs[freqkeep], self.bandwidths[freqkeep],
                                [freqkeep.index(n) for n in (set(self.rfi_channels) & set(freqkeep))],
                                self.dump_rate)
    
    def merge(self, channels_per_band):
        """Merge frequency channels into bands.
        
        This applies the *channels_per_band* mapping to the frequency data.
        Each band centre frequency is the mean of the corresponding channel
        group's frequencies, while the band bandwidth is the sum of the
        corresponding channel group's bandwidths. Any band containing an
        RFI-flagged channel is RFI-flagged too.
        
        Parameters
        ----------
        channels_per_band : List of lists of ints, optional
            List of lists of channel indices (one list per band), indicating
            which channels are averaged together to form each band.
        
        """
        # Each band centre frequency is the mean of the corresponding channel centre frequencies
        self.freqs = np.array([self.freqs[chans].mean() for chans in channels_per_band], dtype='double')
        # Each band bandwidth is the sum of the corresponding channel bandwidths
        self.bandwidths = np.array([self.bandwidths[chans].sum() for chans in channels_per_band],
                                   dtype='double')
        # If the band contains *any* RFI-flagged channel, it is RFI-flagged too
        self.rfi_channels = np.array([(len(set(chans) & set(self.rfi_channels)) > 0) 
                                      for chans in channels_per_band], dtype='bool').nonzero()[0].tolist()
        return self

#--------------------------------------------------------------------------------------------------
#--- CLASS :  CompoundScan
#--------------------------------------------------------------------------------------------------

class CompoundScan(object):
    """Container for the data of a compound scan.
    
    Parameters
    ----------
    scanlist : list of :class:`scan.Scan` objects
        List of scan objects
    target : :class:`katpoint.Target` object, or string
        The target of this compound scan, or its description string
    beam : :class:`beam_baseline.BeamPatternFit` object, optional
        Object that describes fitted beam
    baseline : :class:`fitting.Polynomial2DFit` object, optional
        Object that describes fitted baseline
    
    """
    def __init__(self, scanlist, target, beam=None, baseline=None):
        self.scans = scanlist
        if isinstance(target, basestring):
            self.target = katpoint.construct_target(target)
        else:
            self.target = target
        self.beam = beam
        self.baseline = baseline

    def __eq__(self, other):
        """Equality comparison operator."""
        if len(self.scans) != len(other.scans):
            return False
        for self_scan, other_scan in zip(self.scans, other.scans):
            if self_scan != other_scan:
                return False
        return (self.target.get_description() == other.target.get_description())
    
    def __ne__(self, other):
        """Inequality comparison operator."""
        return not self.__eq__(other)
