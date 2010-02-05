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
        Sequence of channel/band centre frequencies, in MHz
    bandwidths : real array-like, shape (*F*,)
        Sequence of channel/band bandwidths, in MHz
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
        # Make sure freqkeep is a list, as we need to call its .index method
        freqkeep = np.asarray(freqkeep).tolist()
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
        Object that describes initial fitted baseline
    dataset : :class:`DataSet` object, optional
        Parent data set of which this compound scan forms a part

    """
    def __init__(self, scanlist, target, beam=None, baseline=None, dataset=None):
        self.scans = scanlist
        if isinstance(target, katpoint.Target):
            self.target = target
        else:
            self.target = katpoint.construct_target(target)
        self.beam = beam
        self.baseline = baseline
        self.dataset = dataset

    def __eq__(self, other):
        """Equality comparison operator."""
        if len(self.scans) != len(other.scans):
            return False
        for self_scan, other_scan in zip(self.scans, other.scans):
            if self_scan != other_scan:
                return False
        return (self.target.description == other.target.description)

    def __ne__(self, other):
        """Inequality comparison operator."""
        return not self.__eq__(other)

    def __str__(self):
        """Verbose human-friendly string representation of compound scan object."""
        descr = ["target='%s' [%s]" % (self.target.name, self.target.body_type)]
        if self.baseline:
            descr[0] += ', initial baseline offset=%f' % (self.baseline.poly[-1],)
        if self.beam:
            descr[0] += ', beam height=%f' % (self.beam.height,)
        for scan_ind, scan in enumerate(self.scans):
            descr.append('%4d: %s' % (scan_ind, str(scan)))
        return '\n'.join(descr)

    def __repr__(self):
        """Short human-friendly string representation of compound scan object."""
        return "<scape.CompoundScan target='%s' scans=%d at 0x%x>" % (self.target.name, len(self.scans), id(self))

    def baseline_height(self):
        """Estimate height of fitted baseline (at fitted beam center).

        This estimates the height of the fitted baseline (if any) at the beam
        center (or the target position, if no beam is fitted). It takes into
        account scan-based baselines in the case of a refined beam.

        Returns
        -------
        height : float or None
            Estimated baseline height, in data units (None if no baseline)

        """
        if self.beam and self.beam.is_valid:
            # Refined beam has at least 2 per-scan baselines - obtain weighted average closest to beam center
            if self.beam.refined:
                dist_to_center = np.tile(np.inf, len(self.scans))
                closest_time = np.zeros(len(self.scans))
                # Find sample in time in each scan that is closest to beam center
                for n, scan in enumerate(self.scans):
                    if scan.baseline:
                        dist_sq = (scan.target_coords[0] - self.beam.center[0]) ** 2 + \
                                  (scan.target_coords[1] - self.beam.center[1]) ** 2
                        closest_sample = dist_sq.argmin()
                        dist_to_center[n] = np.sqrt(dist_sq[closest_sample])
                        closest_time[n] = scan.timestamps[closest_sample]
                # Pick the closest two samples (in different scans)
                closest_scan = dist_to_center.argmin()
                dist_closest = dist_to_center[closest_scan]
                assert dist_closest < np.inf, 'Beam is refined but no scan-based baselines found'
                dist_to_center[closest_scan] = np.inf
                next_closest_scan = dist_to_center.argmin()
                dist_next_closest = dist_to_center[next_closest_scan]
                assert dist_next_closest < np.inf, 'Beam is refined but less than 2 scan-based baselines found'
                # Return a weighted sum of the per-scan baseline heights at the closest two samples
                # Baseline height is linear combination - assumes beam center is *between* nearest two scans
                baseline_closest = self.scans[closest_scan].baseline(closest_time[closest_scan])
                baseline_next_closest = self.scans[next_closest_scan].baseline(closest_time[next_closest_scan])
                return (dist_next_closest * baseline_closest + dist_closest * baseline_next_closest) / \
                       (dist_closest + dist_next_closest)
            else:
                # Return compound scan-based baseline height at beam center
                return self.baseline(np.expand_dims(self.beam.center, 1))
        elif self.baseline:
            # Without a beam, return the baseline height at the target position
            return self.baseline([[0.], [0.]])
        else:
            # Without no baseline or beam, return None
            return None
