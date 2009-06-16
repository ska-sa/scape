"""Container for the data of a single-dish experiment."""

import os.path
import logging

import numpy as np

import katpoint
from .compoundscan import CompoundScan
from .gaincal import calibrate_gain
from .beam_baseline import fit_beam_and_baseline

# Try to import all available formats
try:
    from .xdmfits import load_dataset as xdmfits_load
    xdmfits_found = True
except ImportError:
    xdmfits_found = False
try:
    from .hdf5 import load_dataset as hdf5_load
    from .hdf5 import save_dataset as hdf5_save
    hdf5_found = True
except ImportError:
    hdf5_found = False

logger = logging.getLogger("scape.dataset")

#--------------------------------------------------------------------------------------------------
#--- CLASS :  DataSet
#--------------------------------------------------------------------------------------------------

class DataSet(object):
    """Container for the data of a single-dish experiment.
    
    This is the top-level container for the data of a single-dish experiment.
    Given a data filename, the initialiser determines the appropriate file format
    to use, based on the file extension. If the filename is blank, the
    :class:`DataSet` can also be directly initialised from its constituent
    parts, which is useful for simulations and creating the data sets from
    scratch. The :class:`DataSet` object contains a list of compound scans, as
    well as the correlator configuration, antenna details and noise diode model.
    
    Parameters
    ----------
    filename : string
        Name of data set file, or blank string if the other parameters are given
    compscanlist : list of :class:`compoundscan.CompoundScan` objects, optional
        List of compound scans
    data_unit : {'raw', 'K', 'Jy'}, optional
        Physical unit of power data
    corrconf : :class:`compoundscan.CorrelatorConfig` object, optional
        Correlator configuration object
    antenna : string, optional
        Description string of antenna that produced the data set
    nd_data : :class:`gaincal.NoiseDiodeModel` object, optional
        Noise diode model
    kwargs : dict, optional
        Extra keyword arguments are passed to selected :func:`load_dataset` function
    
    Attributes
    ----------
    freqs : real array, shape (*F*,)
        Centre frequency of each channel/band, in Hz (same as in *corrconf*)
    bandwidths : real array-like, shape (*F*,)
        Bandwidth of each channel/band, in Hz (same as in *corrconf*)
    rfi_channels : list of ints
        RFI-flagged channel indices (same as in *corrconf*)
    dump_rate : float
        Correlator dump rate, in Hz (same as in *corrconf*)
    scans : list of :class:`scan.Scan` objects
        Flattened list of all scans in data set
    
    Raises
    ------
    ImportError
        If file extension is known, but appropriate module would not import
    ValueError
        If file extension is unknown
    KeyError
        If the antenna name is unknown
    
    """
    def __init__(self, filename, compscanlist=None, data_unit=None, corrconf=None,
                 antenna=None, nd_data=None, **kwargs):
        if filename:
            ext = os.path.splitext(filename)[1]
            if ext == '.fits':
                if not xdmfits_found:
                    raise ImportError('XDM FITS support could not be loaded - please check xdmfits module')
                compscanlist, data_unit, corrconf, antenna, nd_data = xdmfits_load(filename, **kwargs)
            elif (ext == '.h5') or (ext == '.hdf5'):
                if not hdf5_found:
                    raise ImportError('HDF5 support could not be loaded - please check hdf5 module')
                compscanlist, data_unit, corrconf, antenna, nd_data = hdf5_load(filename, **kwargs)
            else:
                raise ValueError("File extension '%s' not understood" % ext)
        self.compscans = compscanlist
        self.data_unit = data_unit
        self.corrconf = corrconf
        self.antenna = katpoint.construct_antenna(antenna)
        self.noise_diode_data = nd_data
        # Create scan list
        self.scans = []
        for compscan in self.compscans:
            self.scans.extend(compscan.scans)
        # Calculate target coordinates for all scans. This functionality is here at the highest level
        # because it involves interaction between the DataSet, CompoundScan and Scan levels, while the results
        # need to be stored at a Scan level.
        for compscan in self.compscans:
            for scan in compscan.scans:
                scan.calc_target_coords(compscan.target, self.antenna)
    
    def __eq__(self, other):
        """Equality comparison operator."""
        if len(self.compscans) != len(other.compscans):
            return False
        for self_compscan, other_compscan in zip(self.compscans, other.compscans):
            if self_compscan != other_compscan:
                return False
        return (self.data_unit == other.data_unit) and (self.corrconf == other.corrconf) and \
               (self.antenna.get_description() == other.antenna.get_description()) and \
               (self.noise_diode_data == other.noise_diode_data)
    
    def __ne__(self, other):
        """Inequality comparison operator."""
        return not self.__eq__(other)
    
    # Provide properties to access the attributes of the correlator configuration directly
    # This uses the same trick as in stats.MuSigmaArray to create the properties, which
    # leads to less class namespace clutter, but more pylint uneasiness (shame).
    # pylint: disable-msg=E0211,E0202,W0612,W0142
    def freqs():
        """Class method which creates freqs property."""
        doc = 'Centre frequency of each channel/band, in Hz.'
        def fget(self):
            return self.corrconf.freqs
        def fset(self, value):
            self.corrconf.freqs = value
        return locals()
    freqs = property(**freqs())
    
    # pylint: disable-msg=E0211,E0202,W0612,W0142
    def bandwidths():
        """Class method which creates bandwidths property."""
        doc = 'Bandwidth of each channel/band, in Hz.'
        def fget(self):
            return self.corrconf.bandwidths
        def fset(self, value):
            self.corrconf.bandwidths = value
        return locals()
    bandwidths = property(**bandwidths())
    
    # pylint: disable-msg=E0211,E0202,W0612,W0142
    def rfi_channels():
        """Class method which creates rfi_channels property."""
        doc = 'List of RFI-flagged channels.'
        def fget(self):
            return self.corrconf.rfi_channels
        def fset(self, value):
            self.corrconf.rfi_channels = value
        return locals()
    rfi_channels = property(**rfi_channels())
    
    # pylint: disable-msg=E0211,E0202,W0612,W0142
    def dump_rate():
        """Class method which creates dump_rate property."""
        doc = 'Correlator dump rate, in Hz.'
        def fget(self):
            return self.corrconf.dump_rate
        def fset(self, value):
            self.corrconf.dump_rate = value
        return locals()
    dump_rate = property(**dump_rate())
    
    def convert_to_coherency(self):
        """Convert power data to coherency format.
        
        This is a convenience function to convert the data of the underlying
        scans to coherency format.
        
        """
        for scan in self.scans:
            scan.convert_to_coherency()
    
    def convert_to_stokes(self):
        """Convert power data to Stokes parameter format.
        
        This is a convenience function to convert the data of the underlying
        scans to Stokes parameter format.
        
        """
        for scan in self.scans:
            scan.convert_to_stokes()
    
    def save(self, filename, **kwargs):
        """Save data set object to file.
        
        This automatically figures out which file format to use based on the
        file extension.
        
        Parameters
        ----------
        filename : string
            Name of output file
        kwargs : dict, optional
            Extra keyword arguments are passed on to underlying save function
        
        Raises
        ------
        IOError
            If output file already exists
        ValueError
            If file extension is unknown or unsupported
        ImportError
            If file extension is known, but appropriate module would not import
        
        """
        if os.path.exists(filename):
            raise IOError('File %s already exists - please remove first!' % filename)
        ext = os.path.splitext(filename)[1]
        if ext == '.fits':
            raise ValueError('XDM FITS writing support not implemented')
        elif (ext == '.h5') or (ext == '.hdf5'):
            if not hdf5_found:
                raise ImportError('HDF5 support could not be loaded - please check hdf5 module')
            hdf5_save(self, filename, **kwargs)
        else:
            raise ValueError("File extension '%s' not understood" % ext)
    
    def select(self, labelkeep=None, flagkeep=None, freqkeep=None, copy=False):
        """Select subset of data set, based on scan label, flags and frequency.
        
        This returns a data set with a possibly reduced number of time samples,
        frequency channels/bands and scans, based on the selection criteria.
        Since each scan potentially has a different number of time samples,
        it is less useful to filter directly on sample index. Instead, the
        flags are used to select a subset of time samples in each scan. The
        list of flags are ANDed together to determine which parts are kept. It
        is also possible to invert flags by prepending a ~ (tilde) character.
        
        Based on the value of *copy*, the new data set contains either a view of
        the original data or a copy. All criteria are optional, and with no
        parameters the returned data set is unchanged. This can be used to make
        a copy of the data set.
        
        Parameters
        ----------
        labelkeep : list of strings, optional
            All scans with labels in this list will be kept. The default is
            None, which means all labels are kept.
        flagkeep : list of strings, optional
            List of flags used to select time ranges in each scan. The time
            samples for which all the flags in the list are true are kept.
            Individual flags can be negated by prepending a ~ (tilde) character.
            The default is None, which means all time samples are kept.
        freqkeep : sequence of bools or ints, optional
            Sequence of indicators of which frequency channels/bands to keep
            (either integer indices or booleans that are True for the values to
            be kept). The default is None, which keeps all channels/bands.
        copy : {False, True}, optional
            True if the new scan is a copy, False if it is a view
        
        Returns
        -------
        dataset : :class:`DataSet` object
            Data set with selection of scans with possibly smaller data arrays.
        
        Raises
        ------
        KeyError
            If flag in *flagkeep* is unknown
        
        """
        compscanlist = []
        for compscan in self.compscans:
            scanlist = []
            for scan in compscan.scans:
                # Convert flag selection to time sample selection
                if flagkeep is None:
                    timekeep = None
                else:
                    # By default keep all time samples
                    timekeep = np.tile(True, len(scan.timestamps))
                    for flag in flagkeep:
                        invert = False
                        # Flags prepended with ~ get inverted
                        if flag[0] == '~':
                            invert = True
                            flag = flag[1:]
                        # Ignore unknown flags
                        try:
                            flag_data = scan.flags[flag]
                        except KeyError:
                            raise KeyError("Unknown flag '%s'" % flag)
                        if invert:
                            timekeep &= ~flag_data
                        else:
                            timekeep &= flag_data
                if (labelkeep is None) or (scan.label in labelkeep):
                    scanlist.append(scan.select(timekeep, freqkeep, copy))
            if scanlist:
                compscanlist.append(CompoundScan(scanlist, compscan.target.get_description()))
        return DataSet(None, compscanlist, self.data_unit, self.corrconf.select(freqkeep),
                       self.antenna.get_description(), self.noise_diode_data)
    
    
    def remove_rfi_channels(self):
        """Remove RFI-flagged channels from data set, returning a copy.
        
        This is a convenience function that selects all the non-RFI-flagged
        frequency channels in the data set and returns a copy.
        
        """
        non_rfi = list(set(range(len(self.freqs))) - set(self.rfi_channels))
        d = self.select(freqkeep=non_rfi, copy=True)
        DataSet.__init__(self, None, d.compscans, d.data_unit, d.corrconf, d.antenna.get_description(), d.noise_diode_data)
        return self
    
    def convert_power_to_temperature(self, randomise=False, **kwargs):
        """Convert raw power into temperature (K) based on noise injection.
        
        This is a convenience function that converts the raw power measurements
        in the data set to temperatures, based on the change in levels caused by
        switching the noise diode on and off. At the same time it corrects for
        different gains in the X and Y polarisation receiver chains and for
        relative phase shifts between them. It should be called before averaging
        the data, as gain calibration should happen on the finest available
        frequency scale.
        
        Parameters
        ----------
        randomise : {False, True}, optional
            True if noise diode data and smoothing should be randomised
        kwargs : dict, optional
            Extra keyword arguments are passed to underlying :mod:`gaincal` functions

        Returns
        -------
        dataset : :class:`DataSet` object
            Data set with calibrated power data

        """
        # Only operate on raw data
        if self.data_unit != 'raw':
            logger.error("Expected raw power data to convert to temperature, got data with units '" +
                         self.data_unit + "' instead.")
            return self
        return calibrate_gain(self, randomise, **kwargs)
    
    def average(self, channels_per_band='all', time_window=1):
        """Average data in time and/or frequency.

        If *channels_per_band* is not `None`, the frequency channels are grouped
        into bands, and the power data is merged and averaged within each band.
        Each band contains the average power of its constituent channels. If
        *time_window* is larger than 1, the power data is averaged in time in
        non-overlapping windows of this length, and the rest of the time-varying
        data is averaged accordingly. The default behaviour is to average all
        channels into one band, in line with the continuum focus of :mod:`scape`.
        
        Parameters
        ----------
        channels_per_band : List of lists of ints, optional
            List of lists of channel indices (one list per band), indicating
            which channels are averaged together to form each band. If this is
            the string 'all', all channels are averaged together into 1 band.
            If this is None, no averaging is done (each channel becomes a band).
        time_window : int, optional
            Window length in samples, within which to average data in time. If
            this is 1 or None, no averaging is done in time.
        
        Returns
        -------
        dataset : :class:`DataSet` object
            Data set with averaged power data
        
        Notes
        -----
        The average power is simpler to use than the total power in each band,
        as total power is dependent on the bandwidth of each band. This method
        should be called *after* :meth:`convert_power_to_temperature`.
        
        """
        # The string 'all' means average all channels together
        if channels_per_band == 'all':
            channels_per_band = [range(len(self.freqs))]
        # None means no frequency averaging (band == channel)
        if channels_per_band is None:
            channels_per_band = np.expand_dims(range(len(self.freqs)), axis=1).tolist()
        # None means no time averaging too
        if (time_window is None) or (time_window < 1):
            time_window = 1
        # Prune all empty bands
        channels_per_band = [chans for chans in channels_per_band if len(chans) > 0]
        for scan in self.scans:
            # Average power data along frequency axis first
            num_bands = len(channels_per_band)
            band_data = np.zeros((scan.data.shape[0], num_bands, 4), dtype=scan.data.dtype)
            for band_index, band_channels in enumerate(channels_per_band):
                band_data[:, band_index, :] = scan.data[:, band_channels, :].mean(axis=1)
            scan.data = band_data
            # Now average along time axis, if required
            if time_window > 1:
                # Cap the time window length by the scan length, and only keep an integral number of windows
                num_samples = scan.data.shape[0]
                window = min(max(time_window, 1), num_samples)
                new_len = num_samples // window
                cutoff = new_len * window
                scan.data = scan.data[:cutoff, :, :].reshape((new_len, window, num_bands, 4)).mean(axis=1)
                # Also adjust other time-dependent arrays
                scan.timestamps = scan.timestamps[:cutoff].reshape((new_len, window)).mean(axis=1)
                # The record arrays are more involved - form view, average, and reassemble fields
                num_fields = len(scan.pointing.dtype.names)
                view = scan.pointing.view(dtype=np.float32).reshape((num_samples, num_fields))
                view = view[:cutoff, :].reshape((new_len, window, num_fields)).mean(axis=1)
                scan.pointing = view.view(scan.pointing.dtype).squeeze()
                # All flags in a window are OR'ed together to form the 'average'
                num_fields = len(scan.flags.dtype.names)
                view = scan.flags.view(dtype=np.bool).reshape((num_samples, num_fields))
                view = (view[:cutoff, :].reshape((new_len, window, num_fields)).sum(axis=1) > 0)
                scan.flags = view.view(scan.flags.dtype).squeeze()
                if not scan.target_coords is None:
                    scan.target_coords = scan.target_coords[:, :cutoff].reshape((2, new_len, window)).mean(axis=2)
        if time_window > 1:
            self.corrconf.dump_rate /= time_window
        self.corrconf.merge(channels_per_band)
        return self
    
    def fit_beams_and_baselines(self, band=0, **kwargs):
        """Simultaneously fit beams and baselines to all compound scans.
        
        This fits a beam pattern and baseline to the total power data of all the
        scans comprising each compound scan, and stores the resulting fitted function
        in each CompoundScan object. Only one frequency band is used.
        
        Parameters
        ----------
        band : int, optional
            Frequency band in which to fit beam and baseline
        kwargs : dict, optional
            Extra keyword arguments are passed to underlying :mod:`beam_baseline`
            functions
        
        Returns
        -------
        dataset : :class:`DataSet` object
            Data set with fitted beam/baseline functions added
        
        """
        # FWHM Beamwidth for circular dish is 1.03 lambda / D
        expected_width = 1.03 * katpoint.lightspeed / self.freqs[band] / self.antenna.diameter
        for compscan in self.compscans:
            compscan.beam, compscan.baseline = fit_beam_and_baseline(compscan, expected_width, band=band, **kwargs)
        return self
