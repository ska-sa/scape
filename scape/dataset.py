"""Container for the data of a single-dish experiment."""

import os.path

import numpy as np

import scan
import coord

# Try to import all available formats
try:
    import xdmfits
    xdmfits_found = True
except ImportError:
    xdmfits_found = False
try:
    import hdf5
    hdf5_found = True
except ImportError:
    hdf5_found = False

class DataSet(object):
    """Container for the data of a single-dish experiment.
    
    This is the top-level container for the data of a single-dish experiment.
    Given a data filename, the initialiser determines the appropriate file format
    to use, based on the file extension. If the filename is blank, the
    :class:`DataSet` can also be directly initialised from its constituent
    parts, which is useful for simulations and creating the data sets from
    scratch. The :class:`DataSet` object contains a list of scans, as well as
    the spectral configuration, antenna details and noise diode characteristics.
    
    Parameters
    ----------
    filename : string
        Name of data set file, or blank string if the other parameters are given
    scanlist : list of :class:`scan.Scan` objects, optional
        List of scans
    data_unit : {'raw', 'K', 'Jy'}, optional
        Physical unit of power data
    spectral : :class:`scan.SpectralConfig` object, optional
        Spectral configuration object
    antenna : string, optional
        Name of antenna that produced the data set
    nd_data : :class:`gaincal.NoiseDiodeBase` object, optional
        Noise diode model
    
    Raises
    ------
    ImportError
        If file extension is known, but appropriate module would not import
    ValueError
        If file extension is unknown
    
    """
    def __init__(self, filename, scanlist=None, data_unit=None, spectral=None, antenna=None, nd_data=None):
        if filename:
            ext = os.path.splitext(filename)[1]
            if ext == '.fits':
                if not xdmfits_found:
                    raise ImportError('XDM FITS support could not be loaded - please check xdmfits module')
                scanlist, data_unit, spectral, antenna, nd_data = xdmfits.load_dataset(filename)
            elif (ext == '.h5') or (ext == '.hdf5'):
                if not hdf5_found:
                    raise ImportError('HDF5 support could not be loaded - please check hdf5 module')
                scanlist, data_unit, spectral, antenna, nd_data = hdf5.load_dataset(filename)
            else:
                raise ValueError("File extension '%s' not understood" % ext)
        self.scans = scanlist
        self.data_unit = data_unit
        self.spectral = spectral
        try:
            self.antenna = coord.antenna_catalogue[antenna]
        except KeyError:
            raise KeyError("Unknown antenna '%s'" % antenna)
        self.noise_diode_data = nd_data
    
    # Provide properties to access the attributes of the spectral configuration directly
    # This uses the same trick as in stats.MuSigmaArray to create the properties, which
    # leads to less class namespace clutter, but more pylint uneasiness (shame).
    def freqs():
        """Class method which creates freqs property."""
        doc = 'Frequency of each channel/band, in Hz.'
        def fget(self):
            return self.spectral.freqs
        def fset(self, value):
            self.spectral.freqs = value
        return locals()
    freqs = property(**freqs())
    
    def bandwidths():
        """Class method which creates bandwidths property."""
        doc = 'Bandwidth of each channel/band, in Hz.'
        def fget(self):
            return self.spectral.bandwidths
        def fset(self, value):
            self.spectral.bandwidths = value
        return locals()
    bandwidths = property(**bandwidths())
    
    def rfi_channels():
        """Class method which creates rfi_channels property."""
        doc = 'List of RFI-corrupted channels.'
        def fget(self):
            return self.spectral.rfi_channels
        def fset(self, value):
            self.spectral.rfi_channels = value
        return locals()
    rfi_channels = property(**rfi_channels())
    
    def channels_per_band():
        """Class method which creates channels_per_band property."""
        doc = 'List of channel index lists, one per frequency band.'
        def fget(self):
            return self.spectral.channels_per_band
        def fset(self, value):
            self.spectral.channels_per_band = value
        return locals()
    channels_per_band = property(**channels_per_band())
    
    def dump_rate():
        """Class method which creates dump_rate property."""
        doc = 'Correlator dump rate, in Hz.'
        def fget(self):
            return self.spectral.dump_rate
        def fset(self, value):
            self.spectral.dump_rate = value
        return locals()
    dump_rate = property(**dump_rate())
    
    def subscans(self):
        """List of all subscans in data set."""
        subscanlist = []
        for s in self.scans:
            subscanlist.extend(s.subscans)
        return subscanlist
    
    def convert_to_coherency(self):
        """Convert power data to coherency format."""
        for ss in self.subscans():
            ss.convert_to_coherency()
    
    def convert_to_stokes(self):
        """Convert power data to Stokes parameter format."""
        for ss in self.subscans():
            ss.convert_to_stokes()
    
    def select(self, labelkeep=None, flagkeep=None, freqkeep=None, copy=False):
        """Select subset of data set, based on subscan label, flags and frequency.
        
        This returns a data set with a possibly reduced number of time samples,
        frequency channels/bands and subscans, based on the selection criteria.
        Since each subscan potentially has a different number of time samples,
        it is less useful to filter directly on sample index. Instead, the
        flags are used to select a subset of time samples in each subscan. The
        list of flags are ANDed together to determine which parts are kept. It
        is also possible to invert flags by prepending a ~ (tilde) character.
        
        Based on the value of *copy*, the new data set contains either a view of
        the original data or a copy. All criteria are optional, and with no
        parameters the returned data set is unchanged. This can be used to make
        a copy of the data set.
        
        Parameters
        ----------
        labelkeep : list of strings, optional
            All subscans with labels in this list will be kept. The default is
            None, which means all labels are kept.
        flagkeep : list of strings, optional
            List of flags used to select time ranges in each subscan. The time
            samples for which all the flags in the list are true are kept.
            Individual flags can be negated by prepending a ~ (tilde) character.
            The default is None, which means all time samples are kept.
        freqkeep : sequence of bools or ints, optional
            Sequence of indicators of which frequency channels/bands to keep
            (either integer indices or booleans that are True for the values to
            be kept). The default is None, which keeps all channels/bands.
        copy : {False, True}, optional
            True if the new subscan is a copy, False if it is a view
        
        Returns
        -------
        dataset : :class:`DataSet` object
            Data set with selection of subscans with possibly smaller data arrays.
        
        Raises
        ------
        KeyError
            If flag in *flagkeep* is unknown
        
        """
        scanlist = []
        for s in self.scans:
            subscanlist = []
            for ss in s.subscans:
                # Convert flag selection to time sample selection
                if flagkeep is None:
                    timekeep = None
                else:
                    # By default keep all time samples
                    timekeep = np.tile(True, len(ss.timestamps))
                    for flag in flagkeep:
                        invert = False
                        # Flags prepended with ~ get inverted
                        if flag[0] == '~':
                            invert = True
                            flag = flag[1:]
                        # Ignore unknown flags
                        try:
                            flag_data = ss.flags[flag]
                        except KeyError:
                            raise KeyError("Unknown flag '%s'" % flag)
                        if invert:
                            timekeep &= ~flag_data
                        else:
                            timekeep &= flag_data
                if (labelkeep is None) or (ss.label in labelkeep):
                    subscanlist.append(ss.select(timekeep, freqkeep, copy))
            if subscanlist:
                scanlist.append(scan.Scan(subscanlist, s.target.name))
        return DataSet(None, scanlist, self.data_unit, self.spectral.select(freqkeep),
                       self.antenna.name, self.noise_diode_data)
    
    def convert_power_to_temp(self, func):
        """Convert raw power into temperature (K) using conversion function.

        The main parameter is a callable object with the signature
        ``factor = func(time)``, which provides an interpolated conversion
        factor function. The conversion factor returned by func should be
        an array of shape (*T*, *F*, 4), where *T* is the number of timestamps
        and *F* is the number of channels. This will be multiplied with the
        power data in coherency form to obtain temperatures. This should be
        called *before* merge_channels_into_bands and
        fit_and_subtract_baseline, as gain calibration should happen on the
        finest available frequency scale.

        Parameters
        ----------
        func : function, signature ``factor = func(time)``
            The power-to-temperature conversion factor as a function of time

        """
        if func is None:
            return self
        # Only operate on raw data
        if self.data_unit != 'raw':
            logger.warning("Expected raw power data to convert to temperature, got data with units '" +
                           self.data_unit + "' instead.")
            return self
        originally_stokes = self.is_stokes
        # Convert coherency power to temperature, and restore Stokes/coherency status
        self.convert_to_coherency()
        self.data *= func(self.timestamps)
        if originally_stokes:
            self.convert_to_stokes()
        self.data_unit = 'K'
        return self

    def merge_channels_into_bands(self):
        """Merge frequency channels into bands.

        The frequency channels are grouped into bands, and the power data is
        merged and averaged within each band. Each band contains the average
        power of its constituent channels. The average power is simpler to use
        than the total power in each band, as total power is dependent on the
        bandwidth of each band. The channels_per_band mapping contains a list
        of lists of channel indices, indicating which channels belong to each
        band. This method should be called *after* convert_power_to_temp and
        *before* fit_and_subtract_baseline.

        """
        # Merge and average power data into new array (keep same type as original data, which may be complex)
        band_data = np.zeros((self.data.shape[0], len(self.channels_per_band), 4),
                             dtype=self.data.dtype)
        for band_index, band_channels in enumerate(self.channels_per_band):
            band_data[:, band_index, :] = self.data[:, band_channels, :].mean(axis=1)
        self.data = band_data
        # Each band centre frequency is the mean of the corresponding channel centre frequencies
        self.freqs = np.array([self.freqs[chans].mean() for chans in self.channels_per_band], dtype='double')
        # Each band bandwidth is the sum of the corresponding channel bandwidths
        self.bandwidths = np.array([self.bandwidths[chans].sum() for chans in self.channels_per_band],
                                   dtype='double')
        return self
