"""Container for the data of a single-dish experiment."""

import os.path
import logging

import numpy as np

from .scan import Scan
from .coord import antenna_catalogue
from .gaincal import calibrate_gain

# Try to import all available formats
try:
    from .xdmfits import load_dataset as xdmfits_dataset
    xdmfits_found = True
except ImportError:
    xdmfits_found = False
try:
    from .hdf5 import load_dataset as hdf5_dataset
    hdf5_found = True
except ImportError:
    hdf5_found = False

logger = logging.getLogger("scape.dataset")

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
    kwargs : dict, optional
        Extra keyword arguments are passed to selected :func:`load_dataset` function
    
    Attributes
    ----------
    freqs : real array, shape (*F*,)
        Centre frequency of each channel/band, in Hz (same as in *spectral*)
    bandwidths : real array-like, shape (*F*,)
        Bandwidth of each channel/band, in Hz (same as in *spectral*)
    rfi_channels : list of ints
        RFI-flagged channel indices (same as in *spectral*)
    channels_per_band : List of lists of ints
        List of lists of channel indices (one list per band), indicating which
        channels belong to each band (same as in *spectral*)
    dump_rate : float
        Correlator dump rate, in Hz (same as in *spectral*)
    subscans : list of :class:`subscan.SubScan` objects
        Flattened list of all subscans in data set
    
    Raises
    ------
    ImportError
        If file extension is known, but appropriate module would not import
    ValueError
        If file extension is unknown
    KeyError
        If the antenna name is unknown
    
    """
    def __init__(self, filename, scanlist=None, data_unit=None, spectral=None,
                 antenna=None, nd_data=None, **kwargs):
        if filename:
            ext = os.path.splitext(filename)[1]
            if ext == '.fits':
                if not xdmfits_found:
                    raise ImportError('XDM FITS support could not be loaded - please check xdmfits module')
                scanlist, data_unit, spectral, antenna, nd_data = xdmfits_dataset(filename, **kwargs)
            elif (ext == '.h5') or (ext == '.hdf5'):
                if not hdf5_found:
                    raise ImportError('HDF5 support could not be loaded - please check hdf5 module')
                scanlist, data_unit, spectral, antenna, nd_data = hdf5_dataset(filename, **kwargs)
            else:
                raise ValueError("File extension '%s' not understood" % ext)
        self.scans = scanlist
        self.data_unit = data_unit
        self.spectral = spectral
        try:
            self.antenna = antenna_catalogue[antenna]
        except KeyError:
            raise KeyError("Unknown antenna '%s'" % antenna)
        self.noise_diode_data = nd_data
        # Create subscan list
        self.subscans = []
        for s in self.scans:
            self.subscans.extend(s.subscans)
    
    # Provide properties to access the attributes of the spectral configuration directly
    # This uses the same trick as in stats.MuSigmaArray to create the properties, which
    # leads to less class namespace clutter, but more pylint uneasiness (shame).
    # pylint: disable-msg=E0211,E0202,W0612,W0142
    def freqs():
        """Class method which creates freqs property."""
        doc = 'Centre frequency of each channel/band, in Hz.'
        def fget(self):
            return self.spectral.freqs
        def fset(self, value):
            self.spectral.freqs = value
        return locals()
    freqs = property(**freqs())
    
    # pylint: disable-msg=E0211,E0202,W0612,W0142
    def bandwidths():
        """Class method which creates bandwidths property."""
        doc = 'Bandwidth of each channel/band, in Hz.'
        def fget(self):
            return self.spectral.bandwidths
        def fset(self, value):
            self.spectral.bandwidths = value
        return locals()
    bandwidths = property(**bandwidths())
    
    # pylint: disable-msg=E0211,E0202,W0612,W0142
    def rfi_channels():
        """Class method which creates rfi_channels property."""
        doc = 'List of RFI-flagged channels.'
        def fget(self):
            return self.spectral.rfi_channels
        def fset(self, value):
            self.spectral.rfi_channels = value
        return locals()
    rfi_channels = property(**rfi_channels())
    
    # pylint: disable-msg=E0211,E0202,W0612,W0142
    def channels_per_band():
        """Class method which creates channels_per_band property."""
        doc = 'List of channel index lists, one per frequency band.'
        def fget(self):
            return self.spectral.channels_per_band
        def fset(self, value):
            self.spectral.channels_per_band = value
        return locals()
    channels_per_band = property(**channels_per_band())
    
    # pylint: disable-msg=E0211,E0202,W0612,W0142
    def dump_rate():
        """Class method which creates dump_rate property."""
        doc = 'Correlator dump rate, in Hz.'
        def fget(self):
            return self.spectral.dump_rate
        def fset(self, value):
            self.spectral.dump_rate = value
        return locals()
    dump_rate = property(**dump_rate())
    
    def convert_to_coherency(self):
        """Convert power data to coherency format."""
        for ss in self.subscans:
            ss.convert_to_coherency()
    
    def convert_to_stokes(self):
        """Convert power data to Stokes parameter format."""
        for ss in self.subscans:
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
                scanlist.append(Scan(subscanlist, s.target.name))
        return DataSet(None, scanlist, self.data_unit, self.spectral.select(freqkeep),
                       self.antenna.name, self.noise_diode_data)
    
    def remove_rfi_channels(self):
        """Remove RFI-flagged channels from data set, returning a copy."""
        non_rfi = list(set(range(len(self.freqs))) - set(self.rfi_channels))
        d = self.select(freqkeep=non_rfi, copy=True)
        DataSet.__init__(self, None, d.scans, d.data_unit, d.spectral, d.antenna.name, d.noise_diode_data)
        return self
    
    def convert_power_to_temperature(self, randomise=False, **kwargs):
        """Convert raw power into temperature (K) based on noise injection.
        
        This converts the raw power measurements in the data set to temperatures,
        based on the change in levels caused by switching the noise diode on and
        off. At the same time it corrects for different gains in the X and Y
        polarisation receiver chains and for relative phase shifts between them.
        It should be called *before* :meth:`merge_channels_into_bands`, as gain
        calibration should happen on the finest available frequency scale.
        
        Parameters
        ----------
        randomise : {False, True}, optional
            True if noise diode data and smoothing should be randomised
        kwargs : dict, optional
            Extra keyword arguments are passed to underlying :mod:`gaincal` functions

        """
        # Only operate on raw data
        if self.data_unit != 'raw':
            logger.warning("Expected raw power data to convert to temperature, got data with units '" +
                           self.data_unit + "' instead.")
            return self
        return calibrate_gain(self, randomise, **kwargs)
    
    def merge_channels_into_bands(self):
        """Merge frequency channels into bands.

        The frequency channels are grouped into bands, and the power data is
        merged and averaged within each band. Each band contains the average
        power of its constituent channels. The average power is simpler to use
        than the total power in each band, as total power is dependent on the
        bandwidth of each band. This method should be called *after*
        :meth:`convert_power_to_temperature`.
        
        """
        # Prune all empty bands
        self.channels_per_band = [chans for chans in self.channels_per_band if len(chans) > 0]
        for ss in self.subscans:
            # Merge and average power data into new array
            band_data = np.zeros((ss.data.shape[0], len(self.channels_per_band), 4), dtype=ss.data.dtype)
            for band_index, band_channels in enumerate(self.channels_per_band):
                band_data[:, band_index, :] = ss.data[:, band_channels, :].mean(axis=1)
            ss.data = band_data
        self.spectral.merge()
        return self
