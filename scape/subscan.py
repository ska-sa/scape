"""Container for the data of a single subscan.

A *subscan* is the *minimal amount of data taking that can be commanded at the
script level,* in accordance with the ALMA Science Data Model. This includes
a single linear sweep across a source, one line in an OTF map, a single
pointing for Tsys measurement, a noise diode on-off measurement, etc. It
contains several *integrations*, or correlator samples, and forms part of an
overall *scan*.

This module provides the :class:`SubScan` class, which encapsulates all data
and actions related to a single subscan across a point source, or a single
subscan at a certain pointing. All actions requiring more than one subscan are
grouped together in :class:`scan.Scan` instead.

Functionality: power conversion,...

"""

import logging

import numpy as np

import coord

logger = logging.getLogger("scape.subscan")

class SubScan(object):
    """Container for the data of a single subscan.

    The main data member of this class is the 3-D :attr:`data` array, which
    stores power (autocorrelation) measurements as a function of time, frequency
    channel/band and polarisation index. The array can take one of two forms:

    - Stokes parameters (I, Q, U, V), which are always real in the case of
      a single dish. I is additionally non-negative, being the total power.

    - Coherencies (XX, YY, 2*Re{XY}, 2*Im{XY}), where XX and YY are real and
      non-negative polarisation powers. For a single dish, the YX cross-coherency
      is the complex conjugate of XY, and therefore does not need to be stored.
      Additionally, since U = 2 * Re{XY} and V = 2 * Im{XY}, conversion to
      Stokes parameters is very simple and efficient.

    The class also stores pointing data (azimuth/elevation/rotator angles),
    timestamps, flags, the spectral configuration and the target object, which
    permits the calculation of any relevant coordinates.

    The number of time samples are indicated by *T* and the number of frequency
    channels/bands are indicated by *F* below. The Attributes section lists
    variables that do not directly correspond to parameters in the constructor.

    Parameters
    ----------
    data : real array, shape (*T*, *F*, 4)
        Stokes/coherency measurements. If the data is in Stokes form, the
        polarisation order on the last dimension is (I, Q, U, V). If the data
        is in coherency form, the order is (XX, YY, 2*Re{XY}, 2*Im{XY}).
    is_stokes : bool
        True if data is in Stokes parameter form, False if in coherency form
    data_unit : {'raw', 'K', 'Jy'}
        Physical unit of power data
    timestamps : real array, shape (*T*,)
        Sequence of timestamps, one per integration (in seconds since epoch)
    pointing : real record array, shape (*T*,)
        Pointing coordinates, with one record per integration (in radians).
        The real-valued fields are 'az', 'el' and optionally 'rot', for
        azimuth, elevation and rotator angle, respectively.
    flags : bool record array, shape (*T*,)
        Flags, with one record per integration. The field names correspond with
        the flag names.
    freqs : real array-like, length *F*
        Sequence of channel/band centre frequencies (in Hz)
    bandwidths : real array, shape (*F*,)
        Array of channel/band bandwidths (in Hz)
    *rfi_channels : list of ints
        RFI-flagged channel indices
    channels_per_band : List of lists of ints
        List of lists of channel indices (one list per band), indicating which
        channels belong to each band
    *dump_rate : float
        Correlator dump rate (in Hz)
    *target : string
        Name of the target of this subscan
    *antenna : string
        Name of antenna that did the subscan
    label : string
        Subscan label, used to distinguish e.g. normal and cal subscans
    path : string
        Filename or HDF5 path from which subscan was loaded

    Attributes
    ----------
    target_coords : real array, shape (*T*, 2)
        Spherical projection coordinates of subscan pointing

    """
    def __init__(self, data, is_stokes, data_unit, timestamps, pointing, flags,
                 freqs, bandwidths, rfi_channels, channels_per_band, dump_rate,
                 target, antenna, label, path):
        self.data = data
        self.is_stokes = is_stokes
        self.data_unit = data_unit
        self.timestamps = timestamps
        self.pointing = pointing
        self.flags = flags
        # Keep as doubles to prevent precision issues
        self.freqs = np.asarray(freqs, dtype='double')
        self.bandwidths = bandwidths
        self.rfi_channels = rfi_channels
        self.channels_per_band = channels_per_band
        self.dump_rate = dump_rate
        # Interpret source and antenna name strings and return relevant objects
        self.target = coord.construct_source(target)
        try:
            self.antenna = coord.antenna_catalogue[antenna]
        except KeyError:
            raise KeyError("Unknown antenna '%s'" % antenna)
        self.label = label
        self.path = path
        self.target_coords = coord.sphere_to_plane(self.target, self.antenna,
                                                   pointing['az'], pointing['el'], timestamps)
    
    def coherency(self, key):
        """Calculate specific coherency from data.

        Parameters
        ----------
        key : {'XX', 'XY', 'YX', 'YY'}
            Coherency to calculate

        Returns
        -------
        coherency : real/complex array, shape (*T*, *F*)
            The array is real for XX and YY, and complex for XY and YX.
        
        Raises
        ------
        KeyError
            If *key* is not one of the allowed coherency names
        
        """
        if key == 'XX':
            if not self.is_stokes:
                return self.data[:, :, 0]
            else:
                return 0.5 * (self.data[:, :, 0] + self.data[:, :, 1])
        elif key == 'YY':
            if not self.is_stokes:
                return self.data[:, :, 1]
            else:
                return 0.5 * (self.data[:, :, 0] - self.data[:, :, 1])
        elif key == 'XY':
            return 0.5 * (self.data[:, :, 2] + 1.0j * self.data[:, :, 3])
        elif key == 'YX':
            return 0.5 * (self.data[:, :, 2] - 1.0j * self.data[:, :, 3])
        else:
            raise KeyError("Coherency key should be one of 'XX', 'XY', 'YX', or 'YY'")

    def stokes(self, key):
        """Calculate specific Stokes parameter from data.
        
        Parameters
        ----------
        key : {'I', 'Q', 'U', 'V'}
            Stokes parameter to calculate
        
        Returns
        -------
        stokes : real array, shape (*T*, *F*)
            Specified Stokes parameter as a function of time and frequency
        
        Raises
        ------
        KeyError
            If *key* is not one of the allowed Stokes parameter names
        
        """
        # If data is already in Stokes form, just return appropriate subarray
        if self.is_stokes:
            stokes_order = ['I', 'Q', 'U', 'V']
            try:
                index = stokes_order.index(key)
            except ValueError:
                raise KeyError("Stokes key should be one of 'I', 'Q', 'U' or 'V'")
            return self.data[:, :, index]
        else:
            if key == 'I':
                return self.data[:, :, 0] + self.data[:, :, 1]
            elif key == 'Q':
                return self.data[:, :, 0] - self.data[:, :, 1]
            elif key == 'U':
                return self.data[:, :, 2]
            elif key == 'V':
                return self.data[:, :, 3]
            else:
                raise KeyError("Stokes key should be one of 'I', 'Q', 'U' or 'V'")

    def convert_to_coherency(self):
        """Convert data to coherency form (idempotent).

        If the data is already in coherency form, do nothing.

        """
        if self.is_stokes:
            data_i, data_q = self.data[:, :, 0].copy(), self.data[:, :, 1].copy()
            self.data[:, :, 0] = 0.5 * (data_i + data_q)
            self.data[:, :, 1] = 0.5 * (data_i - data_q)
            self.is_stokes = False
        return self

    def convert_to_stokes(self):
        """Convert data to Stokes parameter form (idempotent).

        If the data is already in Stokes form, do nothing.

        """
        if not self.is_stokes:
            data_xx, data_yy = self.data[:, :, 0].copy(), self.data[:, :, 1].copy()
            self.data[:, :, 0] = data_xx + data_yy
            self.data[:, :, 1] = data_xx - data_yy
            self.is_stokes = True
        return self
    
    def select(self, timekeep=None, freqkeep=None, copy=False):
        """Select a subset of time and frequency indices in data matrix.
        
        This creates a new :class:`SubScan` object that contains a subset of the
        rows and columns of the data matrix. This allows time samples and/or
        frequency channels/bands to be discarded. If *copy* is False, the data
        is selected via a masked array or view, and the returned object is a
        view on the original data. If *copy* is True, the data matrix and all
        associated coordinate vectors are reduced to a smaller size and copied.
        
        Parameters
        ----------
        timekeep : sequence of bools or ints, optional
            Sequence of indicators of which time samples to keep (either integer
            indices or booleans that are True for the values to be kept). The
            default is None, which keeps everything.
        freqkeep : sequence of bools or ints, optional
            Sequence of indicators of which frequency channels/bands to keep
            (either integer indices or booleans that are True for the values to
            be kept). The default is None, which keeps everything.
        copy : {False, True}, optional
            True if the new subscan is a copy, False if it is a view
        
        Returns
        -------
        sub : :class:`SubScan` object
            Subscan with reduced data matrix (either masked array or smaller copy)
        
        """
        # Use advanced indexing to create a smaller copy of the data matrix
        if copy:
            # If data matrix is kept intact, make a straight copy - probably faster
            if (timekeep is None) and (freqkeep is None):
                selected_data = self.data.copy()
                timekeep = np.arange(len(self.timestamps))
                freqkeep = np.arange(len(self.freqs))
            else:
                # Convert boolean selection vectors (and None) to indices
                if timekeep is None:
                    timekeep = np.arange(len(self.timestamps))
                elif np.asarray(timekeep).dtype == 'bool':
                    timekeep = np.asarray(timekeep).nonzero()[0]
                if freqkeep is None:
                    freqkeep = np.arange(len(self.freqs))
                elif np.asarray(freqkeep).dtype == 'bool':
                    freqkeep = np.asarray(freqkeep).nonzero()[0]
                selected_data = self.data[np.atleast_2d(timekeep).transpose(), np.atleast_2d(freqkeep), :]
            return SubScan(selected_data, self.is_stokes, self.data_unit,
                           self.timestamps[timekeep], self.pointing[timekeep], self.flags[timekeep],
                           self.freqs[freqkeep], self.bandwidths[freqkeep], self.rfi_channels,
                           [self.channels_per_band[n] for n in xrange(len(self.channels_per_band)) if n in freqkeep],
                           self.dump_rate, self.target.name, self.antenna.name, self.label, self.path)
        # Create a shallow view of data matrix via a masked array or view
        else:
            # If data matrix is kept intact, rather just return a view instead of masked array
            if (timekeep is None) and (freqkeep is None):
                selected_data = self.data
            else:
                # Normalise the selection vectors to select elements via bools instead of indices
                if timekeep is None:
                    timekeep1d = np.tile(True, len(self.timestamps))
                else:
                    timekeep1d = np.tile(False, len(self.timestamps))
                    timekeep1d[timekeep] = True
                if freqkeep is None:
                    freqkeep1d = np.tile(True, len(self.freqs))
                else:
                    freqkeep1d = np.tile(False, len(self.freqs))
                    freqkeep1d[freqkeep] = True
                # Create 3-D mask matrix of same shape as data, with rows and columns masked
                timekeep3d = np.atleast_3d(timekeep1d).transpose((1, 0, 2))
                freqkeep3d = np.atleast_3d(freqkeep1d).transpose((0, 1, 2))
                polkeep3d = np.atleast_3d([True, True, True, True]).transpose((0, 2, 1))
                keep3d = np.kron(timekeep3d, np.kron(freqkeep3d, polkeep3d))
                selected_data = np.ma.array(self.data, mask=~keep3d)
            return SubScan(selected_data, self.is_stokes, self.data_unit,
                           self.timestamps, self.pointing, self.flags,
                           self.freqs, self.bandwidths, self.rfi_channels, self.channels_per_band,
                           self.dump_rate, self.target.name, self.antenna.name, self.label, self.path)
    
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
