"""Container for the data of a single scan.

A *scan* is the *minimal amount of data taking that can be commanded at the
script level,* which corresponds to the *subscan* of the ALMA Science Data Model.
This includes a single linear sweep across a source, one line in an OTF map, a
single pointing for Tsys measurement, a noise diode on-off measurement, etc. It
contains several *integrations*, or correlator samples, and forms part of an
overall *compound scan*.

This module provides the :class:`Scan` class, which encapsulates all data
and actions related to a single scan across a point source, or a single
scan at a certain pointing. All actions requiring more than one scan are
grouped together in :class:`compoundscan.CompoundScan` instead.

Functionality: power conversion,...

"""

import numpy as np
import time

from .stats import minimise_angle_wrap
from katpoint import rad2deg

#--------------------------------------------------------------------------------------------------
#--- CLASS :  Scan
#--------------------------------------------------------------------------------------------------

class Scan(object):
    """Container for the data of a single scan.

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
    timestamps and flags, which all vary as a function of time. The number of
    time samples are indicated by *T* and the number of frequency channels/bands
    are indicated by *F* below.

    Parameters
    ----------
    data : float32 array, shape (*T*, *F*, 4)
        Stokes/coherency measurements. If the data is in Stokes form, the
        polarisation order on the last dimension is (I, Q, U, V). If the data
        is in coherency form, the order is (XX, YY, 2*Re{XY}, 2*Im{XY}).
    is_stokes : bool
        True if data is in Stokes parameter form, False if in coherency form
    timestamps : float64 array, shape (*T*,)
        Sequence of timestamps, one per integration (in UTC seconds since epoch).
        These timestamps should be in the *middle* of each integration.
    pointing : float32 record array, shape (*T*,)
        Pointing coordinates, with one record per integration (in radians).
        The real-valued fields are 'az', 'el' and optionally 'rot', for
        azimuth, elevation and rotator angle, respectively. The pointing should
        be valid for the *middle* of each integration.
    flags : bool record array, shape (*T*,)
        Flags, with one record per integration. The field names correspond to
        the flag names.
    enviro_ambient : record array, shape (*Ta*,) or None
        Slowly-varying environmental measurements, containing *Ta* records.
        The first field ('timestamp') is a timestamp in UTC seconds since epoch,
        and the rest of the field names correspond to ambient variables.
    enviro_wind : record array, shape (*Tw*,) or None
        Wind speed and direction measurements, containing *Tw* records.
        The first field ('timestamp') is a timestamp in UTC seconds since epoch,
        and the rest of the field names correspond to wind variables.
    label : string
        Scan label, used to distinguish e.g. normal and cal scans
    path : string
        Filename or HDF5 path from which scan was loaded
    target_coords : real array, shape (2, *T*), optional
        Coordinates on projected plane, with target as reference, in radians
    baseline : :class:`fitting.Polynomial1DFit` object, optional
        Object that describes fitted baseline

    """
    def __init__(self, data, is_stokes, timestamps, pointing, flags, enviro_ambient,
                 enviro_wind, label, path, target_coords=None, baseline=None):
        self.data = data
        self.is_stokes = is_stokes
        self.timestamps = timestamps
        self.pointing = pointing
        self.flags = flags
        self.enviro_ambient = enviro_ambient
        self.enviro_wind = enviro_wind
        self.label = label
        self.path = path
        self.target_coords = target_coords
        self.baseline = baseline

    def __eq__(self, other):
        """Equality comparison operator."""
        # Make sure that both data sets have the same polarisation format first
        if self.is_stokes != other.is_stokes:
            if other.is_stokes:
                self.convert_to_stokes()
                data_same = np.all(self.data == other.data)
                self.convert_to_coherency()
                if not data_same:
                    return False
            else:
                self.convert_to_coherency()
                data_same = np.all(self.data == other.data)
                self.convert_to_stokes()
                if not data_same:
                    return False
        else:
            if not np.all(self.data == other.data):
                return False
        # Because of conversion to degrees and back during saving and loading, the last (8th)
        # significant digit of the float32 pointing values may change - do approximate comparison.
        # Since pointing is used to calculate target coords, this is also only approximately equal.
        # Timestamps and pointing are also converted to and from the start and middle of each sample,
        # which causes extra approximateness... (pointing now only accurate to arcminutes, but time
        # should be OK up to microseconds)
        return np.all(self.flags == other.flags) and (self.label == other.label) and \
               np.all(self.enviro_ambient == other.enviro_ambient) and \
               np.all(self.enviro_wind == other.enviro_wind) and \
               np.allclose(self.timestamps, other.timestamps, atol=1e-6) and \
               np.allclose(self.pointing.view(np.float32), other.pointing.view(np.float32), 1e-4) and \
               np.allclose(self.target_coords, other.target_coords, atol=1e-6)

    def __ne__(self, other):
        """Inequality comparison operator."""
        return not self.__eq__(other)

    def __str__(self):
        """Verbose human-friendly string representation of scan object."""
        mean_az = rad2deg(minimise_angle_wrap(self.pointing['az']).mean())
        mean_el = rad2deg(minimise_angle_wrap(self.pointing['el']).mean())
        return "'%s', data=%s, start='%s', az=%.1f, el=%.1f, path='%s'" % \
               (self.label, self.data.shape, time.strftime('%Y/%m/%d %H:%M:%S', time.gmtime(self.timestamps[0])),
                mean_az, mean_el, self.path)

    def __repr__(self):
        """Short human-friendly string representation of scan object."""
        return "<scape.Scan '%s' data=%s at 0x%x>" % (self.label, self.data.shape, id(self))

    def calc_target_coords(self, target, antenna=None):
        """Calculate target coordinates, based on target and antenna objects.

        Parameters
        ----------
        target : :class:`katpoint.Target` object
            Target object which is scanned across, obtained from CompoundScan
        antenna : :class:`katpoint.Antenna` object, optional
            Antenna object for antenna that does scanning, obtained from DataSet

        Returns
        -------
        target_coords : real array, shape (2, *T*)
            Coordinates on projected plane, with target as reference, in radians

        """
        # Fix over-the-top elevations (projections can only handle elevations in range +- 90 degrees)
        over_the_top = (self.pointing['el'] > np.pi / 2.0) & (self.pointing['el'] < np.pi)
        self.pointing['az'][over_the_top] += np.pi
        self.pointing['el'][over_the_top] = np.pi - self.pointing['el'][over_the_top]
        target_x, target_y = target.sphere_to_plane(self.pointing['az'], self.pointing['el'],
                                                    self.timestamps, antenna)
        self.target_coords = np.vstack((target_x, target_y))
        return self.target_coords

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

    def pol(self, key):
        """Convenience function to return desired coherency or Stokes parameter.

        Parameters
        ----------
        key : {'I', 'Q', 'U', 'V', 'XX', 'XY', 'YX', 'YY'}

        Returns
        -------
        pol : real/complex array, shape (*T*, *F*)
            Coherency / Stokes parameter as a function of time and frequency.
            It is complex for XY and YX and real for the rest.

        Raises
        ------
        KeyError
            If *key* is not one of the allowed coherency / Stokes parameter names

        """
        if key in ('I', 'Q', 'U', 'V'):
            return self.stokes(key)
        elif key in ('XX', 'XY', 'YX', 'YY'):
            return self.coherency(key)
        else:
            raise KeyError("Polarisation key should be one of 'I', 'Q', 'U', 'V', 'XX', 'XY', 'YX' or 'YY'")

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

    def swap_x_and_y(self):
        """Swap the X and Y polarisations around."""
        if self.is_stokes:
            # The sign of Q changes, as XX - YY => YY - XX
            self.data[:, :, 1] *= -1.0
        else:
            # Swap XX and YY
            temp = self.data[:, :, 0].copy()
            self.data[:, :, 0] = self.data[:, :, 1]
            self.data[:, :, 1] = temp
        # The sign of V changes, as Im(XY) => Im(YX), and YX is conjugate of XY
        self.data[:, :, 3] *= -1.0
        return self

    def select(self, timekeep=None, freqkeep=None, copy=False):
        """Select a subset of time and frequency indices in data matrix.

        This creates a new :class:`Scan` object that contains a subset of the
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
            True if the new scan is a copy, False if it is a view

        Returns
        -------
        scan : :class:`Scan` object
            Scan with reduced data matrix (either masked array or smaller copy)

        """
        # Use advanced indexing to create a smaller copy of the data matrix
        if copy:
            # If data matrix is kept intact, make a straight copy - probably faster
            if (timekeep is None) and (freqkeep is None):
                selected_data = self.data.copy()
                timekeep = np.arange(self.data.shape[0])
                freqkeep = np.arange(self.data.shape[1])
            else:
                # Convert boolean selection vectors (and None) to indices
                if timekeep is None:
                    timekeep = np.arange(self.data.shape[0])
                elif np.asarray(timekeep).dtype == 'bool':
                    timekeep = np.asarray(timekeep).nonzero()[0]
                if freqkeep is None:
                    freqkeep = np.arange(self.data.shape[1])
                elif np.asarray(freqkeep).dtype == 'bool':
                    freqkeep = np.asarray(freqkeep).nonzero()[0]
                selected_data = self.data[np.atleast_2d(timekeep).transpose(), np.atleast_2d(freqkeep), :]
            target_coords = self.target_coords
            if not target_coords is None:
                target_coords = target_coords[:, timekeep]
            return Scan(selected_data, self.is_stokes, self.timestamps[timekeep], self.pointing[timekeep],
                        self.flags[timekeep], self.enviro_ambient, self.enviro_wind,
                        self.label, self.path, target_coords, self.baseline)
        # Create a shallow view of data matrix via a masked array or view
        else:
            # If data matrix is kept intact, rather just return a view instead of masked array
            if (timekeep is None) and (freqkeep is None):
                selected_data = self.data
            else:
                # Normalise the selection vectors to select elements via bools instead of indices
                if timekeep is None:
                    timekeep1d = np.tile(True, self.data.shape[0])
                else:
                    timekeep1d = np.tile(False, self.data.shape[0])
                    timekeep1d[timekeep] = True
                if freqkeep is None:
                    freqkeep1d = np.tile(True, self.data.shape[1])
                else:
                    freqkeep1d = np.tile(False, self.data.shape[1])
                    freqkeep1d[freqkeep] = True
                # Create 3-D mask matrix of same shape as data, with rows and columns masked
                timekeep3d = np.atleast_3d(timekeep1d).transpose((1, 0, 2))
                freqkeep3d = np.atleast_3d(freqkeep1d).transpose((0, 1, 2))
                polkeep3d = np.atleast_3d([True, True, True, True]).transpose((0, 2, 1))
                keep3d = np.kron(timekeep3d, np.kron(freqkeep3d, polkeep3d))
                selected_data = np.ma.array(self.data, mask=~keep3d)
            return Scan(selected_data, self.is_stokes, self.timestamps, self.pointing, self.flags,
                        self.enviro_ambient, self.enviro_wind, self.label, self.path,
                        self.target_coords, self.baseline)
