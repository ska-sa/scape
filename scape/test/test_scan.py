"""Unit tests for the scan module."""
# pylint: disable-msg=C0103

import time
import unittest

import numpy as np

import scape
import katpoint

# Mueller matrix that performs coherency -> Stokes transform
stokes_from_coh = np.array([[1,   0,  0,  1],
                            [1,   0,  0, -1],
                            [0,   1,  1,  0],
                            [0, -1j, 1j,  0]])
# Mueller matrix that performs Stokes -> coherency transform (inverse of above)
coh_from_stokes = 0.5 * np.array([[1,  1, 0,   0],
                                  [0,  0, 1,  1j],
                                  [0,  0, 1, -1j],
                                  [1, -1, 0,   0]])

def transform_pol(scan, *args):
    """Return data transformed by series of Jones / generalised Mueller matrices.

    This applies a series of transformation matrices to the correlation data
    of the scan and returns the output array. The scan data is not changed
    by this operation. The matrices are applied from right to left, as they
    would be written in mathematical notation. Each matrix is in one of the
    following forms:

    - Single Jones matrix with shape (2, 2). This is converted to the
      equivalent 4x4 generalised Mueller matrix using the Kronecker product.
      This matrix is independent of time and frequency.

    - Single generalised Mueller matrix with shape (4, 4). It is called
      *generalised* because a proper Mueller matrix specifically operates on
      the Stokes parameters, while this matrix may also operate on the
      coherency vector. The matrix is also independent of time and frequency.

    - Series of generalised Mueller matrices, with shape (*TT*, *FF*, 4, 4).
      If the matrices depend on time, the first dimension should contain
      *T* elements (same as that of the data array). If the matrices depend
      on frequency, the second dimension should contain *F* elements (same
      as that of the data array). Independent dimensions should contain one
      element.

    The matrices are automatically extended to the full time and frequency
    range via broadcasting. After transformation, the output array has the
    standard coherency ordering on its polarisation axis, which is (*I*, *Q*,
    *U*, *V*) for Stokes parameters, (*XX*, *XY*, *YX*, *YY*) for sky
    coherencies and (*VV*, *VH*, *HV*, *HH*) for mount coherencies (since
    V correspond to X and H to Y for a parallactic angle of zero). If no
    transformation matrices are supplied, the default mount coherencies are
    still returned in this reordered format. The output array is complex for
    both interferometric and single-dish data.

    Parameters
    ----------
    scan : :class:`scape.Scan` object
        Scan that provides data to be transformed
    args : list of arrays, optional
        List of transforms to apply to data (applied from right to left)

    Returns
    -------
    data : complex128 array, shape (*T*, *F*, 4)
        Transformed correlation data array, in standard coherency order

    """
    # Retrieve data from scan module
    scape_pol_sd, scape_pol_if, mount_coh = scape.scan.scape_pol_sd, scape.scan.scape_pol_if, scape.scan.mount_coh
    # Create complex data array in standard coherency order (where V = X and H = Y)
    if scan.has_autocorr:
        data_HV = scan.data[:, :, scape_pol_sd.index('ReHV')] + 1j * scan.data[:, :, scape_pol_sd.index('ImHV')]
        data = np.dstack((scan.data[:, :, scape_pol_sd.index('VV')], data_HV.conj(), data_HV,
                          scan.data[:, :, scape_pol_sd.index('HH')]))
    else:
        data = scan.data[:, :, [scape_pol_if.index(p) for p in mount_coh]]
    # Iterate through matrix operators, from right to left
    for matrix in reversed(args):
        # Turn 4-element polarisation vector (on last dimension of data) effectively into row vector.
        # This allows pointwise multiplication with 4-dimensional matrix, which allows a different
        # transformation matrix per time and frequency bin to be applied in a single step.
        data = np.expand_dims(data, 2)
        matrix = np.array(matrix, dtype=np.complex128)
        if matrix.shape == (2, 2):
            matrix = np.kron(matrix, matrix.conj())
        if matrix.shape == (4, 4):
            matrix = np.expand_dims(np.expand_dims(matrix, 0), 0)
        assert matrix.ndim == 4, 'Transformation matrix should have a shape of (2, 2), (4, 4) or (T, F, 4, 4)'
        # Do a manual matrix multiplication on the last dimension (actually many 4x4 transforms in parallel)
        data = (matrix * data).sum(axis=-1)
    return data

def generic_pol(scan, key):
    """Extract a specific polarisation term from correlation data, the slow way.

    This duplicates the functionality of :meth:`scape.Scan.pol`, but uses generic
    Jones / Mueller transformations which turned out to be too slow. It is still
    useful as a double-check of the calculations.

    Parameters
    ----------
    scan : :class:`scape.Scan` object
        Scan to extract data from
    key : {'VV', 'VH', 'HV', 'HH', 'XX', 'XY', 'YX', 'YY', 'I', 'Q', 'U', 'V'}
        Polarisation term to extract

    Returns
    -------
    pol_data : float64 or complex128 array, shape (*T*, *F*)
        Polarisation term as a function of time and frequency

    Raises
    ------
    KeyError
        If *key* is not one of the allowed polarisation terms

    """
    # Retrieve data from scan module
    scape_pol_sd, scape_pol_if = scape.scan.scape_pol_sd, scape.scan.scape_pol_if
    sky_coh, stokes = scape.scan.sky_coh, scape.scan.stokes

    # Mount coherencies are the easiest to extract - simply pick correct subarray (mostly)
    if key in scape_pol_sd:
        # Re{HV} and Im{HV} are not explicitly stored in interferometer data - extract them from HV instead
        if not self.has_autocorr and key in ('ReHV', 'ImHV'):
            HV = self.data[:, :, scape_pol_if.index('HV')]
            return HV.real if key == 'ReHV' else HV.imag
        else:
            return self.data[:, :, scape_pol_sd.index(key)]
    elif key in scape_pol_if:
        # HV and VH are not explicitly stored in single-dish data - calculate them instead
        if self.has_autocorr and key in ('HV', 'VH'):
            ReHV, ImHV = self.data[:, :, scape_pol_sd.index('ReHV')], self.data[:, :, scape_pol_sd.index('ImHV')]
            return ReHV + 1j * ImHV if key == 'HV' else ReHV - 1j * ImHV
        else:
            return self.data[:, :, scape_pol_if.index(key)]

    # The rest of the polarisation terms are sky-based, and need parallactic angle correction.
    # The mount rotation on the sky is equivalent to a *negative* parallactic angle rotation, and
    # is compensated for by rotating the data through the *positive* parallactic angle itself.
    if key in sky_coh:
        # Construct series of 2x2 Jones rotation matrices, of shape (*T*, 2, 2)
        cospa, sinpa = np.cos(scan.parangle), np.sin(scan.parangle)
        rot_jones = np.array([[cospa, -sinpa], [sinpa, cospa]], dtype=np.complex128).transpose([2, 0, 1])
        # Construct corresponding series of 4x4 generalised Mueller matrices, of shape (*T*, 1, 4, 4)
        rot_genmueller = np.expand_dims(np.array([np.kron(J, J.conj()) for J in rot_jones]), 1)
        # Rotate data to sky coordinate system
        data = transform_pol(scan, rot_genmueller)[:, :, sky_coh.index(key)]
        return data.real if scan.has_autocorr and key in ('XX', 'YY') else data
    elif key in stokes:
        # Construct series of 2x2 rotation matrices, of shape (*T*, 2, 2)
        cos2pa, sin2pa = np.cos(2.0 * scan.parangle), np.sin(2.0 * scan.parangle)
        rot_mat = np.array([[cos2pa, -sin2pa], [sin2pa, cos2pa]], dtype=np.complex128).transpose([2, 0, 1])
        # Construct series of 4x4 Mueller rotation matrices, of shape (*T*, 1, 4, 4)
        rot_mueller = np.zeros((len(cos2pa), 1, 4, 4), dtype=np.complex128)
        rot_mueller[:, :, 0, 0] = rot_mueller[:, :, 3, 3] = 1.0
        rot_mueller[:, :, 1:3, 1:3] = np.expand_dims(rot_mat, 1)
        data = transform_pol(scan, rot_mueller, stokes_from_coh)[:, :, stokes.index(key)]
        return data.real if scan.has_autocorr else data
    else:
        raise KeyError("Polarisation key should be one of %s" % list(set(scape_pol_sd + scape_pol_if + sky_coh + stokes)),)

def assert_almost_equal(actual, desired, decimal=15):
    """Assert actual is close to desired, within relative tolerance.

    The *actual* and *desired* inputs are assumed to be float64 or complex128
    arrays, and *decimal* is the number of mantissa decimals that match (i.e.
    it is a relative tolerance, unlike the decimal of
    :func:`np.testing.assert_almost_equal`).

    """
    if np.isrealobj(actual):
        np.testing.assert_almost_equal(actual / desired, 1.0, decimal)
    else:
        np.testing.assert_almost_equal(actual.real / desired.real, 1.0, decimal)
        np.testing.assert_almost_equal(actual.imag / desired.imag, 1.0, decimal)

class ScanTestCases(unittest.TestCase):
    """Create scan and check it."""

    def setUp(self):
        """Create scan."""

        time_start = '2009/06/26 20:00:00 SAST'
        num_channels = 16
        dump_rate = 10.
        samples_per_scan = 101

        # Create sinusoidal power data with different values in each frequency bin and polarisation
        data_t = 10. + 2. * np.cos(2. * np.pi * np.arange(0.0, 1.0, 1.0 / samples_per_scan, dtype=np.float64))
        data_tf = np.outer(data_t, np.arange(100., num_channels + 100., 1., dtype=np.float64) / 100.)
        # This is single-dish [HH, VV, Re{HV}, Im{HV}] format
        self.data_sd = np.dstack((data_tf, 2.0 * data_tf, 0.3 * (data_tf - 10.), 0.1 * (data_tf - 10.)))
        # This is interferometer [HH, VV, HV, VH] format
        self.data_if = np.dstack((1.0 * data_tf           + 0.6j * np.flipud(data_tf),
                                  2.0 * data_tf           + 1.5j * np.fliplr(data_tf),
                                  0.3 * (data_tf - 10.) + 0.003j * np.fliplr(data_tf),
                                  0.1 * (data_tf - 10.) + 0.002j * np.flipud(data_tf)))
        time_start = time.mktime(time.strptime(time_start, '%Y/%m/%d %H:%M:%S %Z'))
        timestamps = time_start + np.arange(samples_per_scan) / dump_rate
        az = katpoint.deg2rad(np.arange(-20., 20., 40. / samples_per_scan, dtype=np.float32))
        el = katpoint.deg2rad(np.arange(40., 50., 10. / samples_per_scan, dtype=np.float32))
        pointing = np.rec.fromarrays([az, el], names='az,el')
        flags = np.rec.fromarrays([np.tile(True, samples_per_scan),
                                   np.tile(False, samples_per_scan)], names='valid,nd_on')
        target_coords = np.zeros((2, samples_per_scan))
        parangle = katpoint.deg2rad(np.arange(-180., 180., 360. / samples_per_scan))
        # Scan containing single-dish data
        self.scan_sd = scape.Scan(self.data_sd, timestamps, pointing, flags, 'scan', 'scan', target_coords, parangle)
        # Scan containing interferometer data
        self.scan_if = scape.Scan(self.data_if, timestamps, pointing, flags, 'scan', 'scan', target_coords, parangle)

    def test_pol(self):
        """Test Scan polarisation transforms and extraction."""
        # Check mount coherencies - should be exactly equal, as this is the native format
        np.testing.assert_equal(self.scan_sd.pol('HH'), self.data_sd[:, :, 0])
        np.testing.assert_equal(self.scan_sd.pol('VV'), self.data_sd[:, :, 1])
        np.testing.assert_equal(self.scan_sd.pol('HV'), self.data_sd[:, :, 2] + 1j * self.data_sd[:, :, 3])
        np.testing.assert_equal(self.scan_sd.pol('VH'), self.data_sd[:, :, 2] - 1j * self.data_sd[:, :, 3])
        np.testing.assert_equal(self.scan_if.pol('HH'), self.data_if[:, :, 0])
        np.testing.assert_equal(self.scan_if.pol('VV'), self.data_if[:, :, 1])
        np.testing.assert_equal(self.scan_if.pol('HV'), self.data_if[:, :, 2])
        np.testing.assert_equal(self.scan_if.pol('VH'), self.data_if[:, :, 3])
        # Check sky coherencies against slow but generic calculation
        assert_almost_equal(self.scan_sd.pol('XX'), generic_pol(self.scan_sd, 'XX'))
        assert_almost_equal(self.scan_sd.pol('YY'), generic_pol(self.scan_sd, 'YY'))
        assert_almost_equal(self.scan_sd.pol('XY'), generic_pol(self.scan_sd, 'XY'), decimal=14)
        assert_almost_equal(self.scan_sd.pol('YX'), generic_pol(self.scan_sd, 'YX'), decimal=14)
        assert_almost_equal(self.scan_if.pol('XX'), generic_pol(self.scan_if, 'XX'))
        assert_almost_equal(self.scan_if.pol('YY'), generic_pol(self.scan_if, 'YY'))
        assert_almost_equal(self.scan_if.pol('XY'), generic_pol(self.scan_if, 'XY'), decimal=14)
        assert_almost_equal(self.scan_if.pol('YX'), generic_pol(self.scan_if, 'YX'), decimal=13)
        # Check Stokes parameters against slow but generic calculation
        np.testing.assert_equal(self.scan_sd.pol('I'), generic_pol(self.scan_sd, 'I'))
        np.testing.assert_equal(self.scan_sd.pol('Q'), generic_pol(self.scan_sd, 'Q'))
        np.testing.assert_equal(self.scan_sd.pol('U'), generic_pol(self.scan_sd, 'U'))
        np.testing.assert_equal(self.scan_sd.pol('V'), generic_pol(self.scan_sd, 'V'))
        np.testing.assert_equal(self.scan_if.pol('I'), generic_pol(self.scan_if, 'I'))
        np.testing.assert_equal(self.scan_if.pol('Q'), generic_pol(self.scan_if, 'Q'))
        np.testing.assert_equal(self.scan_if.pol('U'), generic_pol(self.scan_if, 'U'))
        np.testing.assert_equal(self.scan_if.pol('V'), generic_pol(self.scan_if, 'V'))
        # Check some "invariants"
        np.testing.assert_equal(self.scan_sd.pol('HH') + self.scan_sd.pol('VV'), self.scan_sd.pol('I'))
        assert_almost_equal(self.scan_sd.pol('XX') + self.scan_sd.pol('YY'), self.scan_sd.pol('I'))
        np.testing.assert_equal(1j * (self.scan_sd.pol('HV') - self.scan_sd.pol('VH')), self.scan_sd.pol('V'))
        np.testing.assert_equal(1j * (self.scan_sd.pol('YX') - self.scan_sd.pol('XY')), self.scan_sd.pol('V'))
        np.testing.assert_equal(self.scan_if.pol('HH') + self.scan_if.pol('VV'), self.scan_if.pol('I'))
        assert_almost_equal(self.scan_if.pol('XX') + self.scan_if.pol('YY'), self.scan_if.pol('I'))
        np.testing.assert_equal(1j * (self.scan_if.pol('HV') - self.scan_if.pol('VH')), self.scan_if.pol('V'))
        assert_almost_equal(1j * (self.scan_if.pol('YX') - self.scan_if.pol('XY')), self.scan_if.pol('V'), decimal=13)
        # Check absolute powers
        np.testing.assert_equal(self.scan_sd.pol('absI'), self.scan_sd.pol('I'))
        np.testing.assert_equal(self.scan_if.pol('absI'), np.abs(self.scan_if.pol('HH')) + np.abs(self.scan_if.pol('VV')))
        np.testing.assert_equal(self.scan_sd.pol('absHH'), self.scan_sd.pol('HH'))
        np.testing.assert_equal(self.scan_if.pol('absHH'), np.abs(self.scan_if.pol('HH')))
        np.testing.assert_equal(self.scan_sd.pol('absVV'), self.scan_sd.pol('VV'))
        np.testing.assert_equal(self.scan_if.pol('absVV'), np.abs(self.scan_if.pol('VV')))

    def test_select(self):
        """Test Scan time/frequency selection."""
        data = self.scan_sd.data
        # Check verbatim copies
        self.assertTrue((self.scan_sd.select(copy=True).data == data).all())
        self.assertEqual(id(self.scan_sd.select(copy=False).data), id(data))
        # Test time selection via indices and masks
        tselect = [0, 50, 100]
        self.assertTrue((self.scan_sd.select(timekeep=tselect, copy=True).data == data[tselect, :, :]).all())
        tmask = np.tile(False, data.shape[0])
        tmask[tselect] = True
        self.assertTrue((self.scan_sd.select(timekeep=tselect, copy=True).data == data[tselect, :, :]).all())
        self.assertRaises(IndexError, self.scan_sd.select, timekeep=[0, 50, 1000], copy=True)
        self.assertRaises(IndexError, self.scan_sd.select, timekeep=np.tile(False, 2 * data.shape[0]), copy=True)
        # Test frequency selection via indices and masks
        fselect = [0, 8, 15]
        self.assertTrue((self.scan_sd.select(freqkeep=fselect, copy=True).data == data[:, fselect, :]).all())
        fmask = np.tile(False, data.shape[1])
        fmask[fselect] = True
        self.assertTrue((self.scan_sd.select(freqkeep=fselect, copy=True).data == data[:, fselect, :]).all())
        self.assertRaises(IndexError, self.scan_sd.select, freqkeep=[0, 50, 1000], copy=True)
        self.assertRaises(IndexError, self.scan_sd.select, freqkeep=np.tile(False, 2 * data.shape[1]), copy=True)
