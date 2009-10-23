"""Tests for the dataset module."""
# pylint: disable-msg=C0103

import unittest
import os

import numpy as np

from scape import dataset, compoundscan, scan, gaincal

class SaveLoadTestCases(unittest.TestCase):
    """Save and reload data set to verify that file was written successfully."""

    def setUp(self):
        # Create dummy data set
        num_samples, num_chans = 100, 16
        data = np.random.randn(num_samples, num_chans, 4).astype(np.float32)
        timestamps = 1231760940.0 + np.arange(num_samples, dtype=np.float64)
        pointing = np.rec.fromarrays(0.01 * np.random.randn(3, num_samples).astype(np.float32),
                                     names='az,el,rot')
        flags = np.rec.fromarrays((np.random.randn(2, num_samples) > 0.0), names='valid,nd_on')
        enviro_ambient = np.rec.array([(timestamps[0], 35.1, 1020.4, 31.0)],
                                      dtype=[('timestamp', np.float64), ('temperature', np.float32),
                                             ('pressure', np.float32), ('humidity', np.float32)])
        enviro_wind = np.rec.array([(timestamps[0], 2.0, 45.3)],
                                   dtype=[('timestamp', np.float64),
                                          ('wind_speed', np.float32), ('wind_direction', np.float32)])
        s1 = scan.Scan(data, False, timestamps, pointing, flags, enviro_ambient, enviro_wind, 'test', 'generated')
        s2 = s1.select(timekeep=s1.flags['valid'], copy=True)
        s3 = s1.select(copy=True)
        s4 = s1.select(timekeep=s1.flags['nd_on'], copy=True)
        cs1 = compoundscan.CompoundScan([s1, s2], 'Sun, special')
        cs2 = compoundscan.CompoundScan([s3, s4], 'Moon, special')
        freqs = 1e3 + np.arange(num_chans, dtype=np.float64)
        corrconf = compoundscan.CorrelatorConfig(freqs, np.tile(1.0, num_chans).astype(np.float64),
                                                 [2, 5, 9], 1.0)
        temp = np.column_stack((freqs, 20.0 + np.random.randn(len(freqs))))
        nd_data = gaincal.NoiseDiodeModel(temp, temp)
        pm = np.ones(20)
        self.d = dataset.DataSet('', [cs1, cs2], 'raw', corrconf, 'Test, 0, 0, 0, 15.0', nd_data, pm)
        self.filename = 'scape_test_dataset.h5'
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_hdf5_save_load(self):
        # Skip test if HDF5 support is not available
        try:
            self.d.save(self.filename)
            d2 = dataset.DataSet(self.filename)
            self.assertEqual(self.d, d2, "Dataset loaded from HDF5 file is not the same as the one saved to file.")
        except ImportError:
            pass

    def tearDown(self):
        try:
            os.remove(self.filename)
        except OSError:
            pass
