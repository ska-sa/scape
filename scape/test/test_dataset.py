###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
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
        enviro = {'temperature' : np.rec.array([timestamps[0], np.float32(35.1), 'nominal'],
                                               names=('timestamp', 'value', 'status')),
                  'pressure' : np.rec.array([timestamps[0], np.float32(1020.2), 'nominal'],
                                            names=('timestamp', 'value', 'status')),
                  'humidity' : np.rec.array([timestamps[0], np.float32(21.5), 'nominal'],
                                            names=('timestamp', 'value', 'status')),
                  'wind_speed' : np.rec.array([timestamps[0], np.float32(2.0), 'nominal'],
                                               names=('timestamp', 'value', 'status')),
                  'wind_direction' : np.rec.array([timestamps[0], np.float32(45.3), 'nominal'],
                                                  names=('timestamp', 'value', 'status'))}
        s1 = scan.Scan(data, timestamps, pointing, flags, 'test', 'generated')
        s2 = s1.select(timekeep=s1.flags['valid'], copy=True)
        s3 = s1.select(copy=True)
        s4 = s1.select(timekeep=s1.flags['nd_on'], copy=True)
        cs1 = compoundscan.CompoundScan([s1, s2], 'Sun, special')
        cs2 = compoundscan.CompoundScan([s3, s4], 'Moon, special', 'overthemoon')
        freqs = 1e3 + np.arange(num_chans, dtype=np.float64)
        rfi_channels = [2, 5, 9]
        channel_select = list(set(range(num_chans)) - set(rfi_channels))
        corrconf = compoundscan.CorrelatorConfig(freqs, np.tile(1.0, num_chans).astype(np.float64),
                                                 channel_select, 1.0)
        nd_h_model = gaincal.NoiseDiodeModel(freqs, 20.0 + np.random.randn(len(freqs)),
                                             antenna='Test', pol='H', diode='coupler', date='2010-10-20')
        nd_v_model = gaincal.NoiseDiodeModel(freqs, 20.0 + np.random.randn(len(freqs)),
                                             antenna='Test', pol='V', diode='coupler', date='2010-10-20')
        self.d = dataset.DataSet('', [cs1, cs2], '007', 'tester', 'Test', 'counts',
                                 corrconf, 'Test, 0, 0, 0, 15.0', None, nd_h_model, nd_v_model, enviro)
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
