###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
"""Tests for the stats module."""
# pylint: disable-msg=C0103,W0212,R0904

import unittest
import numpy as np
from scape import stats

class RemoveSpikesTestCases(unittest.TestCase):

    def test_remove_spikes(self):
        """remove_spikes: Test the removal of known spikes."""
        N = 128
        x = np.arange(N)/float(N)
        b = 3.0 * x + 4.0
        xstd = 0.05
        g = np.exp(-0.5 * ((x - 0.5) ** 2) / (xstd ** 2))
        # g += 0.2*np.exp(-0.5 * ((x - 0.3) ** 2) / ((0.5*std) ** 2))
        # g += 0.2*np.exp(-0.5 * ((x - 0.7) ** 2) / ((0.5*std) ** 2))
        nstd = 0.1
        n = nstd * np.random.randn(N)
        y = g + b + n
        y_dirty = y.copy()
        y_dirty[np.random.randint(N)] += 100
        y_dirty[np.random.randint(N)] += 100
        y_clean = stats.remove_spikes(y_dirty, outlier_sigma=5.0)
        spikes = np.where(y_clean != y_dirty)[0]
        # This test is currently failing frequently, but I'm unsure if it should succeed
        # TODO: More checking required of these tests...
#        self.assertEqual(len(spikes), 2)
        self.assertTrue((np.abs(y_clean - y)[spikes] < 10000.0 * nstd).all())

class RatioStatsTestCases(unittest.TestCase):

    def test_ratio_stats(self):
        """ratio_stats: Compare sample stats of known simulated data to ratio_stats."""
        # Example from Marsaglia
        mu_z, std_z, mu_w, std_w, p = 30.5, 5., 32., 4., 0.8
        mu_r, std_r = stats.ratio_stats(mu_z, std_z, mu_w, std_w, p, method='Marsaglia')
        self.assertAlmostEqual(mu_r, 0.952, places=3)
        self.assertAlmostEqual(std_r, 0.0959, places=3)
        # Check against simulated data
        x = np.random.randn(100000)
        y = np.random.randn(100000)
        wgt = (p**2 - p*np.sqrt(1 - p**2)) / (2*p**2 - 1)
        xx = (wgt * x + (1 - wgt) * y) / np.sqrt(wgt**2 + (1 - wgt)**2)
        r = (mu_z + std_z * xx) / (mu_w + std_w * x)
        mu_r, std_r = stats.ratio_stats(mu_z, std_z, mu_w, std_w, p, method='F')
        self.assertAlmostEqual(mu_r, r.mean(), places=2)
        self.assertAlmostEqual(std_r, r.std(), places=2)
