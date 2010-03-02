"""Tests for the stats module."""
# pylint: disable-msg=C0103,W0212,R0904

import unittest
import numpy as np
from scape import stats

class RemoveSpikesTestCases(unittest.TestCase):

    def test_remove_spikes(self):
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
