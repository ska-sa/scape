###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
"""Tests for the gaincal module."""
# pylint: disable-msg=C0103

import unittest
import numpy as np
import StringIO

from scape import gaincal

class NoiseDiodeModelTestCases(unittest.TestCase):
    """Load a noise diode model from and compare."""

    def setUp(self):
        self.freq = np.arange(1190562500, 1960250000, 781250)
        self.model = lambda freq: 5. + 2. * (freq - self.freq[0]) / (self.freq[-1] - self.freq[0])
        self.temp = self.model(self.freq)
        self.file_data = """# antenna = ant1
# pol = H
# diode = coupler
# date = 2010-08-05
# interp = Polynomial1DFit(max_degree=1)
#
# freq [Hz], T_nd [K]
"""
        self.file_data += '\n'.join(['%d, %.3f' % row for row in zip(self.freq, self.temp)])
        self.csv_file = StringIO.StringIO(self.file_data)

    def test_load_nd_model(self):
        nd = gaincal.NoiseDiodeModel(self.csv_file)
        np.testing.assert_equal(nd.freq, self.freq / 1e6)
        np.testing.assert_almost_equal(nd.temp, self.temp, decimal=3)
        self.assertEqual(nd.antenna, 'ant1')
        self.assertEqual(nd.pol, 'H')
        self.assertEqual(nd.diode, 'coupler')
        self.assertEqual(nd.date, '2010-08-05')
        self.assertEqual(nd.interp, 'Polynomial1DFit(max_degree=1)')
        freq = np.arange(1e9, 2e9, 1e6)
        np.testing.assert_almost_equal(nd.temperature(freq / 1e6), self.model(freq), decimal=5)
        nd2 = gaincal.NoiseDiodeModel(nd.freq, nd.temp, interp='Polynomial1DFit(max_degree=1)',
                                      antenna='ant1', pol='H', diode='coupler', date='2010-08-05')
        self.assertEqual(nd, nd2)
