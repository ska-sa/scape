###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: cam@ska.ac.za                                                       #
# Copyright @ 2013 SKA SA. All rights reserved.                               #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################
"""Unit test suite for scape."""

import unittest
# pylint: disable-msg=W0403
import test_stats
import test_fitting
import test_gaincal
import test_scan
import test_scape
import test_xdmfits

def suite():
    loader = unittest.TestLoader()
    testsuite = unittest.TestSuite()
    testsuite.addTests(loader.loadTestsFromModule(test_stats))
    testsuite.addTests(loader.loadTestsFromModule(test_fitting))
    testsuite.addTests(loader.loadTestsFromModule(test_gaincal))
    testsuite.addTests(loader.loadTestsFromModule(test_scan))
    testsuite.addTests(loader.loadTestsFromModule(test_scape))
    testsuite.addTests(loader.loadTestsFromModule(test_xdmfits))
    return testsuite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
