"""Unit test suite for scape."""

import unittest
# pylint: disable-msg=W0403
import test_projection
import test_stats
import test_fitting

def suite():
    loader = unittest.TestLoader()
    testsuite = unittest.TestSuite()
    testsuite.addTests(loader.loadTestsFromModule(test_projection))
    testsuite.addTests(loader.loadTestsFromModule(test_stats))
    testsuite.addTests(loader.loadTestsFromModule(test_fitting))
    return testsuite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
