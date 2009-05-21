import unittest
import test_projection

def suite():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromModule(test_projection))
    suite.addTests(loader.loadTestsFromModule(test_stats))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
