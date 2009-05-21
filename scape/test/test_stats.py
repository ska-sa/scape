"""Tests for the stats module."""

import copy
import unittest
import numpy as np
from scape import stats

class MuSigmaArrayTestCases(unittest.TestCase):
    
    def setUp(self):
        self.scalar_mu = 0
        self.scalar_sigma = 1
        self.vector_mu = [0,0,0]
        self.vector_sigma = [1,1,1]
        self.array_mu = np.zeros((2,3))
        self.array_sigma = np.ones((2,3))
    
    def test_scalar(self):
        ms_scalar = stats.MuSigmaArray(self.scalar_mu, self.scalar_sigma)
        np.testing.assert_equal(ms_scalar, self.scalar_mu)
        np.testing.assert_equal(ms_scalar.mu, self.scalar_mu)
        np.testing.assert_equal(ms_scalar.sigma, self.scalar_sigma)
    
    def test_vector(self):
        self.assertRaises(TypeError, stats.MuSigmaArray, [1,1,1], [2,2])
        ms_vector = stats.MuSigmaArray(self.vector_mu, self.vector_sigma)
        np.testing.assert_array_equal(ms_vector, self.vector_mu)
        np.testing.assert_array_equal(ms_vector.mu, self.vector_mu)
        np.testing.assert_array_equal(ms_vector.sigma, self.vector_sigma)
    
    def test_array(self):
        ms_array = stats.MuSigmaArray(self.array_mu, self.array_sigma)
        np.testing.assert_array_equal(ms_array, self.array_mu)
        np.testing.assert_array_equal(ms_array.mu, self.array_mu)
        np.testing.assert_array_equal(ms_array.sigma, self.array_sigma)
        ms_vector = stats.MuSigmaArray(self.vector_mu, self.vector_sigma)
        np.testing.assert_array_equal(ms_array[0].mu, ms_vector.mu)
        np.testing.assert_array_equal(ms_array[0].sigma, ms_vector.sigma)
        np.testing.assert_array_equal(ms_array[0:2].mu, ms_array.mu)
        np.testing.assert_array_equal(ms_array[0:2].sigma, ms_array.sigma)
        
    def test_copy(self):
        ms_array = stats.MuSigmaArray(self.array_mu, self.array_sigma)
        ms_array_shallow = copy.copy(ms_array)
        ms_array_deep = copy.deepcopy(ms_array)
        np.testing.assert_array_equal(ms_array.mu, ms_array_shallow.mu)
        np.testing.assert_array_equal(ms_array.sigma, ms_array_shallow.sigma)
        np.testing.assert_array_equal(ms_array.mu, ms_array_deep.mu)
        np.testing.assert_array_equal(ms_array.sigma, ms_array_deep.sigma)
        ms_array[0,0] = 1
        ms_array.sigma[0,0] = 0
        np.testing.assert_array_equal(ms_array.mu, ms_array_shallow.mu)
        np.testing.assert_array_equal(ms_array.sigma, ms_array_shallow.sigma)
        self.assertNotEqual(ms_array.mu[0,0], ms_array_deep.mu[0,0])
        self.assertNotEqual(ms_array.sigma[0,0], ms_array_deep.sigma[0,0])
    
    def test_stack(self):
        ms_scalar = stats.MuSigmaArray([self.scalar_mu], [self.scalar_sigma])
        ms_vector = stats.MuSigmaArray(self.vector_mu, self.vector_sigma)
        ms_array = stats.MuSigmaArray(self.array_mu, self.array_sigma)
        ms_concat = stats.ms_concatenate((ms_scalar, ms_scalar, ms_scalar))
        ms_hstack = stats.ms_hstack((ms_scalar, ms_scalar, ms_scalar))
        ms_vstack = stats.ms_vstack((ms_vector, ms_vector))
        np.testing.assert_array_equal(ms_concat.mu, ms_vector.mu)
        np.testing.assert_array_equal(ms_concat.sigma, ms_vector.sigma)
        np.testing.assert_array_equal(ms_hstack.mu, ms_vector.mu)
        np.testing.assert_array_equal(ms_hstack.sigma, ms_vector.sigma)
        np.testing.assert_array_equal(ms_vstack.mu, ms_array.mu)
        np.testing.assert_array_equal(ms_vstack.sigma, ms_array.sigma)
