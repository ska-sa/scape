## xdmsbelib/stats.py
#
# Statistics routines used by XDM software backend.
#
# copyright (c) 2007 SKA/KAT. All rights reserved.
# @author Rudolph van der Merwe <rudolph@ska.ac.za>
# @date 2007-03-13

# pylint: disable-msg=C0103

from __future__ import division
import numpy as np
import scipy as sp
sp.pkgload('stats')
import misc
import copy
import logging

logger = logging.getLogger("xdmsbe.xdmsbelib.stats")


#=========================================================================================================
#===                                       CLASSES                                                     ===
#=========================================================================================================

#---------------------------------------------------------------------------------------------------------
#--- CLASS :  DistributionStatsArray
#------------------------------------

## Class used as a structure to represent an array of scalar random variable distribution
# statistics

class DistributionStatsArray(object):
    
    # pylint: disable-msg=R0903
    
    ## @var mean
    # array of mean values
    
    ## @var var
    # array of variance values
    
    ## @var num
    # array of number of samples values
    
    ## @var shape
    # the shape of the arrays as a tuple
    
    ## @var timeStamp
    # timestamp associated with data
    
    ## Initialiser/constructor. Note that the arrays should all have the same dimensions.
    # It is possible that they are all scalars in the degenerative case.
    # @param self  the current object
    # @param mean  array of mean values
    # @param var   array of variance values
    # @param num   array of number of samples
    # @param timeStamp timestamp associated with data
    # @param shape desired shape of statistics arrays
    # pylint: disable-msg=R0913
    def __init__(self, mean=None, var=None, num=None, timeStamp=None, shape=None):
        
        if shape:
            if mean or var or num or timeStamp:
                logger.warn("shape arg overrides mean, var and num args")
            self.mean = np.zeros(shape, 'double')
            self.var = np.zeros(shape, 'double')
            self.num = np.zeros(shape, 'double')
            self.timeStamp = None
        else:
            if mean == None or var == None or num == None:
                message = "Must specify all of mean, var and num"
                #logger.error(message)
                raise ValueError, message
            self.mean = np.atleast_1d(mean)
            self.var = np.atleast_1d(var)
            self.num = np.atleast_1d(num)
            self.timeStamp = None
        # Check that all inputs have correct dimensions
        dimSet = set([self.mean.shape, self.var.shape, self.num.shape])
        assert(len(dimSet) == 1)
        self.shape = list(dimSet)[0]
    
    ## Add statistics for a given index of the array by computing directly from a sample array of data.
    # @param self  the current object
    # @param data  numpy.ndarray of sample values
    # @param timeStamps timestamps of data block samples
    # @param index the index in the stats array for the mean, var and num
    def add_data(self, data, timeStamps, index):
        # An empty data array results in a None
        # (this happens e.g. if all channels in a band are marked as RFI)
        if data.size > 0:
            self.mean[index] = np.mean(data.ravel())
            self.var[index] = np.cov(data.ravel())
        else:
            self.mean[index] = np.None
            self.var[index] = np.None
        #self.var[index] = 1/(data.size-1)*((data - self.mean[index])**2)
        self.num[index] = data.size
        self.timeStamp = np.mean(timeStamps.ravel())
        return self
    
    ## Find instances of zeros variance and set it to some small finite value
    # @param self  the current object
    # @param minVar  minimum variance floor
    def floor_zero_var(self, minVar):
        self.var[self.var == 0] = minVar
        return self


#---------------------------------------------------------------------------------------------------------
#--- CLASS :  SpStatsFuncWrapper
#---------------------------------

## Function wrapper class used by sp_stats function. This class wraps an arbitrary mathematical
#  function such that the sp_stats() function has a consistent interface to the function in
#  question.
class SpStatsFuncWrapper(object):
    
    # pylint: disable-msg=R0902,W0212,W0142
    
    ## Specify the function to be wrapped as well as the number and (numpy) shapes of the input parameters
    # and output variables
    # @param    self            The object pointer
    # @param    func            Function handle
    # @param    inputShapeDict  Dict of {paramName: np.shape} tuples specifying the input parameter shapes
    # @param    outputShapeList List of np.shape tuples specifying the output (return) variables shapes
    # @param    constantDict    Dict of {paramName: constant} parameters
    def __init__(self, func, inputShapeDict, outputShapeList, constantDict=None):
        self._func = func
        self._inputShapeDict = inputShapeDict
        self._outputShapeDict = {}
        for i, shape in enumerate(outputShapeList):
            self._outputShapeDict['ov'+str(i)] = shape
        self._constantDict = constantDict
        self._inputIdxDict = {}
        self._outputIdxDict = {}
        self._numInputs = len(inputShapeDict)
        self._numOutputs = len(outputShapeList)
        self._inputDim = SpStatsFuncWrapper._init_idx_dict(self._inputShapeDict, self._inputIdxDict)
        self._outputDim = SpStatsFuncWrapper._init_idx_dict(self._outputShapeDict, self._outputIdxDict)
    
    ## Call overloading so that the class can be used directly as a function,
    # i.e. if foo = FuncWrapper(), then foo(x) will call this method.
    # @param    self        The object pointer
    # @param    X           Vectorized input argument
    # @return   Y           Y = VECTORIZE(self._func(DEVECTORIZE(X)))
    def __call__(self, X):
        (_dim, numSigmaPoints) = X.shape
        Y = np.zeros((self._outputDim, numSigmaPoints), 'double')
        for k in xrange(numSigmaPoints):
            Xk = X[:, k]
            paramDict = self.devectorize_input(Xk)
            if self._constantDict:
                for key, constant in self._constantDict.items():
                    paramDict[key] = constant
            # pylint: disable-msg=W0142
            output = self._func(**paramDict)
            if self._numOutputs == 1:
                output = (output,)
            Y[:, k] = self.vectorize_output(*output)
        return Y
    
    ## Devectorize an input vector
    # @param self the current object
    # @param X input vector
    # @return dictionary of {paramName: np.array}
    def devectorize_input(self, X):
        return SpStatsFuncWrapper._devectorize(X, self._inputShapeDict, self._inputIdxDict)
    
    ## Devectorize an output vector
    # @param self the current object
    # @param Y output vector
    # @return devectorized output of function
    def devectorize_output(self, Y):
        result = SpStatsFuncWrapper._devectorize(Y, self._outputShapeDict, self._outputIdxDict)
        if len(result) == 1:
            return result.values()[0]
        else:
            resultList = []
            for i in np.arange(self._numOutputs):
                resultList.append(result['ov'+str(i)])
            return resultList
    
    ## Vectorize a dictionary of input arrays
    # @param self the current object
    # @param kwds dictionary of {paramName: np.array}
    # @return vector representation of dictionary
    def vectorize_input(self, **kwds):
        return SpStatsFuncWrapper._vectorize(self._inputIdxDict, self._inputDim, **kwds)
    
    ## Vectorize a list of output arrays
    # @param self the current object
    # @param outputs list of output np.array objects
    # @return vector representation of list
    def vectorize_output(self, *outputs):
        outputDict = {}
        for i, output in enumerate(outputs):
            outputDict['ov'+str(i)] = output
        return SpStatsFuncWrapper._vectorize(self._outputIdxDict, self._outputDim, **outputDict)
    
    @staticmethod
    def _init_idx_dict(shapeDict, idxDict):
        idx = 0
        for key, shape in shapeDict.items():
            size = int(np.prod(shape))
            idxDict[key] = (idx, idx + size)
            idx += size
        return idx
    
    @staticmethod
    def _devectorize(Z, shapeDict, indexDict):
        result = {}
        for key, index in indexDict.items():
            sliceZ = Z[index[0]: index[1]]
            result[key] = np.reshape(sliceZ, shapeDict[key])
        return result
    
    @staticmethod
    def _vectorize(indexDict, dim, **kwds):
        assert(indexDict.keys() == kwds.keys())
        result = np.zeros(dim, 'double')
        for key, index in indexDict.items():
            result[index[0]: index[1]] = np.ravel(kwds[key])
        return result


#---------------------------------------------------------------------------------------------------------
#--- CLASS :  MuSigmaArray
#--------------------------

## Container that bundles mean and standard deviation of N-dimensional data.
# This is a subclass of numpy.ndarray, which adds a sigma data member.
# The idea is that the main array is the mean, while sigma contains the standard deviation
# of each corresponding element of the main array. Sigma should therefore have the same shape
# as the main array (mean). This approach allows the object itself to be the mean (e.g. x),
# while the standard deviation can be accessed as x.sigma. This makes the standard use of the
# mean (required in most calculations) cleaner, while still allowing the mean and standard
# deviation to travel together instead of as two arrays.
#
# Alternative solutions:
# - A class with .mu and .sigma arrays (cumbersome when using mu)
# - A dict with 'mu' and 'sigma' keys and array values (ditto)
# - A tuple containing two arrays (mean still has to be extracted first)
# - Extending the mu array to contain another dimension (can be misleading)
# pylint: disable-msg=R0903
class MuSigmaArray(np.ndarray):
    ## Object creation.
    # This casts the mu array to the current subclass.
    # @param cls   The current class
    # @param mu    The mean array (which becomes the main array of this object)
    # @param sigma The standard deviation array (default=None)
    # pylint: disable-msg=W0613
    def __new__(cls, mu, sigma=None):
        return np.asarray(mu).view(cls)
    ## Initialiser.
    # This checks sigma to ensure its dimensions are compatible with mu.
    # @param self  The current object
    # @param mu    The mean array (which becomes the main array of this object)
    # @param sigma The standard deviation array (default=None)
    # pylint: disable-msg=W0613
    def __init__(self, mu, sigma=None):
        ## @var _sigma
        # Standard deviation of each element in main array (internal variable set via property).
        self._sigma = None
        ## @var sigma
        # Standard deviation of each element in main array (property).
        self.sigma = sigma
    
    ## Class method which creates mean property.
    # This is a nice way to create Python properties. It prevents clutter of the class namespace with
    # getter and setter methods, while being more readable than lambda notation. The function below is
    # effectively hidden by giving it the same name as the eventual property. Pylint gets queasy here,
    # for obvious reasons.
    # @return Dictionary containing property getter and setter methods, and doc string
    # pylint: disable-msg=E0211,E0202,W0212,W0612
    def mu():
        doc = 'Mean array.'
        def fget(self):
            return self.view(np.ndarray)
        def fset(self, value):
            self = value
        return locals()
    ## @var mu
    # Mean array. This is merely for convenience, to restrict the object to be an numpy.ndarray.
    # Normal access to the object also provides the mean.
    # pylint: disable-msg=W0142,W1001
    mu = property(**mu())
    
    ## Class method which creates sigma property.
    # @return Dictionary containing property getter and setter methods, and doc string
    # pylint: disable-msg=E0211,E0202,W0212,W0612
    def sigma():
        doc = 'Standard deviation of each element in main array.'
        def fget(self):
            return self._sigma
        def fset(self, value):
            if value != None:
                value = np.asarray(value)
                if value.shape != self.shape:
                    raise TypeError, "X.sigma should have the same shape as X (i.e. " +  \
                                     str(self.shape) + " instead of " + str(value.shape) + " )"
            self._sigma = value
        return locals()
    ## @var sigma
    # Standard deviation of each element in main array (property).
    # pylint: disable-msg=W0142,W1001
    sigma = property(**sigma())
    
    ## Official string representation
    # @param self  The current object
    def __repr__(self):
        return self.__class__.__name__ + '(' + repr(self.mu) + ',' + repr(self.sigma) + ')'
    ## Informal string representation
    # @param self  The current object
    def __str__(self):
        return 'mu    = ' + str(self.mu) + '\nsigma = ' + str(self.sigma)
    ## Index both arrays at the same time
    def __getitem__(self, value):
        if self.sigma == None:
            return MuSigmaArray(self.mu[value], None)
        else:
            return MuSigmaArray(self.mu[value], self.sigma[value])
    ## Index both arrays at the same time
    def __getslice__(self, first, last):
        if self.sigma == None:
            return MuSigmaArray(self.mu[first:last], None)
        else:
            return MuSigmaArray(self.mu[first:last], self.sigma[first:last])
    ## Shallow copy operation
    def __copy__(self):
        return MuSigmaArray(self.mu, self.sigma)
    ## Deep copy operation
    def __deepcopy__(self, memo):
        return MuSigmaArray(copy.deepcopy(self.mu, memo), copy.deepcopy(self.sigma, memo))
    
## Concatenate MuSigmaArrays.
# @param msaList List of MuSigmaArrays to concatenate
# @return MuSigmaArray that is concatenation of list
def ms_concatenate(msaList):
    muList = [msa.mu for msa in msaList]
    sigmaList = [msa.sigma for msa in msaList]
    # If any sigma is None, discard the rest
    if None in sigmaList:
        return MuSigmaArray(np.concatenate(muList), None)
    else:
        return MuSigmaArray(np.concatenate(muList), np.concatenate(sigmaList))

## Stack MuSigmaArrays horizontally.
# @param msaList List of MuSigmaArrays to stack
# @return MuSigmaArray that is horizontal stack of list
def ms_hstack(msaList):
    muList = [msa.mu for msa in msaList]
    sigmaList = [msa.sigma for msa in msaList]
    # If any sigma is None, discard the rest
    if None in sigmaList:
        return MuSigmaArray(np.hstack(muList), None)
    else:
        return MuSigmaArray(np.hstack(muList), np.hstack(sigmaList))

## Stack MuSigmaArrays vertically.
# @param msaList List of MuSigmaArrays to stack
# @return MuSigmaArray that is vertical stack of list
def ms_vstack(msaList):
    muList = [msa.mu for msa in msaList]
    sigmaList = [msa.sigma for msa in msaList]
    # If any sigma is None, discard the rest
    if None in sigmaList:
        return MuSigmaArray(np.vstack(muList), None)
    else:
        return MuSigmaArray(np.vstack(muList), np.vstack(sigmaList))

## Determine second-order statistics from data.
# Convenience function to return second-order statistics of data along given axis as a MuSigmaArray.
# @param data Numpy data array of arbitrary shape
# @param axis Index of axis along which stats are calculated (will be averaged away in the process) [0]
# @return MuSigmaArray containing data stats, of same dimension as data, but without given axis
def mu_sigma(data, axis=0):
    return MuSigmaArray(data.mean(axis=axis), data.std(axis=axis))

## Determine second-order statistics from data, using more robust order statistics.
# Convenience function to return second-order statistics of data along given axis as a MuSigmaArray.
# These are determined via the median and interquartile range.
# @param data Numpy data array of arbitrary shape
# @param axis Index of axis along which stats are calculated (will be averaged away in the process) [0]
# @return MuSigmaArray containing data stats, of same dimension as data, but without given axis
def robust_mu_sigma(data, axis=0):
    data = np.asarray(data)
    # Create sequence of axis indices with specified axis at the front, and the rest following it
    moveAxisToFront = range(len(data.shape))
    moveAxisToFront.remove(axis)
    moveAxisToFront = [axis] + moveAxisToFront
    # Create copy of data sorted along specified axis, and reshape so that the specified axis becomes the first one
    sortedData = np.sort(data, axis=axis).transpose(moveAxisToFront)
    # Obtain quartiles
    perc25 = sortedData[int(0.25 * len(sortedData))]
    perc50 = sortedData[int(0.50 * len(sortedData))]
    perc75 = sortedData[int(0.75 * len(sortedData))]
    # Conversion factor from interquartile range to standard deviation (based on normal pdf)
    iqrToStd = 0.741301109253
    return MuSigmaArray(perc50, iqrToStd * (perc75 - perc25))

## Determine second-order statistics of periodic (angular or directional) data.
# Convenience function to return second-order statistics of data along given axis as a MuSigmaArray.
# This handles periodic variables, which exhibit the problem of wrap-around and therefore are unsuited for
# the normal mu_sigma function. The period with which the values repeat can be explicitly specified,
# otherwise the data is assumed to be radians.
# The mean is in the range -period/2 ... period/2, and the maximum standard deviation is about period/4.
#
# Reference:
# Yamartino, R.J. (1984). "A Comparison of Several "Single-Pass" Estimators of the Standard Deviation 
# of Wind Direction". Journal of Climate and Applied Meteorology 23: 1362-1366.
# 
# @param data   Numpy data array of arbitrary shape, containing angles (typically in radians)
# @param axis   Index of axis along which stats are calculated (will be averaged away in the process) [0]
# @param period Period with which data values repeat [2.0 * np.pi]
# @return MuSigmaArray containing data stats, of same dimension as data, but without given axis
def periodic_mu_sigma(data, axis=0, period=2.0*np.pi):
    data = np.asarray(data, dtype='double')
    # Create sequence of axis indices with specified axis at the front, and the rest following it
    moveAxisToFront = range(len(data.shape))
    moveAxisToFront.remove(axis)
    moveAxisToFront = [axis] + moveAxisToFront
    # Create copy of data, and reshape so that the specified axis becomes the first one
    data = data.copy().transpose(moveAxisToFront)
    # Scale data so that one period becomes 2*pi, the natural period for angles
    scale = 2.0 * np.pi / period
    data *= scale
    # Calculate a "safe" mean on the unit circle
    mu = np.arctan2(np.sin(data).mean(axis=0), np.cos(data).mean(axis=0))
    deltaAng = data - mu
    # Wrap angle differences into interval -pi ... pi
    deltaAng = (deltaAng + np.pi) % (2.0 * np.pi) - np.pi
    # Calculate variance using standard formula with a second correction term
    sigma2 = (deltaAng ** 2.0).mean(axis=0) - (deltaAng.mean(axis=0) ** 2.0)
    # Scale answers back to original data range
    return MuSigmaArray(mu / scale, np.sqrt(sigma2) / scale)

#=========================================================================================================
#=== FUNCTIONS                                                                                         ===
#=========================================================================================================

#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  check_model_fit_agn
#-------------------------------------

## Check whether a data set fits a specific model with additive Gaussian noise.
#  A D'Agostino-Pearson "Goodness-of-Fit" test is used.
#
# Check if the following generative assumption holds for a given data set.
#
#     y = f(x) + n
#
# Here f() is an arbitrary function (specified as input parameter) and n is additive Gaussian (normal) noise
# with zero mean and specified standard deviation.
#
# @param      xMu               input random variable dataset (mean)
# @param      xSig              input random variable dataset (standard deviation)
# @param      y                 output random variable dataset
# @param      func              function f()
# @param      alpha             tail probability [0.05]
# @param      expResStd         tuple(expected residual standard deviation, degrees-of-freedom of estimate)
#                               If not specified, ignore this part of the test.
# @return     fit               boolean indicating if model fit data or not
# @return     pVal              p-value of fit
# @return     k2                D'Agostino-Pearson K2 statistic
# @return     chiSqT            Chi-Squared threshold value for given alpha
# @return     resMuCorrect      Is the residual zero mean?
# @return     resStdCorrect     If expResStd is specified, this return flag will indicate if the residual's std ise as
#                               expected
#
# @todo       Expected residual variance test should make use of correct statistical formulation. Currently it is too
#             simplistic.
#

def check_model_fit_agn(xMu=None, xSig=None, y=None, func=None, alpha=0.05, expResStd=None):
    
    # pylint: disable-msg=R0913,R0914
    
    # If X has any probabilistic spread we need to use a sigma-point approach to propagate the distribution
    if xSig:
        yPred = sp_stats(func, xMu, xSig**2)
    else:
        yPred = func(xMu)
    
    # Calculate difference between predicted and observed data (residuals)
    res = y - yPred
    
    # Now perform a D'Agostino-Pearson normality test on the residuals
    k2, pVal = sp.stats.normaltest(res)
    
    # Calculate Chi-square test threshold for given alpha value. k2 is distributed as Chi-Squared with 2 degrees
    # of freedom
    
    chiSqThreshold = sp.stats.chi2.isf(alpha, 2)
    
    resGaussian = k2 < chiSqThreshold           # Is res Gaussian?
    
    # Statistics of residual: We expect residual to be zero mean with std equal to expResStd (if specified)
    numSamples = len(res)
    resMu = res.mean()
    resStd = res.std()
    
    tVal = sp.stats.t.isf(alpha/2, numSamples-1)   # Student-T distribution critical value
    intVal = tVal * resStd/np.sqrt(numSamples)
    meanConfInterval = [resMu-intVal, resMu+intVal]
    
    resMuCorrect = (meanConfInterval[0] <= 0) and (meanConfInterval[1] >= 0)   # Is residual zero mean?
    
    if not expResStd:
        return resGaussian, pVal, k2, chiSqThreshold, resMuCorrect
    else:
        xDOF = numSamples -1
        yDOF = expResStd[1]
        fValXY = sp.stats.f.isf(alpha/2, xDOF, yDOF)
        fValYX = sp.stats.f.isf(alpha/2, yDOF, xDOF)
        resStdCorrect = ((resStd**2)/(expResStd[0]**2) <= fValXY) and ((expResStd[0]**2)/(resStd**2) <= fValYX)
        return resGaussian, pVal, k2, chiSqThreshold, resMuCorrect, resStdCorrect


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  propagate_scalar_stats
#--------------------------------------

## Propagate statistics through a function to determine the resultant stats. Wrapper for
# sp_stats function for the special case of scalar independent random variables.
# @todo Write a new wrapper around sp_stats so that externally we can pass in the lists of
#       means and covariances and get back the resultant means and covariances, ie hide the
#       vectorizing from the user.
# @param func the function that is going to be applied to the stats
# @param muX input mean vector
# @param sigmaX input standard deviation vector
# @return muY output mean vector
# @return sigmaY output standard deviation vector
def propagate_scalar_stats(func, muX, sigmaX):
    
    # NOTE that this can only be used like this because we have scalar and independent
    # random variables.
    muY, covY = sp_stats(func, muX, np.diag(sigmaX**2))
    
    muY = func.devectorize_output(muY)
    sigmaY = func.devectorize_output(np.sqrt(np.diag(covY)))
    
    return muY, sigmaY


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  sp_stats
#-------------------------

## Sigma-Point Statistics for general (correlated) RV.
#
# @param    func        function handle
# @param    muX         mean of input RV (random variable)
# @param    covX        covariance of input RV
# @param    h           CDT length [sqrt(3)]
#
# @return   (muY, covY) mean and covariance of output RV
#
# Note:     Y = func(X)
#
# For more detail regarding the mathematics behind this algorithm, please see:
#
#
# copyright    (c) 2004 Rudolph van der Merwe. All rights reserved.
# @author       Rudolph van der Merwe [rudolphv@gmail.com] on 2004-03-01.
# @todo Swap order of indices, ie use row vectors for means

# pylint: disable-msg=R0914

def sp_stats(func, muX, covX, h=np.sqrt(3)):
    
    muX = np.atleast_1d(muX)
    covX = np.atleast_1d(covX)
    
    L = muX.size
    N = 2*L + 1
    
    # Build sigma-point set
    X = np.zeros((L, 2*L+1), dtype='double')
    
    X[:, 0] = np.ravel(muX)
    
    if L == 1:
        S = np.atleast_2d(np.sqrt(covX))
    else:
        S = np.atleast_2d(np.linalg.cholesky(covX))
    
    muXX = muX[:, np.newaxis]
    X[:, 1:L+1] = muXX + h*S
    X[:, L+1:] = muXX - h*S
    
    # Weights
    hh = h**2
    wm = np.zeros(N)       # mean weights
    wm[0] = (hh - L)/hh
    wm[1:] = 1/(2*hh)
    wc1 = np.zeros(L)      # cov 1 weights
    wc2 = np.zeros(L)      # cov 2 weights
    wc1[:] = 1/(4*hh)
    wc2[:] = (hh-1)/(4*(hh**2))
    
    # Propagate sigma points
    Y = np.atleast_2d(func(X))
    
    muY = np.sum(wm[np.newaxis, :] * Y, 1)
    
    A = Y[:, 1:L+1] - Y[:, L+1:]
    B = Y[:, 1:L+1] + Y[:, L+1:] - 2*(Y[:, 0])[:, np.newaxis]
    
    covY = np.dot(wc1[np.newaxis, :]*A, A.transpose()) + np.dot(wc2[np.newaxis, :]*B, B.transpose())
    
    return muY, covY


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  sp_uncorrelated_stats
#--------------------------------------

## Sigma-Point Statistics for set of uncorrelated scalar random variables (RVs).
# This function propagates the statistics of a set of uncorrelated scalar RVs through an arbitrary function
# func, and returns the statistics of the corresponding output RVs. It is assumed that both the input and
# output RVs are uncorrelated with each other, allowing the use of a single mean and standard deviation value
# to characterise each RV. The function should have the signature Y = func(X), where X is an L-dimensional array
# (shape = (L,)) and Y is an M-dimensional array (shape=(M,)).
#
# @param    func      Function handle
# @param    muSigmaX  MuSigmaArray object containing mean and standard deviation vectors of input RVs (shape (L,))
# @param    h         CDT length [sqrt(3)]
# @return   muSigmaY  MuSigmaArray object containing mean and standard deviation vectors of output RVs (shape (M,))
#
# copyright (c) 2007 SKA/KAT. All rights reserved.
# @author Ludwig Schwardt <ludwig@ska.ac.za>
# @date 2007-09-19

# pylint: disable-msg=R0914

def sp_uncorrelated_stats(func, muSigmaX, h=np.sqrt(3)):
    # Extract mean and standard deviation
    if not isinstance(muSigmaX, MuSigmaArray):
        message = 'sp_uncorrelated_stats expects MuSigmaArray object for muSigmaX.'
        logger.error(message)
        raise TypeError, message
    muX = np.atleast_1d(muSigmaX.mu)
    # No sigma info allows a quick shortcut - only the mean is propagated
    if muSigmaX.sigma == None:
        return MuSigmaArray(func(muX), None)
    sigmaX = np.atleast_1d(muSigmaX.sigma)
    
    # Dimension of RV vector (number of scalar RVs)
    L = muX.size
    # Number of sigma points (which lie at +-sigma on each dimension, and one at the mean)
    N = 2*L + 1
    
    # Build sigma-point set
    X = np.zeros((N, L), dtype='double')
    X[0] = muX
    S = np.diag(sigmaX)
    muXX = muX[np.newaxis, :]
    X[1:L+1] = muXX + h*S
    X[L+1:] = muXX - h*S
    
    # Weights
    hh = h**2
    wm = np.zeros(N)       # mean weights
    wm[0] = (hh - L)/hh
    wm[1:] = 1/(2*hh)
    wc1 = np.zeros(L)      # sigma 1 weights
    wc1[:] = 1/(4*hh)
    wc2 = np.zeros(L)      # sigma 2 weights
    wc2[:] = (hh-1)/(4*(hh**2))
    
    # Propagate sigma points (the real work)
    Y = np.array([func(sigmaPoint) for sigmaPoint in X])
    # Calculate stats of output RVs
    muY = np.sum(wm[:, np.newaxis] * Y, axis=0)
    A = Y[1:L+1] - Y[L+1:]
    B = Y[1:L+1] + Y[L+1:] - 2*Y[0][np.newaxis, :]
    sigmaY = np.sqrt(np.sum(wc1[:, np.newaxis]*A*A + wc2[:, np.newaxis]*B*B, axis=0))
    
    return MuSigmaArray(muY, sigmaY)


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  calc_conf_interval_diff2means
#-----------------------------------------------

## Calculate confidence interval for the difference of two means using a Student-t test.
#  (See pg. 305 of Hogg & Tanis, "Probability and Statistical Inference")
#
# @note all parameters must have the same dimensions.
#
# @param    a1      sample stats (DistributionStatsArray) of first distribution
# @param    a2      sample stats (DistributionStatsArray) of second distribution
# @param    alpha   significance level between 0 and 1 (typically 0.05)
#
# @return   deltaMeanConfInterval   measured confidence intervals
# @return   r                       number of degrees of freedom of the Student-t distribution
#                                   used for hypothesis test

def calc_conf_interval_diff2means(a1, a2, alpha):
    
    # Check that all inputs have same dimensions
    dimSet = set([a1.shape, a2.shape])
    assert(len(dimSet) == 1)
    dimShape = misc.tuple_append(2, a1.shape)
    
    # First we estimate the r-value of the Student-t distrubution (the number of degrees
    # of freedom)
    r = np.floor((a2.var/a2.num + a1.var/a1.num)**2 / \
                 ((1/(a2.num-1)) * (a2.var/a2.num)**2 + \
                  (1/(a1.num-1)) * (a1.var/a1.num)**2))
    # Now use this to calculate the confidence interval
    t_alpha_over_2_r = np.abs(sp.stats.t.isf(1-alpha/2, r))
    cDel = t_alpha_over_2_r * np.sqrt(a2.var/a2.num + a1.var/a1.num)
    deltaMeanConfInterval = np.zeros(dimShape, 'double')
    deltaMean = a1.mean - a2.mean
    deltaMeanConfInterval[0] = deltaMean - cDel
    deltaMeanConfInterval[1] = deltaMean + cDel
    
    return deltaMeanConfInterval, r


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  calc_conf_interval_diff_diff2means
#----------------------------------------------------

## Calculate linearity of bands
# (See pg. 305 of Hogg & Tanis, "Probability and Statistical Inference") using a Student-t test.
# This checks that the difference between the first two distributions is the same as the
# difference between the second two distributions.
#
# @param  a1    sample stats (DistributionStatsArray) of first distribution or first group
# @param  a2    sample stats (DistributionStatsArray) of first distribution or second group
# @param  b1    sample stats (DistributionStatsArray) of second distribution or first group
# @param  b2    sample stats (DistributionStatsArray) of second distribution or second group
# @param  alpha Student-T test alpha value
#
# @return deltaMeanConfInterval     measured confidence intervals
# @return r                         number of degrees of freedom of the Student-t distribution
#                                   used for hypothesis test

def calc_conf_interval_diff_diff2means(a1, a2, b1, b2, alpha):
    
    # Check that all inputs have same dimensions
    dimSet = set([a1.shape, a2.shape, b1.shape, b2.shape])
    assert(len(dimSet) == 1)
    dimShape = misc.tuple_append(2, a1.shape)
    
    # Calculate confidence interval for deltaPhot - deltaPcold
    # First we estimate the r-value of the Student-t distrubution (the number of degrees of freedom)
    r = np.floor((a2.var/a2.num + a1.var/a1.num + b2.var/b2.num + b1.var/b1.num)**2 / \
                 ( (1/(a2.num-1)) * (a2.var/a2.num)**2 + (1/(a1.num-1)) * (a1.var/a1.num)**2 + \
                   (1/(b2.num-1)) * (b2.var/b2.num)**2 + (1/(b1.num-1)) * (a1.var/b1.num)**2 ) )
    # Now use this to calculate the confidence interval
    t_alpha_over_2_r = np.abs(sp.stats.t.isf(1-alpha/2, r))
    cDel = t_alpha_over_2_r * np.sqrt(a2.var/a2.num + a1.var/a1.num + b2.var/b2.num + b1.var/b1.num)
    
    deltaDeltaMean = (b2.mean - b1.mean) - (a2.mean - a1.mean)
    
    deltaDeltaMeanConfInterval = np.zeros(dimShape, 'double')
    deltaDeltaMeanConfInterval[0] = deltaDeltaMean - cDel
    deltaDeltaMeanConfInterval[1] = deltaDeltaMean + cDel
    
    return deltaDeltaMeanConfInterval, r


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  check_equality_of_means
#-----------------------------------------

## Check if two means are equal by ensuring that the confidence interval straddles 0
#
# @param deltaDeltaMeanConfInterval output of calc_conf_interval_diff_diff2means
#
# @return boolean result for each interval in array

def check_equality_of_means(deltaDeltaMeanConfInterval):
    
    return (deltaDeltaMeanConfInterval[0] <= 0.0) & (deltaDeltaMeanConfInterval[1] >= 0.0)


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  interpolate_noise_diode_profile
#------------------------------------------------

## Calculate interpolated noise diode temperatures at a list of desired frequencies
#  given known values.
#
# @param    knownFreqs      list of known frequencies
# @param    knownTemps      list of known temperatures in same order as knownFreqs
# @param    desiredFreqs    list of desired frequencies for temperatures
#
# @return   list of desired temperatures corresponding to desiredFreqs

def interpolate_noise_diode_profile(knownFreqs, knownTemps, desiredFreqs):
    knownFreqs = np.atleast_1d(knownFreqs)
    knownTemps = np.atleast_1d(knownTemps)
    desiredFreqs = np.atleast_1d(desiredFreqs)
    assert(len(knownFreqs.shape) == 1)
    assert(knownFreqs.shape == knownTemps.shape)
    assert(knownFreqs.size > 0)
    if knownFreqs.size < 4:
        return np.repeat(knownTemps.mean(), desiredFreqs.size)
    elif (knownFreqs.size == 4):
        polynomialDegree = 0
    else:
        polynomialDegree = 1
    p1 = np.polyfit(knownFreqs, knownTemps, deg=polynomialDegree)
    return np.polyval(p1, desiredFreqs)
