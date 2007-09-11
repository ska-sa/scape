## xdmsbelib/stats.py
#
# statistics routines used by XDM software backend.
#
# copyright (c) 2007 SKA/KAT. All rights reserved.
# @author Rudolph van der Merwe [rudolph@ska.ac.za] on 2007-03-13.

# pylint: disable-msg=C0103

from __future__ import division
import numpy as np
import scipy as sp
sp.pkgload('stats')
import misc
import logging

logger = logging.getLogger("xdmsbe.xdmsbelib.stats")

#=======================================================================================================
#=== FUNCTIONS 
#=======================================================================================================

#-------------------------------------------------------------------------------------------------------
#--- FUNCTION :  check_model_fit_agn
#-------------------------------------

## Check whether a data set fits a specific model with additive Gaussian noise.
#  A D'Agostino-Pearson "Goodness-of-Fit" test is used. 
#
# Check if the following generative assumption holds for a given data set.
#
#     y = f(x) + n
#
# Here f() is an arbitray function (specified as input parameter) and n is additive Gaussian (normal) noise
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

## Sigma-Point Statistics
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


#---------------------------------------------------------------------------------------------------------------------
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



#---------------------------------------------------------------------------------------------------------------------
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



#---------------------------------------------------------------------------------------------------------------------
#--- FUNCTION :  check_equality_of_means
#-----------------------------------------

## Check if two means are equal by ensuring that the confidence interval straddles 0
#
# @param deltaDeltaMeanConfInterval output of calc_conf_interval_diff_diff2means
#
# @return boolean result for each interval in array

def check_equality_of_means(deltaDeltaMeanConfInterval):
    
    return (deltaDeltaMeanConfInterval[0] <= 0.0) & (deltaDeltaMeanConfInterval[1] >= 0.0)


#---------------------------------------------------------------------------------------------------------------------
#--- FUNCTION :  interpolate_tipping_curve
#------------------------------------------------

## Calculate interpolated tippinbg curve system temperatures at a list of desired elevation angles
#  given known values.
#
# @param    knownElAngs      array of elevation angles (in rad) at which tipping curve is known
# @param    knownFreqs       array of band frequencies at which tipping curve is known    
# @param    knownTsys        2D array of known system temperatures, indexed by knownElAngs x knownFreqs
# @param    desiredElAngs    array of desired elevation angles (in rad) for system temperatures
# @param    desiredFreqs     array of desired frequencies for system temperatures
#
# @return   interpolatedTsys 2D array of interpolated system temperatures (interpolated tipping curve)

def interpolate_tipping_curve(knownElAngs, knownFreqs, knownTsys, desiredElAngs, desiredFreqs):
    
    import scipy.sandbox.delaunay as delaunay
    
    freqScale = np.std(np.ravel(knownFreqs))
    angleScale = np.std(np.ravel(knownElAngs)) 
    
    # Scale variables to comparable ranges
    kElAngs = knownElAngs / angleScale
    dElAngs = desiredElAngs / angleScale
    kFreqs = knownFreqs / freqScale
    dFreqs = desiredFreqs / freqScale
    
    # Get 2D interpolation grid
    yi = np.tile(dFreqs[np.newaxis, :], (len(dElAngs), 1))
    xi = np.tile(dElAngs[:, np.newaxis], (1, len(dFreqs)))    
    
    y = np.tile(kFreqs[np.newaxis, :], (len(kElAngs), 1))
    x = np.tile(kElAngs[:, np.newaxis], (1, len(kFreqs)))
        
    # triangulate data
    tri = delaunay.Triangulation(np.ravel(x), np.ravel(y))
    
    # interpolate data
    interp = tri.nn_interpolator(np.ravel(knownTsys))
    
    interpolatedTsys = interp(xi, yi)
    
    # knownElAngs = np.atleast_1d(knownElAngs)
    # knownTsys = np.atleast_1d(knownTsys)
    # desiredElAngs = np.atleast_1d(desiredElAngs)
    # assert(len(knownElAngs.shape) == 1)
    # assert(knownElAngs.shape == knownTsys.shape)
    # assert(knownElAngs.size > 5)
    # polynomialDegree = 4    
    # p1 = np.polyfit(knownElAngs, knownTsys, deg=polynomialDegree)
    # return np.polyval(p1, desiredElAngs)
    
    return interpolatedTsys


#---------------------------------------------------------------------------------------------------------------------
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



#---------------------------------------------------------------------------------------------------------------------
#--- FUNCTION :  calc_power_stats

## Calculate statistics for a particular block of power data looping over bands and stokes.
#  RFI channels are excluded.
#
# @param    bandNoRfiChannelList   list of non-rfi channels per band
# @param    numStokes              number of stokes dimensions
# @param    dataObj                fitsreader.SelectedPower object containing list of power arrays[time][channel], 
#                                  one per stokes dimension
#
# @return   DistributionStatsArray

def calc_power_stats(bandNoRfiChannelList, numStokes, dataObj):
    statsArray = DistributionStatsArray(shape=(numStokes, len(bandNoRfiChannelList)))
    for b, bandChannels in enumerate(bandNoRfiChannelList):
        for s in xrange(numStokes):
            data = (dataObj.powerData[s])[:, bandChannels]
            timeStamps = dataObj.timeSamples
            statsArray.add_data(data, timeStamps, (s, b))
    return statsArray



#=====================================================================================================================
#===  CLASSES  
#=====================================================================================================================

#-----------------------------------------------------------------------------------------------------------------
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

#-----------------------------------------------------------------------------------------------------------------
#--- CLASS :  MultiVariateGaussian
#------------------------------------

## Multi variate gaussian class
# pylint: disable-msg=R0903
class MultiVariateGaussian(object):
    
    ## Initialiser/Constructor
    # @param self   the current object
    # @param mu     mean vector
    # @param cov    covariance matrix
    def __init__(self, mu, cov):
        self._mu = mu
        self._cov = cov
    



#-----------------------------------------------------------------------------------------------------------------
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
    


