## xdmsbelib/misc.py
#
# General utility routines used by XDM software backend.
#
# @author Richard Lord [rlord@ska.ac.za] & Rudolph van der Merwe [rudolph@ska.ac.za] on 2007-01-11.
# @copyright (c) 2007 SKA/KAT. All rights reserved.

from __future__ import division
import xdmsbe.xdmpyfits as pyfits
from conradmisclib.transforms import deg_to_rad
from optparse import OptionParser
import numpy as np
import numpy.linalg as linalg
import scipy.stats as sp_stats
import logging
import logging.config
import os.path
import csv
import re
import sys
import ConfigParser
import simcor as sc

logger = logging.getLogger("xdmsbe.xdmsbelib.misc")

# pylint: disable-msg=C0103

#=======================================================================================================
#--- CONSTANTS -----------------------------------------------------------------------------------------
#=======================================================================================================

c = lightSpeed = 299792458.0       # Speed of light [m/s] (exact)
omegaEarthRad  = 7.2921158553e-5   # Earth's rotation rate [rad/s] WGS84
boltzmannK     = 1.3806505e-23     # Boltzmann constant [J/K]
planckH        = 6.6260755e-34     # Planck constant [J s]
Jy             = 1e-26             # Radio flux density [W / m^2 / Hz]
AU             = 1.49597870691e11  # Astronomical unit [m]
lightyear      = 9.460536207e15    # Light year [m]
parsec         = 3.08567802e16     # Parsec [m]
siderealYear   = 365.2564          # Sidereal year [days]
tropicalYear   = 365.2422          # Tropical year [days]
gregorianYear  = 365.2425          # Gregorian year [days]
earthMass      = 5.9736e24         # Mass of Earth [kg]
earthRadius    = 6371e3            # Radius of Earth [m]
earthGM        = 398600.4415e9     # G * mass_earth   [m^3/s^2]
radiusEquator  = 6378.137e3        # WGS 84, IAG-GRS 80
radiusPolar    = 6356.75231425e3   # WGS 84, IAG-GRS 80
sunMass        = 1.9891e30         # Mass of Sun [kg]
sunRadius      = 6.96265e8         # Radius of Sun [m]
sunLuminosity  = 3.827e26          # Sun luminosity [W]
gravConstantG  = 6.6726e-11        # Gravitational constant [m^3 / (kg s^2)

# Stokes to coherency vector transformation matrix
stokesToCoherency = 0.5*np.array([[  1,   1,  0,  0],  \
                                  [  0,   0,  1,  1j], \
                                  [  0,   0,  1, -1j], \
                                  [  1,  -1,  0,  0]], 'complex128')

# Coherency vector to Stokes transformation matrix
coherencyToStokes =     np.array([[  1,   0,   0,  1],  \
                                  [  1,   0,   0, -1],  \
                                  [  0,   1,   1,  0],  \
                                  [  0,  -1j, +1j, 0]], 'complex128')


#=======================================================================================================
#--- FUNCTIONS -----------------------------------------------------------------------------------------
#=======================================================================================================

#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  config_logging
#---------------------------------------------------------------------------------------------------------

## Configure the logging support to either use the default (built-in) or configure using an external
# loggin config file
#
# @param logConfFile Logging configuration file [None]. If not specified, the default is used.
def config_logging(logConfFile=None):
    if logConfFile:
        try:
            logging.config.fileConfig(logConfFile)
        except ConfigParser.NoSectionError:
            message = "Logging configuration file not found or wrong format! Using built-in default."
            config_logging()
            logger.error(message)
    else:
        logging.basicConfig(level=logging.INFO,
                            stream=sys.stdout,
                            format="%(asctime)s - %(name)s - %(filename)s:%(lineno)s - %(levelname)s - %(message)s")


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  reduction_script_options
#---------------------------------------------------------------------------------------------------------

## Option parser for all reduction scripts.
# This parses the standard options for each reduction script, and returns the main argument (the FITS filename).
# @param    expType      Experiment type
# @param    disabledOpts String indicating which options are disabled for this experiment ['', meaning none]
def reduction_script_options(expType, disabledOpts=''):
    
    # parse command line options
    expUsage = "usage: %prog [options] FITS_file"
    parser = OptionParser(expUsage, description = "Data reduction backend for '" + expType + "' experiment")
    if 'a' not in disabledOpts:
        parser.add_option('-a', '--alpha', action='store', type='float', dest='alpha', \
                          default=0.001, help="Alpha value for statistical tests")
    if 'c' not in disabledOpts:
        parser.add_option('-c', '--config', dest='logConfigFile', action='store', type='string', default=None, \
                          metavar='LOGFILE', help='use LOGFILE for logging configuration')
    if 'l' not in disabledOpts:
        parser.add_option('-l', '--linearity', action = 'store_true', dest = 'testLinearity', \
                          default=False, help="Do linearity test")
    if 'p' not in disabledOpts:
        parser.add_option('-p', '--showplots', action='store_true', dest='showPlots', \
                          default=False, help="Display plots")
    if 's' not in disabledOpts:
        parser.add_option('-s', '--save', action='store_true', dest='saveResults', \
                          default=False, help="Save all output to file(s)")
    if 't' not in disabledOpts:
        parser.add_option('-t', '--tar', action='store_true', dest='tarResults', \
                          default=False, help="Tarball all output file(s)")
    if 'v' not in disabledOpts:
        parser.add_option('-v', '--verbose', action='store_true', dest='verbose', \
                          default=False, help="Display data reduction results on stdout")
    if 'z' not in disabledOpts:
        parser.add_option('-z', '--zip', action='store', type='string', dest='zipTar', default='none', \
                          help="Compress tarball, valid values: 'bz2', 'gz' 'none'. Default is 'none'")
    
    opts, args = parser.parse_args()
    
    if len(args) != 1:
        parser.print_help()
        sys.exit(1)
    
    # Configure logging
    config_logging(opts.logConfigFile)
    
    try:
        fitsFileName = args[0]
    except IndexError:
        message = "You must provide a FITS filename as argument."
        logger.error(message)
        raise IndexError, message
    
    return opts, args, fitsFileName


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  experiment_label
#---------------------------------------------------------------------------------------------------------

## Extracts experiment label from first part of FITS filename.
# This assumes that the FITS filename has the structure experimentlabel_xxxx.fits, where xxxx is a set
# of digits indicating the order of the file in the FITS set.
# @param     fileName  FITS filename
# @return    expLabel  Experiment label
#
def experiment_label(fileName):
    
    fileRoot = os.path.splitext(os.path.basename(fileName))[0]
    lastUnderscore = fileRoot.rfind('_')
    if lastUnderscore >= 0:
        return fileRoot[:lastUnderscore]
    else:
        return fileRoot


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  fmt_seq_num
#---------------------------------------------------------------------------------------------------------

## Format a sequence number to always have n digits, e.g.
#        fmt_seq_num(2, 3)  -> 002
#        fmt_seq_num(12, 3) -> 012
#        fmt_seq_num(5, 4)  -> 0005
# @param     seqNum    sequence number
# @param     digits    number of digits in returned sequence string
# @return    seqStr    sequence string
#
def fmt_seq_num(seqNum, digits):
    
    assert seqNum < 10**digits
    seqStr = str(seqNum)
    seqStr = seqStr.rjust(digits, '0')
    return seqStr


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  load_fits
#---------------------------------------------------------------------------------------------------------

## Open and verify FITS file
#
# @param     fileName     FITS file name
# @param     hduNames     Set of required HDU names. If not specified, no verificaiton will be done.
#
# @return    hduList      header-data-unit list

def load_fits(fileName, hduNames=None):
    try:
        hdulist = pyfits.open(fileName, memmap=False)        # open FITS file and extract HDU list
        try:
            hdulist.verify(option='exception')
        except pyfits.VerifyError:
            hdulist.close()
            logger.error('File does not comply with FITS standard.')
    except IOError, e:
        logger.error("FITS file '%s' does not exist." % fileName)
        raise e
    
    if hduNames:
        # Build list of missing HDU names
        hduNamesPresent = set([x.name for x in hdulist])
        hduNamesMissing = set.difference(hduNames, hduNamesPresent)
        
        # Report missing HDU's
        if len(hduNamesMissing):
            hdulist.close()
            message = "Missing HDUs : %s" % str(hduNamesMissing)[5:-2]
            logger.error(message)
            raise ValueError, message
    
    return hdulist


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  extract_svg_from_fits
#---------------------------------------------------------------------------------------------------------

## Extract SVG image from FITS file
#
# @param     fileName       input FITS file name
# @param     hduList        input HDU list (handle to already open FITS file)
# @param     svgFileName    Output filename for SVG file. If none specified, no file is written.
# @param     svgHduName     Name of HDU containing SVG data [default = SVG]
#
# @return    svgCharData    List of chars containing the SVG data
#
# Specify input as either an input FITS filename OR a HDU list.
#

def extract_svg_from_fits(fileName = False, hduList = False, svgFileName = False, svgHduName = "SVG"):
    
    if fileName:
        hduList = load_fits(fileName)
    
    if hduList:
        
        try:
            svgHDU = hduList[svgHduName]
        except KeyError:
            logger.error("FITS file does not contain SVG HDU.")
        
        svgData = svgHDU.data
        
        if svgFileName:
            try:
                svgData.tofile(svgFileName)
            except IOError:
                logger.error("I/O Error writing SVG file")
        
        try:
            svgCharData = list(svgData.field('CHAR'))
        except KeyError:
            logger.error("SVG table did not contain a column named CHAR as expected")
    
    return svgCharData



#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  hdr_keyval_copy
#---------------------------------------------------------------------------------------------------------

## Copy a set of key-value pairs from one FITS HDU header to another
#
# @param     hdrS       HDU header (source)
# @param     hdrD       HDU header (destination)
# @param     keyList    list of keys (and their values) to copy
#
# @return    hdr        updated HDU header
# @todo Copy comment in loop

def hdr_keyval_copy(hdrS, hdrD, keyList):
    
    keyValueCommentList = [card.ascardimage() for card in hdrS.ascardlist()]
    sourceKeyList = [val[0] for val in keyValueCommentList]
    
    for key in keyList:
        if key in sourceKeyList:
            hdrD.update(key, hdrS[key], '')
        else:
            logger.warn("Key %s not found in list" % key)
    
    return hdrD


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  set_or_check
#---------------------------------------------------------------------------------------------------------

## Set a dictionary key to val if it is not yet set, otherwise check for equality with val.
#
# @param     dic    input dictionary
# @param     key    dictionary key to set/check
# @param     val    value to set/check dic[key] against
#
# @return    True or False

def set_or_check(dic, key, val):
    return np.all(val == dic.setdefault(key, val))


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  set_or_check_param_continuity
#---------------------------------------------------------------------------------------------------------

## Set parameter if it doesn't exist in dictionary; otherwise, check if it equals given value.
# If the parameter paramName does not yet exist in workDict, it is added to the dictionary with
# value paramValue. Otherwise, its existing value is checked against paramValue, and a ValueError
# exception is raised if they differ. It is assumed that each run of this function refers to a
# different FITS file. Same as set_or_check(), with FITS-specific error messages on a failed check.
# @param workDict   dictionary of current values
# @param paramName  name of the parameter
# @param paramValue new value for the parameter
# pylint: disable-msg=C0103
def set_or_check_param_continuity(workDict, paramName, paramValue):
    if not(set_or_check(workDict, paramName, paramValue)):
        message = "Parameter %s changed from one FITS file to the next." % paramName
        logger.error(message)
        raise ValueError, message


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  tuple_append
#---------------------------------------------------------------------------------------------------------

## Extend a shape tuple by appending x to it
# @param tuple1 first tuple
# @param tuple2 second tuple
# @return tuple containing concatenation of two input tuples
def tuple_append(tuple1, tuple2):
    try:
        tempList = list(tuple1)
    except TypeError:
        tempList = [tuple1]
    try:
        tempList.extend(tuple2)
    except TypeError:
        tempList.append(tuple2)
    return tuple(tempList)


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  dict_concatenate
#---------------------------------------------------------------------------------------------------------

## Concantenate a list of dictionaries into a single dictionary. We assume there is no duplicate keys
#
# @param    dictList    List of dictionaries
# @return   dictOut     new concatenated dictionary

def dict_concatenate(dictList):
    dictOut = {}
    for d in dictList:
        dictOut.update(d)
    return dictOut


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION : dict_regex_select
#---------------------------------------------------------------------------------------------------------

##  Select a subset of a dictionary's (key, value) pairs based on a regular expression based key.
#
# @param     inDict          The dictionary object
# @param     regexStr      The regular expression to serve as key
# @return    outDict       Subset dictionary based on selection
def dict_regex_select(inDict, regexStr):
    outDict = {}
    for (key, val) in inDict.items():
        if len(re.findall(regexStr, key)):
            outDict[key] = val
    return outDict


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  a_matrix_sqrt
#---------------------------------------------------------------------------------------------------------

## Obtains matrix square root S of positive semi-definite Hermitian square matrix A so that S * S' = A.
# The input matrix A should be square, Hermitian and positive semi-definite (i.e. the eigenvalues of A should
# all be real and non-negative). This is "a" matrix square root, as matrix factorisations are not unique.
# This function uses eigenvalue decomposition to factorise A = U * D * U', and then forms the square root as
# S = U * sqrt(D) * U'.
#
# @param a Input matrix (should be square, Hermitian, and positive semi-definite)
# @return s Matrix which is square root of a, so that numpy.dot(s, s.conj().transpose()) == a
def a_matrix_sqrt(a):
    d, u = np.linalg.eig(np.asarray(a))
    # If a has negative or imaginary eigenvalues, quit
    tol = np.abs(max(d)) * 1e-13
    if np.any(d < 0) or np.any(np.abs(d.imag) > tol):
        raise np.linalg.linalg.LinAlgError, "Matrix has negative or imaginary eigenvalues - cannot take square root."
    return np.dot(np.dot(u, np.diag(np.sqrt(d.real))), u.conj().transpose())


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :    construct_stokes
#---------------------------------------------------------------------------------------------------------

## Converts comprehendable specification of polarised electromagnetic wave to equivalent Stokes parameters.
#
#  @param    stokesI        Stokes parameter I, which is the total power.
#  @param    posAngle_deg   Position angle, in degrees. For linear and elliptical polarisations, this is the 
#                           orientation of the electromagnetic wave in the plane perpendicular to the direction 
#                           of propagation.
#  @param    fracLinPol     The fraction of the wave's power that is linearly polarised.
#  @param    fracCircPol    The fraction of the wave's power that is circularly polarised. The sign of this
#                           quantity indicates the handedness of the polarisation (plus is left-handed, minus 
#                           is right-handed, as per the IEEE definition).
#
#  @return   stokes         Stokes parameter representation of the specified wave.
def construct_stokes(stokesI, posAngle_deg, fracLinPol, fracCircPol):
    #Note: TotFracPol.^2 = FracLinPol.^2 + FracCircPol.^2
    stokesQ = fracLinPol * stokesI * np.cos(2.0 * deg_to_rad(posAngle_deg))
    stokesU = fracLinPol * stokesI * np.sin(2.0 * deg_to_rad(posAngle_deg))
    stokesV = fracCircPol * stokesI
    return np.array([stokesI, stokesQ, stokesU, stokesV])


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  randn_complex
#---------------------------------------------------------------------------------------------------------

## Generates a 2-D matrix of independent zero-mean, complex Gaussian random variables.
#
# @param    numSamples    Number of samples (number of rows in output)
# @param    dim           Dimension of random variables (number of columns in output)
# @param    complexType   Data type of samples (default = 'complex128')
# @param    power         Total signal power (default = 1.0)
# @param    whiten        Flag indicating whether data should be whitened by its sample covariance matrix
# @return   data          Generated samples, as an array of shape (numSamples, dim)
def randn_complex(numSamples, dim, complexType='complex128', power=1.0, whiten=True):
    data = np.ndarray((numSamples, dim), dtype=complexType)
    scaleFact = np.sqrt(power / dim / 2.0)
    data.real = np.random.normal(0.0, scale=scaleFact, size=(numSamples, dim))
    data.imag = np.random.normal(0.0, scale=scaleFact, size=(numSamples, dim))
    #P = np.zeros((dim, dim), dtype='complex128')
    if whiten:
        pass
        # for k in np.arange(numSamples):
        #     P = P + np.outer(sigBuf[:, k], sigBuf[:, k].conj())
        # P = (1 / (numSamples - 1)) * P
        # A = np.linalg.inv(np.linalg.cholesky(P)) # Whiten the data
        # sigBuf = np.dot(A, sigBuf)
    #sigBuf *= scale
    return data


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  rand_complex_wishart
#---------------------------------------------------------------------------------------------------------

## Generates a sequence of random matrices from the complex Wishart distribution.
# The complex Wishart distribution is the distribution of the scatter matrix of a sequence of "degsFreedom"
# independent zero-mean complex Gaussian RVs with dimension "dim" and covariance matrix "covMat". It is
# therefore also the distribution of the estimate of the covariance matrix of the underlying multivariate 
# Gaussian RV. It is a multivariate version of the chi-squared distribution. The function generates a
# sequence of "numSamples" square (dim x dim) nonnegative-definite Hermitian matrices. If covMat is None,
# a standard underlying Gaussian RV is assumed. If it is required that the Wishart RVs have a mean of
# exactly covMat, divide the output by (2.0 * degsFreedom).
#  
# References (taken from Matlab function wishrnd):
# Krzanowski, W.J. (1990), Principles of Multivariate Analysis, Oxford.
# Smith, W.B., and R.R. Hocking (1972), "Wishart variate generator,"
#  Applied Statistics, v. 21, p. 341.  (Algorithm AS 53)
#  Available in Fortran at http://lib.stat.cmu.edu/apstat/53
#
# @param    numSamples    Number of samples
# @param    dim           Dimension of square random matrices
# @param    degsFreedom   Wishart degrees of freedom (number of Gaussians added together)
# @param    covMat        Covariance matrix of underlying Gaussian RVs (proportional to Wishart mean)
# @param    complexType   Data type of samples (default = 'complex128')
# @return   data          Generated samples, as an array of shape (numSamples, dim, dim)
def rand_complex_wishart(numSamples, dim, degsFreedom, covMat=None, complexType='complex128'):
    # First create a standardised central Wishart RV
    data = np.zeros((numSamples, dim, dim), dtype=complexType)
    # For small degrees of freedom, follow the direct (quadratic form) approach
    if (degsFreedom <= 81 + dim) and (degsFreedom == int(degsFreedom)):
        for sampleInd in xrange(numSamples):
            stdWishartSqrt = randn_complex(degsFreedom, dim, complexType=complexType, power=2.0*dim)
            data[sampleInd] = np.dot(stdWishartSqrt.transpose(), stdWishartSqrt.conj())
    # Otherwise use the Smith & Hocking method (also known as Algorithm AS 53 from apstat)
    else:
        # Load diagonal elements with square root of chi-square variates
        for d in xrange(dim):
            data[:, d, d] = sp_stats.chi.rvs(2*degsFreedom - 2*d, size=numSamples)
        # Obtain all lower triangle indices, and load them with standard complex Gaussian RVs
        for sampleInd in xrange(numSamples):
            lower = np.tri(dim, k=-1).ravel() > 0
            data[sampleInd].ravel()[lower] = randn_complex(dim*(dim-1)//2, 1, complexType=complexType, power=2.0*dim)
            data[sampleInd] = np.dot(data[sampleInd], data[sampleInd].conj().transpose())
           
    # Apply desired underlying covariance matrix (hopefully it's well behaved...)
    if covMat != None:
        sqrtCovMat = a_matrix_sqrt(covMat)
        for sampleInd in xrange(numSamples):
            data[sampleInd] = np.dot(sqrtCovMat, np.dot(data[sampleInd], sqrtCovMat.conj().transpose()))
    # Clean up diagonals
    for d in xrange(dim):
        data[:, d, d].imag = 0.0
        
    return data


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  gen_std_correlator_samples
#---------------------------------------------------------------------------------------------------------

## Generate standardised random samples simulating the output of a single-dish correlator.
# Each sample is a 4-dimensional coherency vector, of the form [rxx, rxy, ryx, ryy]. The underlying signal
# is assumed to be totally unpolarised with total power of two, which corresponds to a Stokes vector 
# of [2, 0, 0, 0]. The mean values of rxx and ryy are therefore 1.0 each (unity power per polarisation channel).
# The resulting samples can be transformed to any desired power value and polarisation by multiplying them with
# the appropriate matrix. The more accurate 'wishart' method samples from the true distribution of the correlator
# outputs, while the 'normal' method is an approximation, but hopefully faster. The 'wishart' method returns
# complex-valued samples, while 'normal' returns real-valued samples.
#
# @param numSamples           Number of samples to generate
# @param numSamplesIntegrated Number of voltage samples that were averaged into one power sample by correlator
# @param method               Method to use (slower but more accurate 'wishart', or faster 'normal' (default))
# @return data                Generated samples, as an array of shape (numStokes, numSamples)
def gen_std_correlator_samples(numSamples, numSamplesIntegrated, method='normal'):
    
    numStokes = 4
    if method == 'wishart':
        data = rand_complex_wishart(numSamples, 2, numSamplesIntegrated) / 2.0 / numSamplesIntegrated
        data = data.reshape(numSamples, numStokes).transpose()
    else:
        # The old way of simulating received power (polarisation info bad, but faster)
        # Simulate power samples via normally-distributed voltages
        data = np.random.normal(1.0, 1.0 / np.sqrt(2.0 * numSamplesIntegrated), (numStokes, numSamples)) ** 2.0
        # Reduce XY and YX data to virtually zero (assume unpolarised source)
        data[1, :] *= 1e-10
        data[2, :] *= 1e-10
        
    return data


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  gen_polarized_power
#---------------------------------------------------------------------------------------------------------

## Generate power measurements that have a specified polarisation characteristic. The power measurements
# are simulated as they would be generated by an integrative correlator.
#
# @param    numPowerSamples                 dimension of random variables
# @param    numVoltSamplesPerIntPeriod      number of samples integrated
# @param    S                               2x2 matrix consisting of Jones matrices and coherency-to-Stokes matrix
# @param    desiredTotalPower               desired total power of generated signal
# @param    outputFormat                    'stokes' or 'coherency' vectors
# @return   sigBuf                          data sample buffer
# pylint: disable-msg=W0102,R0914

def gen_polarized_power(numPowerSamples, numVoltSamplesPerIntPeriod, S, desiredTotalPower=1.0, outputFormat='stokes'):
    
    linR = np.array(S.ravel().real, dtype='double')
    linI = np.array(S.ravel().imag, dtype='double')
    
    xx = np.zeros((numPowerSamples), dtype='complex128')
    xy = np.zeros((numPowerSamples), dtype='complex128')
    yx = np.zeros((numPowerSamples), dtype='complex128')
    yy = np.zeros((numPowerSamples), dtype='complex128')
    
    xxR = np.array(xx.real, dtype='double')
    xxI = np.array(xx.imag, dtype='double')
    xyR = np.array(xy.real, dtype='double')
    xyI = np.array(xy.imag, dtype='double')
    yxR = np.array(yx.real, dtype='double')
    yxI = np.array(yx.imag, dtype='double')
    yyR = np.array(yy.real, dtype='double')
    yyI = np.array(yy.imag, dtype='double')
    
    success = sc.simcor(long(numPowerSamples), long(numVoltSamplesPerIntPeriod), float(desiredTotalPower), 
                        linR, linI, xxR, xxI, xyR, xyI, yxR, yxI, yyR, yyI)
    if success:
        xx.real = xxR
        xx.imag = xxI
        xy.real = xyR
        xy.imag = xyI
        yx.real = yxR
        yx.imag = yxI
        yy.real = yyR
        yy.imag = yyI
        sigBuf = np.concatenate([xx[np.newaxis, :], xy[np.newaxis, :], yx[np.newaxis, :], yy[np.newaxis, :]], 0)
    else:
        logger.error("Correlator simulator did not run successfully!")
    
    if (outputFormat.lower() == 'stokes'):
        sigBuf = np.dot(coherencyToStokes, sigBuf)

    return sigBuf


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  calc_ant_eff_area_and_gain
#---------------------------------------------------------------------------------------------------------

## Calculate the effective area and gain of an antenna
#
# @param    pointSourceSens     antenna point source sensitivity as a function of frequency (one per band) [Jy/K]
# @param    freq                center frequency of bands
# @return   effArea             effective area per frequency band [m^2]
# @return   antGain             antenna gain per frequency band [dB]
def calc_ant_eff_area_and_gain(pointSourceSens, freq):
    effArea = boltzmannK / (pointSourceSens*Jy)
    waveLength = lightSpeed / freq
    antGain = (4.0 * np.pi * effArea) / (waveLength**2)
    return effArea, 10.0 * np.log10(antGain)


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION : unwrap_angles
#---------------------------------------------------------------------------------------------------------

## Unwrap a sequence of angles so that they do not straddle the 0 / 360 or -180 / 180 degree boundaries.
# This assumes that the range of angle values is less than 180 degrees. If the angles are within 90 degrees
# of 180/-180 (i.e. in the range (-180,-90) U (90,180)), they are remapped to the range (90, 270). On the other 
# hand, if the angles are within 90 degrees of 0/360 (i.e. in the range (0,90) U (270,360)), they are remapped 
# to the range (-90, 90).
#
# @param    angles              Sequence of angle values, in degrees
# @return   Modified sequence of angle values that avoids any angle boundaries causing wrap-around
def unwrap_angles(angles):
    angles = np.atleast_1d(np.asarray(angles))
    # This assumes that the angles wrap at +180 and -180, with some positive and negative angles
    if np.all((angles < -90.0) + (angles > 90.0)) and (angles.min() < -90.0) and (angles.max() > 90.0):
        angles = angles % 360.0
    # This assumes that the angles wrap at 0 and 360, with some angles at either end
    elif np.all((angles < 90.0) + (angles > 270.0)) and (angles.min() < 90.0) and (angles.max() > 270.0):
        angles = (angles + 180.0) % 360.0 - 180.0
    return angles


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  gaussian_ellipses
#---------------------------------------------------------------------------------------------------------

## Determine ellipse(s) representing 2-D Gaussian function.
#
# @param    mean            2-dimensional mean vector
# @param    cov             2x2 covariance matrix
# @param    contour         Contour height of ellipse(s), as a (list of) factor(s) of the peak value [0.5]
#                           For a factor sigma of the standard deviation, use e^(-0.5*sigma^2) here
# @param    numPoints       Number of points on each ellipse [200]
# @return   ellipses        Array of dimension len(contour) x numPoints x 2, containing 2-D ellipse coordinates

def gaussian_ellipses(mean, cov, contour=0.5, numPoints=200):

    mean = np.asarray(mean)
    cov = np.asarray(cov)
    contour = np.atleast_1d(np.asarray(contour))
    if (mean.shape != (2,)) or (cov.shape != (2, 2)):
        message = 'Mean and covariance should be 2-dimensional, with shapes (2,) and (2,2) instead of' \
                  + str(mean.shape) + ' and ' + str(cov.shape)
        logger.error(message)
        raise ValueError, message
    # Create parametric circle
    t = np.linspace(0, 2*np.pi, numPoints)
    circle = np.vstack((np.cos(t), np.sin(t)))
    # Determine and apply transformation to ellipse
    eigVal, eigVec = linalg.eig(cov)
    circleToEllipse = np.dot(eigVec, np.diag(np.sqrt(eigVal)))
    baseEllipse = np.real(np.dot(circleToEllipse, circle))
    ellipses = []
    for cnt in contour:
        ellipse = np.sqrt(-2*np.log(cnt)) * baseEllipse + mean[:, np.newaxis]
        ellipses.append(ellipse.transpose())
    return np.array(ellipses)


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  read_spectrum_analyzer_data
#---------------------------------------------------------------------------------------------------------

## Function to read noise diode profile measured by spectrum analyzer
# @param filename the noise diode profile filename 
#
def read_spectrum_analyzer_data(filename):
    # pylint: disable-msg=R0914

    try:
        fileHandle = open(filename, 'r')
    except IOError:
        msg = "Filename (%s) does not exist or cannot be read" % filename
        logger.error(msg)
        sys.exit(1)
    
    lineList = [line for line in csv.reader(fileHandle, delimiter=';')]

    # Close file
    fileHandle.close()

    valuesInd = [ind for ind in xrange(len(lineList)) if lineList[ind][0] == 'Values']
    if len(valuesInd) == 0:
        logger.error("Could not find 'Values' field in %s" % filename)
        sys.exit(1)
    
    headerEnd = valuesInd[0]
    header = dict([line[:2] for line in lineList[:headerEnd]])
    data = np.array([[np.double(line[0]), np.double(line[1])] for line in lineList[headerEnd+1:]])
    freqs_Hz, power_dBm = data[:, 0], data[:, 1]
    
    # Convert power to temperature
    resBW = np.double(header['RBW'])
    mW_to_W = 1e3
    temp_K = np.power(10.0, power_dBm / 10.0) / mW_to_W / (boltzmannK * resBW)
    
    return freqs_Hz, temp_K


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  read_coupling_data
#---------------------------------------------------------------------------------------------------------

## Function to read coupling parameter data
# @param filename the filename of the coupling parameter file
#
def read_coupling_data(filename):
    # pylint: disable-msg=R0912

    try:
        fileHandle = open(filename, 'r')
    except IOError:
        msg = "Filename (%s) does not exist or cannot be read" % filename
        logger.error(msg)
        sys.exit(1)
    
    # Strip empty lines at beginning
    line = fileHandle.readline()
    while line.strip().split() == []:
        line = fileHandle.readline()
    
    # Strip comments and empty lines
    lineList = line.strip().split()
    while lineList[0] == '%':
        line = fileHandle.readline()
        while line.strip().split() == []:
            line = fileHandle.readline()
        lineList = line.strip().split()
    
    # Read number of frequencies and number of rotation stage angles
    while lineList[0] != 'Number':
        line = fileHandle.readline()
        while line.strip().split() == []:
            line = fileHandle.readline()
        lineList = line.strip().split()
    if len(lineList) == 4:
        if (lineList[0] == 'Number') & (lineList[2] == 'frequencies:'):
            numFrequencies = np.long(lineList[3])
    line = fileHandle.readline()
    lineList = line.strip().split()
    while lineList[0] != 'Number':
        line = fileHandle.readline()
        while line.strip().split() == []:
            line = fileHandle.readline()
        lineList = line.strip().split()
    if len(lineList) == 6:
        if (lineList[0] == 'Number') & (lineList[2] == 'rotation'):
            numRotations = np.long(lineList[5])
    
    # Read rotator angle values
    while lineList[0] != 'Angle:':
        line = fileHandle.readline()
        while line.strip().split() == []:
            line = fileHandle.readline()
        lineList = line.strip().split()
    rotAngle_deg = np.array([float(s) for s in lineList[1:]])
    
    # Read until 'Frequency' keyword
    while lineList[0] != 'Frequency':
        line = fileHandle.readline()
        while line.strip().split() == []:
            line = fileHandle.readline()
        lineList = line.strip().split()
    line = fileHandle.readline()
    lineList = line.strip().split()
    
    # Initialise
    coupleFreq = np.zeros(numFrequencies)
    coupleAmpl = np.zeros((numRotations, numFrequencies))
    couplePhase = np.zeros((numRotations, numFrequencies))
    
    # Read data
    for i in range(numFrequencies):
        lineList = line.strip().split()
        coupleFreq[i] = np.float(lineList[0]) * 1e6
        for k in range(numRotations):
            coupleAmpl[k, i] = np.float(lineList[1 + (k*2)])
            couplePhase[k, i] = np.float(lineList[2 + (k*2)])
        line = fileHandle.readline()

    # Close file
    fileHandle.close()

    return rotAngle_deg, coupleFreq, coupleAmpl, couplePhase


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  read_cal_data
#---------------------------------------------------------------------------------------------------------

## Function to read calibration FITS file data (noise diode and floodlight illuminator)
# @param opts option parser options
#
def read_cal_data(opts):
    
    # Read data
    freqs_Hz, temp_K = read_spectrum_analyzer_data(opts.measured)
    rotAngleB0p1_deg, coupleFreqB0p1, coupleAmplB0p1, couplePhaseB0p1 = read_coupling_data(opts.b0p1)
    rotAngleB0p2_deg, coupleFreqB0p2, coupleAmplB0p2, couplePhaseB0p2 = read_coupling_data(opts.b0p2)
    rotAngleB1p1_deg, coupleFreqB1p1, coupleAmplB1p1, couplePhaseB1p1 = read_coupling_data(opts.b1p1)
    rotAngleB1p2_deg, coupleFreqB1p2, coupleAmplB1p2, couplePhaseB1p2 = read_coupling_data(opts.b1p2)
    
    # Check if data is consistent
    if not ((rotAngleB0p1_deg == rotAngleB0p2_deg).all() and (rotAngleB0p1_deg == rotAngleB1p1_deg).all() and
        (rotAngleB0p1_deg == rotAngleB1p2_deg).all()):
        msg = "Rotation angles are not consistent among coupling data files"
        logger.error(msg)
        sys.exit(1)
    if not ((coupleFreqB0p1 == coupleFreqB0p2).all() and (coupleFreqB0p1 == coupleFreqB1p1).all() and
        (coupleFreqB0p1 == coupleFreqB1p2).all()):
        msg = "Frequency values are not consistent among coupling data files"
        logger.error(msg)
        sys.exit(1)
    
    return freqs_Hz, temp_K, rotAngleB0p1_deg, coupleFreqB0p1, coupleAmplB0p1, couplePhaseB0p1, coupleAmplB0p2, \
           couplePhaseB0p2, coupleAmplB1p1, couplePhaseB1p1, coupleAmplB1p2, couplePhaseB1p2
    


    



