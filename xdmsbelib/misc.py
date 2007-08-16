## xdmsbelib/misc.py
#
# General utility routines used by XDM software backend.
#
# @author Richard Lord [rlord@ska.ac.za] & Rudolph van der Merwe [rudolph@ska.ac.za] on 2007-01-11.
# @copyright (c) 2007 SKA/KAT. All rights reserved.

from __future__ import division
import xdmsbe.xdmpyfits as pyfits
import numpy as np
import logging
import re

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
siderealYear   = 365.2564          # idereal year [days]
tropicalYear   = 365.2422          # ropical year [days]
gregorianYear  = 365.2425          # regorian year [days]
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
stokes2coherencyMatrix = 0.5*np.array([[  1,   1,  0,  0], \
                                       [  0,   0,  1,  1j], \
                                       [  0,   0,  1, -1j], \
                                       [  1,  -1,  0,  0]],'complex128')

coherency2stokesMatrix = np.array([[1, 0, 0, 1],  \
                                   [1, 0, 0, -1], \
                                   [0, 1, 1, 0], \
                                   [0, -1j, +1j, 0]], 'complex128')

#=======================================================================================================
#--- FUNCTIONS -----------------------------------------------------------------------------------------
#=======================================================================================================

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
#--- FUNCTION :  get_power
#---------------------------------------------------------------------------------------------------------

## Calculate and return XDM power measurements
# @param     numChannels           number of channels
# @param     numIntegrationBins    number of samples that have been integrated together
# @param     temperature           received power expressend in temperature [K]
# @param     channelBandwidth      channel bandwidth [Hz]
# @param     rxGain                receiver gain [dB]
# @return    power                 2-D array of measured power [W]
# pylint: disable-msg=R0913
#
def get_power(numChannels, numIntegrationBins, temperature, channelBandwidth, rxGain):
    np.random.seed()
    numSamples = len(temperature)
    pMean  =  np.array(temperature) * (boltzmannK * channelBandwidth * pow(10.0, rxGain / 10.0))
    pSigma = pMean * (np.sqrt(2.0) / np.sqrt(numIntegrationBins))
    vMean  = np.sqrt(pMean)
    vSigma = (0.5 / np.sqrt(pMean)) * pSigma
    power = np.random.normal(vMean.repeat(numChannels).reshape(numSamples, numChannels), 
                             vSigma.repeat(numChannels).reshape(numSamples, numChannels), 
                             (numSamples, numChannels)) ** 2.0

#    power = vMean**2 * np.ones((numSamples, numChannels), dtype = 'float')
#    vMean2, vCov = spStats(np.sqrt, pMean, (pSigma ** 2.0), h=np.sqrt(3))
#    print vMean, vMean2
#    print vSigma, np.sqrt(vCov)
#    print

    return power


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
        logger.error("FITS file '%s' does not exist. Exiting!" % fileName)
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
#-----------------------------

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
#----------------------------------------

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


#---------------------------------------------------------------------------------------------
#--- FUNCTION : dict_regex_select 
#--------------------------------

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
#--- FUNCTION :  default_usage
#---------------------------------------------------------------------------------------------------------

## Common experiment usage parameters for option parser
#
# @param    parser    parser object of type OptionParser
def default_usage(parser):
    parser.add_option('-v', '--verbose', action='store_true', dest='verbose', 
                      default=False, help="Display data reduction results on stdout")
    parser.add_option('-p', '--showplots', action='store_true', dest='showPlots', 
                      default=False, help="Display Tsys plot")
    parser.add_option('-s', '--save', action='store_true', dest='saveResults', 
                      default=False, help="Save all output to file(s)")
    parser.add_option('-t', '--tar', action='store_true', dest='tarResults', 
                      default=False, help="Tarball all output file(s)")
    parser.add_option('-z', '--zip', action='store', type='string', dest='zipTar', 
                      default='none', 
                      help="Compress tarball, valid values: 'bz2', 'gz' 'none'. Default is 'none'")

#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  randn_complex
#---------------------------------------------------------------------------------------------------------

## Return a new array of zeros mean, complex Gaussian random variables of a given shape
# specified by tuple shape and a given power level (per )
#
# @param    dim           dimension of random variables
# @param    numSamples    number of samples
# @param    complexType   Data type of samples (default = 'complex128')
# @param    power         total signal power
# @param    whiten        flag indicating whether data should be whitened by its sample covariance matrix
# @return   sigBuf        data sample buffer
def randn_complex(dim, numSamples, complexType='complex128', power=1, whiten=True):
    shapeTuple = (dim, numSamples)
    sigBuf = np.ndarray(shapeTuple, dtype=complexType)
    scale = np.sqrt(power / dim)
    sigBuf.real = np.random.standard_normal(shapeTuple)
    sigBuf.imag = np.random.standard_normal(shapeTuple)
    P = np.zeros((dim, dim), dtype='complex128')    
    if whiten:
        for k in np.arange(numSamples):
            P = P + np.outer(sigBuf[:, k], sigBuf[:, k].conj())
        P = (1 / (numSamples - 1)) * P
        A = np.linalg.inv(np.linalg.cholesky(P)) # Whiten the data
        sigBuf = np.dot(A, sigBuf)
    sigBuf *= scale
    return sigBuf
    

#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  gen_polarized_power
#---------------------------------------------------------------------------------------------------------

## Generate power measurements that has a specified polarization characteristic. The power measurments 
# are simulated as they would be generated by an integrative correlator.
#
# @param    numPowerSamples                 dimension of random variables
# @param    numVoltSamplesPerIntPeriod      number of samples
# @param    desiredTotalPower               desired total power of generated signal
# @param    stokesVector                    Desired Stokes parameters of polarized signal  ( default=[1,0,0,0] )
# @param    outputFormat                    'stokes' or 'coherency' vectors             ()
# @return   sigBuf                          data sample buffer
# pylint: disable-msg=W0102,R0914
def gen_polarized_power(numPowerSamples, numVoltSamplesPerIntPeriod, desiredTotalPower=1, 
                        stokesVector=[1, 0, 0, 0], outputFormat='stokes'):
    
    RVec = np.dot(stokes2coherencyMatrix, np.array(stokesVector)) # Coherency vector 
    Rxx = RVec[0]
    Rxy = RVec[1]
    Ryx = RVec[2]
    Ryy = RVec[3]
    P = np.array([[Rxx, Rxy], [Ryx, Ryy]],'complex128')  # Correlation matrix for given Stokes parameters
    # Have to check for matrices which are non-positive definite
    if ((P[0, 1]==0) and (P[1, 0]==0)) and ((P[0, 0]==0) or (P[1, 1]==0)):
        S = np.sqrt(P)
    else:
        S = np.linalg.cholesky(P)    
    SS = np.array(np.outer(S, np.transpose(np.conjugate(S))), dtype='complex128')     # Complex Tensor product of S
        
    e_u_pseudo_int = np.zeros((4, numPowerSamples), dtype='complex128')
    scale = 1 / (4 * numVoltSamplesPerIntPeriod)
    e_u_pseudo_int[0, :] = scale * np.random.chisquare(2*numVoltSamplesPerIntPeriod, (numPowerSamples))
    e_u_pseudo_int[3, :] = scale * np.random.chisquare(2*numVoltSamplesPerIntPeriod, (numPowerSamples))
    
    sigBuf = np.dot(SS, e_u_pseudo_int)
    
    totalPower = np.mean(np.abs(sigBuf[0, :]) + np.abs(sigBuf[3, :]))
     
    sigBuf *= (desiredTotalPower / totalPower)
    
    if outputFormat == 'stokes':
        sigBuf = np.dot(coherency2stokesMatrix, sigBuf)
    
    return sigBuf


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  calc_ant_eff_area_and_gain
#----------------------------------------------------

## Calculate the effective area and gain of an antenna
#
# @param    pointSourceSens     antenna point source sensitivity as a function of frequency (one per band) [Jy/K] 
# @param    freq                center frequency of bands
# @return   effArea             effective area per frequency band [m^2]
# @return   antGain             antenna gain per frequency band [dB]
def calc_ant_eff_area_and_gain(pointSourceSens, freq):
    effArea = boltzmannK / (pointSourceSens*Jy)
    waveLength = lightSpeed / freq
    antGain = (4 * np.pi * effArea) / (waveLength**2)
    return effArea, 10*np.log10(antGain)