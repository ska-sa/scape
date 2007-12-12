## @file gain_cal_data.py
#
# Classes that contain gain calibration and noise diode data.
#
# copyright (c) 2007 SKA/KAT. All rights reserved.
# @author Ludwig Schwardt <ludwig@ska.ac.za>
# @date 2007-11-12

import xdmsbe.xdmsbelib.fitting as fitting
import xdmsbe.xdmsbelib.single_dish_data as sdd
from conradmisclib.transforms import deg_to_rad
import numpy as np
import logging

logger = logging.getLogger("xdmsbe.xdmsbelib.gain_cal_data")

#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  NoiseDiodeData
#----------------------------------------------------------------------------------------------------------------------

## A container for noise diode calibration data.
# This allows the (randomised) calculation of the noise diode temperature from the tables stored in a FITS file.
# pylint: disable-msg=R0903
class NoiseDiodeData(object):
    ## Initialiser/constructor.
    # Extract noise diode calibration data from FITS file, which includes the power spectrum of the diode and the 
    # coupling strength between the diode and the specific feed's input ports. This pre-calculates an interpolation
    # function for the coupling strength, which is considered constant, while the power spectrum is only interpolated
    # when the actual noise diode temperature is requested.
    # @param self       The current object
    # @param fitsReader Object used for reading from a FITS file (already initialised to a specific file)
    def __init__(self, fitsReader):
        primaryHeader = fitsReader.get_primary_header()
        ## @var noiseDiodeType
        # Type of noise diode (typically 0=injected, 1=floodlight)
        self.noiseDiodeType = int(primaryHeader['NdType'])
        ## @var feedId
        # Feed ID number (typically 0=main feed, 1=offset feed)
        self.feedId = int(primaryHeader['FeedID'])
        
        try:
            spectrumTable = fitsReader.get_hdu_data('ND_Spectrum')
        except KeyError:
            logger.error('No noise diode power spectrum calibration data found!')
            raise RuntimeError
        ## @var spectrumFreqs_Hz
        # Sequence of frequencies at which noise diode power spectrum is characterised, in Hz
        self.spectrumFreqs_Hz = np.array(spectrumTable.field('Freq'), dtype = 'double')
        ## @var spectrumTemp_K
        # Noise diode power spectrum measurements, as temperature in K
        self.spectrumTemp_K = np.array(spectrumTable.field('Temp'), dtype = 'double')
        
        try:
            couplingTable = fitsReader.get_hdu_data('ND_Coupling')
        except KeyError:
            logger.error('No noise diode coupling calibration data found!')
            raise RuntimeError
        # Sequence of frequencies at which coupling between noise diode and feed is characterised, in Hz
        couplingFreqs_Hz = np.array(couplingTable.field('Freq'), dtype = 'double')
        # Sequence of rotator angles at which coupling between noise diode and feed is characterised, in radians
        couplingRotAngs_rad = deg_to_rad(np.array(couplingTable.field('RotAngle'), dtype = 'double'))
        # Array of coupling strengths between noise diode and feed's X and Y inputs, in dB, 
        # as a function of frequency and rotator angle
        couplingGainX_dB = np.array(couplingTable.field('B%dP1amp' % (self.feedId)), dtype = 'double')
        couplingGainY_dB = np.array(couplingTable.field('B%dP2amp' % (self.feedId)), dtype = 'double')
        # Structure of FITS file requires all columns in a table to have the same length
        # Frequency and angle vectors typically have different lengths - they should also match sides of gain array
        # Enforce this relationship here (this removes any zero padding induced by FITS table restriction)
        couplingFreqs_Hz = couplingFreqs_Hz[:couplingGainX_dB.shape[0]]
        couplingRotAngs_rad = couplingRotAngs_rad[:couplingGainX_dB.shape[1]]
        # Extend range of angles a little to each side, to ensure good interpolation wherever angles wrap around
        couplingRotAngs_rad = np.concatenate((couplingRotAngs_rad[-2:] - 2.0*np.pi, couplingRotAngs_rad, 
                                              couplingRotAngs_rad[:2] + 2.0*np.pi))
        couplingGainX_dB = np.hstack((couplingGainX_dB[:, -2:], couplingGainX_dB, couplingGainX_dB[:, :2]))
        couplingGainY_dB = np.hstack((couplingGainY_dB[:, -2:], couplingGainY_dB, couplingGainY_dB[:, :2]))
        
        # Pre-fit interpolation functions for coupling strength on a regular grid, as this is considered to be constant
        ## @var couplingInterpX
        # Interpolation function for coupling strength between noise diode and X input of feed, as a function
        # of frequency and rotator angle
        self.couplingInterpX = fitting.Spline2DGridFit()
        self.couplingInterpX.fit((couplingFreqs_Hz, couplingRotAngs_rad), couplingGainX_dB)
        ## @var couplingInterpY
        # Interpolation function for coupling strength between noise diode and Y input of feed, as a function
        # of frequency and rotator angle
        self.couplingInterpY = fitting.Spline2DGridFit()
        self.couplingInterpY.fit((couplingFreqs_Hz, couplingRotAngs_rad), couplingGainY_dB)
    
    ## Obtain noise diode temperature.
    # Obtain interpolated noise diode temperature at desired frequencies and rotator angles. Optionally, randomise
    # the smooth fit to the noise diode power spectrum, to provide some confidence info as part of a larger
    # Monte Carlo iteration. The function returns a (2,F,A)-shaped array, where the first dimension represents
    # the feed input ports (X and Y polarisations), F is the number of frequencies and A the number of angles.
    # @param self       The current object
    # @param freqs_Hz   Frequency (or sequence of frequencies) at which to evaluate temperature, in Hz
    # @param rotAng_rad Rotator angle (or sequence of angles) at which to evaluate temperature, in rads
    # @param randomise  True if noise diode spectrum smoothing should be randomised [False]
    # @return Noise diode temperature as an array of shape (2,F,A), with F number of freqs and A number of angles
    def temperature(self, freqs_Hz, rotAng_rad, randomise=False):
        # Fit a smooth polynomial to noise diode power spectrum measurements, with optional perturbation
        # The polynomial degree was determined by eye-balling the fit to the "gold standard" noise diode data
        spectrumInterp = fitting.Polynomial1DFit(maxDegree = 20)
        spectrumInterp.fit(self.spectrumFreqs_Hz, self.spectrumTemp_K)
        if randomise:
            spectrumInterp = fitting.randomise(spectrumInterp, self.spectrumFreqs_Hz, self.spectrumTemp_K, 'shuffle')
        # Evaluate the smoothed spectrum at the desired frequencies
        smoothSpectrum = spectrumInterp(freqs_Hz)
        # Evaluate the coupling strength on the grid of frequency and angle values, and convert from dB to linear
        coupling = np.array([self.couplingInterpX((freqs_Hz, rotAng_rad)), \
                             self.couplingInterpY((freqs_Hz, rotAng_rad))])
        coupling = np.power(10.0, coupling / 10.0)
        # Combine spectrum with coupling strengths, for each input port (X and Y)
        return smoothSpectrum[np.newaxis, :, np.newaxis] * coupling

#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  GainCalibrationData
#----------------------------------------------------------------------------------------------------------------------

## A container for gain calibration data obtained from power measurements.
# This allows the (randomised) calculation of the power-to-temperature conversion function.
# pylint: disable-msg=R0902,R0903
class GainCalibrationData(object):
    ## Initialiser/constructor
    # Extract adjacent "noise diode on" and "noise diode off" power data blocks from given list. Make sure data
    # is in coherency form, and extract the relevant second-order statistics (mean and stdev). Also store the
    # rotator angle and timestamp of each pair of "on-off" blocks.
    # @param self           The current object
    # @param powerBlockDict Dictionary containing blocks of power data used for gain calibration
    def __init__(self, powerBlockDict):
        # Find pairs of "noise diode off" and "noise diode on" blocks (with same esn and data id)
        noiseDiodeOffLabels = [k for k in powerBlockDict.iterkeys() if k.endswith("On")]
        noiseDiodeOnLabels = [k for k in powerBlockDict.iterkeys() if k.endswith("OnND")]
        # Collect power data blocks in pairs
        deltaPairs = [{'off' : powerBlockDict[off], 'on' : powerBlockDict[on]} \
                      for off in noiseDiodeOffLabels for on in noiseDiodeOnLabels if on[:-2] == off]
        if len(deltaPairs) == 0:
            logger.error('No noise diode on+off pairs found - cannot do gain calibration!')
            raise RuntimeError
        
        ## @var timeSamples
        # List of averaged time instants, close to the centre of each on-off pair
        self.timeSamples = [np.mean((np.median(pair['off'].timeSamples), np.median(pair['on'].timeSamples))) \
                            for pair in deltaPairs]
        ## @var rotAng_rad
        # List of averaged rotator angles, one for each on-off pair
        self.rotAng_rad = [np.mean((np.median(pair['off'].rotAng_rad), np.median(pair['on'].rotAng_rad))) \
                           for pair in deltaPairs]
        ## @var freqs_Hz
        # List of channel frequencies where power is measured, in Hz
        self.freqs_Hz = deltaPairs[0]['off'].freqs_Hz
        
        # Only XX and YY info are used for calibration, as this is all that is known about noise diode
        diodeOffData = [np.array([pair['off'].coherency('XX'), pair['off'].coherency('YY')]) \
                        for pair in deltaPairs]
        diodeOnData = [np.array([pair['on'].coherency('XX'), pair['on'].coherency('YY')]) \
                       for pair in deltaPairs]
        ## @var diodeOffMean
        # Mean of coherencies when noise diode is off, as an array of shape (numOnOffPairs, 2, numFreqs)
        self.diodeOffMean = np.array([x.mean(axis=1) for x in diodeOffData])
        ## @var diodeOnMean
        # Mean of coherencies when noise diode is on, as an array of shape (numOnOffPairs, 2, numFreqs)
        self.diodeOnMean = np.array([x.mean(axis=1) for x in diodeOnData])
        ## @var diodeOffSigma
        # Standard deviation of estimated mean coherencies when noise diode is off, as an array of shape 
        # (numOnOffPairs, 2, numFreqs). This is the spread on the final result of the "off" block - the
        # the estimated mean power in the block. Since the estimated mean of data is less variable than
        # the data itself, we have to divide the data sigma by sqrt(N).
        self.diodeOffSigma = np.array([(x.std(axis=1) / np.sqrt(x.shape[1])) for x in diodeOffData])
        ## @var diodeOnSigma
        # Standard deviation of estimated mean coherencies when noise diode is on, as an array of shape 
        # (numOnOffPairs, 2, numFreqs). This is the spread on the final result of the "on" block - the
        # the estimated mean power in the block. Since the estimated mean of data is less variable than
        # the data itself, we have to divide the data sigma by sqrt(N).
        self.diodeOnSigma = np.array([(x.std(axis=1) / np.sqrt(x.shape[1])) for x in diodeOnData])
        
        ## @var badChannels
        # List of channel indices of channels than are "uncalibratable", because the noise diode data is
        # suspect on either polarisation or in any on-off pair. The noise diode data is disregarded if the
        # lowest power values with the noise diode switched on is conceivably lower than the highest values
        # with the diode switched off.
        self.badChannels = [ind for ind in range(self.diodeOnMean.shape[2]) \
                            if np.any(self.diodeOnMean[:, :, ind] - 3.0*self.diodeOnSigma[:, :, ind] < \
                                      self.diodeOffMean[:, :, ind] + 3.0*self.diodeOffSigma[:, :, ind])]
    
    ## Obtain power-to-temperature conversion function.
    # Calculate a power-to-temperature conversion factor (Fpt) for each stored noise-diode-on-off pair,
    # based on temperature of given noise diode. Interpolate these Fpt factors as a function of time
    # (to compensate for amplifier gain drifts), which becomes the power-to-temperature conversion function.
    # @param self       The current object
    # @param noiseDiode Noise diode characteristics
    # @param maxDegree  Maximum polynomial degree for power-to-temperature interpolation [1]
    # @param randomise  True if noise diode characteristics and power measurements should be randomised [False]
    # @return Power-to-temperature conversion function, as a function of time which returns a 
    # (4, numTimes, numFreqs)-shaped array (polarisation type by time by frequency) for a sequence of time instants
    # pylint: disable-msg=R0914
    def power_to_temp_func(self, noiseDiode, maxDegree=1, randomise=False):
        diodeOffPower = self.diodeOffMean.copy()
        diodeOnPower = self.diodeOnMean.copy()
        # Perturb it if required
        if randomise:
            diodeOffPower += self.diodeOffSigma * np.random.standard_normal(diodeOffPower.shape)
            diodeOnPower += self.diodeOnSigma * np.random.standard_normal(diodeOnPower.shape)
        deltaPower = diodeOnPower - diodeOffPower
        # Force delta power to be positive, otherwise nonsensical negative (or infinite) temperatures may result
        goodChannels = list(set.difference(set(range(len(self.freqs_Hz))), set(self.badChannels)))
        if np.any(deltaPower[:, :, goodChannels] <= 0.0):
            logger.warning('Some delta power values are negative or zero during gain calibration, reset to 1e-20...')
        deltaPower[deltaPower <= 0.0] = 1e-20
        # Obtain noise diode temperature for each channel frequency and the rotator angle of each on-off block
        # Transpose to get a (2, numBlocks, numFreqs) array
        noiseDiodeTemp = noiseDiode.temperature(self.freqs_Hz, self.rotAng_rad, randomise).transpose((0, 2, 1))
        # Calculate Fpt factors for X and Y input ports (XX and YY polarisations), of shape (numBlocks, numFreqs)
        powerToTempX = noiseDiodeTemp[0] / deltaPower[:, 0, :]
        powerToTempY = noiseDiodeTemp[1] / deltaPower[:, 1, :]
        # Extend these factors to full polarisation set (XX, YY, XY and YX) to form (4, numBlocks, numFreqs) array
        powerToTempFactors = np.zeros((4, len(self.rotAng_rad), len(self.freqs_Hz)))
        polIndex = sdd.power_index_dict(False)
        powerToTempFactors[polIndex['XX']] = powerToTempX
        powerToTempFactors[polIndex['YY']] = powerToTempY
        powerToTempFactors[polIndex['XY']] = np.sqrt(powerToTempX) * np.sqrt(powerToTempY)
        powerToTempFactors[polIndex['YX']] = powerToTempFactors[polIndex['XY']]
        # Power-to-temp conversion factor is inverse of gain, which is assumed to change linearly over time
        powerToTempFunc = fitting.Independent1DFit(fitting.ReciprocalFit( \
                          fitting.Polynomial1DFit(maxDegree=maxDegree)), axis=1)
        # Obtain interpolation function as a function of time
        powerToTempFunc.fit(np.array(self.timeSamples), powerToTempFactors)
        return powerToTempFunc
