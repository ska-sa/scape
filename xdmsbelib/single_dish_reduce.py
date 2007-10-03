## @file single_dish_reduce.py
#
# Routines for reducing single-dish data.
#
# copyright (c) 2007 SKA/KAT. All rights reserved.
# @author Ludwig Schwardt <ludwig@ska.ac.za>, Rudolph van der Merwe <rudolph@ska.ac.za>,
#         Robert Crida <robert.crida@ska.ac.za>
# @date 2007-08-28

# pylint: disable-msg=C0103,R0902

import xdmsbe.xdmsbelib.fitsreader as fitsreader
import xdmsbe.xdmsbelib.interpolator as interpolator
from xdmsbe.xdmsbelib import tsys
from xdmsbe.xdmsbelib import stats
import xdmsbe.xdmsbelib.misc as misc
from acsm.coordinate import Coordinate
import acsm.transform
import numpy as np
import numpy.random as random
import logging
import copy

logger = logging.getLogger("xdmsbe.xdmsbelib.single_dish_reduce")

#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  StandardSourceScan
#----------------------------------------------------------------------------------------------------------------------

## Container for the components of a standard scan across a source.
# This struct contains all the parts that make up a standard scan:
# - information about the source (name and position)
# - information about the antenna (beamwidth)
# - the main scan segment to be calibrated
# - powerToTemp conversion info, as derived from gain cal data at the start and end of the scan
# - the actual powerToTemp function used to calibrate power measurements for whole scan
# - scan segments used to fit a baseline
# - the baseline function that will be subtracted from main scan segment
# pylint: disable-msg=R0903
class StandardSourceScan(object):
    ## @var sourceName
    # Name of source in scan (useful for obtaining source flux density)
    sourceName = None
    ## @var sourceAzAng_deg
    # Source azimuth angle in degrees
    sourceAzAng_deg = None
    ## @var sourceElAng_deg
    # Source elevation angle in degrees
    sourceElAng_deg = None
    ## @var antennaBeamwidth_deg
    # Antenna half-power beamwidth in degrees
    antennaBeamwidth_deg = None
    ## @var mainData
    # SingleDishData object for main scan segment
    mainData = None
    ## @var powerToTempTimes
    # Array of times where Fpt factors were measured
    powerToTempTimes = None
    ## @var powerToTempFactors
    # Array of Fpt factors, per time, band and polarisation type
    powerToTempFactors = None
    ## @var powerToTempFunc
    # Interpolated power-to-temperature conversion function (Fpt as a function of time)
    powerToTempFunc = None
    ## @var baselineDataList
    # List of SingleDishData objects that contain empty sky around a source and will be used for baseline fitting
    baselineDataList = None
    ## @var baselineUsesElevation
    # True if baseline is fit to elevation angle (preferred), False if azimuth angle is used instead
    baselineUsesElevation = None
    ## @var baselineFunc
    # Interpolated baseline function (power as a function of angle)
    baselineFunc = None

#----------------------------------------------------------------------------------------------------------------------
#--- FUNCTIONS
#----------------------------------------------------------------------------------------------------------------------

## Loads a list of standard scans across various point sources from a chain of FITS files.
# @param fitsFileName      Name of initial file in FITS chain
# @param alpha             Alpha value for statistical tests used in gain calibration
# @return stdScanList      List of StandardSourceScan objects, one per scan through a source
# @return rawPowerScanList List of SingleDishData objects, containing copies of all raw data blocks
# pylint: disable-msg=R0914,R0915
def load_point_source_scan_list(fitsFileName, alpha):
    fitsIter = fitsreader.FitsIterator(fitsFileName)
    workDict = {}
    stdScanList = []
    rawPowerScanList = []
    print "                 **** Loading scan data from FITS files ****\n"
    while True:
        try:
            stdScan = StandardSourceScan()
            fitsReaderPreCal = fitsIter.next()
            
            print "==============================================================================================\n"
            print "                             **** SCAN %d ****\n" % len(stdScanList)
            
            #..................................................................................................
            # Extract the first gain calibration chunk
            #..................................................................................................
            expSeqNum = fitsReaderPreCal.expSeqNum
            
            preCalResDict = tsys.process(fitsreader.SingleShotIterator(fitsReaderPreCal), \
                                         testLinearity = False, alpha = alpha)
            
            misc.set_or_check_param_continuity(workDict, 'numBands', preCalResDict['numBands'])
            misc.set_or_check_param_continuity(workDict, 'bandFreqs', preCalResDict['bandFreqs'])
            
            #..................................................................................................
            # Extract the initial part, which is assumed to be of a piece of empty sky preceding the source
            #..................................................................................................
            fitsReaderPreScan = fitsIter.next()
            
            if expSeqNum != fitsReaderPreScan.expSeqNum:
                logger.error("Unexpected change in experiment sequence number!")
            
            # Extract data and masks
            dataIdNameList = ['scan']
            dataSelectionList = [('PreBaselineScan', {'RX_ON_F': True, 'ND_ON_F': False, 'VALID_F': True}, False)]
            preScanDict = fitsReaderPreScan.extract_data(dataIdNameList, dataSelectionList, perBand=True)
            
            preScanData = preScanDict.values()[0]
            
            misc.set_or_check_param_continuity(workDict, 'numBands', len(preScanData.bandFreqs))
            misc.set_or_check_param_continuity(workDict, 'bandFreqs', preScanData.bandFreqs)
            
            #..................................................................................................
            # This is the main part of the scan, which contains the calibrator source
            #..................................................................................................
            fitsReaderScan = fitsIter.next()
            
            if expSeqNum != fitsReaderScan.expSeqNum:
                logger.error("Unexpected change in experiment sequence number!")
            
            # Extract data and masks
            dataIdNameList = ['scan']
            dataSelectionList = [('MainScan', {'RX_ON_F': True, 'ND_ON_F': False, 'VALID_F': True}, False)]
            scanDict = fitsReaderScan.extract_data(dataIdNameList, dataSelectionList, perBand=True)
            
            mainScanName, mainScanData = scanDict.items()[0]
            # Get the source name and position from the main scan, as well as the antenna beamwidth
            stdScan.sourceName = fitsReaderScan.select_masked_column('TARGET', 'Name', mask=None)[0]
            stdScan.sourceAzAng_deg = fitsReaderScan.select_masked_column('TARGET', 'Azimuth', mask=None)[0]
            stdScan.sourceElAng_deg = fitsReaderScan.select_masked_column('TARGET', 'Elevation', mask=None)[0]
            stdScan.antennaBeamwidth_deg = fitsReaderScan.select_masked_column('CONSTANTS', 'Beamwidth', mask=None)[0]
            
            misc.set_or_check_param_continuity(workDict, 'numBands', len(mainScanData.bandFreqs))
            misc.set_or_check_param_continuity(workDict, 'bandFreqs', mainScanData.bandFreqs)
            
            #..................................................................................................
            # Extract the final part, which is assumed to be of a piece of empty sky following the source
            #..................................................................................................
            fitsReaderPostScan = fitsIter.next()
            
            if expSeqNum != fitsReaderPostScan.expSeqNum:
                logger.error("Unexpected change in experiment sequence number!")
            
            # Extract data and masks
            dataIdNameList = ['scan']
            dataSelectionList = [('PostBaselineScan', {'RX_ON_F': True, 'ND_ON_F': False, 'VALID_F': True}, False)]
            postScanDict = fitsReaderPostScan.extract_data(dataIdNameList, dataSelectionList, perBand=True)
            
            postScanData = postScanDict.values()[0]
            
            misc.set_or_check_param_continuity(workDict, 'numBands', len(postScanData.bandFreqs))
            misc.set_or_check_param_continuity(workDict, 'bandFreqs', postScanData.bandFreqs)
            
            #..................................................................................................
            # Now extract the second gain calibration chunk
            #..................................................................................................
            fitsReaderPostCal = fitsIter.next()
            
            if expSeqNum != fitsReaderPostCal.expSeqNum:
                logger.error("Unexpected change in experiment sequence number!")
            
            postCalResDict = tsys.process(fitsreader.SingleShotIterator(fitsReaderPostCal), \
                                          testLinearity = False, alpha = alpha)
            
            misc.set_or_check_param_continuity(workDict, 'numBands', postCalResDict['numBands'])
            misc.set_or_check_param_continuity(workDict, 'bandFreqs', postCalResDict['bandFreqs'])
            
            #..................................................................................................
            # Now package the data in objects
            #..................................................................................................
            logger.info("Found scan ID : %s" % (mainScanName))
            logger.info("Source name and coordinates [az, el] = %s, [%5.3f, %5.3f]" \
                        % (stdScan.sourceName, stdScan.sourceAzAng_deg, stdScan.sourceElAng_deg))
            logger.info("Scan start coordinate [az, el] = [%5.3f, %5.3f]" \
                        % (mainScanData.azAng[0] * 180.0 / np.pi, mainScanData.elAng[0] * 180.0 / np.pi))
            logger.info("Scan stop coordinate  [az, el] = [%5.3f, %5.3f]" \
                        % (mainScanData.azAng[-1] * 180.0 / np.pi, mainScanData.elAng[-1] * 180.0 / np.pi))
            
            # Set up power-to-temperature conversion factors
            # Check if variation in Fpt is significant - choose linear interp in that case, else constant interp
            fptDiff = np.abs(preCalResDict['Fpt'] - postCalResDict['Fpt'])
            fptTotSig = np.sqrt(preCalResDict['Fpt_sigma']**2 + postCalResDict['Fpt_sigma']**2)
            msDiff = fptTotSig - fptDiff
            isLinear = msDiff.mean(axis=1) > 0.0
            preCalResDict['Fpt'] = preCalResDict['Fpt'][:, np.newaxis, :]
            postCalResDict['Fpt'] = postCalResDict['Fpt'][:, np.newaxis, :]
            preCalResDict['Fpt_sigma'] = preCalResDict['Fpt_sigma'][:, np.newaxis, :]
            postCalResDict['Fpt_sigma'] = postCalResDict['Fpt_sigma'][:, np.newaxis, :]
            if np.all(isLinear):
                logger.info("Pre and Post gain cal within stable bounds. Using average gain.")
                fptMean = 0.5*(preCalResDict['Fpt'] + postCalResDict['Fpt'])
                fptSigma = np.sqrt(preCalResDict['Fpt_sigma']**2 + postCalResDict['Fpt_sigma']**2)
                fptTime = 0.5*(preCalResDict['Fpt_time'] + postCalResDict['Fpt_time'])
            else:
                logger.info("Pre and Post gain cal outside stable bounds. Fitting linear gain profile.")
                fptMean = np.concatenate((preCalResDict['Fpt'], postCalResDict['Fpt']), axis=1)
                fptSigma = np.concatenate((preCalResDict['Fpt_sigma'], postCalResDict['Fpt_sigma']), axis=1)
                fptTime = np.array((preCalResDict['Fpt_time'], postCalResDict['Fpt_time']))
            
            # Save raw power objects
            rawList = []
            for dataBlock in preCalResDict['data'].itervalues():
                rawList.append(dataBlock)
            rawList.append(copy.deepcopy(preScanData))
            rawList.append(copy.deepcopy(mainScanData))
            rawList.append(copy.deepcopy(postScanData))
            for dataBlock in postCalResDict['data'].itervalues():
                rawList.append(dataBlock)
            rawPowerScanList.append(rawList)
            
            # Set up standard scan object (rest of members will be filled in during calibration)
            stdScan.mainData = mainScanData
            stdScan.baselineDataList = [preScanData, postScanData]
            stdScan.powerToTempTimes = fptTime
            stdScan.powerToTempFactors = stats.MuSigmaArray(fptMean, fptSigma)
            stdScanList.append(stdScan)
        
        except StopIteration:
            break
    
    return stdScanList, rawPowerScanList


## Calibrate a single scan, by correcting gain drifts and subtracting a baseline.
# @param stdScan   StandardSourceScan object containing scan and all auxiliary scans and info for calibration
# @param randomise True if fits should be randomised, as part of a larger Monte Carlo run
# @return SingleDishData object containing calibrated scan
def calibrate_scan(stdScan, randomise):
    # Set up power-to-temp conversion factors (optionally randomising it)
    p2tFactors = stdScan.powerToTempFactors
    if randomise:
        p2tFactors = p2tFactors.mean + p2tFactors.sigma * random.standard_normal(p2tFactors.shape)
    # Power-to-temp conversion factor is inverse of gain, which is assumed to change linearly over time
    stdScan.powerToTempFunc = interpolator.Independent1DFit(interpolator.ReciprocalFit( \
                              interpolator.Polynomial1DFit(maxDegree=1)), axis=1)
    stdScan.powerToTempFunc.fit(stdScan.powerToTempTimes, p2tFactors)
    # Convert baseline data to temperature, and concatenate them into a single structure
    for ind, baselineData in enumerate(stdScan.baselineDataList):
        baselineData.convert_power_to_temp(stdScan.powerToTempFunc)
        if ind == 0:
            allBaselineData = copy.deepcopy(baselineData)
        else:
            allBaselineData.append(baselineData)
    # Fit baseline on a coordinate with sufficient variation (elevation angle preferred)
    elAngMin = allBaselineData.elAng.min()
    elAngMax = allBaselineData.elAng.max()
    azAngMin = allBaselineData.azAng.min()
    azAngMax = allBaselineData.azAng.max()
    # Require variation on the order of an antenna beam width to fit higher-order polynomial
    if elAngMax - elAngMin > stdScan.antennaBeamwidth_deg * np.pi / 180.0:
#        logger.info('Baseline fit to elevation angle (range = '+str(elAngMax - elAngMin)+' deg)')
        stdScan.baselineUsesElevation = True
        interp = interpolator.Independent1DFit(interpolator.Polynomial1DFit(maxDegree=3), axis=1)
        interp.fit(allBaselineData.elAng, allBaselineData.powerData)
        if randomise:
            interp = interpolator.randomise(interp, allBaselineData.elAng, allBaselineData.powerData, 'shuffle')
        stdScan.baselineFunc = interp
    elif azAngMax - azAngMin > stdScan.antennaBeamwidth_deg * np.pi / 180.0:
#        logger.info('Baseline fit to azimuth angle (range = '+str(azAngMax - azAngMin)+' deg)')
        stdScan.baselineUsesElevation = False
        interp = interpolator.Independent1DFit(interpolator.Polynomial1DFit(maxDegree=1), axis=1)
        interp.fit(allBaselineData.azAng, allBaselineData.powerData)
        if randomise:
            interp = interpolator.randomise(interp, allBaselineData.azAng, allBaselineData.powerData, 'shuffle')
        stdScan.baselineFunc = interp
    else:
#        logger.info('Baseline fit to elevation angle, but as a constant (too little variation in angles)')
        stdScan.baselineUsesElevation = True
        interp = interpolator.Independent1DFit(interpolator.Polynomial1DFit(maxDegree=0), axis=1)
        interp.fit(allBaselineData.elAng, allBaselineData.powerData)
        if randomise:
            interp = interpolator.randomise(interp, allBaselineData.elAng, allBaselineData.powerData, 'shuffle')
        stdScan.baselineFunc = interp
    # Calibrate the main segment of scan
    calibratedScan = copy.deepcopy(stdScan.mainData.convert_power_to_temp(stdScan.powerToTempFunc))
    return calibratedScan.subtract_baseline(stdScan.baselineFunc, stdScan.baselineUsesElevation)


## Fit a beam pattern to total power data in 2-D target coordinate space.
# This is the original version, which works OK for strong sources, but struggles on weaker ones (unless
# all points in the scan are used in the fit).
# @param targetCoords 2-D coordinates in target space, as an (N,2)-shaped numpy array
# @param totalPower   Total power values, as an (N,M)-shaped numpy array (M = number of bands)
# @param randomise    True if fits should be randomised, as part of a larger Monte Carlo run
# @return List of Gaussian interpolator functions fitted to power data, one per band
def fit_beam_pattern_old(targetCoords, totalPower, randomise):
    interpList = []
    for band in range(totalPower.shape[1]):
        bandPower = totalPower[:, band]
        # Find main blob, defined as all points with a power value above some factor of the peak power
        # A factor of roughly 0.25 seems to provide the best combined accuracy for the peak height and location
        peakVal = bandPower.max()
        insideBlob = bandPower > 0.25 * peakVal
        blobPoints = targetCoords[insideBlob, :]
        # Find the centroid of the main blob
        weights = bandPower[insideBlob]
        centroid = np.dot(weights, blobPoints)/weights.sum()
        # Use the average distance from the blob points to the centroid to estimate standard deviation
        diffVectors = blobPoints - centroid[np.newaxis, :]
        distToCentroid = np.sqrt((diffVectors * diffVectors).sum(axis=1))
        initStDev = 2 * distToCentroid.mean()
        interp = interpolator.GaussianFit(centroid, initStDev*np.ones(centroid.shape), peakVal)
        # Fit Gaussian beam only to points within blob, where approximation is more valid (more accurate?)
        interp.fit(blobPoints, weights)
        # Or fit Gaussian beam to all points in scan (this can provide a better fit with very noisy data)
        # interp.fit(targetCoords, bandPower)
        if randomise:
            interp = interpolator.randomise(interp, blobPoints, weights, 'shuffle')
            # interp = interpolator.randomise(interp, targetCoords, bandPower, 'shuffle')
        interpList.append(interp)
    return interpList


## Fit a beam pattern to total power data in 2-D target coordinate space.
# This fits a Gaussian shape to power data, with the peak location as initial mean, and a factor of the beamwidth 
# as initial standard deviation. It uses all power values in the fitting process, instead of only the points within
# the half-power beamwidth of the peak, as suggested in [1]. This seems to be more robust for weak sources, but
# with more samples close to the peak, things might change again.
#
# [1] "Reduction and Analysis Techniques", Ronald J. Maddalena, in Single-Dish Radio Astronomy: Techniques and 
#     Applications, ASP Conference Series, Vol. 278, 2002
#
# @param targetCoords 2-D coordinates in target space, as an (N,2)-shaped numpy array
# @param totalPower   Total power values, as an (N,M)-shaped numpy array (M = number of bands)
# @param beamWidth    Antenna half-power beamwidth (in target coordinate scale)
# @param randomise    True if fits should be randomised, as part of a larger Monte Carlo run
# @return List of Gaussian interpolator functions fitted to power data, one per band
# @todo Fit other shapes, such as Airy
# @todo Fix width of Gaussian during fitting process to known beamwidth
# @todo Only fit Gaussian to points within half-power beamwidth of peak (not robust enough, need more samples)
def fit_beam_pattern(targetCoords, totalPower, beamWidth, randomise):
    interpList = []
    for band in range(totalPower.shape[1]):
        bandPower = totalPower[:, band]
        # Find peak power position and value
        peakInd = bandPower.argmax()
        peakVal = bandPower[peakInd]
        peakPos = targetCoords[peakInd, :]
        # Use peak location as initial estimate of mean, and peak value as initial height
        # A Gaussian function reaches half its peak value at 1.183*sigma => should equal beamWidth/2 for starters
        interp = interpolator.GaussianFit(peakPos, 0.423*beamWidth*np.ones(peakPos.shape), peakVal)
        # Fit Gaussian beam to all points in scan, which is not only better for weak sources, but also
        # provides more accurate estimates of peak than only fitting to points close to the peak
        interp.fit(targetCoords, bandPower)
        if randomise:
            interp = interpolator.randomise(interp, targetCoords, bandPower, 'shuffle')
        interpList.append(interp)
    return interpList


## Fit a beam pattern to multiple scans of a single calibrator source, after first calibrating the scans.
# @param stdScanList     List of StandardSourceScan objects
# @param randomise       True if fits should be randomised, as part of a larger Monte Carlo run
# @return beamFuncList   List of Gaussian beam functions, one per band
# @return calibScanList  List of data objects containing the fully calibrated main segments of each scan
# @return stdScanList    List of modified scan objects, after power-to-temp conversion but before baseline subtraction
def calibrate_and_fit_beam_pattern(stdScanList, randomise):
    calibScanList = []
    targetCoords = []
    totalPowerData = []
    for stdScan in stdScanList:
        calibratedScan = calibrate_scan(stdScan, randomise)
        targetCoords.append(calibratedScan.targetCoords)
        totalPowerData.append(calibratedScan.total_power())
        calibScanList.append(calibratedScan)
    targetCoords = np.concatenate(targetCoords)[:, 0:2]
    totalPowerData = np.concatenate(totalPowerData)
#    beamFuncList = fit_beam_pattern_old(targetCoords, totalPowerData, randomise)
    targetBeamwidth = stdScanList[0].antennaBeamwidth_deg * np.pi / 180.0
    beamFuncList = fit_beam_pattern(targetCoords, totalPowerData, targetBeamwidth, randomise)
    return beamFuncList, calibScanList, stdScanList


## Extract all information from an unresolved point source scan.
# This reduces the power data obtained from multiple scans across a point source. In the process, the data is
# calibrated to remove receiver gain drifts, baselines are removed, and a beam pattern is fitted to the combined
# scans. For general (unnamed) sources, the estimated source position in mount coordinates is returned.
# Additionally, for known calibrator sources, the antenna gain and effective area can be estimated from the
# known source flux density.
# @param stdScanList    List of StandardSourceScan objects, describing scans across a single point source
# @param randomise      True if fits should be randomised, as part of a larger Monte Carlo run [False]
# @return resultList    List of results, including antenna effective area and gain, and estimated source coordinates
# @return beamFuncList  List of Gaussian beam functions, one per band
# @return calibScanList List of data objects containing the fully calibrated main segments of each scan
# @return stdScanList   List of modified scan objects, after power-to-temp conversion but before baseline subtraction
# pylint: disable-msg=R0914
def reduce_point_source_scan(stdScanList, randomise=False):
    # Calibrate across all scans, and fit a beam pattern to estimate source position and strength
    beamFuncList, calibScanList, stdScanList = calibrate_and_fit_beam_pattern(stdScanList, randomise)
    bandFreqs = calibScanList[0].bandFreqs
    sourceName = stdScanList[0].sourceName
    # The antenna effective area and friends can only be calculated for sources with known flux densities
    sourcePowerFluxDensity = deltaT = pointSourceSensitivity = effArea = antGain = None
    if sourceName != 'None':
        # Get source flux density for each frequency band (based on Ott tables)
        sourcePowerFluxDensity = np.array([misc.ottflux(sourceName, freq / 1e6) for freq in bandFreqs])
        # Calculate antenna effective area and gain, per band
        deltaT = np.array([beamFunc.height for beamFunc in beamFuncList])
        pointSourceSensitivity = sourcePowerFluxDensity / deltaT
        effArea, antGain = misc.calc_ant_eff_area_and_gain(pointSourceSensitivity, bandFreqs)
    # For general point sources, it is still possible to estimate pointing error (use first main scan as reference)
    targetSys = calibScanList[0].targetCoordSystem
    mountSys = calibScanList[0].mountCoordSystem
    targetToMount = acsm.transform.get_factory_instance().get_transformer(targetSys, mountSys)
    # Average the beam centres across all bands to obtain source position estimate
    beamCentre = np.array([beamFunc.mean for beamFunc in beamFuncList]).mean(axis=0)
    targetCoordinate = Coordinate(targetSys, beamCentre.tolist() + [0.0])
    # The time of the source peak is taken to be in the middle of the first main scan
    mountCoordinate = targetToMount.transform_coordinate(targetCoordinate, calibScanList[0].timeSamples.mean())
    estmSourceAngs = mountCoordinate.get_vector()[0:2] * 180.0 / np.pi
    resultList = sourcePowerFluxDensity, deltaT, pointSourceSensitivity, effArea, antGain, estmSourceAngs
    return resultList, beamFuncList, calibScanList, stdScanList


## Extract point source information, with quantified statistical uncertainty.
# This reduces the power data obtained from multiple scans across a point source. In the process, the data is
# calibrated to remove receiver gain drifts, baselines are removed, and a beam pattern is fitted to the combined
# scans. For general (unnamed) sources, the estimated source position in mount coordinates is returned.
# Additionally, for known calibrator sources, the antenna gain and effective area can be estimated from the
# known source flux density. All final results are provided with standard deviations, which are estimated from
# the data set itself via resampling or sigma-point techniques.
# @param stdScanList    List of StandardSourceScan objects, describing scans across a single point source
# @param method         Technique used to obtain statistics ('resample' (recommended default) or 'sigma-point')
# @param numSamples     Number of samples in resampling process (more is better but slower) [10]
# @return resultList    List of results, including antenna effective area and gain, and estimated source coordinates
# @return beamFuncList  List of Gaussian beam functions, one per band
# @return calibScanList List of data objects containing the fully calibrated main segments of each scan
# @return tempScanList  List of modified scan objects, after power-to-temp conversion but before baseline subtraction
# pylint: disable-msg=W0612
def reduce_point_source_scan_with_stats(stdScanList, method='resample', numSamples=10):
    # Do the reduction without any variations first, to obtain scan lists and other constant outputs
    res, beamFuncList, calibScanList, tempScanList = reduce_point_source_scan(copy.deepcopy(stdScanList), False)
    sourcePowerFluxDensity, dT, pss, effA, antG, estmA = res
    
    # Resampling technique for estimating standard deviations
    if method == 'resample':
        deltaTList = [dT]
        pssList = [pss]
        effAreaList = [effA]
        antGainList = [antG]
        azAngList = [estmA[0]]
        elAngList = [estmA[1]]
        # Re-run the reduction routine, which randomises itself internally each time
        for n in xrange(numSamples-1):
            results = reduce_point_source_scan(copy.deepcopy(stdScanList), randomise=True)
            pfd, dT, pss, effA, antG, estmA = results[0]
            deltaTList.append(dT)
            pssList.append(pss)
            effAreaList.append(effA)
            antGainList.append(antG)
            azAngList.append(estmA[0])
            elAngList.append(estmA[1])
        # Obtain statistics and save in combined data structure
        deltaT = stats.MuSigmaArray(np.array(deltaTList).mean(axis=0), np.array(deltaTList).std(axis=0))
        pointSourceSensitivity = stats.MuSigmaArray(np.array(pssList).mean(axis=0), np.array(pssList).std(axis=0))
        effArea = stats.MuSigmaArray(np.array(effAreaList).mean(axis=0), np.array(effAreaList).std(axis=0))
        antGain = stats.MuSigmaArray(np.array(antGainList).mean(axis=0), np.array(antGainList).std(axis=0))
        estmAzAng = stats.MuSigmaArray(np.array(azAngList).mean(axis=0), np.array(azAngList).std(axis=0))
        estmElAng = stats.MuSigmaArray(np.array(elAngList).mean(axis=0), np.array(elAngList).std(axis=0))
    
    # Sigma-point technique for estimating standard deviations
    elif method == 'sigma-point':
        # Currently only Fpt factors are sigma'ed, as the full data set will be too computationally intensive
        fptMean = np.array([stdScan.powerToTempFactors.mean for stdScan in stdScanList])
        fptSigma = np.array([stdScan.powerToTempFactors.sigma for stdScan in stdScanList])
        fptMuSigma = stats.MuSigmaArray(fptMean.ravel(), fptSigma.ravel())
        
        ## Wraps reduction code for sigma stats calculation.
        # @param fptFactors Vector of input power-to-temp factors
        # @param scanList   Main list of scan objects, installed via default value (don't assign it something else!)
        # @return Vector of output values
        # pylint: disable-msg=W0102
        def wrapper_func(fptFactors, scanList=stdScanList):
            # Don't trash the original scan list, as this will be re-used by several runs of this function
            scanListCopy = copy.deepcopy(scanList)
            # Reshape and install Fpt values into scan list
            fptShape = [len(scanListCopy)] + list(scanListCopy[0].powerToTempFactors.shape)
            for index, fptFactor in enumerate(fptFactors.reshape(fptShape)):
                scanListCopy[index].powerToTempFactors = fptFactor
            # Run reduction code with updated scan list, and ravel the results that require sigmas
            results = reduce_point_source_scan(scanListCopy, False)
            return np.concatenate(results[0][1:6])
        
        # Do sigma-point evaluation, and extract relevant outputs
        resMuSigma = stats.sp_uncorrelated_stats(wrapper_func, fptMuSigma)
        
        numBands = len(stdScanList[0].mainData.bandFreqs)
        deltaT = resMuSigma[0:numBands]
        pointSourceSensitivity = resMuSigma[numBands:2*numBands]
        effArea = resMuSigma[2*numBands:3*numBands]
        antGain = resMuSigma[3*numBands:4*numBands]
        estmAzAng = resMuSigma[4*numBands]
        estmElAng = resMuSigma[4*numBands+1]
    
    else:
        raise ValueError, "Unknown stats method '" + method + "'."
    
    resultList = sourcePowerFluxDensity, deltaT, pointSourceSensitivity, effArea, antGain, (estmAzAng, estmElAng)
    return resultList, beamFuncList, calibScanList, tempScanList
