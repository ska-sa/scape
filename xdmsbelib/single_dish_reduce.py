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
import logging
import copy

logger = logging.getLogger("xdmsbe.xdmsbelib.single_dish_reduce")

#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  StandardSourceScan
#----------------------------------------------------------------------------------------------------------------------

## Container for the components of a standard scan across a source.
# This struct contains all the parts that make up a standard scan:
# - information about the source (name and position)
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
    ## @var sourceAzAng
    # Source azimuth angle in degrees
    sourceAzAng = None
    ## @var sourceElAng
    # Source elevation angle in degrees
    sourceElAng = None
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
            dataSelectionList = [('CalSourcePreScan', {'RX_ON_F': True, 'ND_ON_F': False, 'VALID_F': True}, False)]
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
            dataSelectionList = [('CalSourceScan', {'RX_ON_F': True, 'ND_ON_F': False, 'VALID_F': True}, False)]
            scanDict = fitsReaderScan.extract_data(dataIdNameList, dataSelectionList)
            
            mainScanName, mainScanData = scanDict.items()[0]
            # Get the source name and position from the main scan
            stdScan.sourceName = fitsReaderScan.select_masked_column('TARGET', 'Name', mask=None)[0]
            stdScan.sourceAzAng = fitsReaderScan.select_masked_column('TARGET', 'Azimuth', mask=None)[0]
            stdScan.sourceElAng = fitsReaderScan.select_masked_column('TARGET', 'Elevation', mask=None)[0]
            
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
            dataSelectionList = [('CalSourcePostScan', {'RX_ON_F': True, 'ND_ON_F': False, 'VALID_F': True}, False)]
            postScanDict = fitsReaderPostScan.extract_data(dataIdNameList, dataSelectionList)
            
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
            logger.info("Start coordinate [az, el] = [%5.3f, %5.3f]" % (mainScanData.azAng[0], \
                                                                        mainScanData.elAng[0]))
            logger.info("Stop coordinate  [az, el] = [%5.3f, %5.3f]" % (mainScanData.azAng[-1], \
                                                                        mainScanData.elAng[-1]))
            
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
# @param stdScan StandardSourceScan object containing scan and all auxiliary scans and info for calibration
# @return SingleDishData object containing calibrated scan
def calibrate_scan(stdScan):
    # Power-to-temp conversion factor is inverse of gain, which is assumed to change linearly over time
    stdScan.powerToTempFunc = interpolator.Independent1DFit(interpolator.ReciprocalFit( \
                              interpolator.Polynomial1DFit(maxDegree=1)), axis=1)
    stdScan.powerToTempFunc.fit(stdScan.powerToTempTimes, stdScan.powerToTempFactors)
#    print stdScan.powerToTempTimes, stdScan.powerToTempFactors
#    for inter in stdScan.powerToTempFunc._interps.ravel():
#        print inter._interp._mean, inter._interp.poly
    # Convert baseline data to temperature, and concatenate them into a single structure
    for ind, baselineData in enumerate(stdScan.baselineDataList):
        baselineData.convert_power_to_temp(stdScan.powerToTempFunc)
        if ind == 0:
            allBaselineData = copy.deepcopy(baselineData)
        else:
            allBaselineData.append(baselineData)
    # Fit baseline on a coordinate with sufficient variation (elevation angle preferred)
    elAngMin = allBaselineData.elAng.min() * 180.0 / np.pi
    elAngMax = allBaselineData.elAng.max() * 180.0 / np.pi
    azAngMin = allBaselineData.azAng.min() * 180.0 / np.pi
    azAngMax = allBaselineData.azAng.max() * 180.0 / np.pi
    antennaBeamwidth_deg = 1.0
    # Require variation on the order of an antenna beam width to fit higher-order polynomial
    if elAngMax - elAngMin > antennaBeamwidth_deg:
#        logger.info('Baseline fit to elevation angle (range = '+str(elAngMax - elAngMin)+' deg)')
        stdScan.baselineUsesElevation = True
        stdScan.baselineFunc = interpolator.Independent1DFit(interpolator.Polynomial1DFit(maxDegree=3), axis=1)
        stdScan.baselineFunc.fit(allBaselineData.elAng, allBaselineData.powerData)
    elif azAngMax - azAngMin > antennaBeamwidth_deg:
#        logger.info('Baseline fit to azimuth angle (range = '+str(azAngMax - azAngMin)+' deg)')
        stdScan.baselineUsesElevation = False
        stdScan.baselineFunc = interpolator.Independent1DFit(interpolator.Polynomial1DFit(maxDegree=1), axis=1)
        stdScan.baselineFunc.fit(allBaselineData.azAng, allBaselineData.powerData)
    else:
#        logger.info('Baseline fit to elevation angle, but as a constant (too little variation in angles)')
        stdScan.baselineUsesElevation = True
        stdScan.baselineFunc = interpolator.Independent1DFit(interpolator.Polynomial1DFit(maxDegree=0), axis=1)
        stdScan.baselineFunc.fit(allBaselineData.elAng, allBaselineData.powerData)
    # Calibrate the main segment of scan
    calibratedScan = copy.deepcopy(stdScan.mainData.convert_power_to_temp(stdScan.powerToTempFunc))
    return calibratedScan.subtract_baseline(stdScan.baselineFunc, stdScan.baselineUsesElevation)


## Fit a beam pattern to total power data in 2-D target coordinate space.
# @param targetCoords 2-D coordinates in target space, as an (N,2)-shaped numpy array
# @param totalPower   Total power values, as an (N,M)-shaped numpy array (M = number of bands)
# @return Gaussian interpolator function fitted to power data
def fit_beam_pattern(targetCoords, totalPower):
    interpList = []
    for band in range(totalPower.shape[1]):
        bandPower = totalPower[:, band]
        # Find main blob, defined as all points with a power value above some factor of the peak power
        # Maybe this should be the points within the half-power beamwidth of the peak?
        # A factor of roughly 0.25 seems to provide the best combined accuracy for the peak height and location
        peakVal = bandPower.max()
        blobThreshold = 0.25 * peakVal
        insideBlob = bandPower > blobThreshold
        blobPoints = targetCoords[insideBlob, :]
        # Find the centroid of the main blob
        weights = bandPower[insideBlob]
        centroid = np.dot(weights, blobPoints)/weights.sum()
        # Use the average distance from the blob points to the centroid to estimate standard deviation
        diffVectors = blobPoints - centroid[np.newaxis, :]
        distToCentroid = np.sqrt((diffVectors * diffVectors).sum(axis=1))
        initStDev = 2 * distToCentroid.mean()
        interp = interpolator.GaussianFit(centroid, initStDev*np.ones(centroid.shape), peakVal)
        # Fit Gaussian beam only to points within blob, where approximation is more valid (more accurate)
        interp.fit(blobPoints, weights)
        # Or fit Gaussian beam to all points in scan (this can provide a better fit with very noisy data)
        # interp.fit(targetCoords, bandPower)
        interpList.append(interp)
    return interpList


## Fit a beam pattern to multiple scans of a single calibrator source, after first calibrating the scans.
# @param stdScanList     List of StandardSourceScan objects
# @return stdScanList    List of modified scan objects, after power-to-temp conversion but before baseline subtraction
# @return calibScanList  List of data objects containing the fully calibrated main segments of each scan
# @return targetCoords   2-D coordinates in target space of all calibrated main scans, as an (N,2)-shaped numpy array
# @return totalPowerData Total power values of all calibrated main scans, as an (N,M)-shaped numpy array
# @return beamFuncList   List of Gaussian beam functions, one per band
def calibrate_and_fit_beam_pattern(stdScanList):
    calibScanList = []
    targetCoords = []
    totalPowerData = []
    for stdScan in stdScanList:
        calibratedScan = calibrate_scan(stdScan)
        targetCoords.append(calibratedScan.targetCoords)
        totalPowerData.append(calibratedScan.total_power())
        calibScanList.append(calibratedScan)
    targetCoords = np.concatenate(targetCoords)[:, 0:2]
    totalPowerData = np.concatenate(totalPowerData)
    beamFuncList = fit_beam_pattern(targetCoords, totalPowerData)
    return stdScanList, calibScanList, targetCoords, totalPowerData, beamFuncList


## Extract all information from an unresolved point source scan.
# This reduces the power data obtained from multiple scans across a point source. In the process, the data is
# calibrated to remove receiver gain drifts, baselines are removed, and a beam pattern is fitted to the combined
# scans. For general (unnamed) sources, the estimated source position in mount coordinates is returned.
# Additionally, for known calibrator sources, the antenna gain and effective area can be estimated from the
# known source flux density.
# @param stdScanList     List of StandardSourceScan objects, describing scans across a single point source
# @return A whole bunch of stuff
# pylint: disable-msg=R0914
def reduce_point_source_scan(stdScanList):
    # Calibrate across all scans, and fit a beam pattern to estimate source position and strength
    stdScanList, calibScanList, targetCoords, totalPower, beamFuncList = calibrate_and_fit_beam_pattern(stdScanList)
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
    estmSourceAzAng, estmSourceElAng = mountCoordinate.get_vector()[0:2] * 180.0 / np.pi
    return sourcePowerFluxDensity, deltaT, pointSourceSensitivity, effArea, antGain, \
           (estmSourceAzAng, estmSourceElAng), stdScanList, calibScanList, targetCoords, totalPower, beamFuncList
