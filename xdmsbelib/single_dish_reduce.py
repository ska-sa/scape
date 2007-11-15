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
import xdmsbe.xdmsbelib.fitting as fitting
import xdmsbe.xdmsbelib.gain_cal_data as gain_cal_data
from xdmsbe.xdmsbelib import stats
import xdmsbe.xdmsbelib.misc as misc
from conradmisclib.transforms import rad_to_deg, deg_to_rad
from acsm.coordinate import Coordinate
import acsm.transform.transformfactory as transformfactory
import numpy as np
import logging
import copy

logger = logging.getLogger("xdmsbe.xdmsbelib.single_dish_reduce")


#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  StandardSourceScan
#----------------------------------------------------------------------------------------------------------------------


## Container for the components of a standard scan across a source.
# This struct contains all the parts that make up a standard scan:
# - information about the source (name)
# - information about the antenna (beamwidth)
# - the main scan segment to be calibrated
# - a summary of gain cal data recorded at the start and end of the scan, and noise diode characteristics
# - the actual powerToTemp function used to calibrate power measurements for whole scan
# - scan segments used to fit a baseline (if available)
# - the baseline function that will be subtracted from main scan segment
# pylint: disable-msg=R0903
class StandardSourceScan(object):
    ## @var sourceName
    # Description of underlying target/source being scanned
    sourceName = None
    ## @var antennaBeamwidth_deg
    # Antenna half-power beamwidth in degrees
    antennaBeamwidth_deg = None
    ## @var mainData
    # SingleDishData object for main scan segment
    mainData = None
    ## @var noiseDiodeData
    # Object storing noise diode characteristics
    noiseDiodeData = None
    ## @var gainCalData
    # Object storing summary of power measurements made before and after the scan, for gain calibration
    gainCalData = None
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
#--- CLASS :  TargetToInstantMountTransform
#----------------------------------------------------------------------------------------------------------------------


## Converter which converts target to "instantaneous" mount coordinates, for a set of scans through a single target.
# This simplifies the conversion of coordinates for a set of scans through a single target. It assumes that all 
# measurements in the target coordinate space happen simultaneously, at the median time of the scan list.
# The target is therefore assumed to move very little during the scan process.The class provides a callable object 
# that does the transformation, and can be passed around to functions (the reason for its existence).
class TargetToInstantMountTransform(object):
    ## Initialiser.
    # @param self        The current object
    # @param stdScanList List of StandardSourceScan objects, one per scan through a target
    def __init__(self, stdScanList):
        # Use first main scan as reference for coordinate system
        mountSys = stdScanList[0].mainData.mountCoordSystem
        ## @var targetSys
        # Target coordinate system (assumed the same for all scans in list)
        self.targetSys = stdScanList[0].mainData.targetObject.get_coordinate_system()
        ## @var targetToInstantMount
        # Coordinate transformer object
        self.targetToInstantMount = transformfactory.get_transformer(self.targetSys, mountSys)
        # The measurement time instant is taken to be the median time of all the main segments in the scan list
        allTimes = np.concatenate([stdScan.mainData.timeSamples for stdScan in stdScanList])
        ## @var timeStamp
        # Time instant at which target is deemed to be observed - all measurements are assumed to be simultaneous
        self.timeStamp = np.median(allTimes)

    ## Convert coordinates from target space to "instantaneous" mount space.
    # This currently assumes that the mount has a horizontal coordinate system, and it discards any rotator angle.
    # The output vector is therefore (az, el) in degrees.
    # @param self        The current object
    # @param targetCoord Vector of coordinates in target coordinate system
    # @return Vector of coordinates in mount coordinate system, of the form (az, el) in degrees
    def __call__(self, targetCoord):
        targetCoordinate = Coordinate(self.targetSys, targetCoord)
        mountCoordinate = self.targetToInstantMount.transform_coordinate(targetCoordinate, self.timeStamp)
        return rad_to_deg(mountCoordinate.get_vector()[0:2])
    
    ## Dimension of target coordinate vector.
    # @param self The current object
    # @return Dimension of target coordinate vector
    def get_target_dimensions(self):
        return self.targetSys.get_dimensions()
    
    ## Position of target itself in "instantaneous" mount coordinates, at the median time instant.
    # This transforms the origin of the target coordinate system, which represents the position of the target itself
    # at the median time instant, to mount coordinates.
    # @param self The current object
    # @return Position of target in mount coordinates
    def origin(self):
        return self(np.zeros(self.get_target_dimensions()))


#----------------------------------------------------------------------------------------------------------------------
#--- FUNCTIONS
#----------------------------------------------------------------------------------------------------------------------


## Checks consistency of data blocks loaded from a set of FITS files.
# This checks that all the data blocks that make up a standard source scan have consistent parameters.
# @param dataBlocks List of SingleDishData objects containing raw power data of a standard scan
# @param dataLabels List of block labels (or ID strings), identifying each block
def check_data_consistency(dataBlocks, dataLabels):
    # Nothing to check
    if len(dataBlocks) == 0:
        return
    referenceBlock = dataBlocks[0]
    for block in dataBlocks[1:]:
        if np.any(block.bandFreqs != referenceBlock.bandFreqs):
            message = "Channel frequencies of two blocks in standard scan differ: " + str(block.bandFreqs) + \
                      " vs. " + str(referenceBlock.bandFreqs)
            logger.error(message)
            raise ValueError, message
    # Skip 'esn' and convert everything up to first '_' as experiment sequence number
    expSeqNums = [int(label[3:label.find('_')]) for label in dataLabels]
    if len(set(expSeqNums)) != 1:
        message = "Different experiment sequence numbers found within a standard scan: " + str(expSeqNums)
        logger.error(message)
        raise ValueError, message


## Loads a list of standard scans across various point sources from a chain of FITS files.
# @param fitsFileName      Name of initial file in FITS chain
# @param fitBaseline       True if baseline fitting is required [True]
# @return stdScanList      List of StandardSourceScan objects, one per scan through a source
# @return rawPowerScanList List of SingleDishData objects, containing copies of all raw data blocks
# pylint: disable-msg=R0912,R0914,R0915
def load_point_source_scan_list(fitsFileName, fitBaseline=True):
    fitsIter = fitsreader.FitsIterator(fitsFileName)
    stdScanList = []
    rawPowerScanList = []
    noiseDiodeData = None
    print "                 **** Loading scan data from FITS files ****\n"
    # Iterate through multiple standard scans
    while True:
        stdScan = StandardSourceScan()
        # Put this before printing the banner below, otherwise the banner will appear once too many
        try:
            fitsReaderPreCal = fitsIter.next()
        except StopIteration:
            break
        
        print "==============================================================================================\n"
        print "                             **** SCAN %d ****\n" % len(stdScanList)
        
        #..................................................................................................
        # Extract the first gain calibration chunks
        #..................................................................................................
        # Load noise diode characteristics from first FITS file
        if noiseDiodeData == None:
            noiseDiodeData = gain_cal_data.NoiseDiodeData(fitsReaderPreCal)
        
        dataIdNameList = ['cold', 'hot', 'cal']
        dataSelectionList = [('PreCalOff', {'RX_ON_F': False, 'ND_ON_F': False, 'VALID_F': True}, False), \
                             ('PreCalOn', {'RX_ON_F': True, 'ND_ON_F': False, 'VALID_F': True}, False), \
                             ('PreCalOnND', {'RX_ON_F': True, 'ND_ON_F': True, 'VALID_F': True}, False)]
        preCalDict = fitsReaderPreCal.extract_data(dataIdNameList, dataSelectionList, perBand=True)
        
        print "PreCalibration data blocks:  ", preCalDict.keys()
        
        #..................................................................................................
        # Extract the initial part, which is assumed to be of a piece of empty sky preceding the source
        #..................................................................................................
        if fitBaseline:
            try:
                fitsReaderPreScan = fitsIter.next()
            except StopIteration:
                break
            
            dataIdNameList = ['scan']
            dataSelectionList = [('PreBaselineScan', {'RX_ON_F': True, 'ND_ON_F': False, 'VALID_F': True}, False)]
            preScanDict = fitsReaderPreScan.extract_data(dataIdNameList, dataSelectionList, perBand=True)
            
            preScanData = preScanDict.values()[0]
            
            print "PreBaselineScan data blocks: ", preScanDict.keys()
        
        #..................................................................................................
        # This is the main part of the scan, which contains the calibrator source
        #..................................................................................................
        try:
            fitsReaderScan = fitsIter.next()
        except StopIteration:
            break
        
        dataIdNameList = ['scan']
        dataSelectionList = [('MainScan', {'RX_ON_F': True, 'ND_ON_F': False, 'VALID_F': True}, False)]
        mainScanDict = fitsReaderScan.extract_data(dataIdNameList, dataSelectionList, perBand=True)
        
        mainScanData = mainScanDict.values()[0]
        # Get the source name and position from the main scan, as well as the antenna beamwidth
        stdScan.sourceName = fitsReaderScan.get_primary_header()['Target']
        stdScan.antennaBeamwidth_deg = fitsReaderScan.select_masked_column('CONSTANTS', 'Beamwidth', mask=None)[0]
        
        print "MainScan data blocks:        ", mainScanDict.keys()
        print "Source name:", stdScan.sourceName
        print "Scan start coordinate [az, el] = [%5.3f, %5.3f]" \
              % (rad_to_deg(mainScanData.azAng_rad[0]), rad_to_deg(mainScanData.elAng_rad[0]))
        print "Scan stop coordinate  [az, el] = [%5.3f, %5.3f]" \
              % (rad_to_deg(mainScanData.azAng_rad[-1]), rad_to_deg(mainScanData.elAng_rad[-1]))
        
        #..................................................................................................
        # Extract the final part, which is assumed to be of a piece of empty sky following the source
        #..................................................................................................
        if fitBaseline:
            try:
                fitsReaderPostScan = fitsIter.next()
            except StopIteration:
                break
            
            dataIdNameList = ['scan']
            dataSelectionList = [('PostBaselineScan', {'RX_ON_F': True, 'ND_ON_F': False, 'VALID_F': True}, False)]
            postScanDict = fitsReaderPostScan.extract_data(dataIdNameList, dataSelectionList, perBand=True)
            
            postScanData = postScanDict.values()[0]
            
            print "PostBaselineScan data blocks:", postScanDict.keys()
            
        #..................................................................................................
        # Now extract the second gain calibration chunks
        #..................................................................................................
        try:
            fitsReaderPostCal = fitsIter.next()
        except StopIteration:
            break
        
        dataIdNameList = ['cold', 'hot', 'cal']
        dataSelectionList = [('PostCalOff', {'RX_ON_F': False, 'ND_ON_F': False, 'VALID_F': True}, False), \
                             ('PostCalOn', {'RX_ON_F': True, 'ND_ON_F': False, 'VALID_F': True}, False), \
                             ('PostCalOnND', {'RX_ON_F': True, 'ND_ON_F': True, 'VALID_F': True}, False)]
        postCalDict = fitsReaderPostCal.extract_data(dataIdNameList, dataSelectionList, perBand=True)
        
        print "PostCalibration data blocks: ", postCalDict.keys()
        
        #..................................................................................................
        # Now package the data in objects
        #..................................................................................................
        
        # Save raw power objects (deep-copying the ones that will be affected by calibration)
        rawBlocks = []
        rawLabels = []
        rawBlocks += preCalDict.values()
        rawLabels += preCalDict.keys()
        if fitBaseline:
            rawBlocks.append(copy.deepcopy(preScanData))
            rawLabels.append(preScanDict.keys()[0])
        rawBlocks.append(copy.deepcopy(mainScanData))
        rawLabels.append(mainScanDict.keys()[0])
        if fitBaseline:
            rawBlocks.append(copy.deepcopy(postScanData))
            rawLabels.append(postScanDict.keys()[0])
        rawBlocks += postCalDict.values()
        rawLabels += postCalDict.keys()
        
        check_data_consistency(rawBlocks, rawLabels)
        
        # Set up standard scan object (rest of members will be filled in during calibration)
        stdScan.mainData = mainScanData
        stdScan.noiseDiodeData = noiseDiodeData
        calDict = {}
        calDict.update(preCalDict)
        calDict.update(postCalDict)
        stdScan.gainCalData = gain_cal_data.GainCalibrationData(calDict)
        if fitBaseline:
            stdScan.baselineDataList = [preScanData, postScanData]
        
        # Successful standard scan finally gets added to lists here - this ensures lists are in sync
        rawPowerScanList.append(rawBlocks)
        stdScanList.append(stdScan)
        
    return stdScanList, rawPowerScanList


## Calibrate a single scan, by correcting gain drifts and subtracting a baseline.
# This also modifies the stdScan object, by converting its data from raw power to temperature values. If this
# is not desired, use deepcopy on stdScan before calling this function. The baseline is only subtracted from
# the main scan (which is returned).
# @param stdScan   StandardSourceScan object containing scan and all auxiliary scans and info for calibration
# @param randomise True if fits should be randomised, as part of a larger Monte Carlo run
# @return SingleDishData object containing calibrated main scan
def calibrate_scan(stdScan, randomise):
    # Set up power-to-temp conversion function (optionally randomising it)
    stdScan.powerToTempFunc = stdScan.gainCalData.power_to_temp_func(stdScan.noiseDiodeData, randomise=randomise)
    # Convert the main segment of scan to temperature, and make a copy to preserve original data
    calibratedScan = copy.deepcopy(stdScan.mainData.convert_power_to_temp(stdScan.powerToTempFunc))
    # Without baseline data segments, the calibration is done
    if stdScan.baselineDataList == None:
        return calibratedScan
    # Convert baseline data to temperature, and concatenate them into a single structure
    for ind, baselineData in enumerate(stdScan.baselineDataList):
        baselineData.convert_power_to_temp(stdScan.powerToTempFunc)
        if ind == 0:
            allBaselineData = copy.deepcopy(baselineData)
        else:
            allBaselineData.append(baselineData)
    # Fit baseline on a coordinate with sufficient variation (elevation angle preferred)
    elAngMin = allBaselineData.elAng_rad.min()
    elAngMax = allBaselineData.elAng_rad.max()
    azAngMin = allBaselineData.azAng_rad.min()
    azAngMax = allBaselineData.azAng_rad.max()
    # Require variation on the order of an antenna beam width to fit higher-order polynomial
    if elAngMax - elAngMin > deg_to_rad(stdScan.antennaBeamwidth_deg):
        stdScan.baselineUsesElevation = True
        interp = fitting.Independent1DFit(fitting.Polynomial1DFit(maxDegree=3), axis=1)
        interp.fit(allBaselineData.elAng_rad, allBaselineData.powerData)
        if randomise:
            interp = fitting.randomise(interp, allBaselineData.elAng_rad, allBaselineData.powerData, 'shuffle')
        stdScan.baselineFunc = interp
    elif azAngMax - azAngMin > deg_to_rad(stdScan.antennaBeamwidth_deg):
        stdScan.baselineUsesElevation = False
        interp = fitting.Independent1DFit(fitting.Polynomial1DFit(maxDegree=1), axis=1)
        interp.fit(allBaselineData.azAng_rad, allBaselineData.powerData)
        if randomise:
            interp = fitting.randomise(interp, allBaselineData.azAng_rad, allBaselineData.powerData, 'shuffle')
        stdScan.baselineFunc = interp
    else:
        stdScan.baselineUsesElevation = True
        interp = fitting.Independent1DFit(fitting.Polynomial1DFit(maxDegree=0), axis=1)
        interp.fit(allBaselineData.elAng_rad, allBaselineData.powerData)
        if randomise:
            interp = fitting.randomise(interp, allBaselineData.elAng_rad, allBaselineData.powerData, 'shuffle')
        stdScan.baselineFunc = interp
    # Subtract baseline and return calibrated data
    return calibratedScan.subtract_baseline(stdScan.baselineFunc, stdScan.baselineUsesElevation)


## Fit a beam pattern to total power data in 2-D target coordinate space.
# This is the original version, which works OK for strong sources, but struggles on weaker ones (unless
# all points in the scan are used in the fit).
# @param targetCoords 2-D coordinates in target space, as an (N,2)-shaped numpy array
# @param totalPower   Total power values, as an (N,M)-shaped numpy array (M = number of bands)
# @param randomise    True if fits should be randomised, as part of a larger Monte Carlo run
# @return List of Gaussian fitting functions fitted to power data, one per band
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
        interp = fitting.GaussianFit(centroid, initStDev*np.ones(centroid.shape), peakVal)
        # Fit Gaussian beam only to points within blob, where approximation is more valid (more accurate?)
        interp.fit(blobPoints, weights)
        # Or fit Gaussian beam to all points in scan (this can provide a better fit with very noisy data)
        # interp.fit(targetCoords, bandPower)
        if randomise:
            interp = fitting.randomise(interp, blobPoints, weights, 'shuffle')
            # interp = fitting.randomise(interp, targetCoords, bandPower, 'shuffle')
        interpList.append(interp)
    return interpList


## Fit a beam pattern to total power data in 2-D target coordinate space.
# This fits a Gaussian shape to power data, with the peak location as initial mean, and an initial standard 
# deviation based on the expected beamwidth. It uses all power values in the fitting process, instead of only the 
# points within the half-power beamwidth of the peak as suggested in [1]. This seems to be more robust for weak 
# sources, but with more samples close to the peak, things might change again. The fitting currently only uses
# the first two values of the target coordinate system (typically ignoring rotator angle).
#
# [1] "Reduction and Analysis Techniques", Ronald J. Maddalena, in Single-Dish Radio Astronomy: Techniques and 
#     Applications, ASP Conference Series, Vol. 278, 2002
#
# @param targetCoords 2-D coordinates in target space, as an (N,2)-shaped numpy array
# @param totalPower   Total power values, as an (N,M)-shaped numpy array (M = number of bands)
# @param beamwidth    Antenna half-power beamwidth (in target coordinate scale)
# @param randomise    True if fits should be randomised, as part of a larger Monte Carlo run
# @return List of Gaussian fitting functions fitted to power data, paired with valid flags (one per band)
# @todo Fit other shapes, such as Airy
# @todo Fix width of Gaussian during fitting process to known beamwidth
# @todo Only fit Gaussian to points within half-power beamwidth of peak (not robust enough, need more samples)
# @todo Incorporate rotator angle?
def fit_beam_pattern(targetCoords, totalPower, beamwidth, randomise):
    interpList = []
    for band in range(totalPower.shape[1]):
        bandPower = totalPower[:, band]
        # Find peak power position and value
        peakInd = bandPower.argmax()
        peakVal = bandPower[peakInd]
        peakPos = targetCoords[peakInd, :]
        # Determine variance of Gaussian beam function which produces the expected half-power beamwidth value
        # A Gaussian function reaches half its peak value at 1.183*sigma => should equal beamWidth/2 for starters
        expectedVar = (beamwidth / 2.0 / 1.183) ** 2.0
        # Use peak location as initial estimate of mean, and peak value as initial height
        interp = fitting.GaussianFit(peakPos, np.tile(expectedVar, peakPos.shape), peakVal)
        # Fit Gaussian beam to all points in scan, which is better for weak sources, but can produce
        # small errors on especially the peak amplitude
        interp.fit(targetCoords, bandPower)
        if randomise:
            interp = fitting.randomise(interp, targetCoords, bandPower, 'shuffle')
        # Mark any fitted beam that has bad location, negative height or wildly unexpected variance as invalid
        # Make sure that more extended sources such as the Sun are not a problem...
        validFit = not np.any(np.isnan(interp.mean)) and (interp.height > 0.0) and \
                   (np.max(interp.var) < 9.0 * expectedVar) and (np.min(interp.var) > expectedVar / 2.0)
        interpList.append((interp, validFit))
    return interpList


## Fit a beam pattern to multiple scans of a single calibrator source, after first calibrating the scans.
# @param stdScanList     List of StandardSourceScan objects (modified by call - converted from power to temp)
# @param randomise       True if fits should be randomised, as part of a larger Monte Carlo run
# @return beamFuncList   List of Gaussian beam functions and valid flags, one per band
# @return calibScanList  List of data objects containing the fully calibrated main segments of each scan
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
    targetBeamwidth = deg_to_rad(stdScanList[0].antennaBeamwidth_deg)
    beamFuncList = fit_beam_pattern(targetCoords, totalPowerData, targetBeamwidth, randomise)
    return beamFuncList, calibScanList


## Extract all information from an unresolved point source scan.
# This reduces the power data obtained from multiple scans across a point source. In the process, the data is
# calibrated to remove receiver gain drifts, baselines are removed, and a beam pattern is fitted to the combined 
# scans, which allows the source position and strength to be estimated. For general (unnamed) sources, the 
# estimated source position is returned as target coordinates. This is effectively the pointing error, as the target 
# is at the origin of the target coordinate system. Additionally, for known calibrator sources, the antenna gain 
# and effective area can be estimated from the known source flux density. If either the gain or pointing estimation 
# did not succeed, the corresponding result is returned as None.
# @param stdScanList      List of StandardSourceScan objects, describing scans across a single point source
#                         (modified by call - converted from power to temp)
# @param randomise        True if fits should be randomised, as part of a larger Monte Carlo run [False]
# @return gainResults     List of gain-based results, of form (sourcePfd, deltaT, sensitivity, effArea, antGain)
# @return pointingResults Estimated point source position in target coordinates, and mount coordinates
# @return beamFuncList    List of Gaussian beam functions and valid flags, one per band
# @return calibScanList   List of data objects containing the fully calibrated main segments of each scan
# pylint: disable-msg=R0914
def reduce_point_source_scan(stdScanList, randomise=False):
    # Calibrate across all scans, and fit a beam pattern to estimate source position and strength
    beamFuncList, calibScanList = calibrate_and_fit_beam_pattern(stdScanList, randomise)
    # List of valid beam functions
    validBeamFuncList = [beamFuncList[ind][0] for ind in range(len(beamFuncList)) if beamFuncList[ind][1]]
    
    # Get source power flux density for each frequency band, if it is known for the target object
    refTarget = stdScanList[0].mainData.targetObject.get_reference_target()
    bandFreqs = calibScanList[0].bandFreqs
    sourcePowerFluxDensity = np.tile(np.nan, (len(bandFreqs)))
    for ind in xrange(len(bandFreqs)):
        try:
            sourcePowerFluxDensity[ind] = refTarget.get_flux_density_Jy(bandFreqs[ind] / 1e6)
        # pylint: disable-msg=W0704
        except AssertionError:
            pass
        except ValueError:
            pass
    # The antenna effective area and friends can only be calculated for sources with known and valid flux densities
    if np.all(np.isnan(sourcePowerFluxDensity)) or np.all(sourcePowerFluxDensity == 0.0) or \
       (len(validBeamFuncList) == 0):
        gainResults = None
    else:
        # Calculate antenna effective area and gain, per band
        deltaT = np.array([beamFuncPair[0].height for beamFuncPair in beamFuncList])
        # Replace bad bands with NaNs - better than discarding them, as this preserves the array shape
        invalidBeamFuncInds = [ind for ind in range(len(beamFuncList)) if not beamFuncList[ind][1]]
        deltaT[invalidBeamFuncInds] = np.nan
        pointSourceSensitivity = sourcePowerFluxDensity / deltaT
        effArea, antGain = misc.calc_ant_eff_area_and_gain(pointSourceSensitivity, bandFreqs)
        gainResults = sourcePowerFluxDensity, deltaT, pointSourceSensitivity, effArea, antGain
    
    # For general point sources, it is still possible to estimate pointing error (if beam fitting succeeded)
    if len(validBeamFuncList) == 0:
        pointingResults = None
    else:
        transform = TargetToInstantMountTransform(stdScanList)
        # Create target coordinate vector of appropriate size, filled with zeros
        targetCoords = np.zeros(transform.get_target_dimensions(), dtype='double')
        meanBeamCoords = np.array([func.mean for func in validBeamFuncList]).mean(axis=0)
        # The first few dimensions (typically 2) of target coordinate vector are loaded with mean beam position
        targetCoords[:meanBeamCoords.size] = meanBeamCoords
        mountCoords = transform(targetCoords)
        pointingResults = targetCoords, mountCoords.tolist()
        
    return gainResults, pointingResults, beamFuncList, calibScanList


## Extract point source information, with quantified statistical uncertainty.
# This reduces the power data obtained from multiple scans across a point source. In the process, the data is
# calibrated to remove receiver gain drifts, baselines are removed, and a beam pattern is fitted to the combined
# scans. For general (unnamed) sources, the estimated source position is returned as target coordinates.
# Additionally, for known calibrator sources, the antenna gain and effective area can be estimated from the
# known source flux density. All final results are provided with standard deviations, which are estimated from
# the data set itself via resampling or sigma-point techniques.
# @param stdScanList      List of StandardSourceScan objects, describing scans across a single point source
# @param method           Technique used to obtain statistics ('resample' (recommended default) or 'sigma-point')
# @param numSamples       Number of samples in resampling process (more is better but slower) [10]
# @return gainResults     List of gain-based results, of form (sourcePfd, deltaT, sensitivity, effArea, antGain)
# @return pointingResults Estimated point source position in target coordinates and mount coordinates
# @return beamFuncList    List of Gaussian beam functions, one per band
# @return calibScanList   List of data objects containing the fully calibrated main segments of each scan
# @return tempScanList    List of modified scan objects, after power-to-temp conversion but before baseline subtraction
# @todo Sigma-point broken, since powerToTempFactors are assumed to have identical shapes - not true if some scans
#       are using linear gain interpolation while others are using constant interpolation (2 factors vs 1)
# pylint: disable-msg=W0612,R0912
def reduce_point_source_scan_with_stats(stdScanList, method='resample', numSamples=10):
    # Do the reduction without any variations first, to obtain scan lists and other constant outputs
    tempScanList = copy.deepcopy(stdScanList)
    gainResults, pointingResults, beamFuncList, calibScanList = reduce_point_source_scan(tempScanList, randomise=False)
    
    # Resampling technique for estimating standard deviations
    if method == 'resample':
        gainList = [gainResults]
        pointingList = [pointingResults]
        # Re-run the reduction routine, which randomises itself internally each time
        for n in xrange(numSamples-1):
            results = reduce_point_source_scan(copy.deepcopy(stdScanList), randomise=True)
            gainList.append(results[0])
            pointingList.append(results[1])
        # Prune lists of any invalid (None) results
        gainList = [gainRes for gainRes in gainList if gainRes != None]
        pointingList = [pointRes for pointRes in pointingList if pointRes != None]
        if len(pointingList) < numSamples:
            logger.warning("Failed to fit Gaussian beam to source '" + stdScanList[0].sourceName + "' in " + 
                           str(numSamples - len(pointingList)) + " of " + str(numSamples) + " resampled cases.")
        # Obtain statistics and save in combined data structure
        if len(gainList) > 0:
            # Array of numSamples x 5 x numBands, containing gain results
            gainArr = np.array(gainList)
            gainResults = gainArr[0, 0], stats.MuSigmaArray(gainArr[:, 1].mean(axis=0), gainArr[:, 1].std(axis=0)), \
                          stats.MuSigmaArray(gainArr[:, 2].mean(axis=0), gainArr[:, 2].std(axis=0)), \
                          stats.MuSigmaArray(gainArr[:, 3].mean(axis=0), gainArr[:, 3].std(axis=0)), \
                          stats.MuSigmaArray(gainArr[:, 4].mean(axis=0), gainArr[:, 4].std(axis=0))
        if len(pointingList) > 0:
            # Two arrays of numSamples x D, containing pointing results in target and mount coordinates
            pointArrT = np.array([pointRes[0] for pointRes in pointingList])
            pointArrM = np.array([pointRes[1] for pointRes in pointingList])
            pointingResults = stats.MuSigmaArray(pointArrT.mean(axis=0), pointArrT.std(axis=0)), \
                              stats.MuSigmaArray(pointArrM.mean(axis=0), pointArrM.std(axis=0))
        
    # Sigma-point technique for estimating standard deviations
    elif method == 'sigma-point':
        raise NotImplementedError, 'Currently broken, see @todo...'
        # Currently only Fpt factors are sigma'ed, as the full data set will be too computationally intensive
        # pylint: disable-msg=W0101
        print [stdScan.powerToTempFactors.mu.shape for stdScan in stdScanList]
        fptMean = np.array([stdScan.powerToTempFactors.mu for stdScan in stdScanList])
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
            # Make sure the output has the correct shape in case the gain/pointing estimation failed, by adding NaNs
            if results[0] == None:
                numBands = len(stdScanList[0].mainData.bandFreqs)
                results[0] = np.tile(np.nan, (5, numBands)).tolist()
            if results[1] == None:
                results[1] = [[np.nan, np.nan, np.nan], [np.nan, np.nan]]
            return np.concatenate(results[0] + results[1])
        
        # Do sigma-point evaluation, and extract relevant outputs
        resMuSigma = stats.sp_uncorrelated_stats(wrapper_func, fptMuSigma)
        
        # Re-assemble results
        numBands = len(stdScanList[0].mainData.bandFreqs)
        if np.all(np.isnan(resMuSigma[numBands:2*numBands].mu)):
            gainResults = None
        else:
            gainResults = resMuSigma[0:numBands].mu, resMuSigma[numBands:2*numBands], \
                          resMuSigma[2*numBands:3*numBands], resMuSigma[3*numBands:4*numBands], \
                          resMuSigma[4*numBands:5*numBands]
        pointingResults = resMuSigma[5*numBands:5*numBands+3], resMuSigma[5*numBands+3:5*numBands+5]
        if np.all(np.isnan(pointingResults.mu)):
            pointingResults = None
    else:
        raise ValueError, "Unknown stats method '" + method + "'."
    
    return gainResults, pointingResults, beamFuncList, calibScanList, tempScanList
