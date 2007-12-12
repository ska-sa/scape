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
import xdmsbe.xdmsbelib.single_dish_data as sdd
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


## Container for the components of a standard scan across a source (or measurement of a fixed pointing).
# This struct contains all the parts that make up a standard scan:
# - information about the source or pointing location (name)
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
    ## @var badChannels
    # A sequence of channel indices, indicating RFI-corrupted frequency channels to be removed
    badChannels = None
    ## @var channelsPerBand
    # A sequence of lists of channel indices, indicating which frequency channels belong to each band
    channelsPerBand = None
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
        # If any NaNs enter the system here, pass the buck instead of tripping up the coordinate transformer
        if np.any(np.isnan(targetCoord)):
            return np.array([np.nan, np.nan])
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


## Pretty print dictionary of power blocks into a string.
# @param dataDict Dictionary of SingleDishData objects containing raw power data of a standard scan
# @return String containing summary of dictionary contents
def print_power_dict(dataDict):
    return ', '.join(['%s(%d)' % (keyval[0], keyval[1].powerData.shape[1]) for keyval in dataDict.iteritems()])


## Checks consistency of data blocks loaded from a set of FITS files.
# This checks that all the data blocks that make up a standard source scan have consistent parameters. The dictionary
# contains SingleDishData objects for each data block, indexed by string labels indicating the block ID.
# Errors are communicated via ValueError exceptions.
# @param dataDict Dictionary of SingleDishData objects containing raw power data of a standard scan
def check_data_consistency(dataDict):
    # Nothing to check
    if len(dataDict) == 0:
        return
    referenceBlock = dataDict.values()[0]
    for block in dataDict.values()[1:]:
        if np.any(block.freqs_Hz != referenceBlock.freqs_Hz):
            message = "Channel frequencies of two blocks in standard scan differ: " + str(block.freqs_Hz) + \
                      " vs. " + str(referenceBlock.freqs_Hz)
            logger.error(message)
            raise ValueError, message
    # Skip 'esn' and convert everything up to first '_' as experiment sequence number
    expSeqNums = [int(label[3:label.find('_')]) for label in dataDict.iterkeys()]
    if len(set(expSeqNums)) != 1:
        message = "Different experiment sequence numbers found within a standard scan: " + str(expSeqNums)
        logger.error(message)
        raise ValueError, message


## Loads a list of standard scans used for Tsys measurements from a chain of FITS files.
# The power data is loaded in coherency form, as this is easier to calibrate (keeping X and Y separate).
# @param fitsFileName      Name of initial file in FITS chain
# @return stdScanList      List of StandardSourceScan objects, one per pointing
# @return rawPowerDictList List of dictionaries containing SingleDishData objects representing all raw data blocks
# pylint: disable-msg=R0914
def load_tsys_pointing_list(fitsFileName):
    stdScanList = []
    rawPowerDictList = []
    noiseDiodeData = None
    print "               **** Loading pointing data from FITS files ****\n"
    # Iterate through multiple experiment sequences
    for fitsExpIter in fitsreader.ExperimentIterator(fitsFileName):
        print "===========================================================================================\n"
        print "                           **** POINTING %d ****\n" % len(stdScanList)
        stdScan = StandardSourceScan()
        rawPowerDict = {}
        # Iterate through a single experiment sequence
        for fitsReaderScan in fitsExpIter:
            # Load noise diode characteristics from first FITS file
            if noiseDiodeData == None:
                noiseDiodeData = gain_cal_data.NoiseDiodeData(fitsReaderScan)
            
            dataIdNameList = ['cold', 'hot', 'cal']
            dataSelectionList = [('Off', {'RX_ON_F': False, 'ND_ON_F': False, 'VALID_F': True}, False), \
                                 ('On', {'RX_ON_F': True, 'ND_ON_F': False, 'VALID_F': True}, False), \
                                 ('OnND', {'RX_ON_F': True, 'ND_ON_F': True, 'VALID_F': True}, False)]
            mainScanDict = fitsReaderScan.extract_data(dataIdNameList, dataSelectionList)
            
            # Load various parameters (the last file in experiment sequence determines the final values)
            stdScan.sourceName = fitsReaderScan.get_primary_header()['Target']
            stdScan.antennaBeamwidth_deg = fitsReaderScan.select_masked_column('CONSTANTS', 'Beamwidth')[0]
            rfiChannels = fitsReaderScan.get_rfi_channels()
            stdScan.channelsPerBand = [x.tolist() for x in fitsReaderScan.select_masked_column('BANDS', 'Channels')]
            
            print "Loading file: ", fitsReaderScan.fitsFilename
            print "Data blocks:  ", print_power_dict(mainScanDict)
            
            rawPowerDict.update(mainScanDict)
        
        check_data_consistency(rawPowerDict)
        
        # Set up standard scan object (rest of members will be filled in during calibration)
        onBlocks = [key for key in rawPowerDict.iterkeys() if key.endswith('On')]
        coldOnBlocks = [key for key in onBlocks if key.find('_cold') >= 0]
        # If there are no "cold" "on" blocks in set, discard the data, as both Tsys measurements and 
        # linearity checks would be impossible (any use cases without "cold" "on" blocks?)
        if len(coldOnBlocks) == 0:
            print "Experiment sequence discarded, as it doesn't contain 'cold' 'on' blocks."
            continue
        # First 'cold' 'on' block is used for Tsys measurement
        stdScan.mainData = rawPowerDict[coldOnBlocks[0]]
        stdScan.noiseDiodeData = noiseDiodeData
        stdScan.gainCalData = gain_cal_data.GainCalibrationData(rawPowerDict)
        # Merge bad channels due to RFI and those due to untrustworthy noise diode data
        stdScan.badChannels = list(set.union(set(rfiChannels), set(stdScan.gainCalData.badChannels)))
        # Successful standard scan finally gets added to lists here - this ensures lists are in sync
        stdScanList.append(stdScan)
        rawPowerDictList.append(rawPowerDict)
        
        print
        print "Pointing name:", stdScan.sourceName
        print "Number of frequency bands:", len(stdScan.channelsPerBand), "(may still decrease due to bad channels)"
        print "Number of channels:", len(stdScan.mainData.freqs_Hz), \
              "of which", len(stdScan.badChannels), "are bad (", len(rfiChannels), "RFI-flagged and", \
              len(stdScan.gainCalData.badChannels), "with insufficient variation between noise diode on/off blocks)"
        
    return stdScanList, rawPowerDictList


## Loads a list of standard scans across various point sources from a chain of FITS files.
# The power data is loaded in coherency form, as this is easier to calibrate (keeping X and Y separate).
# @param fitsFileName      Name of initial file in FITS chain
# @param fitBaseline       True if baseline fitting is required [True]
# @return stdScanList      List of StandardSourceScan objects, one per scan through a source
# @return rawPowerDictList List of dicts of SingleDishData objects, containing copies of all raw data blocks
# pylint: disable-msg=R0912,R0914,R0915
def load_point_source_scan_list(fitsFileName, fitBaseline=True):
    fitsIter = fitsreader.FitsIterator(fitsFileName)
    stdScanList = []
    rawPowerDictList = []
    noiseDiodeData = None
    print "           **** Loading point source scan data from FITS files ****\n"
    # Iterate through multiple standard scans
    while True:
        stdScan = StandardSourceScan()
        # Put this before printing the banner below, otherwise the banner will appear once too many
        try:
            fitsReaderPreCal = fitsIter.next()
        except StopIteration:
            break
        
        print "===========================================================================================\n"
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
        preCalDict = fitsReaderPreCal.extract_data(dataIdNameList, dataSelectionList)
        
        print "Loading file:                ", fitsReaderPreCal.fitsFilename
        print "PreCalibration data blocks:  ", print_power_dict(preCalDict)
        
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
            preScanDict = fitsReaderPreScan.extract_data(dataIdNameList, dataSelectionList)
            
            preScanData = copy.deepcopy(preScanDict.values()[0])
            
            print "Loading file:                ", fitsReaderPreScan.fitsFilename
            print "PreBaselineScan data blocks: ", print_power_dict(preScanDict)
        
        #..................................................................................................
        # This is the main part of the scan, which contains the calibrator source
        #..................................................................................................
        try:
            fitsReaderScan = fitsIter.next()
        except StopIteration:
            break
        
        dataIdNameList = ['scan']
        dataSelectionList = [('MainScan', {'RX_ON_F': True, 'ND_ON_F': False, 'VALID_F': True}, False)]
        mainScanDict = fitsReaderScan.extract_data(dataIdNameList, dataSelectionList)
        
        mainScanData = copy.deepcopy(mainScanDict.values()[0])
        # Load various parameters from the main scan (these should be the same for other scans as well)
        stdScan.sourceName = fitsReaderScan.get_primary_header()['Target']
        stdScan.antennaBeamwidth_deg = fitsReaderScan.select_masked_column('CONSTANTS', 'Beamwidth')[0]
        rfiChannels = fitsReaderScan.get_rfi_channels()
        stdScan.channelsPerBand = [x.tolist() for x in fitsReaderScan.select_masked_column('BANDS', 'Channels')]
        
        print "Loading file:                ", fitsReaderScan.fitsFilename
        print "MainScan data blocks:        ", print_power_dict(mainScanDict)
        
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
            postScanDict = fitsReaderPostScan.extract_data(dataIdNameList, dataSelectionList)
            
            postScanData = copy.deepcopy(postScanDict.values()[0])
            
            print "Loading file:                ", fitsReaderPostScan.fitsFilename
            print "PostBaselineScan data blocks:", print_power_dict(postScanDict)
        
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
        postCalDict = fitsReaderPostCal.extract_data(dataIdNameList, dataSelectionList)
        
        print "Loading file:                ", fitsReaderPostCal.fitsFilename
        print "PostCalibration data blocks: ", print_power_dict(postCalDict)
        
        #..................................................................................................
        # Now package the data in objects
        #..................................................................................................
        
        # Save raw power objects
        rawPowerDict = {}
        rawPowerDict.update(preCalDict)
        rawPowerDict.update(mainScanDict)
        rawPowerDict.update(postCalDict)
        if fitBaseline:
            rawPowerDict.update(preScanDict)
            rawPowerDict.update(postScanDict)
        
        check_data_consistency(rawPowerDict)
        
        # Set up standard scan object (rest of members will be filled in during calibration)
        stdScan.mainData = mainScanData
        stdScan.noiseDiodeData = noiseDiodeData
        calDict = {}
        calDict.update(preCalDict)
        calDict.update(postCalDict)
        stdScan.gainCalData = gain_cal_data.GainCalibrationData(calDict)
        # Merge bad channels due to RFI and those due to untrustworthy noise diode data
        stdScan.badChannels = list(set.union(set(rfiChannels), set(stdScan.gainCalData.badChannels)))
        if fitBaseline:
            stdScan.baselineDataList = [preScanData, postScanData]
        
        # Successful standard scan finally gets added to lists here - this ensures lists are in sync
        rawPowerDictList.append(rawPowerDict)
        stdScanList.append(stdScan)
        
        print
        print "Source name:", stdScan.sourceName
        print "Scan start coordinate [az, el] = [%5.3f, %5.3f]" \
              % (rad_to_deg(mainScanData.azAng_rad[0]), rad_to_deg(mainScanData.elAng_rad[0]))
        print "Scan stop coordinate  [az, el] = [%5.3f, %5.3f]" \
              % (rad_to_deg(mainScanData.azAng_rad[-1]), rad_to_deg(mainScanData.elAng_rad[-1]))
        print "Number of frequency bands:", len(stdScan.channelsPerBand), "(may still decrease due to bad channels)"
        print "Number of channels:", len(stdScan.mainData.freqs_Hz), \
              "of which", len(stdScan.badChannels), "are bad (", len(rfiChannels), "RFI-flagged and", \
              len(stdScan.gainCalData.badChannels), "with insufficient variation between noise diode on/off blocks)"
        
    return stdScanList, rawPowerDictList


## Calibrate a single scan, by correcting gain drifts, combining channels into bands, and subtracting a baseline.
# This also modifies the stdScan object, by converting its data from raw power to temperature values, and combining
# its channels into bands. If this is not desired, use deepcopy on stdScan before calling this function. The baseline
# is only subtracted from the main scan (which is returned).
# @param stdScan   StandardSourceScan object containing scan and all auxiliary scans and info for calibration
# @param randomise True if fits should be randomised, as part of a larger Monte Carlo run
# @return SingleDishData object containing calibrated main scan
def calibrate_scan(stdScan, randomise):
    # Set up power-to-temp conversion function (optionally randomising it)
    stdScan.powerToTempFunc = stdScan.gainCalData.power_to_temp_func(stdScan.noiseDiodeData, randomise=randomise)
    # Convert the main segment of scan to temperature and bands, and make a copy to preserve original data
    stdScan.mainData.convert_power_to_temp(stdScan.powerToTempFunc)
    stdScan.mainData.merge_channels_into_bands(stdScan.channelsPerBand, stdScan.badChannels)
    calibratedScan = copy.deepcopy(stdScan.mainData)
    # Without baseline data segments, the calibration is done
    if stdScan.baselineDataList == None:
        return calibratedScan
    # Convert baseline data to temperature and bands, and concatenate them into a single structure
    for ind, baselineData in enumerate(stdScan.baselineDataList):
        baselineData.convert_power_to_temp(stdScan.powerToTempFunc)
        baselineData.merge_channels_into_bands(stdScan.channelsPerBand, stdScan.badChannels)
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
# @param stdScanList     List of StandardSourceScan objects (modified by call - power converted to temp, and bands)
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
        totalPowerData.append(calibratedScan.stokes('I'))
        calibScanList.append(calibratedScan)
    targetCoords = np.concatenate(targetCoords)[:, 0:2]
    totalPowerData = np.concatenate(totalPowerData)
#    beamFuncList = fit_beam_pattern_old(targetCoords, totalPowerData, randomise)
    targetBeamwidth = deg_to_rad(stdScanList[0].antennaBeamwidth_deg)
    beamFuncList = fit_beam_pattern(targetCoords, totalPowerData, targetBeamwidth, randomise)
    return beamFuncList, calibScanList


## Extract all information from an unresolved point source scan.
# This reduces the power data obtained from multiple scans across a point source. In the process, the data is
# calibrated to remove receiver gain drifts, channels are combined into bands, baselines are removed, and a beam 
# pattern is fitted to the combined scans, which allows the source position and strength to be estimated. 
# For general (unnamed) sources, the estimated source position is returned as target coordinates. This is
# effectively the pointing error, as the target is at the origin of the target coordinate system. Additionally, 
# for known calibrator sources, the antenna gain and effective area can be estimated from the known source flux 
# density. If either the gain or pointing estimation did not succeed, the corresponding result is returned as None.
# @param stdScanList      List of StandardSourceScan objects, describing scans across a single point source
#                         (modified by call - power converted to temp, and channels combined into bands)
# @param randomise        True if fits should be randomised, as part of a larger Monte Carlo run [False]
# @return gainResults     List of gain-based results, of form (sourcePfd, deltaT, sensitivity, effArea, antGain)
# @return pointingResults Estimated point source position in target coordinates, and mount coordinates
# @return beamFuncList    List of Gaussian beam functions and valid flags, one per band
# @return calibScanList   List of data objects containing the fully calibrated main segments of each scan
# pylint: disable-msg=R0914
def reduce_point_source_scan(stdScanList, randomise=False):
    # Calibrate across all scans, and fit a beam pattern to estimate source position and strength
    beamFuncList, calibScanList = calibrate_and_fit_beam_pattern(stdScanList, randomise)
    # List of valid and invalid beam functions
    validBeamFuncList = [beamFuncList[ind][0] for ind in range(len(beamFuncList)) if beamFuncList[ind][1]]
    invalidBeamFuncInds = [ind for ind in range(len(beamFuncList)) if not beamFuncList[ind][1]]
    
    # Get source power flux density for each frequency band, if it is known for the target object
    refTarget = stdScanList[0].mainData.targetObject.get_reference_target()
    bandFreqs = calibScanList[0].freqs_Hz
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
        deltaT[invalidBeamFuncInds] = np.nan
        pointSourceSensitivity = sourcePowerFluxDensity / deltaT
        effArea, antGain = misc.calc_ant_eff_area_and_gain(pointSourceSensitivity, bandFreqs)
        gainResults = sourcePowerFluxDensity, deltaT, pointSourceSensitivity, effArea, antGain
    
    # For general point sources, it is still possible to estimate pointing error (if beam fitting succeeded)
    # The peak temperature of the beam is also useful, for checking the source you are looking at
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
        # Obtain peak beam temperature per band (assumed to be due to point source) - invalid beams replaced with NaNs
        deltaT = np.array([beamFuncPair[0].height for beamFuncPair in beamFuncList])
        deltaT[invalidBeamFuncInds] = np.nan
        pointingResults = targetCoords, mountCoords.tolist(), deltaT
    
    return gainResults, pointingResults, beamFuncList, calibScanList


## Extract point source information, with quantified statistical uncertainty.
# This reduces the power data obtained from multiple scans across a point source. In the process, the data is
# calibrated to remove receiver gain drifts, channels are combined into bands, baselines are removed, and a beam 
# pattern is fitted to the combined scans. For general (unnamed) sources, the estimated source position is returned
# as target coordinates. Additionally, for known calibrator sources, the antenna gain and effective area can be
# estimated from the known source flux density. All final results are provided with standard deviations, which are
# estimated from the data set itself via resampling or sigma-point techniques.
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
            gainResults = gainArr[0, 0], stats.mu_sigma(gainArr[:, 1]), stats.mu_sigma(gainArr[:, 2]), \
                                         stats.mu_sigma(gainArr[:, 3]), stats.mu_sigma(gainArr[:, 4])
        if len(pointingList) > 0:
            # Two arrays of numSamples x D, containing pointing results in target and mount coordinates
            pointArrT = np.array([pointRes[0] for pointRes in pointingList])
            pointArrM = np.array([pointRes[1] for pointRes in pointingList])
            # Array of numSamples x numBands, containing peak beam temperature per band
            pointArrD = np.array([pointRes[2] for pointRes in pointingList])
            pointingResults = stats.mu_sigma(pointArrT), stats.mu_sigma(pointArrM), stats.mu_sigma(pointArrD)
    
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
                numBands = len(stdScanList[0].mainData.freqs_Hz)
                results[0] = np.tile(np.nan, (5, numBands)).tolist()
            if results[1] == None:
                results[1] = [[np.nan, np.nan, np.nan], [np.nan, np.nan], np.tile(np.nan, (numBands)).tolist()]
            return np.concatenate(results[0] + results[1])
        
        # Do sigma-point evaluation, and extract relevant outputs
        resMuSigma = stats.sp_uncorrelated_stats(wrapper_func, fptMuSigma)
        
        # Re-assemble results
        numBands = len(stdScanList[0].mainData.freqs_Hz)
        if np.all(np.isnan(resMuSigma[numBands:2*numBands].mu)):
            gainResults = None
        else:
            gainResults = resMuSigma[0:numBands].mu, resMuSigma[numBands:2*numBands], \
                          resMuSigma[2*numBands:3*numBands], resMuSigma[3*numBands:4*numBands], \
                          resMuSigma[4*numBands:5*numBands]
        pointingResults = resMuSigma[5*numBands:5*numBands+3], resMuSigma[5*numBands+3:5*numBands+5], \
                          resMuSigma[5*numBands+5:6*numBands+5]
        if np.all(np.isnan(pointingResults.mu)):
            pointingResults = None
    else:
        raise ValueError, "Unknown stats method '" + method + "'."
    
    return gainResults, pointingResults, beamFuncList, calibScanList, tempScanList


## Extract Tsys information from multiple pointings.
# This reduces the power data obtained from multiple pointings at cold sky patches. In the process, the data is
# calibrated to remove receiver gain drifts, and channels are combined into bands. It returns an array of
# Tsys measurements, of shape (numPointings, 2, numBands) (2 => X and Y input ports), and the median times
# and elevation angles of each pointing.
# @param stdScanList      List of StandardSourceScan objects, containing multiple pointing measurements
#                         (modified by call - power converted to temp and bands, and possibly randomised mainData)
# @param randomise        True if data should be randomised, as part of a larger Monte Carlo run [False]
# @return tsysResults     List of results, of form (tsys, timeStamps, elAngs_deg)
def reduce_tsys_pointings(stdScanList, randomise=False):
    tsys = []
    timeStamps = []
    elAngs_deg = []
    coherency = sdd.power_index_dict(False)
    for stdScan in stdScanList:
        # Randomise cold power data itself (assumes power has Gaussian distribution...)
        if randomise:
            powerStats = stats.mu_sigma(stdScan.mainData.powerData, axis=1)
            stdScan.mainData.powerData = powerStats.mu[:, np.newaxis, :] + \
                powerStats.sigma[:, np.newaxis, :] * np.random.standard_normal(stdScan.mainData.powerData.shape)
        # Calibrate scan (power-to-temp and channel-to-band conversion)
        calibratedScan = calibrate_scan(stdScan, randomise)
        # Save results
        tsys.append(calibratedScan.powerData.mean(axis=1)[[coherency['XX'], coherency['YY']]].real)
        midPoint = len(calibratedScan.timeSamples) // 2
        timeStamps.append(calibratedScan.timeSamples[midPoint])
        elAngs_deg.append(rad_to_deg(calibratedScan.elAng_rad[midPoint]))
    return (np.array(tsys), np.array(timeStamps), np.array(elAngs_deg))


## Extract Tsys information from multiple pointings, with quantified statistical uncertainty.
# This reduces the power data obtained from multiple pointings at cold sky patches. In the process, the data is
# calibrated to remove receiver gain drifts, and channels are combined into bands. The Tsys results are provided 
# with standard deviations, which are estimated from the data set itself via resampling.
# @param stdScanList      List of StandardSourceScan objects, containing multiple pointing measurements
# @param numSamples       Number of samples in resampling process (more is better but slower) [10]
# @return tsysResults     List of results, of form (tsys, timeStamps, elAngs_deg)
# @return tempScanList    List of modified scan objects, after power-to-temp conversion
def reduce_tsys_pointings_with_stats(stdScanList, numSamples=10):
    # Do the reduction without any variations first, to obtain scan lists and other constant outputs
    tempScanList = copy.deepcopy(stdScanList)
    tsys, timeStamps, elAngs_deg = reduce_tsys_pointings(tempScanList, randomise=False)

    tsysList = [tsys]
    # Re-run the reduction routine, which randomises itself internally each time
    for n in xrange(numSamples-1):
        results = reduce_tsys_pointings(copy.deepcopy(stdScanList), randomise=True)
        tsysList.append(results[0])
    # Obtain statistics
    tsysResults = (stats.mu_sigma(np.array(tsysList), axis=0), timeStamps, elAngs_deg)
    return tsysResults, tempScanList


## Do linearity test on raw power blocks containing noise diode on/off segments at different power levels.
# This checks whether the switching of the noise diode causes the same increase in power at low ("cold") and 
# high ("hot") power levels, using a statistical test. If this is not the case, it may reveal non-linearities
# in the amplifiers. The input dictionary of power data blocks should contain at least "cold_On", "cold_OnND",
# "hot_On" and "hot_OnND" blocks (all in raw uncalibrated form). The input data blocks should be in coherency form.
# The output arrays are typically of shape (2, numChannels), where 2 refers to the X and Y polarisations.
# @param powerBlockDict Dictionary of SingleDishData objects, containing uncalibrated coherencies
# @param alpha          Student-T test alpha value [0.05]
# @return isLinear      Array of shape (2, numChannels) of linearity test result flags (True/False)
# @return confIntervals Array of shape (2, 2, numChannels), where confInterval[0] is the start and confInterval[1]
#                       is the end of the linearity confidence interval for each polarisation and channel
#                       The interval is normalised as a fraction of the "cold" power delta
# @return degsFreedom   Number of degrees of freedom of the Student-t distribution, in (2, numChannels) array
def linearity_test(powerBlockDict, alpha=0.05):
    # Obtain labels for the required blocks
    offLabels = [key for key in powerBlockDict.iterkeys() if key.endswith('On')]
    onLabels = [key for key in powerBlockDict.iterkeys() if key.endswith('OnND')]
    coldOffLabels = [key for key in offLabels if key.find('_cold') >= 0]
    coldOnLabels = [key for key in onLabels if key.find('_cold') >= 0]
    hotOffLabels = [key for key in offLabels if key.find('_hot') >= 0]
    hotOnLabels = [key for key in onLabels if key.find('_hot') >= 0]
    # Bail if some of the required blocks are missing
    if (len(coldOffLabels) == 0) or (len(coldOnLabels) == 0) or (len(hotOffLabels) == 0) or (len(hotOnLabels) == 0):
        return None, None, None
    # Find a common set of labels with the same esn number
    coherency = sdd.power_index_dict(False)
    blockSets = [{'coldOff' : powerBlockDict[coldOff].powerData[[coherency['XX'], coherency['YY']]].real, \
                  'coldOn'  : powerBlockDict[coldOn].powerData[[coherency['XX'], coherency['YY']]].real, \
                  'hotOff'  : powerBlockDict[hotOff].powerData[[coherency['XX'], coherency['YY']]].real, \
                  'hotOn'   : powerBlockDict[hotOn].powerData[[coherency['XX'], coherency['YY']]].real} \
                  for coldOff in coldOffLabels for coldOn in coldOnLabels \
                  for hotOff in hotOffLabels for hotOn in hotOnLabels \
                  if coldOff.split('_')[0] == coldOn.split('_')[0] == hotOff.split('_')[0] == hotOn.split('_')[0]]
    # Bail if there is no common set
    if len(blockSets) == 0:
        return None, None, None
    # Obtain stats of first common set, to satisfy interface of calc_conf_interval_diff_diff2means
    # pylint: disable-msg=W0232
    class Struct:
        pass
    coldOff = Struct()
    coldOff.mean = blockSets[0]['coldOff'].mean(axis=1)
    coldOff.var = blockSets[0]['coldOff'].var(axis=1)
    coldOff.num = blockSets[0]['coldOff'].shape[1]
    coldOff.shape = coldOff.mean.shape
    coldOn = Struct()
    coldOn.mean = blockSets[0]['coldOn'].mean(axis=1)
    coldOn.var = blockSets[0]['coldOn'].var(axis=1)
    coldOn.num = blockSets[0]['coldOn'].shape[1]
    coldOn.shape = coldOn.mean.shape
    hotOff = Struct()
    hotOff.mean = blockSets[0]['hotOff'].mean(axis=1)
    hotOff.var = blockSets[0]['hotOff'].var(axis=1)
    hotOff.num = blockSets[0]['hotOff'].shape[1]
    hotOff.shape = hotOff.mean.shape
    hotOn = Struct()
    hotOn.mean = blockSets[0]['hotOn'].mean(axis=1)
    hotOn.var = blockSets[0]['hotOn'].var(axis=1)
    hotOn.num = blockSets[0]['hotOn'].shape[1]
    hotOn.shape = hotOn.mean.shape
    # Do statistical test
    confIntervals, degsFreedom = stats.calc_conf_interval_diff_diff2means(coldOff, coldOn, hotOff, hotOn, alpha)
    isLinear = stats.check_equality_of_means(confIntervals)
    # Normalise confidence intervals to be the ratio of (hot delta - cold delta) to cold delta
    coldDelta = coldOn.mean - coldOff.mean
    confIntervals /= np.array([coldDelta, coldDelta])
    return isLinear, confIntervals, degsFreedom


## Check if Tsys measurements are stable over time.
# Check the Tsys measurements in each polarisation and frequency band to see if it is normally distributed around
# the mean of the measurements. This is done with a D'Agostino-Pearson "Goodness-of-Fit" test.
# @param tsys      Tsys measurement array of shape (numMeasurements, 2, numBands) (2 => X and Y input ports)
# @param alpha     Chi-squared alpha value for acceptance of null hypothesis [0.05]
# @return isStable Array of shape (2, numBands) of stability test result flags (True/False)
# @todo Check, review and fix expected residual std!!! Currently, this is not included in test.
#       Use Hogg & Tanis ( http://tinyurl.com/2h3uea ) as reference for the statistical tests used here.
def stability_test(tsys, alpha=0.05):
    numBands = tsys.shape[2]
    isStable = np.ndarray((2, numBands), dtype=bool)
    # Loop over frequency bands
    for band in range(numBands):
        # Check X polarisation stability
        tsysX = tsys[:, 0, band]
        fitG, pVal, k_2, chiSqThreshold, muC = stats.check_model_fit_agn(xMu = tsysX, y = tsysX.mean(),
                                                                         func = lambda x: x, alpha=alpha)
        isStable[0, band] = fitG and muC
        # Check Y polarisation stability
        tsysY = tsys[:, 1, band]
        fitG, pVal, k_2, chiSqThreshold, muC = stats.check_model_fit_agn(xMu = tsysY, y = tsysY.mean(),
                                                                         func = lambda x: x, alpha=alpha)
        isStable[1, band] = fitG and muC
    return isStable
