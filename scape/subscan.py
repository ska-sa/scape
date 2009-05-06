## @file single_dish_data.py
#
# Class that contains single-dish power measurements and related pointing and timestamp data.
#
# copyright (c) 2007 SKA/KAT. All rights reserved.
# @author Ludwig Schwardt <ludwig@ska.ac.za>, Rudolph van der Merwe <rudolph@ska.ac.za>,
#         Robert Crida <robert.crida@ska.ac.za>
# @date 2007-08-28

# pylint: disable-msg=C0103,R0902

from acsm.coordinate import Coordinate
import acsm.transform.transformfactory as transformfactory
import xdmsbe.xdmsbelib.misc as misc
import numpy as np
import logging
import copy

logger = logging.getLogger("xdmsbe.xdmsbelib.single_dish_data")

#----------------------------------------------------------------------------------------------------------------------
#--- FUNCTIONS
#----------------------------------------------------------------------------------------------------------------------

## Dictionary that maps polarisation type names to power data matrix indices.
# This allows the use of symbolic names, which are clearer than numeric indices.
# @param isStokes Boolean indicating whether Stokes IQUV (True) or cross power / coherency (False) is required
# @return Dictionary mapping names to indices
def power_index_dict(isStokes):
    if isStokes:
        return {'I': 0, 'Q': 1, 'U': 2, 'V': 3}
    else:
        return {'XX': 0, 'XY': 1, 'YX': 2, 'YY': 3}

#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  SingleDishData
#----------------------------------------------------------------------------------------------------------------------

## A container for single-dish data consisting of polarised power measurements combined with pointing information.
# The main data member of this class is the 3-D powerData array, which stores power (autocorrelation) measurements as a
# function of polarisation index, time and frequency channel/band. This array can take one of two forms:
# - Stokes [I,Q,U,V] parameters, which are always real in the case of a single dish (I = the non-negative total power)
# - Coherencies [XX,YY,XY,YX], where XX and YY are real and non-negative polarisation powers, and XY and YX can be
#   complex in the general case (this makes powerData a potentially complex-valued array).
# The class also stores pointing data (azimuth/elevation/rotator angles), timestamps, a list of frequencies, and
# the mount coordinate system and target object, which permits the calculation of any relevant coordinates.
class SingleDishData(object):
    ## Initialiser/constructor
    #
    # @param    self              The current object
    # @param    powerData         3-D array, of shape (4, number of time samples, number of frequency channels/bands)
    #                             This can be interpreted as 4 data blocks of dimension time x channels/bands
    # @param    isStokes          True if power data is in Stokes [I,Q,U,V] format, or False if in [XX,XY,YX,YY] format
    # @param    timeSamples       Sequence of timestamps for each data block (in seconds since epoch)
    # @param    azAng_rad         Azimuth angle sequence for each data block (in radians)
    # @param    elAng_rad         Elevation angle sequence for each data block (in radians)
    # @param    rotAng_rad        Rotator stage angle sequence for each data block (in radians)
    # @param    freqs_Hz          List of channel/band centre frequencies (in Hz)
    # @param    mountCoordSystem  Mount coordinate system object (see acsm.coordinatesystem module)
    # @param    targetObject      Target object (see acsm.targets module)
    # pylint: disable-msg=R0913
    def __init__(self, powerData, isStokes, timeSamples, azAng_rad, elAng_rad, rotAng_rad, freqs_Hz, \
                 mountCoordSystem=None, targetObject=None):
        ## @var powerData
        # Power data array, of shape (4, number of time samples, number of frequency channels/bands).
        self.powerData = None
        ## @var isStokes
        # True if power data is in Stokes [I,Q,U,V] format, or False if in [XX,XY,YX,YY] format
        self.isStokes = isStokes
        ## @var timeSamples
        # Sequence of timestamps for data block, in seconds since epoch
        self.timeSamples = timeSamples
        ## @var azAng_rad
        # Mount azimuth angle sequence for data block, in radians
        self.azAng_rad = azAng_rad
        ## @var elAng_rad
        # Mount elevation angle sequence for data block, in radians
        self.elAng_rad = elAng_rad
        ## @var rotAng_rad
        # Rotator stage angle for data block, in radians
        self.rotAng_rad = rotAng_rad
        ## @var freqs_Hz
        # List of channel/band centre frequencies, in Hz (keep doubles to prevent precision issues)
        self.freqs_Hz = np.asarray(freqs_Hz, dtype='double')
        
        ## @var mountCoordSystem
        # Coordinate system for mount (azimuth/evelation/rotation angles)
        self.mountCoordSystem = mountCoordSystem
        ## @var targetObject
        # Coordinate system for target output (some projection, typically)
        self.targetObject = targetObject
        ## @var targetCoords
        # Target coordinates, as an NxD array (N = number of samples, D = target coordinate dimension)
        self.targetCoords = None
        self.update_target_coords()
        
        ## @var originalChannels
        # List of original channel indices associated with each frequency channel/band in powerData
        self.originalChannels = range(len(self.freqs_Hz))
        
        self._powerConvertedToTemp = False
        self._channelsMergedIntoBands = False
        self._baselineSubtracted = False
        
        # Set power data to appropriate type, depending on whether it is Stokes or coherency
        if isStokes:
            self.powerData = np.array(np.asarray(powerData).real, dtype='double')
        else:
            self.powerData = np.asarray(powerData, dtype='complex128')
    
    ## Use mount and target coordinate systems (if given) to calculate target coordinates for scan.
    # @param self The current object
    # @return self The updated object
    def update_target_coords(self):
        # Coordinate systems not complete - no target coordinates possible
        if not (self.mountCoordSystem and self.targetObject):
            self.targetCoords = None
            return self
        targetCoordSystem = self.targetObject.get_coordinate_system()
        mountToTarget = transformfactory.get_transformer(self.mountCoordSystem, targetCoordSystem)
        numTimeSamples = len(self.timeSamples)
        self.targetCoords = np.zeros((numTimeSamples, targetCoordSystem.get_dimensions()), dtype='double')
        
        for k in np.arange(numTimeSamples):
            mountCoordinate = Coordinate(self.mountCoordSystem, \
                                         [self.azAng_rad[k], self.elAng_rad[k], self.rotAng_rad[k]])
            targetCoordinate = mountToTarget.transform_coordinate(mountCoordinate, self.timeSamples[k])
            self.targetCoords[k, :] = targetCoordinate.get_vector()
        return self
    
    ## Return power data for specified coherency.
    # Returns coherency data directly, or calculate it if required.
    # @param self The current object
    # @param key  String indicating which coherency to return: 'XX', 'XY', 'YX', or 'YY'
    # @return Matrix of power values, of shape (number of time samples, number of frequency channels/bands)
    def coherency(self, key):
        # If data is already in coherency form, just return appropriate row
        if not self.isStokes:
            return self.powerData[power_index_dict(self.isStokes)[key]]
        else:
            if key == 'XX':
                return 0.5 * (self.stokes('I') +    self.stokes('Q')).real
            elif key == 'XY':
                return 0.5 * (self.stokes('U') + 1j*self.stokes('V'))
            elif key == 'YX':
                return 0.5 * (self.stokes('U') - 1j*self.stokes('V'))
            elif key == 'YY':
                return 0.5 * (self.stokes('I') -    self.stokes('Q')).real
            else:
                raise TypeError, "Invalid coherency key: should be one of XX, XY, YX, or YY"
    
    ## Return power data for specified Stokes parameter.
    # Returns Stokes data directly, or calculate it if required. Stokes 'I' is the total power, for reference.
    # @param self The current object
    # @param key  String indicating which Stokes parameter to return: 'I', 'Q', 'U', or 'V'
    # @return Matrix of power values, of shape (number of time samples, number of frequency channels/bands)
    def stokes(self, key):
        # If data is already in Stokes form, just return appropriate row
        if self.isStokes:
            return self.powerData[power_index_dict(self.isStokes)[key]]
        else:
            if key == 'I':
                return (self.coherency('XX') + self.coherency('YY')).real
            elif key == 'Q':
                return (self.coherency('XX') - self.coherency('YY')).real
            elif key == 'U':
                return (self.coherency('XY') + self.coherency('YX')).real
            elif key == 'V':
                return (self.coherency('XY') - self.coherency('YX')).imag
            else:
                raise TypeError, "Invalid Stokes key: should be one of I, Q, U or V"
    
    ## Convert the contained power buffer from Stokes vectors to coherency vectors.
    # This results in a complex-valued powerData array. If the data is already in coherency form, do nothing.
    # @param self  The current object
    # @return self The current object
    def convert_to_coherency(self):
        if self.isStokes:
            lookup = power_index_dict(False)
            keys = np.array(lookup.keys())[np.argsort(lookup.values())]
            self.powerData = np.array([self.coherency(k) for k in keys], dtype='complex128')
            self.isStokes = False
        return self
    
    ## Convert the contained power buffer from coherency vectors to Stokes vectors.
    # This is forced to result in a real-valued powerData array. If the data is already in Stokes form, do nothing.
    # @param self  The current object
    # @return self The current object
    def convert_to_stokes(self):
        if not self.isStokes:
            lookup = power_index_dict(True)
            keys = np.array(lookup.keys())[np.argsort(lookup.values())]
            self.powerData = np.array([self.stokes(k) for k in keys], dtype='double')
            self.isStokes = True
        return self
    
    ## Appends another data object to the current one.
    # This appends the data of the second object to the main object (including power data,
    # coordinates, timestamps, etc.). It also ensures the two objects are compatible.
    # @param self  The current object
    # @param other The object to append to current object
    # @return self The updated object
    # pylint: disable-msg=W0212
    def append(self, other):
        # Ensure reference targets are the same
#        if self.targetObject.get_reference_target() != other.targetObject.get_reference_target():
#            message = "Cannot concatenate data objects with incompatible reference targets."
#            logger.error(message)
#            raise ValueError, message
        # Ensure mount coordinates (az/el/rot) are compatible
        if self.mountCoordSystem != other.mountCoordSystem:
            message = "Cannot concatenate data objects with incompatible mount coordinate systems."
            logger.error(message)
            raise ValueError, message
        # Ensure objects are in the same state
        if (self._powerConvertedToTemp    != other._powerConvertedToTemp) or \
           (self._channelsMergedIntoBands != other._channelsMergedIntoBands) or \
           (self._baselineSubtracted      != other._baselineSubtracted):
            message = "Data objects are not in the same state, as their flags differ."
            logger.error(message)
            raise ValueError, message
        # Ensure list of frequencies and original channel indices are the same
        if np.any(self.freqs_Hz != other.freqs_Hz) or (self.originalChannels != other.originalChannels):
            message = "Cannot concatenate data objects with different frequency channels/bands."
            logger.error(message)
            raise ValueError, message
        # Convert power data to appropriate format if it differs for the two objects
        if self.isStokes != other.isStokes:
            # First make a copy to prevent unexpected mutation of "other" object
            other = copy.deepcopy(other)
            if self.isStokes:
                other.convert_to_stokes()
            else:
                other.convert_to_coherency()
        # Concatenate coordinate vectors, and power data along time axis
        self.timeSamples = np.concatenate((self.timeSamples, other.timeSamples))
        self.azAng_rad = np.concatenate((self.azAng_rad, other.azAng_rad))
        self.elAng_rad = np.concatenate((self.elAng_rad, other.elAng_rad))
        self.rotAng_rad = np.concatenate((self.rotAng_rad, other.rotAng_rad))
        self.powerData = np.concatenate((self.powerData, other.powerData), axis=1)
        # Convert all target coordinate data to target coord system of first object
        self.update_target_coords()
        return self
    
    ## Convert raw power measurements (W) into temperature (K) using the provided conversion function.
    # The main argument is a callable object with the signature 'factor = func(time)', which provides
    # an interpolated conversion factor function. The conversion factor returned by func should be an array
    # of shape (4, numTimeSamples, numChannels), which will be multiplied with the power data in coherency
    # form to obtain temperatures.
    # @param self The current object
    # @param func The power-to-temperature conversion factor as a function of time
    # @return self The current object
    def convert_power_to_temp(self, func):
        # Already done, so don't do it again
        if self._powerConvertedToTemp:
            return self
        # Obtain interpolated conversion factor
        powerToTempFactor = func(self.timeSamples)
        originallyIsStokes = self.isStokes
        # Convert coherency power to temperature, and restore Stokes/coherency status
        self.convert_to_coherency()
        self.powerData *= powerToTempFactor
        if originallyIsStokes:
            self.convert_to_stokes()
        self._powerConvertedToTemp = True
        return self
    
    ## Merge frequency channels into bands.
    # The frequency channels are grouped into bands, and the power data is merged and averaged within each band.
    # Each band contains the average power of its constituent channels. The average power is simpler to use
    # than the total power in each band, as total power is dependent on the bandwidth of each band.
    # The channelsPerBand mapping contains a list of lists of channel indices, indicating which channels belong
    # to each band.
    # @param self            The current object
    # @param channelsPerBand A sequence of lists of channel indices, indicating which channels belong to each band
    # @return self           The current object
    def merge_channels_into_bands(self, channelsPerBand):
        # Already done, so don't do it again
        if self._channelsMergedIntoBands:
            return self
        if not self._powerConvertedToTemp:
            message = "First convert raw power measurements to temperatures before merging channels into bands."
            logger.error(message)
            raise RuntimeError, message
        # Merge and average power data into new array (keep same type as original data, which may be complex)
        bandPowerData = np.zeros(list(self.powerData.shape[0:2]) + [len(channelsPerBand)], dtype=self.powerData.dtype)
        for bandIndex, bandChannels in enumerate(channelsPerBand):
            bandPowerData[:, :, bandIndex] = self.powerData[:, :, bandChannels].mean(axis=2)
        self.powerData = bandPowerData
        # Each band centre frequency is the mean of the corresponding channel centre frequencies
        self.freqs_Hz = np.array([self.freqs_Hz[chans].mean() for chans in channelsPerBand], dtype='double')
        self.originalChannels = channelsPerBand
        self._channelsMergedIntoBands = True
        return self
    
    ## Subtract a baseline function from the scan data.
    # The main argument is a callable object with the signature 'temp = func(ang)', which provides
    # an interpolated baseline function based on either elevation or azimuth angle.
    # @param self         The current object
    # @param func         The baseline function (as a function of elevation or azimuth angle)
    # @param useElevation True if elevation angle is to be used
    # @return self        The current object
    def subtract_baseline(self, func, useElevation):
        # Already done, so don't do it again
        if self._baselineSubtracted:
            return self
        if not self._powerConvertedToTemp:
            message = "Cannot subtract baseline from unconverted (raw) power measurements."
            logger.error(message)
            raise RuntimeError, message
        if not self._channelsMergedIntoBands:
            message = "Baseline should only be subtracted after frequency channels are merged into bands."
            logger.error(message)
            raise RuntimeError, message
        # Obtain baseline
        if useElevation:
            baselineData = func(self.elAng_rad)
        else:
            baselineData = func(self.azAng_rad)
        # Subtract baseline
        self.powerData -= baselineData
        self._baselineSubtracted = True
        return self
    
    ## Parallactic rotation angle for duration of data.
    # This calculates the parallactic rotation angle experienced by the target at each time instant of the data set.
    # @param self         The current object
    # @return             Array of angles in radians, one per time instant
    def parallactic_rotation(self):
        mountPosition = self.mountCoordSystem.get_attribute("position")
        return misc.parallactic_rotation(self.targetObject, mountPosition, self.timeSamples)
