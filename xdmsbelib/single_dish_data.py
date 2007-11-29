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
import numpy as np
import logging
import copy

logger = logging.getLogger("xdmsbe.xdmsbelib.single_dish_data")

#----------------------------------------------------------------------------------------------------------------------
#--- FUNCTIONS
#----------------------------------------------------------------------------------------------------------------------

## Dictionary that maps polarisation type names to power data matrix indices.
# This allows the use of symbolic names, which are clearer than numeric indices.
# @param stokes Boolean indicating wether Stokes IQUV (True) or cross power / coherency (False) is required
# @return Dictionary mapping names to indices
def power_index_dict(stokes):
    if stokes:
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
    # @param    stokesFlag        True if power data  is in Stokes [I,Q,U,V] format, or False if in [XX,XY,YX,YY] format
    # @param    timeSamples       Sequence of timestamps for each data block (in seconds since epoch)
    # @param    azAng_rad         Azimuth angle sequence for each data block (in radians)
    # @param    elAng_rad         Elevation angle sequence for each data block (in radians)
    # @param    rotAng_rad        Rotator stage angle sequence for each data block (in radians)
    # @param    freqs_Hz          List of channel/band centre frequencies (in Hz)
    # @param    mountCoordSystem  Mount coordinate system object (see acsm.coordinatesystem module)
    # @param    targetObject      Target object (see acsm.targets module)
    # pylint: disable-msg=R0913
    def __init__(self, powerData, stokesFlag, timeSamples, azAng_rad, elAng_rad, rotAng_rad, freqs_Hz, \
                 mountCoordSystem=None, targetObject=None):
        ## @var stokesFlag
        # True if power data is in Stokes [I,Q,U,V] format, or False if in [XX,XY,YX,YY] format
        self.stokesFlag = stokesFlag
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
        ## @var stokesData
        # Look up Stokes visibility values by name
        self.stokesData = {}
        ## @var coherencyData
        # Look up coherency / cross-power data by name
        self.coherencyData = {}
        
        self._powerConvertedToTemp = False
        self._nonRfiChannelsConvertedToBands = False
        self._baselineSubtracted = False
        
        ## @var _powerData
        # Power data array (internal variable).
        self._powerData = None

        ## @var powerData
        # Power data array (property).
        # Power data is set via property setter (set this last to ensure prequisite members are available)
        self.powerData = powerData
        
    ## Get powerData member.
    # @param self  The current object
    # @return Power data array
    def get_power_data(self):
        return self._powerData
    ## Set powerData member.
    # This also sets up the dictionary lookups for the data.
    # It enforces the real nature of I, Q, U, V, XX and YY for single-dish data.
    # @param self  The current object
    # @param powerData Power data array
    def set_power_data(self, powerData):
        # Make into proper 3-D matrix
        self._powerData = np.asarray(powerData)
        # Setup polarisation lookup by name
        self.stokesData = {}
        self.coherencyData = {}
        for k, v in power_index_dict(self.stokesFlag).iteritems():
            if self.stokesFlag:
                self.stokesData[k] = self._powerData[v].real
            else:
                if (k == 'XX') or (k == 'YY'):
                    self.coherencyData[k] = self._powerData[v].real
                else:
                    self.coherencyData[k] = self._powerData[v]
    ## @var powerData
    # Power data array (property).
    powerData = property(get_power_data, set_power_data, doc='Power data array.')
    
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
        if (self._powerConvertedToTemp           != other._powerConvertedToTemp) or \
           (self._nonRfiChannelsConvertedToBands != other._nonRfiChannelsConvertedToBands) or \
           (self._baselineSubtracted             != other._baselineSubtracted):
            message = "Data objects are not in the same state, as their flags differ."
            logger.error(message)
            raise ValueError, message
        # Ensure list of frequencies and original channel indices are the same
        if np.any(self.freqs_Hz != other.freqs_Hz) or (self.originalChannels != other.originalChannels):
            message = "Cannot concatenate data objects with different frequency channels/bands."
            logger.error(message)
            raise ValueError, message
        # Convert power data to appropriate format if it differs for the two objects
        if self.stokesFlag != other.stokesFlag:
            # First make a copy to prevent unexpected mutation of "other" object
            other = copy.deepcopy(other)
            if self.stokesFlag:
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
    
    ## Convert the contained power buffer from Stokes vectors to coherency vectors.
    # This can result in a complex-valued powerData array.
    # @param self  The current object
    # @return self The current object
    def convert_to_coherency(self):
        if self.stokesFlag:
            self.stokesFlag = False
            self.powerData = [0.5*(self.stokesData['I'] + self.stokesData['Q']), \
                              0.5*(self.stokesData['U'] + 1j*self.stokesData['V']), \
                              0.5*(self.stokesData['U'] - 1j*self.stokesData['V']), \
                              0.5*(self.stokesData['I'] - self.stokesData['Q'])]
        return self
    
    ## Convert the contained power buffer from coherency vectors to Stokes vectors.
    # This is forced to result in a real-valued powerData array.
    # @param self  The current object
    # @return self The current object
    def convert_to_stokes(self):
        if not(self.stokesFlag):
            self.stokesFlag = True
            self.powerData = np.array([self.coherencyData['XX']+self.coherencyData['YY'], \
                                       self.coherencyData['XX']-self.coherencyData['YY'], \
                                       self.coherencyData['XY']+self.coherencyData['YX'], \
                                   1j*(self.coherencyData['YX']-self.coherencyData['XY'])]).real
        return self
    
    ## Convert raw power measurements (W) into temperature (K) using the provided conversion function.
    # The main argument is a callable object with the signature 'factor = func(time)', which provides
    # an interpolated conversion factor function.
    # @param self The current object
    # @param func The power-to-temperature conversion factor as a function of time
    # @return self The current object
    def convert_power_to_temp(self, func):
        # Already done, so don't do it again
        if self._powerConvertedToTemp:
            return self
        # Obtain interpolated conversion factor
        powerToTempFactor = func(self.timeSamples)
        # Convert power to temperature
        self.powerData *= powerToTempFactor
        self._powerConvertedToTemp = True
        return self
    
    ## Convert frequency channels to bands, while optionally removing RFI-corrupted channels.
    # The frequency channels are grouped into bands, and the power data is merged and averaged within each band.
    # Each band contains the average power of its constituent channels. The average power is simpler to use
    # than the total power in each band, as total power is dependent on the bandwidth of each band.
    # The channelsPerBand mapping contains a list of lists of channel indices, indicating which channels belong 
    # to each band. Some channels can also be marked as corrupted by RFI, which will remove them from any band.
    # The resulting number of bands might be less than what was requested (or even 0), due to this removal.
    # @param self            The current object
    # @param channelsPerBand A sequence of lists of channel indices, indicating which channels belong to each band
    # @param rfiChannels     A sequence of channel indices, indicating channels to remove [None]
    # @return self           The current object
    def convert_non_rfi_channels_to_bands(self, channelsPerBand, rfiChannels = None):
        # Already done, so don't do it again
        if self._nonRfiChannelsConvertedToBands:
            return self
        if not self._powerConvertedToTemp:
            message = "First convert raw power measurements to temperatures before combining channels into bands."
            logger.error(message)
            raise RuntimeError, message
        # Convert "safer" default value of None to intended empty list
        if rfiChannels == None:
            rfiChannels = []
        # Remove RFI channels from band channel lists, and delete any resulting empty bands
        channelsPerBand = [list(set.difference(set(x), set(rfiChannels))) for x in channelsPerBand]
        channelsPerBand = [x for x in channelsPerBand if len(x) > 0]
        # Merge and average power data into new array (keep same type as original data, which may be complex)
        bandPowerData = np.zeros(list(self.powerData.shape[0:2]) + [len(channelsPerBand)], dtype=self.powerData.dtype)
        for bandIndex, bandChannels in enumerate(channelsPerBand):
            bandPowerData[:, :, bandIndex] = self.powerData[:, :, bandChannels].mean(axis=2)
        self.powerData = bandPowerData
        # Each band centre frequency is the mean of the corresponding channel centre frequencies
        self.freqs_Hz = np.array([self.freqs_Hz[chans].mean() for chans in channelsPerBand], dtype='double')
        self.originalChannels = channelsPerBand
        self._nonRfiChannelsConvertedToBands = True
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
        if not self._nonRfiChannelsConvertedToBands:
            message = "Baseline should only be subtracted after frequency channels are converted to bands."
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
    
    ## The total power in the block of power data (as a function of time and frequency band).
    # This returns a non-negative real-valued array.
    # @param self The current object
    # @return The total power value (Stokes I), as a function of time and frequency band
    def total_power(self):
        if self.stokesFlag:
            return self.stokesData['I']
        else:
            return self.coherencyData['XX'] + self.coherencyData['YY']
