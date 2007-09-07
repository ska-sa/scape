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
import acsm.transform
import numpy as np
import logging

logger = logging.getLogger("xdmsbe.xdmsbelib.single_dish_data")

#------------------------------------
#--- FUNCTIONS
#------------------------------------

## Dictionary that maps polarisation channel names to power data matrix indices.
# This allows the use of symbolic names, which are clearer than numeric indices.
# @param stokes Boolean indicating wether Stokes IQUV (True) or cross power / coherency (False) is required
# @return Dictionary mapping names to indices
def polarisation_name_to_idx(stokes):
    if stokes:
        return {'I': 0, 'Q': 1, 'U': 2, 'V': 3}
    else:
        return {'XX': 0, 'XY': 1, 'YX': 2, 'YY': 3}

#---------------------------------------------------------------------------------------------
#--- CLASS :  SingleDishData
#------------------------------------

## A container for single-dish data consisting of polarised power measurements combined with pointing information.
# The main data member of this class is the powerData list of 2-D arrays, which stores power (autocorrelation) 
# measurements as a function of polarization index, time and frequency band. Additional information contained in
# the class is the pointing data (azimuth/elevation/rotator angles) and timestamps.
class SingleDishData(object):
    ## Initialiser/constructor
    #
    # @param    self              The current object
    # @param    powerData         List of arrays, containing 4 power data blocks of dimension: time x frequency bands
    # @param    stokesFlag        True if power data is in Stokes [I,Q,U,V] format, or False if in [XX,XY,YX,YY] format
    # @param    timeSamples       Sequence of timestamps for each data block (in seconds since epoch)
    # @param    azAng             Azimuth angle sequence for each data block (in radians)
    # @param    elAng             Elevation angle sequence for each data block (in radians)
    # @param    rotAng            Rotator stage angle sequence for each data block (in radians)
    # @param    bandFreqs         List of band centre frequencies (in Hz)
    # @param    mountCoordSystem  Mount coordinate system object (see acsm module)
    # @param    targetCoordSystem Target coordinate system object (see acsm module)
    # pylint: disable-msg=R0913
    def __init__(self, powerData, stokesFlag, timeSamples, azAng, elAng, rotAng, bandFreqs, \
                 mountCoordSystem=None, targetCoordSystem=None):
        ## @var stokesFlag
        # True if power data is in Stokes [I,Q,U,V] format, or False if in [XX,XY,YX,YY] format
        self.stokesFlag = stokesFlag
        ## @var timeSamples
        # Sequence of timestamps for data block
        self.timeSamples = timeSamples
        ## @var azAng
        # Mount azimuth angle sequence for data block
        self.azAng = azAng
        ## @var elAng
        # Mount elevation angle sequence for data block
        self.elAng = elAng
        ## @var rotAng
        # Rotator stage angle for data block
        self.rotAng = rotAng
        ## @var bandFreqs
        # List of band centre frequencies
        self.bandFreqs = bandFreqs
        
        ## @var powerDataSigma
        # Standard deviation of power measurements if known
        self.powerDataSigma = None
        
        ## @var stokesData
        # Look up Stokes visibility values by name
        self.stokesData = {}
        ## @var coherencyData
        # Look up coherency / cross-power data by name
        self.coherencyData = {}
        
        ## @var targetCoords
        # Target coordinates
        self.targetCoords = None
        self._mountCoordSystem = None
        self._targetCoordSystem = None
        self._transformer = None
        self.set_coordinate_systems(mountCoordSystem, targetCoordSystem)
        
        self._powerConvertedToTemp = False
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
    # @return List of power data blocks
    def get_power_data(self):
        return self._powerData
    ## Set powerData member.
    # This also sets up the dictionary lookups for the data.
    # @param self  The current object
    # @param powerData List of power data blocks
    def set_power_data(self, powerData):
        # Make into proper 3-D matrix
        self._powerData = np.asarray(powerData)
        # Setup polarisation lookup by name
        self.stokesData = {}
        self.coherencyData = {}
        for k, v in polarisation_name_to_idx(self.stokesFlag).iteritems():
            if self.stokesFlag:
                self.stokesData[k] = self._powerData[v]
            else:
                self.coherencyData[k] = self._powerData[v]

    ## @var powerData
    # Power data array (property).
    powerData = property(get_power_data, set_power_data, doc='List of power data blocks.')
    
    ## Set mount and target coordinate systems which was used for data capture.
    #
    # @param    self          the current object
    # @param    mountCoordSys Mount coordinate system object (see acsm module)
    # @param    targetCoordSys Target coordinate system object (see acsm module)
    def set_coordinate_systems(self, mountCoordSys, targetCoordSys):
        # Coordinate systems not complete - do nothing
        if not (mountCoordSys and targetCoordSys):
            return self
        self._mountCoordSystem = mountCoordSys
        self._targetCoordSystem = targetCoordSys
        self._transformer = acsm.transform.get_factory_instance().get_transformer(mountCoordSys, targetCoordSys)
        numTimeSamples = len(self.timeSamples)
        self.targetCoords = np.zeros((numTimeSamples, targetCoordSys.get_dimensions()), dtype='double')
        
        for k in np.arange(numTimeSamples):
            mountCoordinate = Coordinate(mountCoordSys, [self.azAng[k], self.elAng[k], self.rotAng[k]])
            targetCoordinate = self._transformer.transform_coordinate(mountCoordinate, self.timeSamples[k])
            self.targetCoords[k, :] = targetCoordinate.get_vector()
        return self
    
    ## Appends second data object to the first one.
    # This appends the data of the second object to the main object (including power data,
    # coordinates, timestamps, etc.). It also ensures the two objects are compatible.
    # @param self  The current object
    # @param other The object to add to current object (will be mutated!)
    # @return self The current object
    # pylint: disable-msg=W0212
    def __iadd__(self, other):
        # Ensure coordinates are compatible
        if (self._mountCoordSystem != other._mountCoordSystem) or \
           (self._targetCoordSystem != other._targetCoordSystem) or \
           (self._transformer != other._transformer):
            raise ValueError, "Cannot concatenate data objects with incompatible coordinate systems."
        # Ensure objects are in the same state
        if (self._powerConvertedToTemp != other._powerConvertedToTemp) or \
           (self._baselineSubtracted != other._baselineSubtracted):
            raise ValueError, "Data objects are not in the same state" + \
                              " (power conversion or baseline subtraction flags differ)."
        # Ensure frequency bands are the same
        if ((self.bandFreqs != other.bandFreqs).any()):
            raise ValueError, "Cannot concatenate data objects with different frequency bands."
        # Convert power data to appropriate format if it differs for the two objects
        if self.stokesFlag != other.stokesFlag:
            if self.stokesFlag:
                other.convert_to_stokes()
            else:
                other.convert_to_coherency()
        # Concatenate coordinate vectors, and power data along time axis
        self.timeSamples = np.concatenate((self.timeSamples, other.timeSamples))
        self.azAng = np.concatenate((self.azAng, other.azAng))
        self.elAng = np.concatenate((self.elAng, other.elAng))
        self.rotAng = np.concatenate((self.rotAng, other.rotAng))
        self.set_coordinate_systems(self._mountCoordSystem, self._targetCoordSystem)
        self.powerData = np.concatenate((self.powerData, other.powerData), axis=1)
        return self
    
    ## Convert the contained power buffer from Stokes vectors to coherency vectors.
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
    # @param self  The current object
    # @return self The current object
    def convert_to_stokes(self):
        if not(self.stokesFlag):
            self.stokesFlag = True
            self.powerData = [self.coherencyData['XX']+self.coherencyData['YY'], \
                              self.coherencyData['XX']-self.coherencyData['YY'], \
                              self.coherencyData['XY']+self.coherencyData['YX'], \
                              1j*(self.coherencyData['YX']-self.coherencyData['XY'])]
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
    
    ## Subtract a baseline function from the scan data.
    # The main argument is a callable object with the signature 'temp = func(elAng)', which provides 
    # an interpolated baseline function.
    # @param self The current object
    # @param func The baseline function (as a function of elevation angle)
    # @return self The current object
    def subtract_baseline(self, func):
        # Already done, so don't do it again
        if self._baselineSubtracted:
            return self
        if not self._powerConvertedToTemp:
            message = "Cannot subtract baseline from unconverted (raw) power measurements."
            logger.error(message)
            raise ValueError, message
        # Obtain baseline
        baselineData = func(self.elAng)
        # Subtract baseline
        self.coherencyData -= baselineData
        self._baselineSubtracted = True
        return self
    
    ## The total power in the block of power data (as a function of time and frequency band).
    # @param self The current object
    # @return The total power value (Stokes I), as a function of time and frequency band
    def total_power(self):
        if self.stokesFlag:
            return self.stokesData['I']
        else:
            return self.coherencyData['XX'] + self.coherencyData['YY']
