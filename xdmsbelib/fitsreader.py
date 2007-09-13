## @file fitsreader.py
# pylint: disable-msg=C0302
#
# Class for reading fits files and extracting data from them. It is implemented
# on top of PyFITS.
#
# copyright (c) 2007 SKA/KAT. All rights reserved.
# @author Rudolph van der Merwe <rudolph@ska.ac.za>, Robert Crida <robert.crida@ska.ac.za>
# @date 2007-05-07

import misc
import logging
import numpy as np
import xdmsbe.fitsgen.fits_generator as fgen
import xdmsbe.xdmsbelib.stats as stats
import xdmsbe.xdmsbelib.single_dish_data as sdd
import cPickle
from acsm.coordinate import Coordinate
import acsm.transform
import os

logger = logging.getLogger("xdmsbe.xdmsbelib.fitsreader")

# pylint: disable-msg=C0103,R0902

#------------------------------------
#--- FUNCTIONS
#------------------------------------

## Copy a HDU from one FITS file to another (or to a FITS file chain)
# @param sourceFile The source FITS file
# @param destFile   The destination FITS file
# @param inHduName  The name of the HDU to copy from source to destination
# @param outHduName If this is set, the HDU is renamed to this in the destination file [None]
# @param chainFiles If this boolean flag is set, the HDU is copied to all FITS files in the FITS file chain starting at
#                   destFile
def hdu_copy(sourceFile, destFile, inHduName, outHduName=None, chainFiles=False):
    
    
    def write_dest_hdu(inHDU, hduListDest, outHduName, outFitsFileName):
        try:
            HDUKey = hduListDest.index_of(outHduName)
            hduListDest.pop(HDUKey)
        except KeyError:    # pylint: disable-msg=W0704
            pass
        hduListDest.append(inHDU)
        
        hduListDest.writeto(outFitsFileName, clobber=True)
    
    
    hduListSource = misc.load_fits(sourceFile, hduNames = set([inHduName]))
    
    try:
        inHDU = hduListSource[inHduName]
        if outHduName:
            inHDU.name = outHduName
    except KeyError, e:
        message = "HDU '%s' not found in source FITS file." % (inHduName)
        logger.error(message)
        raise e
    
    if not(chainFiles):
        
        hduListDest = misc.load_fits(destFile)
        write_dest_hdu(inHDU, hduListDest, outHduName, destFile)
        hduListDest.close()
    
    else:
        
        _temp_iter = FitsIterator(destFile)
        while True:
            try:
                fitsReader = _temp_iter.next()
                hduListDest = fitsReader.get_hdu_list()
                write_dest_hdu(inHDU, hduListDest, outHduName, fitsReader.fitsFilename)
                hduListDest.close()
            except StopIteration:
                break
    
    hduListSource.close()


## Get a dictionary of stokes channels to index mappings
# @param stokes boolean indicating wether IQUV (True) or cross power (False) are required
def get_power_idx_dict(stokes):
    if stokes:
        return {'I': 0, 'Q': 1, 'U': 2, 'V': 3}
    else:
        return {'XX': 0, 'XY': 1, 'YX': 2, 'YY': 3}


#---------------------------------------------------------------------------------------------
#--- CLASS :  SelectedPower
#------------------------------------

## This class is a container for a selected sequence of data along with its corresponding time and
## pointing information.
#

class SelectedPower(object):
    
    ## Initialiser/constructor
    #
    # @param     self       the current object
    # @param    timeSamples Sequence of timestamps for data block
    # @param    azAng       azimuth angle sequence for data block (in radians)
    # @param    elAng       elevation angle sequence for data block (in radians)
    # @param    rotAng      rotator stage angle for data block (in radians)
    # @param    powerData   The selected block of power data, of dimensions: stokes x time x channels
    # @param    stokesFlag  True if power data is in Stokes [I,Q,U,V] format, or False if in [XX,XY,YX,YY] format
    # @param    mountCoordSystem Mount coordinate system object (see acsm module)
    # @param    targetCoordSystem Target coordinate system object (see acsm module)
    # pylint: disable-msg=R0913
    def __init__(self, timeSamples, azAng, elAng, rotAng, powerData, stokesFlag, mountCoordSystem=None, \
                 targetCoordSystem=None):
        ## @var powerData
        # The selected block of power data
        self.powerData = powerData
        ## @var stokesFlag
        # True if power data is in Stokes [I,Q,U,V] format, or False if in [XX,XY,YX,YY] format
        self.stokesFlag = stokesFlag
        ## @var timeSamples
        # Sequence of timestamps for data block
        self.timeSamples = timeSamples
        ## @var azAng
        # mount azimuth angle sequence for data block
        self.azAng = azAng
        ## @var elAng
        # mount elevation angle sequence for data block
        self.elAng = elAng
        ## @var rotAng
        # rotator stage angle for data block
        self.rotAng = rotAng
        
        ## @var powerDataSigma
        # standard deviation of power measurments if known
        self.powerDataSigma = None
        
        ## @var FptProfile
        # Power-to-temperature conversion factor profile for scan data (mean profile)
        self.FptProfile = None
        ## @var FptProfileSigma
        # Power-to-temperature conversion factor profile for scan data (one-sigma profile - standard deviation)
        self.FptProfileSigma = None
        
        self._numTimeSamples = len(self.timeSamples)
        self._midPoint = int(self._numTimeSamples//2)
        
        self._targetCoordSystem = None
        self._mountCoordSystem = None
        
        ## @var targetCoords
        # Target coordinates
        self.targetCoords = None
        
        if (mountCoordSystem and targetCoordSystem):
            self.set_coordinate_systems(mountCoordSystem, targetCoordSystem)
        
        self._mountCoordSys = None
        self._targetCoords = None
        self._transformer = None
        self._targetCoordSys = None
        
        self._FptPostMean = None
        self._FptPostSigma = None
        self._FptPostTime = None
        self._FptPreMean = None
        self._FptPreSigma = None
        self._FptPreTime = None
        self._Fpt_func = None
        
        self._tipCurveX = None
        self._tipCurveY = None
        self._powerConvertedToTemp = False
        self._tipCurveSubtracted = False
        self._channelsConvertedToBands = False
    
    ## Convert the contained power buffer from stokes to coherency vectors
    # @param self the current object
    # @return self the current object
    def convert_to_coherency(self):
        if self.stokesFlag:
            self.stokesFlag = False
            self.powerData = [0.5*(self.powerData[0]+self.powerData[1]), 0.5*(self.powerData[2]+1j*self.powerData[3]), \
                              0.5*(self.powerData[2]-1j*self.powerData[3]), 0.5*(self.powerData[0]-self.powerData[1])]
        return self
    
    ## Convert the contained power buffer from coherency to stokes vectors
    # @param self the current object
    # @return self the current object
    def convert_to_stokes(self):
        if not(self.stokesFlag):
            self.stokesFlag = True
            self.powerData = [self.powerData[0]+self.powerData[3], self.powerData[0]-self.powerData[3], \
                              self.powerData[1]+self.powerData[2], 1j*(self.powerData[2]-self.powerData[1])]
        return self
    
    ## Set the operating tipping curve for the given elevation angle range of the power measurements scan
    #
    # @param    self            The object
    # @param    tipCurveX       The tipping curve (Tsys [K] as a function of elevation angle and band) for the X pol.
    # @param    tipCurveY       The tipping curve (Tsys [K] as a function of elevation angle and band) for the Y pol.
    def set_tipping_curve(self, tipCurveX, tipCurveY):
        self._tipCurveX = tipCurveX
        self._tipCurveY = tipCurveY
        return self
    
    ## Subtract the tipping curve from the scan data
    # @param self the current object
    # @return self the current object
    def subtract_tipping_curve(self):
        # Already done, so don't do it again
        if self._tipCurveSubtracted:
            return self
        if not self._powerConvertedToTemp:
            message = "Cannot subtract tipping curve from uncoverted (raw) power measurments."
            logger.error(message)
            raise ValueError, message
        if (self._tipCurveX == None) or (self._tipCurveY==None):
            message = "Cannot subtract tipping curve. Tipping curve not set."
            logger.error(message)
            raise ValueError, message
        if self.stokesFlag:
            message = "Cannot subtract tipping curve if power is in Stokes vector format. " + \
                      "First convert to coherency vector format."
            logger.error(message)
            raise ValueError, message
        # Subtract tipping curve
        self.powerData[0] -= self._tipCurveX
        self.powerData[3] -= self._tipCurveY
        self._tipCurveSubtracted = True
        return self
    
    ## Set the power-to-temperature conversion factor (Fpt : gain term) for the given selected power object
    # This function takes two inputs. The gain before the scan and/or the gain after the scan. If only one is
    # specified, that is used as the operative gain for the whole scan. If both are specified, a linearly
    # interpolated gain profile is used.
    #
    # @param    self            The current object
    # @param    FptPreMean      The Fpt gain term (mean) before the scan
    # @param    FptPreSigma     The Fpt gain term (standard deviation) before the scan
    # @param    FptPreTime      Timestamp (seconds) of the Fpt data
    # @param    FptPostMean     The Fpt gain term (mean) after the scan
    # @param    FptPostSigma    The Fpt gain term (standard deviation) after the scan
    # @param    FptPostTime     Timestamp (seconds) of the Fpt data
    # @return   self            The current object
    def set_Fpt(self, FptPreMean = None, FptPreSigma = None, FptPreTime = None, \
                FptPostMean = None, FptPostSigma = None, FptPostTime = None):
        self._FptPreMean = FptPreMean
        self._FptPreSigma = FptPreSigma
        self._FptPreTime = FptPreTime
        self._FptPostMean = FptPostMean
        self._FptPostSigma = FptPostSigma
        self._FptPostTime = FptPostTime
        return self
    
    ## Convert the selected power measurements into temperature using the contained power-temp gain profile
    # @param self the current object
    # pylint: disable-msg=R0914,R0915
    def convert_power_to_temp(self):
        
        if not(self._powerConvertedToTemp):
            
            if (self._FptPreMean == None) and (self._FptPostMean == None):
                message = 'Either have to specifiy a positive finite pre-Fpt or post-Fpt or both.'
                logger.error(message)
                raise ValueError, message
            
            def p2t_func(FptPre=None, FptPreTime=None, FptPost=None, FptPostTime=None, \
                         timeSamples=None, powerData=None):
                if FptPre == None:
                    slope = 0.0
                    offset = 1.0/FptPost
                elif FptPost == None:
                    slope = 0.0
                    offset = 1.0/FptPre
                else:
                    slope = (1.0/FptPost - 1.0/FptPre) / (FptPostTime - FptPreTime)
                    offset = 1.0/FptPre - slope * FptPreTime
                powerDataTemp = np.zeros((4, len(timeSamples), offset.shape[1]), dtype='complex128')
                FptProfile = np.zeros((4, len(timeSamples), offset.shape[1]), dtype='complex128')
                for n, t in enumerate(timeSamples):
                    IFpt = slope * t + offset
                    Fpt = IFpt**(-1.0)
                    FptProfile[0, n, :] = Fpt[0, :]
                    FptProfile[1, n, :] = Fpt[1, :]
                    FptProfile[2, n, :] = Fpt[2, :]
                    FptProfile[3, n, :] = Fpt[3, :]
                    powerDataTemp[0, n, :] = Fpt[0, :] * powerData[0][n, :]
                    powerDataTemp[1, n, :] = Fpt[1, :] * powerData[1][n, :]
                    powerDataTemp[2, n, :] = Fpt[2, :] * powerData[2][n, :]
                    powerDataTemp[3, n, :] = Fpt[3, :] * powerData[3][n, :]
                
                return powerDataTemp, FptProfile
            
            
            output1, output2 = p2t_func(self._FptPreMean, self._FptPreTime, self._FptPostMean, self._FptPostTime, \
                                        self.timeSamples, self.powerData)
            
            if self._FptPreMean == None:
                inputShapeDict = {"FptPost": self._FptPostMean.shape}
                outputShapeList = [output1.shape, output2.shape]
                output1 = None
                output2 = None
                constantDict = {"FptPostTime": self._FptPostTime, \
                                "timeSamples": self.timeSamples, "powerData": self.powerData}
                func = stats.SpStatsFuncWrapper(p2t_func, inputShapeDict, outputShapeList, constantDict)
                muX = func.vectorize_input(FptPost=self._FptPostMean)
                sigmaXdiag = func.vectorize_input(FptPost=self._FptPostSigma)
            elif self._FptPostMean == None:
                inputShapeDict = {"FptPre": self._FptPreMean.shape}
                outputShapeList = [output1.shape, output2.shape]
                output1 = None
                output2 = None
                constantDict = {"FptPreTime": self._FptPreTime, \
                                "timeSamples": self.timeSamples, "powerData": self.powerData}
                func = stats.SpStatsFuncWrapper(p2t_func, inputShapeDict, outputShapeList, constantDict)
                muX = func.vectorize_input(FptPre=self._FptPreMean)
                sigmaXdiag = func.vectorize_input(FptPre=self._FptPreSigma)
            else:
                inputShapeDict = {"FptPre": self._FptPreMean.shape, "FptPost": self._FptPostMean.shape}
                outputShapeList = [output1.shape, output2.shape]
                output1 = None
                output2 = None
                constantDict = {"FptPreTime": self._FptPreTime, "FptPostTime": self._FptPostTime, \
                                "timeSamples": self.timeSamples, "powerData": self.powerData}
                func = stats.SpStatsFuncWrapper(p2t_func, inputShapeDict, outputShapeList, constantDict)
                muX = func.vectorize_input(FptPre=self._FptPreMean, FptPost=self._FptPostMean)
                sigmaXdiag = func.vectorize_input(FptPre=self._FptPreSigma, FptPost=self._FptPostSigma)
            
            muY, sigmaY = stats.propagate_scalar_stats(func, muX, sigmaXdiag)
            
            powerDataTempMu = muY[0]
            FptMu = muY[1]
            powerDataTempSigma = sigmaY[0]
            FptSigma = sigmaY[1]
            
            self.powerData = [powerDataTempMu[0, :, :], powerDataTempMu[1, :, :], \
                              powerDataTempMu[2, :, :], powerDataTempMu[3, :, :]]
            self.powerDataSigma = [powerDataTempSigma[0, :, :], powerDataTempSigma[1, :, :], \
                                   powerDataTempSigma[2, :, :], powerDataTempSigma[3, :, :]]
            
            self.FptProfile = [FptMu[0, :, :], FptMu[1, :, :], \
                              FptMu[2, :, :], FptMu[3, :, :]]
            self.FptProfileSigma = [FptSigma[0, :, :], FptSigma[1, :, :], \
                                   FptSigma[2, :, :], FptSigma[3, :, :]]
            
            self._powerConvertedToTemp = True
        
        return self
    
    
    ## Return the nominal power-temperature-gain factor at a given point in time
    # @param self    The current object
    # @param timeVal The timevalue in seconds at which to evaluate the gain function
    def get_Fpt(self, timeVal):
        if self._Fpt_func:
            return self._Fpt_func(timeVal) # pylint: disable-msg=E1102
        else:
            message = "Fpt calibration data not yet set."
            logger.error(message)
            raise ValueError, message
    
    ## Set mount and target coordinate systems which was used for data capture.
    #
    # @param    self          the current object
    # @param    mountCoordSys Mount coordinate system object (see acsm module)
    # @param    targetCoordSys Target coordinate system object (see acsm module)
    def set_coordinate_systems(self, mountCoordSys, targetCoordSys):
        self._mountCoordSys = mountCoordSys
        self._targetCoordSys = targetCoordSys
        self._transformer = acsm.transform.get_factory_instance().get_transformer(mountCoordSys, targetCoordSys)
        
        self.targetCoords = np.zeros((self._numTimeSamples, targetCoordSys.get_dimensions()), dtype='double')
        
        for k in np.arange(self._numTimeSamples):
            mountCoordinate = Coordinate(mountCoordSys, [self.azAng[k], self.elAng[k], self.rotAng[k]])
            targetCoordinate = self._transformer.transform_coordinate(mountCoordinate, self.timeSamples[k])
            self.targetCoords[k, :] = targetCoordinate.get_vector()
        
        return self
    
    ## Returns the median (middle) time stamp for a given block of data
    #
    # @param    self        The current object
    # @return   medianTime  The median (middle) time stamp [s]
    def get_median_time(self):
        return self.timeSamples[self._midpoint]
    
    ## Returns the median (middle) azimuth angle for a given block of data
    #
    # @param    self        The current object
    # @return   medianAzAng The median (middle) azimuth angle [deg]
    def get_median_azimuth(self):
        return self.azAng[self._midpoint]
    
    ## Returns the median (middle) elevation angle for a given block of data
    #
    # @param    self        The current object
    # @return   medianElAng The median (middle) azimuth angle [deg]
    def get_median_elevation(self):
        return self.elAng[self._midpoint]
    
    ## Returns the median (middle) target coordinates for a given block of data
    #
    # @param    self                    The current object
    # @return   medianTargetCoordinate  The median (middle) azimuth angle [deg]
    def get_median_target_coordinate(self):
        if self._targetCoordSystem:
            return self._targetCoords[self._midpoint, :]
        else:
            message = "Target coordinate system has not been set."
            logger.error(message)
            raise AttributeError
    
    ## Returns the median (middle) rotator stage angle for a given block of data
    #
    # @param    self            The current object
    # @return   medianRotAng    The median (middle) rotator stage angle [deg]
    def get_median_rotation(self):
        return self.rotAng[self._midpoint]
    
    ## Returns the maximum power in the selected block of power measurments
    #
    # @param    self            The current object
    # @return   maxVal          The maximum total power value (max Stokes I) Given as a function of bands.
    # @return   timeVal         The corresponding time stamp of the maximum value
    def get_max_total_power(self):
        
        if self.stokesFlag:
            I = self.powerData[0]
        else:
            I = self.powerData[0] + self.powerData[3]
        
        totalPower = np.sum(I, 1)
        maxTotalPower = np.max(totalPower)
        maxIdx = np.where(totalPower == maxTotalPower)[0][0]
        maxTotalPower = self.powerData[0][maxIdx, :]
        timeVal = self.timeSamples[maxIdx]
        
        return maxTotalPower, timeVal
    
    ## Integrate selected power channels into bands excluding RFI corrupted channels
    # @param  self          the current object
    # @param  fitsReader    A FitsReader object containing the relevant channel-to-band mapping and
    #         RFI channel information
    # @return self          the updated current object
    def power_channels_to_bands_ex_rfi(self, fitsReader):
        if not(self._channelsConvertedToBands):
            for k, powerBlock in enumerate(self.powerData):
                _tempData = np.zeros((powerBlock.shape[0], len(fitsReader.bandNoRfiChannelList)), dtype='double')
                for b, bandChannels in enumerate(fitsReader.bandNoRfiChannelList):
                    _tempData[:, b] = np.mean(powerBlock[:, bandChannels], 1)
                self.powerData[k] = _tempData
                self._channelsConvertedToBands = True
        return self



## Class for reading fits files and extracting XDM single-dish data from them.
# It is implemented on top of PyFITS.
class FitsReader(object):
    
    ## Initialiser/constructor.
    # @param self         The current object.
    # @param hduL         Optional hdu list if filename is not specified
    # @param fitsFilename Optional fully qualified filename for loading hdu list (overrides hduL)
    # @param hduNames     Optional set of hdu names to check are in the file
    def __init__(self, hduL=None, fitsFilename=None, hduNames=None):
        
        if fitsFilename:
            if hduL:
                logger.warn("fitsFilename overrides hduL")
            self._workingDir = os.path.dirname(fitsFilename)
            self.fitsFilename = os.path.basename(fitsFilename)
            hduL = misc.load_fits(os.path.join(self._workingDir, self.fitsFilename), hduNames)
            assert(self.fitsFilename == hduL['PRIMARY'].header['CFName'])
        else:
            self._workingDir = os.path.dirname(hduL._HDUList__file.name) # pylint: disable-msg=W0212
        self._hduL = hduL
        self._primHdr = hduL['PRIMARY'].header
        self._msData = self.get_hdu_data('MSDATA')
        
        ## @var fitsFilename
        # The name of the fits file
        self.fitsFilename = self._primHdr['CFName']
        
        ## @var expSeqNum
        # The experiment sequence number associated with this FITS file
        self.expSeqNum = self._primHdr['ExpSeqN']
        
        ## @var rfiChannelList
        # List of rfi channel indexes
        self.rfiChannelList = None
        
        ## @var bandChannelList
        # List of channels per band
        self.bandChannelList = None
        
        ## @var bandNoRfiChannelList
        # List of non-rfi channels per band
        self.bandNoRfiChannelList = None
        self._extract_channel_mappings()
        
        ## @var bandNoRfiFrequencies
        # List of band centre frequencies in Hz (after RFI channels are removed)
        self.bandNoRfiFrequencies = None
        self._calc_band_frequencies()
        
        ## @var polIdxDict
        # Dictionary with the mappings from polarisation names to indexes
        self.polIdxDict = None
        self._extract_polarisation_mappings()
        
        ## @var stokesIdxDict
        # Dictionary with the mappings from stokes names to indexes
        self.stokesIdxDict = None
        self._extract_stokes_mappings()
        
        ## @var dataIdNameList
        # List of data id names
        self.dataIdNameList = None
        
        ## @var dataIdNumList
        # List of data id numbers
        self.dataIdNumList = None
        
        ## @var dataIdNameToNumDict
        # Dictionary with the mapping from id names to numbers
        self.dataIdNameToNumDict = None
        
        ## @var dataIdNumToNameDict
        # Dictionary with the mapping from id numbers to names
        self.dataIdNumToNameDict = None
        
        ## @var dataIdSeqNumList
        # List of sequences numbers for each data id
        self.dataIdSeqNumList = None
        
        ## @var dataIdSeqNumListDict
        # Dictionary with lists of sequences numbers for each data id
        self.dataIdSeqNumListDict = None
        self._extract_data_ids()
        
        self._startTime = np.double(self._primHdr['TEPOCH'])
        self._startTimeOffset = np.double(self._primHdr['TSTART'])
        self._samplePeriod = np.double(self._primHdr['PERIOD'])
        self._numSamples = int(self._primHdr['SAMPLES'])
        
        self._mountCoordSys = self.get_pickle_from_table('Objects', 'Mount')
        self._targetCoordSys = self.get_pickle_from_table('Objects', 'Target')
    
    ## Get access to the primary header.
    # @param self The current object
    # @return The primary header object
    def get_primary_header(self):
        return self.get_hdu_header('PRIMARY')
    
    ## Get the hdu header object.
    # @param self The current object
    # @param hduName The name of the desired hdu
    # @return The specified hdu header object
    def get_hdu_header(self, hduName):
        return self._hduL[hduName].header
    
    ## Get the hdu data object.
    # @param self The current object
    # @param hduName The name of the desired hdu
    # @return The specified hdu data object
    def get_hdu_data(self, hduName):
        return self._hduL[hduName].data
    
    ## Get the HDU list object.
    # @param self The current object
    # @return The HDU list object
    def get_hdu_list(self):
        return self._hduL
    
    ## Get the hdu object.
    # @param self The current object
    # @param hduName The name of the desired hdu
    # @return The specified hdu object
    def get_hdu(self, hduName):
        return self._hduL[hduName]
    
    ## Extract and unpickle a pickled Python object stored in a binary table HDU.
    # @param self The current object
    # @param hduName The name of the desired binary table HDU [objects]
    # @param colName Column name in binary table HDU from which to extract pickled object
    # @return Unpickled object
    def get_pickle_from_table(self, hduName, colName):
        pickleString = (self.get_hdu_data(hduName).field(colName))[0]
        return cPickle.loads(pickleString)
    
    ## Close the fits file.
    # @param self The current object
    def close(self):
        self._hduL.close()
    
    ## Extract list of frequency channels for each frequency band, while discarding RFI-marked channels.
    # @param self The current object
    def _extract_channel_mappings(self):
        self.rfiChannelList = [x[0] for x in self._hduL['RFI'].data.field('Channels')]
        self.bandChannelList = [x.tolist() for x in self._hduL['BANDS'].data.field('Channels')]
        rfiCSet = set(self.rfiChannelList)
        # Remove rfi channels from bandChannelList, and delete any resulting empty bands
        tempBandNoRfiChannelList = [list(set.difference(set(x), rfiCSet)) for x in self.bandChannelList]
        self.bandNoRfiChannelList = [x for x in tempBandNoRfiChannelList if len(x) > 0]
    
    ## Calculate the centre frequency for each band, as the mean of the frequencies of its channels.
    # @param self The current object
    def _calc_band_frequencies(self):
        channelFreqs = self._hduL['CHANNELS'].data.field('Freq')
        self.bandNoRfiFrequencies = np.array([(channelFreqs[x]).mean() for x in self.bandNoRfiChannelList], \
                                             dtype='double')
    
    ## Extract mapping of polarisation type to a corresponding index from FITS file header
    # @param self The current object
    def _extract_polarisation_mappings(self):
        pol0 = str(self.get_primary_header()['Pol0'])
        pol1 = str(self.get_primary_header()['Pol1'])
        self.polIdxDict = {pol0 : 0, pol1 : 1}
        if set.intersection(set([pol0, pol1]), set(['R', 'L'])):
            logger.error('Currently, only linear feeds are supported!')
    
    ## Extract mapping of Stokes / cross-power names to array indices from FITS file header
    # @param self The current object
    def _extract_stokes_mappings(self):
        stokes0 = str(self.get_primary_header()['Stokes0'])
        stokes1 = str(self.get_primary_header()['Stokes1'])
        stokes2 = str(self.get_primary_header()['Stokes2'])
        stokes3 = str(self.get_primary_header()['Stokes3'])
        self.stokesIdxDict = {stokes0 : 0, stokes1 : 1, stokes2 : 2, stokes3 : 3}
    
    ## Get a selection mask given specified columns and their desired values.
    # Uses logical AND to combine the mask for each column specified. Note that
    # currently, the value specified should be of the correct type for the column
    # @param self the current object
    # @param hduName name of HDU to select from
    # @param colValueDict dictionary of column names and desired values to select by
    # @return a mask array (Boolean numarray object)
    # pylint: disable-msg=R0914
    def _get_select_mask(self, hduName, colValueDict):
        # This next block was added after we (Richard & Simon) decided to only ever use one dataId and one dataIdSeqNum
        # per fits file. This means these parameters are constant over one FITS file, resulting in them now being
        # specified in the MSDATA header, and no longer as seperate columns in the MSDATA data table.
        # pylint: disable-msg=W0612
        try:
            dataId = colValueDict.pop('ID')
            if dataId not in self.dataIdNumList:
                mask = []
                return mask
            try:
                dataIdSeqNum = colValueDict.pop('ID_SeqN')
                if dataIdSeqNum not in self.dataIdSeqNumList:
                    mask = []
                    return mask
            except KeyError:
                dataIdSeqNum = None
        except KeyError:
            dataID = None
        
        mask = None
        hdu = self.get_hdu(hduName)
        colNames = [x.lower() for x in hdu.columns.names]
        for col, colVal in colValueDict.items():
            colName = col.lower()
            fitsDataTypeStr = hdu.columns.formats[colNames.index(colName)].lower()
            value = fgen.numerix_array_cast(colVal, fitsDataTypeStr)
            if mask != None:
                mask = mask & (hdu.data.field(colName) == value)
            else:
                mask = (hdu.data.field(colName) == value)
        present = np.any(mask)
        if not present:
            message = 'Mask with colValueDict=%s is EMPTY!' % colValueDict
            logger.error(message)
            raise ValueError, message
        return mask
    
    ## Get only the values from a column that are passed by the mask (ie where mask[i] == True)
    # @param self the current object
    # @param hduName the name of the HDU to select from
    # @param colName the name of the column in the specified HDU
    # @param mask the mask (typically created using _get_select_mask() - Boolean numarray object).
    #             If mask is None (default value), the whole column is returned.
    # @return numarray of column values passed by the mask
    def select_masked_column(self, hduName, colName, mask=None):
        data = self.get_hdu_data(hduName)
        if (mask == None):
            return data.field(colName)
        else:
            return data.field(colName)[mask]
    
    ## Create a SingleDishData object containing power and pointing info, based on a mask.
    # The mask determines which segments of the FITS file are returned. Use stokesType to determine whether
    # the results are returned in IQUV or cross-power/coherency form. The power data is either per channel
    # or per band (minus RFI channels), depending on the perBand flag.
    # @param self       The current object
    # @param mask       The mask (typically created using _get_select_mask() - Boolean numarray object)
    # @param perBand    True for power per band (minus RFI), False for power per channel
    # @param stokesType True for IQUV, False for cross power
    # @return SingleDishData object combining power and pointing info
    # pylint: disable-msg=R0912,R0914
    def _extract_single_dish_data(self, mask=None, perBand=True, stokesType=True):
        ## Integrate per-channel power data according to frequency bands, while excluding RFI-corrupted channels.
        # Each band contains the average power of its constituent channels. The average power is simpler to use
        # than the total power in each band, as total power is dependent on the bandwidth of each band.
        # @param power 2-D array of power values, with dimensions: time x channels
        # @return 2-D array of power values, with dimensions: time x bands
        def power_channels_to_bands_ex_rfi(power):
            bandPower = np.zeros((power.shape[0], len(self.bandNoRfiChannelList)), dtype='double')
            for b, bandChannels in enumerate(self.bandNoRfiChannelList):
                bandPower[:, b] = power[:, bandChannels].mean(axis=1)
            return bandPower
        ## Obtain array of power values from FITS file, with optional mask and conversion to frequency bands.
        # @param name    Name of data column in FITS file
        # @param mask    Mask for selecting various power segments
        # @param perBand True if per-channel data is to be converted to per-band data
        # @return Array of power values
        def get_masked_power_col(name, mask, perBand):
            chanPower = np.array(self.select_masked_column('MSDATA', 'PS' + str(self.stokesIdxDict[name]), mask))
            if perBand:
                return power_channels_to_bands_ex_rfi(chanPower)
            else:
                return chanPower
        
        def get_stokes_power(mask, perBand):
            I = get_masked_power_col('I', mask, perBand)
            Q = get_masked_power_col('Q', mask, perBand)
            U = get_masked_power_col('U', mask, perBand)
            V = get_masked_power_col('V', mask, perBand)
            return I, Q, U, V
        
        def get_cross_power(mask, perBand):
            XX = get_masked_power_col('XX', mask, perBand)
            XY = get_masked_power_col('XY', mask, perBand)
            YX = get_masked_power_col('YX', mask, perBand)
            YY = get_masked_power_col('YY', mask, perBand)
            return XX, XY, YX, YY
        
        if mask == None:
            timeSamples = np.arange(self._numSamples) * self._samplePeriod + self._startTime + self._startTimeOffset
        else:
            timeSamples = np.arange(self._numSamples)[mask] * self._samplePeriod + self._startTime + \
                          self._startTimeOffset
        
        azAng = self.select_masked_column('MSDATA', 'AzAng', mask) / 180.0 * np.pi
        elAng = self.select_masked_column('MSDATA', 'ElAng', mask) / 180.0 * np.pi
        rotAng = self.select_masked_column('MSDATA', 'RotAng', mask) / 180.0 * np.pi
        
        if perBand:
            freqList = self.bandNoRfiFrequencies
        else:
            freqList = self._hduL['CHANNELS'].data.field('Freq')
        
        try:
            # Attempt to read Stokes visibilities from FITS file
            data = sdd.SingleDishData(get_stokes_power(mask, perBand), True, \
                                      timeSamples, azAng, elAng, rotAng, freqList, \
                                      mountCoordSystem=self._mountCoordSys, targetCoordSystem=self._targetCoordSys)
            if not stokesType:
                data.convert_to_coherency()
            return data
        except KeyError:
            try:
                # Attempt to read cross power / coherencies from FITS file
                data = sdd.SingleDishData(get_cross_power(mask, perBand), False, \
                                          timeSamples, azAng, elAng, rotAng, freqList, \
                                          mountCoordSystem=self._mountCoordSys, targetCoordSystem=self._targetCoordSys)
                if stokesType:
                    data.convert_to_stokes()
                return data
            except KeyError, e:
                logger.error("Neither full (XX, XY, YX, YY) nor (I, Q, U, V) power measurements in data set")
                raise e
    
    ## Extract the data ids from the MSDATA block
    # @todo revisit dataIDs == 0, check test code
    def _extract_data_ids(self):
        msHdr = self.get_hdu_header('MSDATA')
        
        # Determine what calibration ID is present in this FITS file
        dataId = msHdr['DATAID']
        
        #dataIdNumList = list(set(self._msData.field('ID')))
        #dataIdNumList.sort()
        dataIdNumList = [dataId]
        
        if len(dataIdNumList) > 0:
            try:
                dataIdNumList.remove(0)
                if len(dataIdNumList) == 0:
                    logger.error("No data IDs present in measurement set meta data")
            # pylint: disable-msg=W0704
            except ValueError:
                pass
        else:
            logger.error("No data IDs present in measurement set meta data")
        
        dataIdNumList = np.array(dataIdNumList, dtype=int)
        
        # Build calibration ID dictionary
        self.dataIdNumList = dataIdNumList
        self.dataIdNameToNumDict = {}
        self.dataIdNumToNameDict = {}
        self.dataIdSeqNumListDict = {}
        self.dataIdNameList = []
        for dataIdNum in dataIdNumList:
            try:
                idName = str(msHdr['ID'+str(dataIdNum)]).lower()
            except KeyError, e:
                logger.error("Data ID %d mapping not found in MSDATA.header as expected." % dataIdNum)
                raise e
            self.dataIdNameList.append(idName)
            self.dataIdNameToNumDict[idName] = dataIdNum
            self.dataIdNumToNameDict[dataIdNum] = idName
            # Determine calibration ID sequence numbers present in data
            dataIdSeqNum = msHdr['SEQUID']
            #mask = self._get_select_mask('MSDATA', {'ID' : dataIdNum})
            #dataIdSeqNumList = np.sort(list(set(self.select_masked_column('MSDATA', 'ID_SeqN', mask))))
            self.dataIdSeqNumList = [dataIdSeqNum]
            self.dataIdSeqNumListDict[idName] = self.dataIdSeqNumList
    
    
    #-------------------------------------------------------------------------------------------------------------------
    #--- Method :  extract_data
    #-------------------------------------------------------------------------------------------------------------------
    
    ## Extract relevant data blocks from FITS file.
    #
    #    This function returns a number of dictionaries containing different sets of data and meta-data
    #    extracted from the FITS file (if available).
    #
    #    @param   self              Object for reading the fits file
    #    @param   dataIdNameList    List of data ID names for which to extract data, i.e. ['hot', 'cold']
    #    @param   dataSelectionList List of data selection 3-tuples of the following form:
    #                                (selectionIdentifierString, selectionDictionary, stokesTypeFlag) , e.g.,
    #                                ('onND', {'RX_ON_F': True, 'ND_ON_F': True, 'VALID_F': True}, False)
    #    @param   perBand           True for power per band (minus RFI), False for power per channel
    #
    #    @return  dataDict    Data dictionary containing a SingleDishData object for each of the selection 
    #                         products of dataIdNameList[i] X dataSelectionList[j]
    #
    
    # pylint: disable-msg=R0912,R0915
    
    def extract_data(self, dataIdNameList, dataSelectionList, perBand=True):
        
        dataDict = {}
        
        for dataIdName in self.dataIdNameList:
            
            if dataIdName in dataIdNameList:
                
                for dataIdSeqNum in self.dataIdSeqNumListDict[dataIdName]:
                    
                    dataIdStr = 'esn' + str(self.expSeqNum) + '_' + dataIdName + str(dataIdSeqNum)
                    #dataIdStr = str(dataId) + '-' + str(dataIdSeqNum)    # This is the old way of doing it...
                    
                    for tagStr, selectDict, stokes in dataSelectionList:
                        selectDict['ID'] = self.dataIdNameToNumDict[dataIdName]
                        selectDict['ID_SeqN'] = dataIdSeqNum
                        mask = self._get_select_mask('MSDATA', selectDict)
                        dataDict[dataIdStr + '_' + tagStr] = self._extract_single_dish_data(mask, perBand, stokes)
        return dataDict



## Class that provides a fitsreader iterator to a single fitsreader object.
# This is needed for cases where we only need to process one file and don't need to iterate through a chain.
# pylint: disable-msg=R0903
class SingleShotIterator(object):
    ## Initialiser/constructor
    # @param self       The current object
    # @param fitsReader The FitsReader object to wrap in this iterator
    def __init__(self, fitsReader):
        self._fitsReader = fitsReader
        self._finished = False
    
    ## Return this object for use in a for loop
    def __iter__(self):
        return self
    
    ## Get the next fits reader for this iterator.
    # This will return the single FitsReader associated with this iterator object.
    # @param self The current object
    # @return a FitsReader object
    def next(self):
        if self._finished:
            raise StopIteration
        self._finished = True
        return self._fitsReader


## Class for iterating over a sequence of FITS files.
# Starts with the given filename and uses the 'NFName' tag in the primary header to determine the next filename.
# Finished at the end of the chain when 'LastFile' is non-zero.
class FitsIterator(object):
    
    # pylint: disable-msg=R0903
    
    ## Initialiser/constructor
    # @param self the current object
    # @param fitsFilename the fully qualified filename for the start of the sequence
    # @param hduNames a list of hdu names that must be present in the files
    def __init__(self, fitsFilename, hduNames=None):
        self._workingDir = os.path.dirname(fitsFilename)
        self._filename = os.path.basename(fitsFilename)
        self._hduNames = hduNames
        self._fitsReader = None
    
    ## Return this object for use in a for loop
    def __iter__(self):
        return self
    
    ## Get the next fits reader for the sequence
    # @param self the current object
    # @return a FitsReader object
    def next(self):
        if not self._fitsReader:
            self._fitsReader = FitsReader(fitsFilename=os.path.join(self._workingDir, self._filename), \
                                          hduNames=self._hduNames)
        else:
            primHdr = self._fitsReader.get_primary_header()
            lastFile = bool(primHdr['LastFile'])
            nextFilename = primHdr['NFName']
            currentFilename = self._fitsReader.fitsFilename
            self._fitsReader.close()
            # check for last file
            if lastFile:
                raise StopIteration
            # get the next filename
            if currentFilename == nextFilename:
                message = "Next FITS filename is sequence is same as current. This will cause an infinite loop!"
                logger.error(message)
                raise ValueError, message
            self._fitsReader = FitsReader(fitsFilename=os.path.join(self._workingDir, nextFilename), \
                                          hduNames=self._hduNames)
            self._filename = self._fitsReader.fitsFilename
        return self._fitsReader


## Class for iterating over a sequence of FITS files in an individual experiment sequence.
# It picks up the sequence number from the first file. Uses FitsIterator for internal
# iterating but checks for continuity of experiment number.
class ExpFitsIterator(object):
    
    # pylint: disable-msg=R0903
    
    ## Initialiser/constructor
    # @param self the current object
    # @param fitsFilename the first filename of the experiment sequence. The experiment
    #                     number will be read from this file.
    # @param hduNames a list of hdu names that must be present in the files
    def __init__(self, fitsFilename, hduNames=None):
        self._workingDir = os.path.dirname(fitsFilename)
        self._expSeqNum = None
        self._iter = FitsIterator(fitsFilename, hduNames)
        
        ## @var nextFilename
        # The name of the first file in the next sequence else None
        self.nextFilename = None
    
    ## Return this object for use in a for loop
    def __iter__(self):
        return self
    
    ## Get the next fits reader for the sequence. Stops if the sequence number changes
    # @param self the current object
    # @return A FitsReader object
    def next(self):
        fitsReader = self._iter.next()
        primHdr = fitsReader.get_primary_header()
        nextExpSeqNum = primHdr['ExpSeqN']
        if self._expSeqNum == None:
            self._expSeqNum = nextExpSeqNum
        elif self._expSeqNum != nextExpSeqNum:
            self.nextFilename = fitsReader.fitsFilename
            raise StopIteration
        return fitsReader


## Iterate over all the experiments in a FITS file sequence.
class ExperimentIterator(object):
    
    # pylint: disable-msg=R0903
    
    ## Initialiser/constructor
    # @param self the current object
    # @param fitsFilename the first filename of the sequence
    # @param hduNames a list of hdu names that must be present in the files
    def __init__(self, fitsFilename, hduNames=None):
        self._fitsFilename = os.path.basename(fitsFilename)
        self._workingDir = os.path.dirname(fitsFilename)
        self._hduNames = hduNames
        self._fitsExpIter = None
    
    ## Return this object for use in a for loop
    def __iter__(self):
        return self
    
    ## Get the next fits experiment iterator for the sequence. Stops if there are no
    # more experiment sequences
    # @param self the current object
    # @return an ExpFitsIterator object
    def next(self):
        if not self._fitsExpIter:
            self._fitsExpIter = ExpFitsIterator(os.path.join(self._workingDir, self._fitsFilename), self._hduNames)
        else:
            nextFilename = self._fitsExpIter.nextFilename
            if nextFilename:
                self._fitsExpIter = ExpFitsIterator(os.path.join(self._workingDir, nextFilename), self._hduNames)
            else:
                raise StopIteration
        return self._fitsExpIter
        