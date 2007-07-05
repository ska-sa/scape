## @file fitsreader.py
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
import cPickle
import re
from acsm.coordinate import Coordinate
import acsm.transform

logger = logging.getLogger("xdmsbe.xdmsbelib.fitsreader")

# pylint: disable-msg=C0103,R0902

#------------------------------------
#--- FUNCTIONS
#------------------------------------

## Copy a HDU from one FITS file to another (or to a FITS file chain)
# @param sourceFile The source FITS file
# @param destFile   The destination FITS file
# @param inHduName  The name of the HDU to copy from source to destination
# @param outHduName If this is set, the HDU is renamed to <outHduName> in the destination file [None]
# @param chainFiles If this boolean flag is set, the HDU is copied to all FITS files in the FITS file chain starting at
#                   destFile
def hdu_copy(sourceFile, destFile, inHduName, outHduName=None, chainFiles=False):
    
    def write_dest_hdu(inHDU, hduListDest, outHduName, outFitsFileName):
        try:
            HDUKey = hduListDest.index_of(outHduName)
            hduListDest.pop(HDUKey)
        except KeyError:
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
#--- CLASS :  SelectedData
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
    # @param    powerData   The selected block of power data
    # @param    stokesFlag  True if power data is in Stokes [I,Q,U,V] format, or False if in [XX,XY,YX,YY] format
    # @param    mountCoordSystem Mount coordinate system object (see acsm module)
    # @param    targetCoordSystem Target coordinate system object (see acsm module)
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
        
        self._numTimeSamples = len(self.timeSamples)
        self._midPoint = int(self._numTimeSamples//2)
        
        self._targetCoordSystem = None
        self._mountCoordSystem = None
        
        if (mountCoordSystem and targetCoordSystem):
            self.set_coordinate_systems(mountCoordSystem, targetCoordSystem)
    
    ## Set mount and target coordinate systems which was used for data capture.
    #
    # @param    mountCoordSys Mount coordinate system object (see acsm module)
    # @param    targetCoordSys Target coordinate system object (see acsm module)
    def set_coordinate_systems(self, mountCoordSys, targetCoordSys):
        self._mountCoordSys = mountCoordSys
        self._targetCoordSys = targetCoordSys
        self._transformer = acsm.transform.get_factory_instance().get_transformer(mountCoordSys, targetCoordSys)
                
        self.targetCoords = np.zeros((self._numTimeSamples, targetCoordSys.get_degrees_of_freedom()), dtype='double')
                
        for k in np.arange(self._numTimeSamples):
            mountCoordinate = Coordinate(mountCoordSys, [self.azAng[k], self.elAng[k], self.rotAng[k]])
            targetCoordinate = self._transformer.transform_coordinate(mountCoordinate, self.timeSamples[k])
            self.targetCoords[k, :] = targetCoordinate.get_vector()
    
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
    
    ## Integrate selected power channels into bands excluding RFI corrupted channels
    def power_channels_to_bands_ex_rfi(self, fitsReader):
        for k, powerBlock in enumerate(self.powerData):
            _tempData = np.zeros((powerBlock.shape[0], len(fitsReader.bandNoRfiChannelList)), dtype='double')
            for b, bandChannels in enumerate(fitsReader.bandNoRfiChannelList):
                _tempData[:, b] = np.mean(powerBlock[:, bandChannels], 1)
            self.powerData[k] = _tempData
        return self 
    


## Class for reading fits files and extracting data from them. It is implemented
# on top of PyFITS.
class FitsReader(object):
    
    ## Initialiser/constructor
    # @param self the current object
    # @param hduL optional hdu list if filename is not specified
    # @param fitsFilename optional filename for loading hdu list (overrides hduL)
    # @param hduNames optional set of hdu names to check are in the file
    def __init__(self, hduL=None, fitsFilename=None, hduNames=None):
        
        if fitsFilename:
            if hduL:
                logger.warn("fitsFilename overrides hduL")
            hduL = misc.load_fits(fitsFilename, hduNames)
            assert(fitsFilename == hduL['PRIMARY'].header['CFName'])
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
        
        ## @var dataIdDict
        # Dictionary with the mapping from id names to numbers
        self.dataIdNameToNumDict = None
        
        ## @var dataIdNumToNameDict
        # Dictionary with the mapping from id numbers to names        
        self.dataIdNumToNameDict = None
        
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
        
    ## Get access to the primary header
    # @param self the current object
    # @return the primary header object
    def get_primary_header(self):
        return self.get_hdu_header('PRIMARY')
    
    ## Get the hdu header object
    # @param self the current object
    # @param hduName the name of the desired hdu
    # @return the specified hdu header object
    def get_hdu_header(self, hduName):
        return self._hduL[hduName].header
    
    ## Get the hdu data object
    # @param self the current object
    # @param hduName the name of the desired hdu
    # @return the specified hdu data object
    def get_hdu_data(self, hduName):
        return self._hduL[hduName].data
    
    ## Get the HDU list object
    # @param self the current object
    # @return the HDU list object
    def get_hdu_list(self):
        return self._hduL
    
    ## Get the hdu object
    # @param self the current object
    # @param hduName the name of the desired hdu
    # @return the specified hdu object
    def get_hdu(self, hduName):
        return self._hduL[hduName]
    
    ## Extract and unpickle a pickled Python object stored in a binary table HDU
    # @param self the current object
    # @param hduName the name of the desired binary table HDU [objects]
    # @param colName Column name in binary table HDU from which to extract pickled object
    # @return Unpickled object
    def get_pickle_from_table(self, hduName, colName):
        pickleString = (self.get_hdu_data(hduName).field(colName))[0]
        return cPickle.loads(pickleString)
    
    ## Close the fits file
    def close(self):
        self._hduL.close()
    
    ## Get a selection mask given specified columns and their desired values.
    # Uses logical AND to combine the mask for each column specified. Note that
    # currently, the value specified should be of the correct type for the column
    # @param self the current object
    # @param hduName name of HDU to select from
    # @param colValueDict dictionary of column names and desired values to select by
    # @return a mask array (Boolean numarray object)
    def get_select_mask(self, hduName, colValueDict):
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
    # @param mask the mask (typically created using get_select_mask() - Boolean numarray object). 
    #             If mask is None (default value), the whole column is returned.
    # @return numarray of column values passed by the mask
    def select_masked_column(self, hduName, colName, mask=None):
        data = self.get_hdu_data(hduName)
        if (mask == None):
            return data.field(colName)
        else:
            return data.field(colName)[mask]
    ## Get only the power values that are passed by the mask. Use stokesType to determine 
    # whether the results are returned in IQUV or cross power product form.
    # @param self the current object
    # @param mask the mask (typically created using get_select_mask() - Boolean numarray object)
    # @param stokesType True for IQUV, False for cross power
    # @return list of np.arrays of power values passed by the mask [I, Q, U, V] or [XX, XY, YX, YY]
    def select_masked_power(self, mask=None, stokesType=True):
        
        def get_masked_power_col(name, mask=None):
            return np.array(self.select_masked_column('MSDATA', 'PS' + str(self.stokesIdxDict[name]), mask))
            #return np.array(self._msData.field('PS' + str(self.stokesIdxDict[name]))[mask])    
        
        def get_stokes_power(mask=None):
            I = get_masked_power_col('I', mask)
            Q = get_masked_power_col('Q', mask)
            U = get_masked_power_col('U', mask)
            V = get_masked_power_col('V', mask)
            return I, Q, U, V
        
        def get_cross_power(mask=None):
            XX = get_masked_power_col('XX', mask)
            XY = get_masked_power_col('XY', mask)
            YX = get_masked_power_col('YX', mask)
            YY = get_masked_power_col('YY', mask)
            return XX, XY, YX, YY
        
                
        if mask == None:
            timeSamples = np.arange(self._numSamples) * self._samplePeriod + self._startTime + self._startTimeOffset
        else:
            timeSamples = np.arange(self._numSamples)[mask] * self._samplePeriod + self._startTime + self._startTimeOffset
            
        azAng = self.select_masked_column('MSDATA', 'AzAng', mask) / 180.0 * np.pi
        elAng = self.select_masked_column('MSDATA', 'ElAng', mask) / 180.0 * np.pi        
        rotAng = self.select_masked_column('MSDATA', 'RotAng', mask) / 180.0 * np.pi        
        
        if stokesType:
            try:
                I, Q, U, V = get_stokes_power(mask)
                return SelectedPower(timeSamples, azAng, elAng, rotAng, powerData=[I, Q, U, V], stokesFlag=True, \
                                     mountCoordSystem=self._mountCoordSys, targetCoordSystem=self._targetCoordSys)
            except KeyError:
                try:
                    XX, XY, YX, YY = get_cross_power(mask)
                    return SelectedPower(timeSamples, azAng, elAng, rotAng, \
                                         powerData=[XX+YY, XX+YY, XY+YX, 1j*(YX-XY)], stokesFlag=True, \
                                         mountCoordSystem=self._mountCoordSys, targetCoordSystem=self._targetCoordSys)
                except KeyError, e:
                    logger.error("Neither full (XX, XY, YX, YY) nor (I, Q, U, V) power measurments in data set")
                    raise e
        else:
            try:
                XX, XY, YX, YY = get_cross_power(mask)
                return SelectedPower(timeSamples, azAng, elAng, rotAng, powerData=[XX, XY, YX, YY], stokesFlag=False, \
                                     mountCoordSystem=self._mountCoordSys, targetCoordSystem=self._targetCoordSys)
            except KeyError:
                try:
                    I, Q, U, V = get_stokes_power(mask)
                    return SelectedPower(timeSamples, azAng, elAng, rotAng, \
                                         powerData=[I+Q, U+1j*V, U-1j*V, U-1j*V], stokesFlag = False, \
                                         mountCoordSystem=self._mountCoordSys, targetCoordSystem=self._targetCoordSys)
                except KeyError, e:
                    logger.error("Neither full (XX, XY, YX, YY) nor (I, Q, U, V) power measurments in data set")
                    raise e
    
    
    def _extract_channel_mappings(self):
        self.rfiChannelList = np.array(self._hduL['RFI'].data.field('CHANNELS')).tolist()[0]
        self.bandChannelList = [x.tolist() for x in self._hduL['BANDS'].data.field('CHANNELS')]
        rfiCSet = set(self.rfiChannelList)
        # Remove rfi channels from bandChannelList
        self.bandNoRfiChannelList = [list(set.difference(set(x), rfiCSet)) for x in self.bandChannelList]
    
    
    def _extract_polarisation_mappings(self):
        pol0 = str(self.get_primary_header()['Pol0'])
        pol1 = str(self.get_primary_header()['Pol1'])
        self.polIdxDict = {pol0 : 0, pol1 : 1}
        if set.intersection(set([pol0, pol1]), set(['R', 'L'])):
            logger.error('Currently, only linear feeds are supported!')
    
    
    def _extract_stokes_mappings(self):
        stokes0 = str(self.get_primary_header()['Stokes0'])
        stokes1 = str(self.get_primary_header()['Stokes1'])
        stokes2 = str(self.get_primary_header()['Stokes2'])
        stokes3 = str(self.get_primary_header()['Stokes3'])
        self.stokesIdxDict = {stokes0 : 0, stokes1 : 1, stokes2 : 2, stokes3 : 3}
    
    
    ## Extract the data ids from the MSDATA block
    # @todo revisit dataIDs == 0, check test code
    def _extract_data_ids(self):
        msHdr = self.get_hdu_header('MSDATA')
        # Determine what calibration ID's are present in the data set
        dataIdNumList = list(set(self._msData.field('ID')))
        dataIdNumList.sort()
        
        if len(dataIdNumList) > 0:
            try:
                dataIdNumList.remove(0)
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
            mask = self.get_select_mask('MSDATA', {'ID' : dataIdNum})
            dataIdSeqNumList = np.sort(list(set(self.select_masked_column('MSDATA', 'ID_SeqN', mask))))
            self.dataIdSeqNumListDict[idName] = dataIdSeqNumList
    
    
    #-------------------------------------------------------------------------------------------------------------------
    #--- Method :  extract_data
    #-------------------------------------------------------------------------------------------------------------------
    
    ## Extract relevant data blocks from FITS file.
    #
    #    This function returns a number of dictionaries containing different sets of data and meta-data
    #    extracted from the FITS file (if available).
    #
    #    @param   fitsReader        Object for reading the fits file
    #    @param   dataIdNameList    List of data ID names for which to extract data, i.e. ['hot', 'cold']
    #    @param   dataSelectionList List of data selection 3-tuples of the following form:
    #                                (selectionIdentifierString, selectionDictionary, stokesTypeFlag) , e.g.,
    #                                ('onND', {'RX_ON_F': True, 'ND_ON_F': True, 'VALID_F': True}, False)
    #
    #    @return  dataDict    Data dictionary containing a SelectedPower object for each of the selection products of
    #                            dataIdNameList[i] X dataSelectionList[j] 
    #
    
    # pylint: disable-msg=R0912,R0915
    
    def extract_data(self, dataIdNameList, dataSelectionList):        
        
        dataDict = {}
        
        for dataIdName in self.dataIdNameList:
            
            if dataIdName in dataIdNameList:
                
                for dataIdSeqNum in self.dataIdSeqNumListDict[dataIdName]:
            
                    dataIdStr = 'esn' + str(self.expSeqNum) + '_' + dataIdName + str(dataIdSeqNum)
                    #dataIdStr = str(dataId) + '-' + str(dataIdSeqNum)    # This is the old way of doing it...
        
                    for tagStr, selectDict, stokes in dataSelectionList:
                        selectDict['ID'] = self.dataIdNameToNumDict[dataIdName]
                        selectDict['ID_SeqN'] = dataIdSeqNum
                        selectMask = self.get_select_mask('MSDATA', selectDict)                    
                        dataDict[dataIdStr + '_' + tagStr] = self.select_masked_power(selectMask, stokesType=stokes)
        return dataDict
    


            
class SingleShotIterator(object):
    
    def __init__(self, fitsReader):
        self._fitsReader = fitsReader
        self._finished = False
        
    def __iter__(self):
        return self
        
    def next(self):
        if self._finished:
            raise StopIteration
        self._finished = True
        return self._fitsReader
        

## Class for iterating over a sequence of FITS files. Starts with the given filename and
# uses the 'NFName' tag in the primary header to determine the next filename. Finished at
# the end of the chain when 'LastFile' is non-zero.
class FitsIterator(object):
    
    # pylint: disable-msg=R0903
    
    ## Initialiser/constructor
    # @param self the current object
    # @param fitsFilename the filename for the start of the sequence
    # @param hduNames a list of hdu names that must be present in the files
    def __init__(self, fitsFilename, hduNames=None):
        self._filename = fitsFilename
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
            self._fitsReader = FitsReader(fitsFilename=self._filename, hduNames=self._hduNames)
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
            self._fitsReader = FitsReader(fitsFilename=nextFilename, hduNames=self._hduNames)
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
    # @return a FitsReader object    
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


## Iterate over all the experiments in a FITS file sequence
class ExperimentIterator(object):
    
    # pylint: disable-msg=R0903
    
    ## Initialiser/constructor
    # @param self the current object
    # @param fitsFilename the first filename of the sequence
    # @param hduNames a list of hdu names that must be present in the files
    def __init__(self, fitsFilename, hduNames=None):
        self._fitsFilename = fitsFilename
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
            self._fitsExpIter = ExpFitsIterator(self._fitsFilename, self._hduNames)
        else:
            nextFilename = self._fitsExpIter.nextFilename
            if nextFilename:
                self._fitsExpIter = ExpFitsIterator(nextFilename, self._hduNames)
            else:
                raise StopIteration
        return self._fitsExpIter
        