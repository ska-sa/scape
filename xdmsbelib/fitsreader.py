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
import xdmsbe.xdmsbelib.single_dish_data as sdd
from conradmisclib.transforms import deg_to_rad
import cPickle
import os

logger = logging.getLogger("xdmsbe.xdmsbelib.fitsreader")

# pylint: disable-msg=C0103,R0902

#---------------------------------------------------------------------------------------------------------
#--- FUNCTIONS
#---------------------------------------------------------------------------------------------------------

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


#---------------------------------------------------------------------------------------------------------
#--- CLASS :  FitsReader
#------------------------

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
        self._samplePeriod = 1.0 / np.double(self._primHdr['DUMPRATE'])
        self._numSamples = int(self._primHdr['SAMPLES'])
        
        self._mountCoordSys = self.get_pickle_from_table('Objects', 'Mount')
        self._targetObject = self.get_pickle_from_table('Objects', 'Target')
    
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
        
        azAng_rad = deg_to_rad(self.select_masked_column('MSDATA', 'AzAng', mask))
        elAng_rad = deg_to_rad(self.select_masked_column('MSDATA', 'ElAng', mask))
        rotAng_rad = deg_to_rad(self.select_masked_column('MSDATA', 'RotAng', mask))
        
        if perBand:
            freqList = self.bandNoRfiFrequencies
        else:
            freqList = self._hduL['CHANNELS'].data.field('Freq')
        
        try:
            # Attempt to read Stokes visibilities from FITS file
            data = sdd.SingleDishData(get_stokes_power(mask, perBand), True, \
                                      timeSamples, azAng_rad, elAng_rad, rotAng_rad, freqList, \
                                      mountCoordSystem=self._mountCoordSys, targetObject=self._targetObject)
            if not stokesType:
                data.convert_to_coherency()
            return data
        except KeyError:
            try:
                # Attempt to read cross power / coherencies from FITS file
                data = sdd.SingleDishData(get_cross_power(mask, perBand), False, \
                                          timeSamples, azAng_rad, elAng_rad, rotAng_rad, freqList, \
                                          mountCoordSystem=self._mountCoordSys, targetObject=self._targetObject)
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


#---------------------------------------------------------------------------------------------------------
#--- CLASS : SingleShotIterator
#-------------------------------

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


#---------------------------------------------------------------------------------------------------------
#--- CLASS : FitsIterator
#-------------------------

## Class for iterating over a sequence of FITS files.
# Starts with the given filename and uses the 'NFName' tag in the primary header to determine the next filename.
# Finished at the end of the chain when 'LastFile' is non-zero.
class FitsIterator(object):
    
    # pylint: disable-msg=R0903
    
    ## Initialiser/constructor.
    # @param self         The current object
    # @param fitsFilename The fully qualified filename for the start of the sequence
    # @param hduNames     A list of hdu names that must be present in the files
    def __init__(self, fitsFilename, hduNames=None):
        self._workingDir = os.path.dirname(fitsFilename)
        self._filename = os.path.basename(fitsFilename)
        self._hduNames = hduNames
        self._fitsReader = None
    
    ## Return this object for use in a for loop.
    # @param self The current object
    def __iter__(self):
        return self
    
    ## Get the next fits reader for the sequence.
    # @param self The current object
    # @return A FitsReader object
    def next(self):
        if not self._fitsReader:
            try:
                self._fitsReader = FitsReader(fitsFilename=os.path.join(self._workingDir, self._filename), \
                                              hduNames=self._hduNames)
            except IOError:
                logger.error('Could not open initial file of FITS sequence.')
                raise StopIteration
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
                message = "Next FITS filename in sequence is same as current. This will cause an infinite loop!"
                logger.error(message)
                raise ValueError, message
            try:
                self._fitsReader = FitsReader(fitsFilename=os.path.join(self._workingDir, nextFilename), \
                                              hduNames=self._hduNames)
            except IOError:
                logger.warning('Expected more files in FITS file sequence, stopping prematurely.')
                raise StopIteration
            self._filename = self._fitsReader.fitsFilename
        return self._fitsReader


#---------------------------------------------------------------------------------------------------------
#--- CLASS : ExpFitsIterator
#----------------------------

## Class for iterating over a sequence of FITS files in an individual experiment sequence.
# It picks up the sequence number from the first file. Uses FitsIterator for internal
# iterating but checks for continuity of experiment number.
class ExpFitsIterator(object):
    
    # pylint: disable-msg=R0903
    
    ## Initialiser/constructor.
    # @param self         The current object
    # @param fitsFilename The first filename of the experiment sequence. The experiment
    #                     number will be read from this file.
    # @param hduNames     A list of hdu names that must be present in the files
    def __init__(self, fitsFilename, hduNames=None):
        self._workingDir = os.path.dirname(fitsFilename)
        self._expSeqNum = None
        self._iter = FitsIterator(fitsFilename, hduNames)
        
        ## @var nextFilename
        # The name of the first file in the next sequence else None
        self.nextFilename = None
    
    ## Return this object for use in a for loop.
    # @param self The current object
    def __iter__(self):
        return self
    
    ## Get the next fits reader for the sequence. Stops if the sequence number changes.
    # @param self The current object
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


#---------------------------------------------------------------------------------------------------------
#--- CLASS : ExperimentIterator
#-------------------------------

## Iterate over all the experiments in a FITS file sequence.
class ExperimentIterator(object):
    
    # pylint: disable-msg=R0903
    
    ## Initialiser/constructor.
    # @param self         The current object
    # @param fitsFilename The first filename of the sequence
    # @param hduNames     A list of hdu names that must be present in the files
    def __init__(self, fitsFilename, hduNames=None):
        self._fitsFilename = os.path.basename(fitsFilename)
        self._workingDir = os.path.dirname(fitsFilename)
        self._hduNames = hduNames
        self._fitsExpIter = None
    
    ## Return this object for use in a for loop.
    # @param self The current object
    def __iter__(self):
        return self
    
    ## Get the next fits experiment iterator for the sequence. Stops if there are no more experiment sequences.
    # @param self The current object
    # @return An ExpFitsIterator object
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
