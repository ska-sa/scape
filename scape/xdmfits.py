"""Read XDM FITS files.

The XDM data set consists of a sequence of consecutively numbered FITS files,
one per subscan. The scans are indicated by assigning the same 'experiment
sequence number' to a group of subscans. The noise diode model is stored in
the first FITS file in the sequence.

Only reading is supported, to encourage a move to later file formats.

"""

import logging
import cPickle
import re
import os.path

import pyfits
import numpy as np
# Needed for pickled target and mount objects
import acsm

import coord
from subscan import SubScan, stokes_order, coherency_order
from scan import Scan
import gaincal

logger = logging.getLogger("scape.xdmfits")

default_nd_filename = None

#--------------------------------------------------------------------------------------------------
#--- CLASS :  NoiseDiodeXDM
#--------------------------------------------------------------------------------------------------

class NoiseDiodeXDM(gaincal.NoiseDiodeBase):
    """A container for noise diode calibration data (XDM FITS version).
    
    This allows the (randomised) calculation of the noise diode temperature from
    the tables stored in a FITS file, as a function of frequency. This is the
    second version, used for XDM after April 2009, which uses a simple
    temperature lookup table as a function of frequency for each feed
    (independent of rotator angle). The spectrum is only interpolated when the
    actual noise diode temperature is requested. This can load data from both
    the data FITS file and the optional cal FITS file. In the latter case the
    feed ID has to be specified (but not in the former case).
    
    Parameters
    ----------
    filename : string
        Name of data or cal FITS file
    feed_id : int
        Feed ID number (0 or 1), which should only be used for cal FITS file

    Raises
    ------
    gaincal.NoiseDiodeNotFound
        If the noise diode tables are not present in the FITS file
    
    """
    def __init__(self, filename, feed_id=None):
        # Open FITS file
        try:
            hdu = pyfits.open(filename)
        except (IOError, TypeError):
            msg = 'The FITS file (%s) cannot be read!' % filename
            logger.error(msg)
            raise IOError(msg)
        # First assume file is a data FITS file, since feed ID will then be extracted from file
        if feed_id is None:
            # Load data FITS file tables
            try:
                feed_id = int(hdu['PRIMARY'].header['FeedID'])
                table_x = hdu['CAL_TEMP_B%dP1' % feed_id].data
                table_y = hdu['CAL_TEMP_B%dP2' % feed_id].data
            except KeyError:
                raise gaincal.NoiseDiodeNotFound('Noise diode tables not found in FITS file')
        else:
            # Load cal FITS file tables instead, which will have feed ID specified externally
            # nd_type = hdu[0].header.get('NAME')
            if (len(hdu) != 5) or \
               (hdu[1].name != 'CAL_TEMP_B0P1') or (hdu[2].name != 'CAL_TEMP_B0P2') or \
               (hdu[3].name != 'CAL_TEMP_B1P1') or (hdu[4].name != 'CAL_TEMP_B1P2'):
                raise gaincal.NoiseDiodeNotFound('Noise diode tables not found in FITS file')
            if feed_id not in [0, 1]:
                msg = 'Feed ID should be 0 (main feed) or 1 (offset feed)'
                logger.error(msg)
                raise ValueError(msg)
            table_x = hdu[2 * feed_id + 1].data
            table_y = hdu[2 * feed_id + 2].data            
        # Store X and Y tables
        self.table_x = np.vstack((np.array(table_x.field('Freq'), dtype='double'), 
                                  np.array(table_x.field('Temp'), dtype='double'))).transpose()
        self.table_y = np.vstack((np.array(table_y.field('Freq'), dtype='double'), 
                                  np.array(table_y.field('Temp'), dtype='double'))).transpose()
        hdu.close()

#--------------------------------------------------------------------------------------------------
#--- FUNCTIONS
#--------------------------------------------------------------------------------------------------

def _acsm_target_name(target):
    """Extract target name from ACSM target object."""
    ref_target = target.get_reference_target()
    if ref_target._name:
        return ref_target._name
    else:
        return ref_target.get_description()

def load_subscan(filename):
    """Load subscan from single XDM FITS file.
    
    Parameters
    ----------
    filename : string
        Name of FITS file
    
    Returns
    -------
    sub : SubScan object
        SubScan based on file
    exp_seq_num : int
        Experiment sequence number associated with subscan
    feed_id : int
        Index of feed used (0 for main feed or 1 for offset feed)
    
    Raises
    ------
    IOError
        If file would not open or is not a proper FITS file
    
    """
    hdu = pyfits.open(filename)
    try:
        hdu.verify(option='exception')
    except pyfits.VerifyError:
        hdu.close()
        raise IOError("File '%s' does not comply with FITS standard" % filename)
    header = hdu['PRIMARY'].header
    
    is_stokes = (header['Stokes0'] == 'I')
    start_time = np.double(header['tEpoch'])
    start_time_offset = np.double(header['tStart'])
    dump_rate = np.double(header['DumpRate'])
    sample_period = 1.0 / dump_rate
    num_samples = int(header['Samples'])
    channel_width = np.double(header['ChannelB'])
    exp_seq_num = int(header['ExpSeqN'])
    feed_id = int(header['FeedID'])
    
    if is_stokes:
        data = np.rec.fromarrays([hdu['MSDATA'].data.field(s) for s in stokes_order], names=stokes_order)
    else:
        data = np.rec.fromarrays([hdu['MSDATA'].data.field(s) for s in coherency_order], names=coherency_order)
    data_unit = 'raw'
    timestamps = np.arange(num_samples) * sample_period + start_time + start_time_offset
    pointing = np.rec.fromarrays([coord.radians(hdu['MSDATA'].data.field(s)) 
                                  for s in ['AzAng', 'ElAng', 'RotAng']],
                                 names=['az', 'el', 'rot'])
    flags = np.rec.fromarrays([np.array(hdu['MSDATA'].data.field(s), 'bool')
                               for s in ['Valid_F', 'ND_ON_F', 'RX_ON_F']],
                              names=['valid', 'nd_on', 'rx_on'])
    freqs = hdu['CHANNELS'].data.field('Freq')
    bandwidths = np.repeat(channel_width, len(freqs))
    rfi_channels = [x[0] for x in hdu['RFI'].data.field('Channels')]
    # The FITS file doesn't like empty lists, so an empty list is represented by [-1] (an invalid index)
    # Therefore, remove any invalid indices, as a further safeguard
    rfi_channels = [x for x in rfi_channels if (x >= 0) and (x < len(freqs))]
    channels_per_band = [x.tolist() for x in hdu['BANDS'].data.field('Channels')]

    target = _acsm_target_name(cPickle.loads(hdu['OBJECTS'].data.field('Target')[0]))
    mount = cPickle.loads(hdu['OBJECTS'].data.field('Mount')[0])
    antenna = mount.get_decorated_coordinate_system().get_attribute('position').get_description().split()[0]
    
    data_header = hdu['MSDATA'].header
    label = str(data_header['ID'+str(data_header['DATAID'])])
    path = filename
    
    return SubScan(data, data_unit, timestamps, pointing, flags,
                   freqs, bandwidths, rfi_channels, channels_per_band, dump_rate,
                   target, antenna, label, path), exp_seq_num, feed_id

def load_dataset(data_filename, nd_filename=None):
    """Load data set from XDM FITS file series.
    
    This loads the XDM data set starting at the given filename and consisting of
    consecutively numbered FITS files. The noise diode model can also be
    overridden. Since this function is usually not called directly, but via the
    DataSet initialiser, the noise diode file should rather be assigned to
    xdmfits.default_nd_filename.
    
    Parameters
    ----------
    data_filename : string
        Name of first FITS file in sequence
    nd_filename : string, optional
        Name of FITS file containing alternative noise diode model
    
    Returns
    -------
    scanlist : list of Scan objects
        List of scans
    nd_data : NoiseDiodeXDM object
        Noise diode model
    
    Raises
    ------
    ValueError
        If data filename does not have expected numbering as part of name
    
    """
    if nd_filename is None:
        nd_filename = default_nd_filename
    match = re.match(r'(.+)_(\d\d\d\d).fits$', data_filename)
    if not match:
        raise ValueError('XDM FITS filenames should have the structure name_dddd.fits, with dddd a four-digit number')
    prefix, file_counter = match.group(1), int(match.group(2))
    filelist = []
    # Add all FITS files with consecutive numbers, starting at the given one
    while os.path.exists('%s_%04d.fits' % (prefix, file_counter)):
        filelist.append('%s_%04d.fits' % (prefix, file_counter))
        file_counter += 1
    # Group all FITS files (= subscans) with the same experiment sequence number into a scan
    subscanlists = {}
    nd_data = None
    for fits_file in filelist:
        logger.info("Loading file:   " + fits_file)
        sub, exp_seq_num, feed_id = load_subscan(fits_file)
        if subscanlists.has_key(exp_seq_num):
            subscanlists[exp_seq_num].append(sub)
        else:
            subscanlists[exp_seq_num] = [sub]
        # Load noise diode characteristics if available
        if nd_data is None:
            # Alternate cal FITS file overrides the data set version
            if nd_filename:
                try:
                    nd_data = NoiseDiodeXDM(nd_filename, feed_id)
                    logger.info("Loaded alternate noise diode characteristics from %s" % nd_filename)
                except gaincal.NoiseDiodeNotFound:
                    logger.warning("Could not load noise diode data from " + nd_filename)
                    # Don't try to load this file again
                    nd_filename = None
            # Fall back to noise diode data in data FITS file
            if nd_filename is None:
                try:
                    nd_data = NoiseDiodeXDM(data_filename)
                    logger.info("Loaded noise diode characteristics")
                except gaincal.NoiseDiodeNotFound:
                    pass
        logger.info("subscan info:   %s '%s' (%d samples, %d channels, %d pols)" % 
                    (sub.label, sub.target.name, sub.data.shape[0], sub.data.shape[1], sub.data.shape[2]))
    # Assemble Scan objects from subscan lists
    scanlist = []
    for subscanlist in subscanlists.itervalues():
        scanlist.append(Scan(subscanlist))
    return scanlist, nd_data
