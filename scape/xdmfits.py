"""Read XDM FITS files.

The XDM data set consists of a sequence of consecutively numbered FITS files,
one per scan. The compound scans are indicated by assigning the same 'experiment
sequence number' to a group of scans. The noise diode model is stored in
the first FITS file in the sequence.

Only reading is supported, to encourage a move to later file formats.

"""

import logging
import cPickle
import re
import os.path

import pyfits
import numpy as np

# First create logger to switch off ACSM messages
acsm_logger = logging.getLogger('acsm')
acsm_logger.setLevel(logging.ERROR)
# Needed for pickled target and mount objects
# pylint: disable-msg=W0611
import acsm

from katpoint import deg2rad, rad2deg, construct_target, construct_antenna, Catalogue
from .scan import Scan, move_start_to_center
from .compoundscan import CompoundScan, CorrelatorConfig
from .gaincal import NoiseDiodeModel, NoiseDiodeNotFound
from .stats import angle_wrap

logger = logging.getLogger("scape.xdmfits")

#--------------------------------------------------------------------------------------------------
#--- CLASS :  NoiseDiodeXDM
#--------------------------------------------------------------------------------------------------

class NoiseDiodeXDM(NoiseDiodeModel):
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
        NoiseDiodeModel.__init__(self)
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
                temperature_x = hdu['CAL_TEMP_B%dP1' % feed_id].data
                temperature_y = hdu['CAL_TEMP_B%dP2' % feed_id].data
            except KeyError:
                raise NoiseDiodeNotFound('Noise diode tables not found in FITS file')
        else:
            # Load cal FITS file tables instead, which will have feed ID specified externally
            # nd_type = hdu[0].header.get('NAME')
            if (len(hdu) != 5) or \
               (hdu[1].name != 'CAL_TEMP_B0P1') or (hdu[2].name != 'CAL_TEMP_B0P2') or \
               (hdu[3].name != 'CAL_TEMP_B1P1') or (hdu[4].name != 'CAL_TEMP_B1P2'):
                raise NoiseDiodeNotFound('Noise diode tables not found in FITS file')
            if feed_id not in [0, 1]:
                msg = 'Feed ID should be 0 (main feed) or 1 (offset feed)'
                logger.error(msg)
                raise ValueError(msg)
            temperature_x = hdu[2 * feed_id + 1].data
            temperature_y = hdu[2 * feed_id + 2].data
        # Store X and Y tables
        self.temperature_x = np.vstack((np.array(temperature_x.field('Freq') / 1e6, dtype='double'),
                                        np.array(temperature_x.field('Temp'), dtype='double'))).transpose()
        self.temperature_y = np.vstack((np.array(temperature_y.field('Freq') / 1e6, dtype='double'),
                                        np.array(temperature_y.field('Temp'), dtype='double'))).transpose()
        hdu.close()

#--------------------------------------------------------------------------------------------------
#--- FUNCTIONS
#--------------------------------------------------------------------------------------------------

def acsm_target_description(target):
    """Create katpoint target description from ACSM target object."""
    # pylint: disable-msg=W0212
    ref_target = target.get_reference_target()
    # Extract TLE from TLE targets
    if isinstance(ref_target, acsm.targets.tle.TLE):
        return 'tle, ' + ref_target._name + '\n' + ref_target._line1 + '\n' + ref_target._line2
    # Look for fixed and stationary targets
    descr = ref_target.get_description()
    match = re.match(r'(.*)EquatorialRaDec\(([BJ\d]+)\)\(\((\d+), (\d+), (\d+)\), \((-?\d+), (-?\d+), (-?\d+)\)\)',
                     descr)
    if match:
        # Extract flux density if available
        if ref_target._fluxDensity is None:
            return "%s, radec %s, %s:%s:%s, %s:%s:%s" % match.groups()
        else:
            min_freq, max_freq = ref_target._fluxDensity._minFreq_MHz, ref_target._fluxDensity._maxFreq_MHz
            coefs = ref_target._fluxDensity._coefficients
            return "%s, radec %s, %s:%s:%s, %s:%s:%s, (%g %g %s)" % \
                   tuple(list(match.groups()) + [min_freq, max_freq, ' '.join('%g' % c for c in coefs)])
    match = re.match(r'(.*)Horizontal\(\((-?\d+), (-?\d+), (-?\d+)\), \((-?\d+), (-?\d+), (-?\d+)\)\)', descr)
    if match:
        return "%s, azel, %s:%s:%s, %s:%s:%s" % match.groups()
    # This is typically a futile thing to return, but will help debug why the above two matches failed
    return descr

def acsm_antenna_description(mount):
    """Create katpoint antenna description from ACSM mount object."""
    descr = mount.get_decorated_coordinate_system().get_attribute('position').get_description()
    match = re.match(r'(.+) Mount WGS84\(\((-?\d+), (-?\d+), (-?\d+)\), \((-?\d+), (-?\d+), (-?\d+)\), (-?[\d\.]+)\)',
                     descr)
    if match:
        descr = "%s, %s:%s:%s, %s:%s:%s, %s" % match.groups()
        # Hard-code the XDM dish size, as this is currently not part of ACSM mount object or stored in FITS
        if match.groups()[0] == 'XDM':
            descr += ', 15.0'
        return descr
    # This is typically a futile thing to return, but will help debug why the above match failed
    return descr

def load_scan(filename):
    """Load scan from single XDM FITS file.

    Parameters
    ----------
    filename : string
        Name of FITS file

    Returns
    -------
    scan : :class:`scan.Scan` object
        Scan based on file
    data_unit : {'raw', 'K', 'Jy'}
        Physical unit of power data
    corrconf : :class:`compoundscan.CorrelatorConfig` object
        Correlator configuration
    target : string
        Description string of the target of this scan
    antenna : string
        Description string of antenna that did the scan
    exp_seq_num : int
        Experiment sequence number associated with scan
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
        data = np.dstack([hdu['MSDATA'].data.field(s) for s in ['I', 'Q', 'U', 'V']])
    else:
        data = np.dstack([hdu['MSDATA'].data.field('XX'), hdu['MSDATA'].data.field('YY'),
                          2.0 * hdu['MSDATA'].data.field('XY').real, 2.0 * hdu['MSDATA'].data.field('XY').imag])
    timestamps = np.arange(num_samples, dtype=np.float64) * sample_period + start_time + start_time_offset
    # Round timestamp to nearest millisecond, as this corresponds to KAT7 accuracy and allows better data comparison
    timestamps = np.round(timestamps * 1000.0) / 1000.0
    pointing = np.rec.fromarrays([deg2rad(hdu['MSDATA'].data.field(s).astype(np.float32))
                                  for s in ['AzAng', 'ElAng', 'RotAng']],
                                 names=['az', 'el', 'rot'])
    # Move timestamps and pointing from start of each sample to the middle
    timestamps, pointing = move_start_to_center(timestamps, pointing, sample_period)
    flags = np.rec.fromarrays([hdu['MSDATA'].data.field(s).astype(np.bool)
                               for s in ['Valid_F', 'ND_ON_F', 'RX_ON_F']],
                              names=['valid', 'nd_on', 'rx_on'])
    # The environmental variables are sampled when the FITS file is written to disk,
    # so it is more appropriate to associate the last timestamp with them
    enviro_ambient = np.rec.array([(timestamps[-1], np.float32(header['Temp']),
                                    np.float32(header['Pressure']), np.float32(header['Humidity']))],
                                  names='timestamp,temperature,pressure,humidity')
    enviro_wind = np.rec.array([(timestamps[-1], np.float32(header['WindSpd']), np.float32(header['WindDir']))],
                               names='timestamp,wind_speed,wind_direction')
    data_header = hdu['MSDATA'].header
    label = str(data_header['ID'+str(data_header['DATAID'])])
    path = filename

    data_unit = 'raw'
    freqs = hdu['CHANNELS'].data.field('Freq') / 1e6
    bandwidths = np.repeat(channel_width, len(freqs)) / 1e6
    rfi_channels = [x[0] for x in hdu['RFI'].data.field('Channels')]
    # The FITS file doesn't like empty lists, so an empty list is represented by [-1] (an invalid index)
    # Therefore, remove any invalid indices, as a further safeguard
    rfi_channels = [x for x in rfi_channels if (x >= 0) and (x < len(freqs))]
    corrconf = CorrelatorConfig(freqs, bandwidths, rfi_channels, dump_rate)

    target = acsm_target_description(cPickle.loads(hdu['OBJECTS'].data.field('Target')[0]))
    antenna = acsm_antenna_description(cPickle.loads(hdu['OBJECTS'].data.field('Mount')[0]))

    return Scan(data, is_stokes, timestamps, pointing, flags, enviro_ambient, enviro_wind, label, path), \
           data_unit, corrconf, target, antenna, exp_seq_num, feed_id

# pylint: disable-msg=W0613
def load_dataset(data_filename, nd_filename=None, catalogue=None, swap_xy=False, **kwargs):
    """Load data set from XDM FITS file series.

    This loads the XDM data set starting at the given filename and consisting of
    consecutively numbered FITS files. The noise diode model can be overridden.
    The target objects of each compound scan can also be refined if they are of
    *radec* type, by looking them up by name or by closest distance in the given
    :class:`katpoint.Catalogue` (or file containing targets). This is useful to
    work around the apparent ra/dec values stored in the ACSM target objects.

    Parameters
    ----------
    data_filename : string
        Name of first FITS file in sequence
    nd_filename : string, optional
        Name of FITS file containing alternative noise diode model
    catalogue : :class:`katpoint.Catalogue` object or string, optional
        Catalogue used to refine ACSM target objects in data set (or filename)
    swap_xy : {False, True}, optional
        True if X and Y polarisations are to be swapped from their FITS definitions
    kwargs : dict, optional
        Extra keyword arguments are ignored, as they usually apply to other formats

    Returns
    -------
    compscanlist : list of :class:`compoundscan.CompoundScan` objects
        List of compound scans
    data_unit : {'raw', 'K', 'Jy'}
        Physical unit of power data
    corrconf : :class:`compoundscan.CorrelatorConfig` object
        Correlator configuration object
    antenna : string
        Description string of antenna that produced the data set
    nd_data : :class:`NoiseDiodeXDM` object
        Noise diode model

    Raises
    ------
    ValueError
        If data filename does not have expected numbering as part of name
    IOError
        If data file does not exist, could not be opened or is invalid FITS file

    Notes
    -----
    The X and Y polarisations may be swapped, since the XDM DBE / correlator
    works in H and V, and took X = H and Y = V (e.g. producing Q = HH - VV),
    while the standard polarisation definitions take X = V and Y = H (the noise
    diode data follows the standard, though). Since the cables to the DBE may
    also be swapped, the default is currently set to False.

    """
    match = re.match(r'(.+)_(\d\d\d\d).fits$', data_filename)
    if not match:
        raise ValueError('XDM FITS filenames should have the structure name_dddd.fits, with dddd a four-digit number')
    prefix, file_counter = match.group(1), int(match.group(2))
    filelist = []
    # Add all FITS files with consecutive numbers, starting at the given one
    while os.path.exists('%s_%04d.fits' % (prefix, file_counter)):
        filelist.append('%s_%04d.fits' % (prefix, file_counter))
        file_counter += 1
    if len(filelist) == 0:
        raise IOError("Data file '%s' not found" % data_filename)
    # Group all FITS files (= scans) with the same experiment sequence number into a compound scan
    scanlists, targets = {}, {}
    nd_data = None
    for fits_file in filelist:
        scan, data_unit, corrconf, target, antenna, exp_seq_num, feed_id = load_scan(fits_file)
        if swap_xy:
            scan.swap_x_and_y()
        if scanlists.has_key(exp_seq_num):
            scanlists[exp_seq_num].append(scan)
        else:
            scanlists[exp_seq_num] = [scan]
        assert not targets.has_key(exp_seq_num) or targets[exp_seq_num] == target, \
               "Each scan in a compound scan is required to have the same target"
        targets[exp_seq_num] = target
        # Load noise diode characteristics if available
        if nd_data is None:
            # Alternate cal FITS file overrides the data set version
            if nd_filename:
                try:
                    nd_data = NoiseDiodeXDM(nd_filename, feed_id)
                    logger.info("Loaded alternate noise diode characteristics from %s" % nd_filename)
                except NoiseDiodeNotFound:
                    logger.warning("Could not load noise diode data from " + nd_filename)
                    # Don't try to load this file again
                    nd_filename = None
            # Fall back to noise diode data in data FITS file
            if nd_filename is None:
                try:
                    nd_data = NoiseDiodeXDM(data_filename)
                    logger.info("Loaded noise diode characteristics from %s" % fits_file)
                except NoiseDiodeNotFound:
                    pass
        target = construct_target(target)
        logger.info("Loaded %s: %s '%s' [%s] (%d samps, %d chans, %d pols)" %
                    (os.path.basename(fits_file), scan.label, target.name, target.body_type,
                     scan.data.shape[0], scan.data.shape[1], scan.data.shape[2]))
    # Assemble CompoundScan objects from scan lists
    compscanlist = []
    for esn, scanlist in scanlists.iteritems():
        target = construct_target(targets[esn])
        # Refine radec target to replace its apparent ra/dec coords with astrometric ones
        if catalogue and (target.body_type == 'radec'):
            # A string argument for catalogue is assumed to be a file name - try to open it blindly
            if isinstance(catalogue, basestring):
                catalogue = Catalogue(file(catalogue), add_specials=False)
            # First attempt named lookup, then try distance-based lookup
            new_target = catalogue[target.name.strip()]
            if new_target:
                logger.info("Replaced original ACSM target '%s' with target '%s' with matching name in catalogue" %
                            (target.name, new_target.name))
                target = new_target
            else:
                ant = construct_antenna(antenna)
                cat_pos = np.array([t.apparent_radec(scan.timestamps[0], ant) for t in catalogue])
                targ_pos = np.array(target.astrometric_radec(scan.timestamps[0], ant))
                pos_error = angle_wrap(cat_pos - targ_pos[np.newaxis, :])
                distance = np.sqrt((pos_error * pos_error).sum(axis=1))
                closest = distance.argmin()
                if distance[closest] < deg2rad(30. / 3600.):
                    logger.info("Replaced original ACSM target '%s' with closest target '%s' in catalogue" %
                                (target.name, catalogue.targets[closest].name))
                    target = catalogue.targets[closest]
                else:
                    logger.warning("No target in catalogue close enough to '%s' (closest is %.1f arcsecs away)" %
                                   (target.name, rad2deg(distance[closest]) * 3600.))
        compscanlist.append(CompoundScan(scanlist, target))
    return compscanlist, data_unit, corrconf, antenna, nd_data
