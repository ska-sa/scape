"""Read HDF5 files using katdal interface."""

import logging
import os.path

import numpy as np
import katdal

from .gaincal import NoiseDiodeModel
from .scan import Scan, scape_pol_if
from .compoundscan import CorrelatorConfig, CompoundScan

logger = logging.getLogger("scape.kathdf5")

# -------------------------------------------------------------------------------------------------
# --- FUNCTION :  load_dataset
# -------------------------------------------------------------------------------------------------

# Mapping of desired fields to KAT sensor names (format version 1)
sensor_name_v1 = {}

# Mapping of desired fields to KAT sensor names (format version 2)
sensor_name_v2 = {'pos_actual_scan': 'pos.actual-scan',
                  'pos_actual_refrac': 'pos.actual-refrac',
                  'pos_actual_pointm': 'pos.actual-pointm',
                  'pos_request_scan': 'pos.request-scan',
                  'pos_request_refrac': 'pos.request-refrac',
                  'pos_request_pointm': 'pos.request-pointm'}

# Mapping of desired fields to KAT sensor names (format version 3)
sensor_name_v3 = {}


# pylint: disable-msg=W0613
def load_dataset(filename, baseline='sd', selected_pointing='pos_actual_scan',
                 noise_diode=None, nd_models=None, time_offset=0.0, **kwargs):
    """Load data set from HDF5 file via katdal interface.

    This loads a data set from an HDF5 file. The file contains all the baselines
    for the experiment, but :mod:`scape` only operates on one baseline at a time.
    The *baseline* parameter selects which one will be loaded from the file.

    Parameters
    ----------
    filename : string or :class:`katdal.DataSet`
        Name of input HDF5 file or katdal dataset object
    baseline : string, optional
        Selected baseline as *<ant1>,<ant2>*, where *<ant1>* is the name of
        the first antenna and *<ant2>* is the name of the second antenna.
        For single-dish data the antenna name is repeated, e.g. 'ant1,ant1'
        or just a single antenna name can be given.
        Alternatively, the baseline may be 'sd' for the first single-dish
        baseline or 'if' for the first interferometric baseline in the file.
    selected_pointing : string, optional
        Identifier of (az, el) sensors that will provide all the pointing data
        of the data set. The actual sensor names are formed by appending '_azim'
        and '_elev' to *selected_pointing*. This is ignored if the data set has
        already been processed to contain a single source of pointing data.
    noise_diode : {None, 'coupler', 'pin'}, optional
        Load the model and on/off flags of this noise diode. The default is to
        pick the noise diode with on/off activity if it is the only one showing
        activity, otherwise defaulting to 'coupler'.
    nd_models : None or string, optional
        Override the noise diode models in the HDF5 file using the ones in this
        directory. This assumes that the model files have the format
        'rx.%(band).%(serialno).%(pol).csv' (e.g. 'rx.l.4.h.csv') if the file
        contains receiver serial number information, else the old format of
        '%(antenna).%(diode).%(pol).csv' (e.g. 'ant1.coupler.h.csv') is used.
    time_offset : float, optional
        Offset to add to correlator timestamps, in seconds
    kwargs : dict, optional
        Extra keyword arguments are ignored, as they typically apply to other formats

    Returns
    -------
    compscanlist : list of :class:`compoundscan.CompoundScan` objects
        List of compound scans
    experiment_id : string
        Experiment ID, a unique string used to link the data files of an
        experiment together with blog entries, etc.
    observer : string
        Name of person that recorded the data set
    description : string
        Short description of the purpose of the data set
    data_unit : {'counts', 'K', 'Jy'}
        Physical unit of power data
    corrconf : :class:`compoundscan.CorrelatorConfig` object
        Correlator configuration object
    antenna : string
        Description string of single-dish antenna or first antenna of baseline pair
    antenna2 : string or None
        Description string of second antenna of baseline pair (None for single-dish)
    nd_h_model, nd_v_model : :class:`NoiseDiodeModel` objects
        Noise diode models for H and V polarisations on first antenna
    enviro : dict of record arrays
        Environmental (weather) measurements. The keys of the dict are strings
        indicating the type of measurement ('temperature', 'pressure', etc),
        while the values of the dict are record arrays with three elements per
        record: 'timestamp', 'value' and 'status'. The 'timestamp' field is a
        timestamp in UTC seconds since epoch, the 'value' field is the
        corresponding value and the 'status' field is a string indicating the
        sensor status.

    Raises
    ------
    katdal.BrokenFile, ValueError
        If file has not been augmented to contain all data fields, or some fields
        are missing

    """
    # Parse antennas involved in explicitly specified baseline
    if baseline not in ('sd', 'if'):
        # Baseline separator is a comma so find the comma (or assume single dish if no comma found)
        parsed_antenna_ids = baseline.split(',')
        if (parsed_antenna_ids is None) or len(parsed_antenna_ids) > 2:
            raise ValueError("Please specify baseline with notation '<ant1>,<ant2>', "
                             "where <ant1> is the name of first antenna and <ant2> is the name of second antenna")
        antA_name, antB_name = parsed_antenna_ids[0], parsed_antenna_ids[-1]

    # Get antennas in file
    file_ants = filename.ants if isinstance(filename, katdal.DataSet) else katdal.get_ants(filename)

    if baseline is 'sd':
        # First single-dish baseline found
        try:
            antA = antB = file_ants[0]
        except IndexError:
            raise ValueError('Could not load first single-dish baseline - no antennas found in file')
        logger.info("Loading single-dish baseline '%s,%s'" % (antA.name, antB.name))
    elif baseline is 'if':
        # First interferometric baseline found
        try:
            antA, antB = file_ants[:2]
        except IndexError:
            raise ValueError('Could not load first interferometric baseline - less than 2 antennas found in file')
        logger.info("Loading interferometric baseline '%s,%s'" % (antA.name, antB.name))
    else:
        # Select antennas involved in explicitly specified baseline
        ant_lookup = dict([(ant.name, ant) for ant in file_ants])
        try:
            antA, antB = ant_lookup[antA_name], ant_lookup[antB_name]
        except KeyError:
            # Check that requested antennas are in data set
            raise ValueError('Requested antenna pair not found in HDF5 file (wanted %s but file only contains %s)'
                             % ([antA_name, antB_name], ant_lookup.keys()))
        logger.info("Loading baseline '%s,%s'" % (antA.name, antB.name))

    antenna, antenna2 = antA.description, antB.description
    if antenna == antenna2:
        antenna2 = None

    # Load the katdal object
    if isinstance(filename, katdal.DataSet):
        d = filename
        filename = d.name
    else:
        d = katdal.open(filename, ref_ant=antA.name, time_offset=time_offset, **kwargs)
        # Turn off exceptions for unknown kwargs
        d.select(strict=False, **kwargs)
    # Load weather sensor data
    enviro = {}
    for quantity in ['temperature', 'pressure', 'humidity', 'wind_speed', 'wind_direction']:
        sensor_values = getattr(d, quantity, None)
        if sensor_values is not None:
            status = np.array(['nominal'] * len(sensor_values))
            tvs = [d.timestamps[:], sensor_values, status]
            names = 'timestamp,value,status'
            enviro[quantity] = np.rec.fromarrays(tvs, names=names)

    # Autodetect the noise diode to use, based on which sensor shows any activity
    if not noise_diode:
        nd_fired = {}
        for nd in ('coupler', 'pin'):
            sensor_name = 'Antennas/%s/nd_%s' % (antA.name, nd)
            nd_fired[nd] = np.any(d.sensor.get(sensor_name).unique_values) if sensor_name in d.sensor else False
        if np.sum(nd_fired.values()) == 1:
            noise_diode = nd_fired.keys()[nd_fired.values().index(True)]
            logger.info("Using '%s' noise diode as it is the only one firing in data set" % noise_diode)
        else:
            noise_diode = 'coupler'
            logger.info("Defaulting to '%s' noise diode (either no or both diodes are firing)" % noise_diode)
    # Load noise diode flags and fit interpolator as a function of time
    nd_sensor = 'Antennas/%s/nd_%s' % (antA.name, noise_diode)
    if nd_sensor not in d.sensor:
        logger.warning("Selected noise diode sensor '%s' not found in HDF5 file - setting nd_on to False" % nd_sensor)
        nd_sensor = None
    # First try to load external noise diode models, if provided
    nd_h_model = nd_v_model = None
    if nd_models:
        if not noise_diode:
            raise ValueError('Unable to pick right noise diode model file, as noise diode could not be identified')
        if not antA.name:
            raise ValueError('Unable to pick right noise diode model file, as antenna could not be identified')
        # Attempt to get active band and its receiver serial number
        rx = d.receivers.get(antA.name, '')
        # Use this for base filename, else fall back to old format
        basename = 'rx.' + rx if rx else '%s.%s' % (antA.name, noise_diode)
        nd_h_file = os.path.join(nd_models, basename + '.h.csv')
        nd_h_model = NoiseDiodeModel(nd_h_file)
        logger.info("Loaded H noise diode model from '%s'" % (nd_h_file,))
        nd_v_file = os.path.join(nd_models, basename + '.v.csv')
        nd_v_model = NoiseDiodeModel(nd_v_file)
        logger.info("Loaded V noise diode model from '%s'" % (nd_v_file,))
    else:
        def nd_dataset_name(ant, pol, nd):
            return ('TelescopeModel/%s/%s_%s_noise_diode_model' % (ant, pol.lower(), nd)) \
                if d.version.startswith('3.') else \
                ('MetaData/Configuration/Antennas/%s/%s_%s_noise_diode_model' % (ant, pol.lower(), nd)) \
                if d.version.startswith('2.') else \
                ('Antennas/Antenna%s/%s/%s_nd_model' % (ant[3:], pol.upper(), nd))
        nd_h_name = nd_dataset_name(antA.name, 'H', noise_diode)
        nd_v_name = nd_dataset_name(antA.name, 'V', noise_diode)
        if nd_h_name in d.sensor.iteritems():
            nd_dataset = d.sensor[nd_h_name]
            nd_h_model = NoiseDiodeModel(nd_dataset[:, 0] / 1e6, nd_dataset[:, 1], **dict(nd_dataset.attrs))
        else:
            logger.warning("Cannot find %s noise diode H polarisation model for antenna %s - using default.",
                           noise_diode, antA.name)
            nd_h_model = NoiseDiodeModel()
        if nd_v_name in d.sensor.iteritems():
            nd_dataset = d.sensor[nd_v_name]
            nd_v_model = NoiseDiodeModel(nd_dataset[:, 0] / 1e6, nd_dataset[:, 1], **dict(nd_dataset.attrs))
        else:
            logger.warning("Cannot find %s noise diode V polarisation model for antenna %s - using default.",
                           noise_diode, antA.name)
            nd_v_model = NoiseDiodeModel()
    # Load correlator configuration group
    num_chans = len(d.channel_freqs)
    corrconf = CorrelatorConfig(d.channel_freqs * 1e-6, np.tile(d.channel_width * 1e-6, num_chans),
                                range(num_chans), 1.0 / d.dump_period)

    # Mapping of polarisation product to corrprod index identifying the pair of inputs multiplied together
    # Each polarisation product has 3 options: normal (corr_id positive), conjugate (corr_id negative), absent (zero)
    pol_to_corr_id = {}
    for pol in scape_pol_if:
        corrprod = ['%s%s' % (antA.name, pol[0].lower()), '%s%s' % (antB.name, pol[1].lower())]
        try:
            # Add one to index to ensure positivity (otherwise 0 and -0 can't be distinguished)
            pol_to_corr_id[pol] = d.corr_products.tolist().index(corrprod) + 1
        except ValueError:
            try:
                # If corrprod not found, swap inputs around and remember to conjugate vis data via minus sign
                pol_to_corr_id[pol] = - d.corr_products.tolist().index(corrprod[::-1]) - 1
            except ValueError:
                pol_to_corr_id[pol] = 0
    # Pointing sensors
    az_sensor = 'Antennas/%s/%sazim' % \
        (antA.name, (sensor_name_v2[selected_pointing] + '-') if d.version.startswith('2.') else
                    (selected_pointing + '_'))
    el_sensor = 'Antennas/%s/%selev' % \
        (antA.name, (sensor_name_v2[selected_pointing] + '-') if d.version.startswith('2.') else
                    (selected_pointing + '_'))

    # Load each compound scan group
    compscanlist = []
    for compscan, label, target in d.compscans():
        # Load each scan group within compound scan
        scanlist = []
        for scan, state, scan_target in d.scans():
            scan_timestamps = d.timestamps[:]
            num_times = len(scan_timestamps)
            # Load correlation data either in float64 (single-dish) form or complex128 (interferometer) form
            # Data has already been normalised by number of samples in integration (accum_per_int)
            # Data of missing feeds are set to zero, as this simplifies the data structure (always 4 products)
            if antenna2 is None:
                # Single-dish uses HH, VV, Re{HV}, Im{HV}
                corr_id = [pol_to_corr_id[pol] for pol in ['HH', 'VV', 'HV', 'HV']]
                scan_data = [d.vis[:, :, cid - 1] if cid > 0 else
                             d.vis[:, :, -cid - 1].conj() if cid < 0 else
                             np.zeros((num_times, num_chans), np.float32) for cid in corr_id]
                scan_data = np.dstack([scan_data[0].real, scan_data[1].real,
                                       scan_data[2].real, scan_data[3].imag]).astype(np.float32)
            else:
                corr_id = [pol_to_corr_id[pol] for pol in scape_pol_if]
                scan_data = [d.vis[:, :, cid - 1] if cid > 0 else
                             d.vis[:, :, -cid - 1].conj() if cid < 0 else
                             np.zeros((num_times, num_chans), np.complex64) for cid in corr_id]
                scan_data = np.dstack(scan_data).astype(np.complex64)
            scan_pointing = np.rec.fromarrays([d.sensor[az_sensor].astype(np.float32) * np.float32(np.pi / 180.0),
                                               d.sensor[el_sensor].astype(np.float32) * np.float32(np.pi / 180.0)],
                                              names=('az', 'el'))
            nd_flags = d.sensor[nd_sensor] if nd_sensor else np.tile(False, scan_timestamps.shape)
            scan_flags = np.rec.fromarrays([nd_flags], names=('nd_on',))

            scanlist.append(Scan(scan_data, scan_timestamps, scan_pointing, scan_flags, state,
                                 '%s scan %d' % (filename, scan)))

        if len(scanlist) > 0:
            compscanlist.append(CompoundScan(scanlist, target, label))

    return compscanlist, d.experiment_id, d.observer, d.description, 'counts', \
        corrconf, antenna, antenna2, nd_h_model, nd_v_model, enviro
