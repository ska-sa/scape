"""Read HDF5 files using katfile interface."""

import logging
import re
import os.path

import numpy as np
import katfile

from .gaincal import NoiseDiodeModel
from .scan import Scan, scape_pol_if
from .compoundscan import CorrelatorConfig, CompoundScan

logger = logging.getLogger("scape.kathdf5")

# Parse baseline string into antenna identifiers
baseline_pattern = re.compile('A(\w+)A(\w+)')

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  load_dataset
#--------------------------------------------------------------------------------------------------

# Mapping of desired fields to KAT sensor names (format version 1)
sensor_name_v1 = {'temperature' : 'enviro_air_temperature',
                  'pressure' : 'enviro_air_pressure',
                  'humidity' : 'enviro_air_relative_humidity',
                  'wind_speed' : 'enviro_wind_speed',
                  'wind_direction' : 'enviro_wind_direction',
                  'coupler_nd_on' : 'rfe3_rfe15_noise_coupler_on',
                  'pin_nd_on' : 'rfe3_rfe15_noise_pin_on'}

# Mapping of desired fields to KAT sensor names (format version 2)
sensor_name_v2 = {'temperature' : 'asc.air.temperature',
                  'pressure' : 'asc.air.pressure',
                  'humidity' : 'asc.air.relative-humidity',
                  'wind_speed' : 'asc.wind.speed',
                  'wind_direction' : 'asc.wind.direction',
                  'coupler_nd_on' : 'rfe3.rfe15.noise.coupler.on',
                  'pin_nd_on' : 'rfe3.rfe15.noise.pin.on',
                  'pos_actual_scan' : 'pos.actual-scan',
                  'pos_actual_refrac' : 'pos.actual-refrac',
                  'pos_actual_pointm' : 'pos.actual-pointm',
                  'pos_request_scan' : 'pos.request-scan',
                  'pos_request_refrac' : 'pos.request-refrac',
                  'pos_request_pointm' : 'pos.request-pointm'}

# pylint: disable-msg=W0613
def load_dataset(filename, baseline='AxAx', selected_pointing='pos_actual_scan',
                 noise_diode=None, nd_models=None, time_offset=0.0, **kwargs):
    """Load data set from HDF5 file via katfile interface.

    This loads a data set from an HDF5 file. The file contains all the baselines
    for the experiment, but :mod:`scape` only operates on one baseline at a time.
    The *baseline* parameter selects which one will be loaded from the file.

    Parameters
    ----------
    filename : string
        Name of input HDF5 file
    baseline : string, optional
        Selected baseline as *AxAy*, where *x* is the number of the first antenna
        and *y* is the number of the second antenna (1-based), and *x* < *y*.
        For single-dish data the antenna number is repeated, e.g. 'A1A1'.
        Alternatively, the baseline may be 'AxAx' for the first single-dish
        baseline or 'AxAy' for the first interferometric baseline in the file.
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
        '%(antenna).%(diode).%(pol).csv' (e.g. 'ant1.coupler.h.csv').
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
    katfile.BrokenFile, ValueError
        If file has not been augmented to contain all data fields, or some fields
        are missing

    """
    d = katfile.open(filename, time_offset=time_offset, **kwargs)

    if baseline in ('AxAx', 'sd'):
        # First single-dish baseline found
        try:
            antA = antB = d.ants[0]
        except IndexError:
            raise ValueError('Could not load first single-dish baseline - no antennas found in file')
        logger.info("Loading single-dish baseline 'A%sA%s'" % (antA.name[3:], antB.name[3:]))
    elif baseline in ('AxAy', 'if'):
        # First interferometric baseline found
        try:
            antA, antB = d.ants[:2]
        except IndexError:
            raise ValueError('Could not load first interferometric baseline - less than 2 antennas found in file')
        logger.info("Loading interferometric baseline 'A%sA%s'" % (antA.name[3:], antB.name[3:]))
    else:
        # Select antennas involved in specified baseline
        parsed_antenna_ids = baseline_pattern.match(baseline)
        if parsed_antenna_ids is None:
            raise ValueError("Please specify baseline with notation 'AxAy', "
                             "where x is identifier of first antenna and y is identifier of second antenna")
        antA, antB = [('ant' + ident) for ident in parsed_antenna_ids.groups()]
        ant_lookup = dict([(ant.name, ant) for ant in d.ants])
        try:
            antA, antB = ant_lookup[antA], ant_lookup[antB]
        except KeyError:
            # Check that requested antennas are in data set
            raise ValueError('Requested antenna pair not found in HDF5 file (wanted %s but file only contains %s)'
                             % ([antA, antB], ant_lookup.keys()))
    antenna, antenna2 = antA.description, antB.description
    if antenna == antenna2:
        antenna2 = None

    # Load weather sensor data
    enviro = {}
    for quantity in ['temperature', 'pressure', 'humidity', 'wind_speed', 'wind_direction']:
        sensor_name = ('Antennas/Antenna%s/%s' % (antA.name[3:], sensor_name_v1[quantity])) if d.version < '2.0' else \
                      ('MetaData/Sensors/Enviro/%s' % (sensor_name_v2[quantity],))
        if sensor_name in d.file:
            enviro[quantity] = d.file[sensor_name].value

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
        nd_h_file = os.path.join(nd_models, '%s.%s.h.csv' % (antA.name, noise_diode))
        nd_h_model = NoiseDiodeModel(nd_h_file)
        logger.info("Loaded H noise diode model from '%s'" % (nd_h_file,))
        nd_v_file = os.path.join(nd_models, '%s.%s.v.csv' % (antA.name, noise_diode))
        nd_v_model = NoiseDiodeModel(nd_v_file)
        logger.info("Loaded V noise diode model from '%s'" % (nd_v_file,))
    else:
        def nd_dataset_name(ant, pol, nd):
            return ('Antennas/Antenna%s/%s/%s_nd_model' % (ant[3:], pol.upper(), nd)) \
                   if d.version < '2.0' else \
                   ('MetaData/Configuration/Antennas/%s/%s_%s_noise_diode_model' % (ant, pol.lower(), nd))
        nd_h_name = nd_dataset_name(antA.name, 'H', noise_diode)
        nd_v_name = nd_dataset_name(antA.name, 'V', noise_diode)
        if nd_h_name in d.file:
            nd_dataset = d.file[nd_h_name]
            nd_h_model = NoiseDiodeModel(nd_dataset[:, 0] / 1e6, nd_dataset[:, 1], **dict(nd_dataset.attrs))
            if nd_v_name not in d.file:
                nd_v_model = NoiseDiodeModel()
        if nd_v_name in d.file:
            nd_dataset = d.file[nd_v_name]
            nd_v_model = NoiseDiodeModel(nd_dataset[:, 0] / 1e6, nd_dataset[:, 1], **dict(nd_dataset.attrs))
            if nd_h_name not in d.file:
                nd_h_model = NoiseDiodeModel()
        if nd_h_model is None and nd_v_model is None:
            raise ValueError("Unknown noise diode '%s'" % (noise_diode,))

    # Load correlator configuration group
    num_chans = len(d.channel_freqs)
    corrconf = CorrelatorConfig(d.channel_freqs * 1e-6, np.tile(d.channel_width * 1e-6, num_chans),
                                range(num_chans), 1.0 / d.dump_period)

    # Mapping of polarisation product to DBE input string identifying the pair of inputs multiplied together
    pol_to_dbestr = dict([(pol, ['%s%s' % (antA.name, pol[0].lower()), '%s%s' % (antB.name, pol[1].lower())])
                          for pol in scape_pol_if])
    pol_to_corr_id = dict([(pol, d.corr_products.tolist().index(pol_to_dbestr[pol])) for pol in scape_pol_if])
    # Pointing sensors
    az_sensor = 'Antennas/%s/%sazim' % (antA.name, (selected_pointing + '_') if d.version < '2.0' else
                                                   (sensor_name_v2[selected_pointing] + '-'))
    el_sensor = 'Antennas/%s/%selev' % (antA.name, (selected_pointing + '_') if d.version < '2.0' else
                                                   (sensor_name_v2[selected_pointing] + '-'))

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
                scan_data = [d.vis[:, :, cid][:, :, 0] if cid >= 0 else np.zeros((num_times, num_chans), np.float32)
                             for cid in corr_id]
                scan_data = np.dstack([scan_data[0].real, scan_data[1].real,
                                       scan_data[2].real, scan_data[3].imag]).astype(np.float32)
            else:
                corr_id = [pol_to_corr_id[pol] for pol in scape_pol_if]
                scan_data = [d.vis[:, :, cid][:, :, 0] if cid >= 0 else np.zeros((num_times, num_chans), np.complex64)
                             for cid in corr_id]
                scan_data = np.dstack(scan_data).astype(np.complex64)
            scan_pointing = np.rec.fromarrays([d.sensor[az_sensor].astype(np.float32) * np.float32(np.pi / 180.0),
                                               d.sensor[el_sensor].astype(np.float32) * np.float32(np.pi / 180.0)],
                                              names=('az', 'el'))
            if nd_sensor:
                nd_flags = d.sensor[nd_sensor]
                # Kill first True noise diode flag, as this interferes with scape's noise diode calibration for now
                nd_on = np.nonzero(nd_flags)[0]
                if len(nd_on) > 0:
                    nd_flags[nd_on[0]] = False
            else:
                nd_flags = np.tile(False, scan_timestamps.shape)
            scan_flags = np.rec.fromarrays([nd_flags], names=('nd_on',))

            scanlist.append(Scan(scan_data, scan_timestamps, scan_pointing, scan_flags, state,
                                 '%s scan %d' % (filename, scan)))

        if len(scanlist) > 0:
            compscanlist.append(CompoundScan(scanlist, target, label))

    return compscanlist, d.experiment_id, d.observer, d.description, 'counts', \
           corrconf, antenna, antenna2, nd_h_model, nd_v_model, enviro
