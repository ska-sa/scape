"""Read and write HDF5 files."""

from __future__ import with_statement

import logging
import re

import h5py
import numpy as np

from .gaincal import NoiseDiodeModel
from .scan import Scan, scape_pol
from .compoundscan import CorrelatorConfig, CompoundScan
from .fitting import PiecewisePolynomial1DFit

logger = logging.getLogger("scape.hdf5")

# Parse baseline string into antenna indices
baseline_pattern = re.compile('A(\d+)A(\d+)')
# Parse antenna name
antenna_name_pattern = re.compile('ant(\d+)')

# Mapping of desired fields to KAT sensor names
sensor_name = {'temperature' : 'enviro_air_temperature',
               'pressure' : 'enviro_air_pressure',
               'humidity' : 'enviro_air_relative_humidity',
               'wind_speed' : 'enviro_wind_speed',
               'wind_direction' : 'enviro_wind_direction',
               'coupler_nd_on' : 'rfe3_rfe15_noise_coupler_on',
               'pin_nd_on' : 'rfe3_rfe15_noise_pin_on'}

def remove_duplicates(sensor, name):
    """Remove duplicate timestamp values from sensor data.

    This sorts the 'timestamp' field of the sensor record array and removes any
    duplicate values, updating the corresponding 'value' and 'status' fields as
    well. If more than one timestamp have the same value, the value and status
    of the last of these timestamps are selected. If the values differ for the
    same timestamp, a warning is logged (and the last one is still picked).

    Parameters
    ----------
    sensor : record array, shape (N,)
        Sensor data record array with fields 'timestamp', 'value' and 'status'
    name : string
        Name of sensor, used in warning message

    Returns
    -------
    unique_sensor : record array, shape (M,)
        Sensor data with duplicate timestamps removed (M <= N)

    """
    x = np.atleast_1d(sensor['timestamp'])
    y = np.atleast_1d(sensor['value'])
    z = np.atleast_1d(sensor['status'])
    # Sort x via mergesort, as it is usually already sorted and stability is important
    sort_ind = np.argsort(x, kind='mergesort')
    x, y = x[sort_ind], y[sort_ind]
    # Array contains True where an x value is unique or the last of a run of identical x values
    last_of_run = np.asarray(list(np.diff(x) != 0) + [True])
    # Discard the False values, as they represent duplicates - simultaneously keep last of each run of duplicates
    unique_ind = last_of_run.nonzero()[0]
    # Determine the index of the x value chosen to represent each original x value (used to pick y values too)
    replacement = unique_ind[len(unique_ind) - np.cumsum(last_of_run[::-1])[::-1]]
    # All duplicates should have the same y and z values - complain otherwise, but continue
    if not np.all(y[replacement] == y) or not np.all(z[replacement] == z):
        logger.warning("Sensor '%s' has duplicate timestamps with different values or statuses" % name)
    return np.rec.fromarrays([x[unique_ind], y[unique_ind], z[unique_ind]], dtype=sensor.dtype)

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  load_dataset
#--------------------------------------------------------------------------------------------------

# pylint: disable-msg=W0613
def load_dataset(filename, baseline='AxAx', selected_pointing='pos_actual_scan',
                 noise_diode='coupler', time_offset=0.0, **kwargs):
    """Load data set from HDF5 file.

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
    noise_diode : {'coupler', 'pin'}, optional
        Load the model and on/off flags of this noise diode
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
    nd_model : :class:`NoiseDiodeModel` object
        Noise diode model
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
    ValueError
        If file has not been augmented to contain all data fields, or some fields
        are missing
    h5py.H5Error
        If HDF5 error occurred

    """
    # pylint: disable-msg=R0914
    with h5py.File(filename, 'r') as f:
        # Only continue if file has been properly augmented
        if not 'augment' in f.attrs:
            raise ValueError('HDF5 file not augmented - please run augment4.py (provided by k7augment package)')

        # Get attributes at the data set level, with defaults
        experiment_id = f.attrs.get('experiment_id', None)
        observer = f.attrs.get('observer', None)
        description = f.attrs.get('description', None)
        data_unit = f.attrs['data_unit']
        data_timestamps_at_sample_centers = f.attrs['data_timestamps_at_sample_centers']

        # Load antennas group
        ants_group = f['Antennas']
        if baseline == 'AxAx':
            # First single-dish baseline found
            try:
                antA = antB = ants_group.keys()[0]
            except IndexError:
                raise ValueError('Could not load first single-dish baseline - no antennas found in file')
            logger.info("Loading single-dish baseline '%s%s'" % (antA.replace('ntenna', ''),
                                                                 antB.replace('ntenna', '')))
        elif baseline == 'AxAy':
            # First interferometric baseline found
            try:
                antA, antB = ants_group.keys()[:2]
            except IndexError:
                raise ValueError('Could not load first interferometric baseline - less than 2 antennas found in file')
            logger.info("Loading interferometric baseline '%s%s'" % (antA.replace('ntenna', ''),
                                                                     antB.replace('ntenna', '')))
        else:
            # Select antennas involved in specified baseline
            parsed_antenna_indices = baseline_pattern.match(baseline)
            if parsed_antenna_indices is None:
                raise ValueError("Please specify baseline with notation 'AxAy', " +
                                 "where x is index of first antenna and y is index of second antenna")
            antA, antB = ['Antenna' + index for index in parsed_antenna_indices.groups()]
        # Check that requested antennas are in data set
        if antA not in ants_group or antB not in ants_group:
            raise ValueError('Requested antenna pair not found in HDF5 file (wanted %s but file only contains %s)'
                             % ([antA, antB], ants_group.keys()))
        antA_group, antB_group = ants_group[antA], ants_group[antB]
        # Get antenna description strings (antenna2 is None for single-dish data)
        antenna, antenna2 = antA_group.attrs['description'], antB_group.attrs['description']
        if antenna2 == antenna:
            antenna2 = None

        # Use antenna A for noise diode info and weather + pointing sensors
        sensors_group = antA_group['Sensors']
        enviro = {}
        for quantity in ['temperature', 'pressure', 'humidity', 'wind_speed', 'wind_direction']:
            sensor = sensor_name[quantity]
            # Environment sensors are optional
            if sensor in sensors_group:
                enviro[quantity] = remove_duplicates(sensors_group[sensor], sensor)
        # First try to load generic noise diode model (typically found in processed / intermediate file)
        try:
            temperature_H = antA_group['H']['nd_model'].value if 'H' in antA_group else np.ones((1, 2))
            temperature_V = antA_group['V']['nd_model'].value if 'V' in antA_group else np.ones((1, 2))
        except KeyError:
            # Now try to load selected noise diode
            try:
                nd_model = noise_diode + '_nd_model'
                temperature_H = antA_group['H'][nd_model].value if 'H' in antA_group else np.ones((1, 2))
                temperature_V = antA_group['V'][nd_model].value if 'V' in antA_group else np.ones((1, 2))
            except KeyError:
                # Find noise diode model datasets common to both 'H' and 'V' (if these feeds exist)
                nd_set = None
                if 'H' in antA_group:
                    nd_set = set([name.rpartition('_nd_model')[0] for name in antA_group['H']])
                    nd_set.discard('')
                if 'V' in antA_group:
                    nd_set2 = set([name.rpartition('_nd_model')[0] for name in antA_group['V']])
                    nd_set2.discard('')
                    nd_set = nd_set2 if nd_set is None else nd_set.intersection(nd_set2)
                raise ValueError("Unknown noise diode '%s', found the following models instead: %s" %
                                 (noise_diode, list(nd_set)))
        nd_model = NoiseDiodeModel(temperature_H, temperature_V)

        # Load correlator configuration group
        corrconf_group = f['Correlator']
        # If center_freqs dataset is available, use it - otherwise, reconstruct it from DBE attributes
        center_freqs = corrconf_group.get('center_freqs', None)
        if center_freqs:
            center_freqs = center_freqs.value / 1e6
            bandwidths = corrconf_group['bandwidths'].value / 1e6
            num_chans = len(center_freqs)
        else:
            band_center = corrconf_group.attrs['center_frequency_hz'] / 1e6
            channel_bw = corrconf_group.attrs['channel_bandwidth_hz'] / 1e6
            num_chans = corrconf_group.attrs['num_freq_channels']
            # Assume that lower-sideband downconversion has been used, which flips frequency axis
            # Also subtract half a channel width to get frequencies at center of each channel
            center_freqs = band_center - channel_bw * (np.arange(num_chans) - num_chans / 2 + 0.5)
            bandwidths = np.tile(np.float64(channel_bw), num_chans)
        channel_select = corrconf_group['channel_select'].value.nonzero()[0].tolist()
        dump_rate = corrconf_group.attrs['dump_rate_hz']
        sample_period = 1.0 / dump_rate
        corrconf = CorrelatorConfig(center_freqs, bandwidths, channel_select, dump_rate)

        # Figure out mapping of antennas and feeds to the correlation products involved
        if 'input_map' in corrconf_group:
            # Obtain DBE input associated with each polarisation of selected antennas (indicating unavailable feeds)
            antA_H = antA_group['H'].attrs['dbe_input'] if 'H' in antA_group else 'UNAVAILABLE'
            antA_V = antA_group['V'].attrs['dbe_input'] if 'V' in antA_group else 'UNAVAILABLE'
            antB_H = antB_group['H'].attrs['dbe_input'] if 'H' in antB_group else 'UNAVAILABLE'
            antB_V = antB_group['V'].attrs['dbe_input'] if 'V' in antB_group else 'UNAVAILABLE'
            # Mapping of polarisation product to DBE input string identifying the pair of inputs multiplied together
            pol_to_dbestr = {'HH' : antA_H + antB_H, 'VV' : antA_V + antB_V,
                             'HV' : antA_H + antB_V, 'VH' : antA_V + antB_H}
            # Correlator mapping of DBE input string to correlation product index (Miriad-style numbering)
            input_map = corrconf_group['input_map'].value
            dbestr_to_corr_id = dict(zip(input_map['dbe_inputs'], input_map['correlator_product_id']))
            # Overall mapping from polarisation product to correlation product index (None for unavailable products)
            pol_to_corr_id = dict([(pol, dbestr_to_corr_id.get(pol_to_dbestr[pol])) for pol in scape_pol])
        else:
            # Simplified mapping, used in processed / intermediate files
            pol_to_corr_id = dict([(pol, n) for n, pol in enumerate(scape_pol)])

        # Load each compound scan group
        compscanlist = []
        for compscan in f['Scans']:
            compscan_group = f['Scans'][compscan]
            compscan_target = compscan_group.attrs.get('target', 'Nothing, special')
            compscan_label = compscan_group.attrs.get('label', '')

            # Load each scan group within compound scan
            scanlist = []
            for scan in compscan_group:
                scan_group = compscan_group[scan]
                data = scan_group['data']
                num_times = len(scan_group['timestamps'].value)

                # Load correlation data either in float64 (single-dish) form or complex128 (interferometer) form
                # Data has already been normalised by number of samples in integration (accum_per_int)
                # Data of missing feeds are set to zero, as this simplifies the data structure (always 4 products)
                if antenna2 is None:
                    # Single-dish uses HH, VV, Re{HV}, Im{HV}
                    corr_id = [pol_to_corr_id[pol] for pol in ['HH', 'VV', 'HV', 'HV']]
                    scan_data = [data[str(cid)] if cid is not None else np.zeros((num_times, num_chans), np.float64)
                                 for cid in corr_id]
                    scan_data = np.dstack([scan_data[0].real, scan_data[1].real,
                                           scan_data[2].real, scan_data[3].imag]).astype(np.float64)
                else:
                    corr_id = [pol_to_corr_id[pol] for pol in scape_pol]
                    scan_data = [data[str(cid)] if cid is not None else np.zeros((num_times, num_chans), np.complex128)
                                 for cid in corr_id]
                    scan_data = np.dstack(scan_data).astype(np.complex128)
                # Convert from millisecs to secs since Unix epoch, and be sure to use float64 to preserve digits
                data_timestamps = scan_group['timestamps'].value.astype(np.float64) / 1000.0 + time_offset
                # Move correlator data timestamps from start of each sample to the middle
                if not data_timestamps_at_sample_centers:
                    data_timestamps += 0.5 * sample_period
                # If data timestamps have problems, warn and discard the scan
                if np.any(data_timestamps < 1000000000.0) or \
                   (len(data_timestamps) > 1 and np.diff(data_timestamps).min() == 0.0):
                    logger.warning("Discarded %s/%s - bad correlator timestamps (duplicates or way out of date)" %
                                   (compscan, scan))
                    continue

                # If per-scan pointing is available, use it - otherwise, select and interpolate original data
                scan_pointing = scan_group['pointing'].value if 'pointing' in scan_group else None
                if scan_pointing is None:
                    interp_coords = []
                    for coord in ('azim', 'elev'):
                        sensor = '%s_%s' % (selected_pointing, coord)
                        if sensor not in sensors_group:
                            raise ValueError("Selected pointing sensor '%s' was not found in HDF5 file" % (sensor,))
                        # Ensure pointing timestamps are unique before interpolation
                        original_coord = remove_duplicates(sensors_group[sensor], sensor)
                        # Linearly interpolate pointing coordinates to correlator data timestamps
                        # As long as azimuth is in natural antenna coordinates, no special angle interpolation required
                        interp = PiecewisePolynomial1DFit(max_degree=1)
                        interp.fit(original_coord['timestamp'], original_coord['value'])
                        interp_coords.append(interp(data_timestamps).astype(np.float32))
                    scan_pointing = np.rec.fromarrays(interp_coords, names=('az', 'el'))
                # Convert contents of pointing from degrees to radians
                # pylint: disable-msg=W0612
                pointing_view = scan_pointing.view(np.float32)
                pointing_view *= np.float32(np.pi / 180.0)

                # If per-scan flags are available, use it - otherwise, select and interpolate original data
                scan_flags = scan_group['flags'].value if 'flags' in scan_group else None
                if scan_flags is None:
                    sensor = sensor_name[noise_diode + '_nd_on']
                    if sensor in sensors_group:
                        # Ensure noise diode timestamps are unique before interpolation
                        nd_on = remove_duplicates(sensors_group[sensor], sensor)
                        # Do step-wise interpolation (as flag is either 0 or 1 and holds its value until it toggles)
                        interp = PiecewisePolynomial1DFit(max_degree=0)
                        interp.fit(nd_on['timestamp'], nd_on['value'].astype(int))
                        scan_flags = np.rec.fromarrays([interp(data_timestamps).astype(bool)], names=('nd_on',))
                    else:
                        logger.warning(("Selected noise diode sensor '%s'" % (sensor,)) +
                                       " not found in HDF5 file - setting nd_on to False")
                        scan_flags = np.rec.fromarrays([np.tile(False, data_timestamps.shape)], names=('nd_on',))
                scan_label = scan_group.attrs.get('label', '')

                scanlist.append(Scan(scan_data, data_timestamps, scan_pointing, scan_flags, scan_label,
                                     filename + '/Scans/%s/%s' % (compscan, scan)))

            if len(scanlist) > 0:
                # Sort scans chronologically, as h5py seems to scramble them based on group name
                scanlist.sort(key=lambda scan: scan.timestamps[0])
                compscanlist.append(CompoundScan(scanlist, compscan_target, compscan_label))

        if len(compscanlist) > 0:
            # Sort compound scans chronologically too
            compscanlist.sort(key=lambda compscan: compscan.scans[0].timestamps[0])
        return compscanlist, experiment_id, observer, description, data_unit, \
               corrconf, antenna, antenna2, nd_model, enviro

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  save_dataset
#--------------------------------------------------------------------------------------------------

def save_dataset(dataset, filename):
    """Save data set to HDF5 file.

    This will overwrite any existing file with the same name.

    Parameters
    ----------
    dataset : :class:`dataset.DataSet` object
        Data set object to save
    filename : string
        Name of output HDF5 file

    """
    with h5py.File(filename, 'w') as f:
        # Attributes at data set level
        f['/'].attrs['augment'] = 'File created by scape'
        f['/'].attrs['experiment_id'] = dataset.experiment_id
        f['/'].attrs['observer'] = dataset.observer
        f['/'].attrs['description'] = dataset.description
        f['/'].attrs['data_unit'] = dataset.data_unit
        f['/'].attrs['data_timestamps_at_sample_centers'] = True

        # Create antennas group
        ants_group = f.create_group('Antennas')
        # If antenna names follow standard pattern, number antenna groups appropriately - otherwise number them 1 and 2
        antenna_index = antenna_name_pattern.match(dataset.antenna.name)
        antA_name = ('Antenna%s' % antenna_index.groups()) if antenna_index is not None else 'Antenna1'
        # Create first antenna group
        antA_group = ants_group.create_group(antA_name)
        antA_group.attrs['description'] = dataset.antenna.description
        # Create second antenna group if the data set is interferometric
        if dataset.antenna2 is not None:
            antenna_index = antenna_name_pattern.match(dataset.antenna2.name)
            antB_name = ('Antenna%s' % antenna_index.groups()) if antenna_index is not None else 'Antenna2'
            antB_group = ants_group.create_group(antB_name)
            antB_group.attrs['description'] = dataset.antenna2.description

        # Create receiver chain groups and enviro sensors for first antenna (other antenna left blank)
        sensors_group = antA_group.create_group('Sensors')
        for quantity in ['temperature', 'pressure', 'humidity', 'wind_speed', 'wind_direction']:
            sensor = sensor_name[quantity]
            # Environment sensors are optional
            if quantity in dataset.enviro:
                sensors_group.create_dataset(sensor, data=dataset.enviro[quantity], compression='gzip')
        # For now, both H and V are created, even if original dataset had only a single feed, to keep things simple
        # The correlator data for the missing feed are already zeros in this situation
        h_group, v_group = antA_group.create_group('H'), antA_group.create_group('V')
        h_group.create_dataset('nd_model', data=dataset.nd_model.temperature_h, compression='gzip')
        v_group.create_dataset('nd_model', data=dataset.nd_model.temperature_v, compression='gzip')

        # Create correlator configuration group
        corrconf_group = f.create_group('Correlator')
        corrconf_group.create_dataset('center_freqs', data=dataset.corrconf.freqs * 1e6, compression='gzip')
        corrconf_group.create_dataset('bandwidths', data=dataset.corrconf.bandwidths * 1e6, compression='gzip')
        select_flags = np.tile(False, len(dataset.corrconf.freqs))
        select_flags[dataset.corrconf.channel_select] = True
        corrconf_group.create_dataset('channel_select', data=select_flags)
        corrconf_group.attrs['dump_rate_hz'] = dataset.corrconf.dump_rate
        # Simplified correlator ID lookup
        pol_to_corr_id = dict([(pol, n) for n, pol in enumerate(scape_pol)])

        scans_group = f.create_group('Scans')
        for compscan_ind, compscan in enumerate(dataset.compscans):
            compscan_group = scans_group.create_group('CompoundScan%d' % compscan_ind)
            compscan_group.attrs['target'] = compscan.target.description
            compscan_group.attrs['label'] = compscan.label

            for scan_ind, scan in enumerate(compscan.scans):
                scan_group = compscan_group.create_group('Scan%d' % scan_ind)
                # Always save power data in complex128 form
                corr_id = [str(pol_to_corr_id[pol]) for pol in scape_pol]
                complex_data = np.rec.fromarrays([scan.pol(key) for key in scape_pol],
                                                 names=corr_id, formats=['complex128'] * 4)
                scan_group.create_dataset('data', data=complex_data, compression='gzip')
                # Save data timestamps in milliseconds
                scan_group.create_dataset('timestamps', data=1000.0 * scan.timestamps, compression='gzip')
                # Convert contents of pointing from radians to degrees, without disturbing original
                # pylint: disable-msg=W0612
                pointing = scan.pointing.copy()
                pointing_view = pointing.view(np.float32)
                pointing_view *= np.float32(180.0 / np.pi)
                scan_group.create_dataset('pointing', data=pointing, compression='gzip')
                scan_group.create_dataset('flags', data=scan.flags, compression='gzip')
                scan_group.attrs['label'] = scan.label
