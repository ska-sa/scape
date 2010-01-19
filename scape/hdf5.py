"""Read and write HDF5 files."""

from __future__ import with_statement

import logging

import h5py
import numpy as np

from .gaincal import NoiseDiodeModel
from .scan import Scan, scape_pol
from .compoundscan import CorrelatorConfig, CompoundScan
from .fitting import PiecewisePolynomial1DFit, Independent1DFit

logger = logging.getLogger("scape.hdf5")

# Mapping from scape polarisation component to corresponding HDF5 correlation product
scape_to_hdf5 = {'HH' : 'AxBx', 'VV' : 'AyBy', 'HV' : 'AxBy', 'VH' : 'AyBx'}

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  load_dataset
#--------------------------------------------------------------------------------------------------

# pylint: disable-msg=W0613
def load_dataset(filename, selected_pointing='actual_scan', **kwargs):
    """Load data set from HDF5 file.

    This loads a data set from an HDF5 file.

    Parameters
    ----------
    filename : string
        Name of input HDF5 file
    selected_pointing : string, optional
        Identifier of (az, el) sensors that will provide all the pointing data
        of the data set. The actual sensor names are formed by appending '_azim'
        and '_elev' to *selected_pointing*. This is ignored if the data set has
        already been processed to contain a single source of pointing data.
    kwargs : dict, optional
        Extra keyword arguments are ignored, as they typically apply to other formats

    Returns
    -------
    compscanlist : list of :class:`compoundscan.CompoundScan` objects
        List of compound scans
    data_unit : {'raw', 'K', 'Jy'}
        Physical unit of power data
    corrconf : :class:`compoundscan.CorrelatorConfig` object
        Correlator configuration object
    antenna : string
        Description string of single-dish antenna or first antenna of baseline pair
    antenna2 : string or None
        Description string of second antenna of baseline pair (None for single-dish)
    nd_data : :class:`NoiseDiodeModel` object
        Noise diode model
    pointing_model : array of float
        Pointing model parameters, nominally in radians

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
            raise ValueError('HDF5 file not augmented - please run k7augment/augment2.py on this file')
        pointing_model = f['pointing_model'].value
        data_unit = f.attrs['data_unit']
        data_timestamps_at_sample_centers = f.attrs.get('data_timestamps_at_sample_centers', False)
        antenna = f.attrs['antenna']
        # Get second antenna of baseline pair (or set to None for single-dish data)
        antenna2 = f.attrs.get('antenna2', None)
        if antenna2 == antenna:
            antenna2 = None
        comment = f.attrs['comment'] # TODO return this as well

        # Load correlator configuration group
        corrconf_group = f['CorrelatorConfig']
        # If center_freqs dataset is available, use it - otherwise, reconstruct it from DBE attributes
        center_freqs = corrconf_group.get('center_freqs', None)
        if center_freqs:
            center_freqs = center_freqs.value / 1e6
            bandwidths = corrconf_group['bandwidths'].value / 1e6
        else:
            band_center = corrconf_group.attrs['center_frequency_hz'] / 1e6
            channel_bw = corrconf_group.attrs['channel_bandwidth_hz'] / 1e6
            num_chans = corrconf_group.attrs['num_freq_channels']
            # Assume that lower-sideband downconversion has been used, which flips frequency axis
            # Also subtract half a channel width to get frequencies at center of each channel
            center_freqs = band_center - channel_bw * (np.arange(num_chans) - num_chans / 2 + 0.5)
            bandwidths = np.tile(np.float64(channel_bw), num_chans)
        rfi_channels = corrconf_group['rfi_channels'].value.nonzero()[0].tolist()
        dump_rate = corrconf_group.attrs['dump_rate']
        sample_period = 1.0 / dump_rate
        corrconf = CorrelatorConfig(center_freqs, bandwidths, rfi_channels, dump_rate)

        # Load noise diode model group
        temperature_x = f['NoiseDiodeModel']['temperature_x'].value
        temperature_y = f['NoiseDiodeModel']['temperature_y'].value
        nd_data = NoiseDiodeModel(temperature_x, temperature_y)

        # Load each compound scan group
        compscanlist = []
        for compscan in f['Scans']:
            compscan_group = f['Scans'][compscan]
            compscan_target = compscan_group.attrs['target']
            compscan_comment = compscan_group.attrs['comment'] # TODO: do something with this

            # Load each scan group within compound scan
            scanlist = []
            for scan in compscan_group:
                scan_group = compscan_group[scan]
                data = scan_group['data'].value
                # Load power data either in float64 (single-dish) form or complex128 (interferometer) form
                # Data has already been normalised by number of samples in integration (accum_per_int)
                if antenna2 is None:
                    scan_data = np.dstack([data[scape_to_hdf5['HH']].real,
                                           data[scape_to_hdf5['VV']].real,
                                           data[scape_to_hdf5['HV']].real,
                                           data[scape_to_hdf5['HV']].imag]).astype(np.float64)
                else:
                    scan_data = np.dstack([data[scape_to_hdf5[p]] for p in scape_pol]).astype(np.complex128)
                # Convert from millisecs to secs since Unix epoch, and be sure to use float64 to preserve digits
                data_timestamps = scan_group['timestamps'].value.astype(np.float64) / 1000.0
                # Move correlator data timestamps from start of each sample to the middle
                if not data_timestamps_at_sample_centers:
                    data_timestamps += 0.5 * sample_period
                # If reduced pointing dataset is available, use it - otherwise, select and interpolate original data
                scan_pointing = scan_group['pointing'].value if 'pointing' in scan_group else None
                if scan_pointing is None:
                    # Select appropriate sensor to use for (az, el) data
                    if selected_pointing.startswith('request_'):
                        scan_pointing = scan_group['requested_pointing'].value
                    else:
                        scan_pointing = scan_group['actual_pointing'].value
                    azel_timestamps = scan_pointing['timestamp']
                    try:
                        original_az = scan_pointing[selected_pointing + '_azim']
                        original_el = scan_pointing[selected_pointing + '_elev']
                    except ValueError:
                        raise ValueError("The selected pointing sensor '%s_{azim,elev}' was not found in HDF5 file" %
                                         (selected_pointing))
                    # Linearly interpolate (az, el) coordinates to correlator data timestamps
                    interp = Independent1DFit(PiecewisePolynomial1DFit(max_degree=1), axis=1)
                    interp.fit(azel_timestamps, [original_az, original_el])
                    scan_pointing = np.rec.fromarrays(interp(data_timestamps).astype(np.float32), names=('az', 'el'))
                # Convert contents of pointing from degrees to radians
                # pylint: disable-msg=W0612
                pointing_view = scan_pointing.view(np.float32)
                pointing_view *= np.float32(np.pi / 180.0)
                scan_flags = scan_group['flags'].value
                scan_enviro_ambient = scan_group['enviro_ambient'].value if 'enviro_ambient' in scan_group else None
                scan_enviro_wind = scan_group['enviro_wind'].value if 'enviro_wind' in scan_group else None
                scan_label = scan_group.attrs['label']
                scan_comment = scan_group.attrs['comment'] # TODO: do something with this

                scanlist.append(Scan(scan_data, data_timestamps, scan_pointing, scan_flags, scan_enviro_ambient,
                                     scan_enviro_wind, scan_label, filename + '/Scans/%s/%s' % (compscan, scan)))

            # Sort scans chronologically, as h5py seems to scramble them based on group name
            scanlist.sort(key=lambda scan: scan.timestamps[0])
            compscanlist.append(CompoundScan(scanlist, compscan_target))

        # Sort compound scans chronologically too
        compscanlist.sort(key=lambda compscan: compscan.scans[0].timestamps[0])
        return compscanlist, data_unit, corrconf, antenna, antenna2, nd_data, pointing_model

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
        f['/'].create_dataset('pointing_model', data=dataset.pointing_model.params)
        f['/'].attrs['data_unit'] = dataset.data_unit
        f['/'].attrs['data_timestamps_at_sample_centers'] = True
        f['/'].attrs['antenna'] = dataset.antenna.description
        if dataset.antenna2 is not None:
            f['/'].attrs['antenna2'] = dataset.antenna2.description
        else:
            f['/'].attrs['antenna2'] = f['/'].attrs['antenna']
        f['/'].attrs['comment'] = ''
        f['/'].attrs['augment'] = 'File created by scape'

        corrconf_group = f.create_group('CorrelatorConfig')
        corrconf_group.create_dataset('center_freqs', data=dataset.corrconf.freqs * 1e6, compression='gzip')
        corrconf_group.create_dataset('bandwidths', data=dataset.corrconf.bandwidths * 1e6, compression='gzip')
        rfi_flags = np.tile(False, len(dataset.corrconf.freqs))
        rfi_flags[dataset.corrconf.rfi_channels] = True
        corrconf_group.create_dataset('rfi_channels', data=rfi_flags)
        corrconf_group.attrs['dump_rate'] = dataset.corrconf.dump_rate
        sample_period = 1.0 / dataset.corrconf.dump_rate

        nd_group = f.create_group('NoiseDiodeModel')
        nd_group.create_dataset('temperature_x', data=dataset.noise_diode_data.temperature_x, compression='gzip')
        nd_group.create_dataset('temperature_y', data=dataset.noise_diode_data.temperature_y, compression='gzip')

        scans_group = f.create_group('Scans')
        for compscan_ind, compscan in enumerate(dataset.compscans):
            compscan_group = scans_group.create_group('CompoundScan%d' % compscan_ind)
            compscan_group.attrs['target'] = compscan.target.description
            compscan_group.attrs['comment'] = ''

            for scan_ind, scan in enumerate(compscan.scans):
                scan_group = compscan_group.create_group('Scan%d' % scan_ind)

                # Always save power data in complex64 form
                hdf5_pol = [scape_to_hdf5[key] for key in scape_pol]
                complex_data = np.rec.fromarrays([scan.pol(key) for key in scape_pol],
                                                 names=hdf5_pol, formats=['complex64'] * 4)
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
                if scan.enviro_ambient:
                    scan_group.create_dataset('enviro_ambient', data=scan.enviro_ambient, compression='gzip')
                if scan.enviro_wind:
                    scan_group.create_dataset('enviro_wind', data=scan.enviro_wind, compression='gzip')

                scan_group.attrs['label'] = scan.label
                scan_group.attrs['comment'] = ''
