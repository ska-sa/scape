"""Read and write HDF5 files."""

from __future__ import with_statement

import logging

import h5py
import numpy as np

from .gaincal import NoiseDiodeModel
from .scan import Scan
from .compoundscan import CorrelatorConfig, CompoundScan

logger = logging.getLogger("scape.hdf5")

#--------------------------------------------------------------------------------------------------
#--- FUNCTIONS
#--------------------------------------------------------------------------------------------------

def load_dataset(filename):
    """Load data set from HDF5 file.
    
    This loads a data set from an HDF5 file.
    
    Parameters
    ----------
    filename : string
        Name of input HDF5 file
    
    Returns
    -------
    compscanlist : list of :class:`compoundscan.CompoundScan` objects
        List of compound scans
    data_unit : {'raw', 'K', 'Jy'}
        Physical unit of power data
    corrconf : :class:`compoundscan.CorrelatorConfig` object
        Correlator configuration object
    antenna : string
        Name of antenna that produced the data set
    nd_data : :class:`NoiseDiodeModel` object
        Noise diode model
    
    """
    # pylint: disable-msg=R0914
    with h5py.File(filename, 'r') as f:
        pointing_model = f['pointing_model'].value  # TODO return this as well
        data_unit = f.attrs['data_unit']
        antenna = f.attrs['antenna']
        comment = f.attrs['comment'] # TODO return this as well
        
        # If center_freqs dataset is available, use it - otherwise, reconstruct it from DBE attributes
        center_freqs = f['CorrelatorConfig'].get('center_freqs', None)
        if center_freqs:
            center_freqs = center_freqs.value
            bandwidths = f['CorrelatorConfig']['bandwidths'].value
        else:
            band_center = f['CorrelatorConfig'].attrs['center_frequency_hz']
            channel_bw = f['CorrelatorConfig'].attrs['bandwidth_hz']
            num_chans = f['CorrelatorConfig'].attrs['num_freq_channels']
            center_freqs = np.arange(band_center - (channel_bw * num_chans / 2.0) + channel_bw / 2.0,
                                     band_center + (channel_bw * num_chans / 2.0) + channel_bw / 2.0,
                                     channel_bw, dtype=np.float64)
            bandwidths = np.tile(channel_bw, num_chans, dtype=np.float64)
        rfi_channels = f['CorrelatorConfig']['rfi_channels'].value.nonzero()[0].tolist()
        dump_rate = f['CorrelatorConfig'].attrs['dump_rate']
        corrconf = CorrelatorConfig(center_freqs, bandwidths, rfi_channels, dump_rate)
        
        temperature_x = f['NoiseDiodeModel']['temperature_x'].value
        temperature_y = f['NoiseDiodeModel']['temperature_y'].value
        nd_data = NoiseDiodeModel(temperature_x, temperature_y)
        
        compscanlist = []
        for compscan in f['Scans']:
            compscan_target = f['Scans'][compscan].attrs['target']
            compscan_comment = f['Scans'][compscan].attrs['comment'] # TODO: do something with this
            
            scanlist = []
            for scan in f['Scans'][compscan]:
                complex_data = f['Scans'][compscan][scan]['data'].value
                assert complex_data.dtype.fields.has_key('XX'), "Power data is not in coherency form"
                # Load power data either in complex64 form or uint32 form
                if complex_data.dtype.fields['XX'][0] == np.complex64:
                    scan_data = np.dstack([complex_data['XX'].real, complex_data['YY'].real,
                                           2.0 * complex_data['XY'].real, 2.0 * complex_data['XY'].imag])
                else:
                    if complex_data.view(np.uint32).max() > 2 ** 24:
                        logger.warning('Uint32 data too large to be accurately represented as 32-bit floats')
                    scan_data = np.dstack([complex_data['XX']['r'].astype(np.float32),
                                           complex_data['YY']['r'].astype(np.float32),
                                           2.0 * complex_data['XY']['r'].astype(np.float32),
                                           2.0 * complex_data['XY']['i'].astype(np.float32)])
                # Convert from millisecs to secs since Unix epoch, and be sure to use float64 to preserve digits
                scan_timestamps = f['Scans'][compscan][scan]['timestamps'].value.astype(np.float64) / 1000.0
                scan_pointing = f['Scans'][compscan][scan]['pointing'].value
                # Convert contents of pointing from degrees to radians
                pointing_view = scan_pointing.view(np.float32)
                pointing_view *= np.pi / 180.0
                scan_flags = f['Scans'][compscan][scan]['flags'].value
                scan_environment = f['Scans'][compscan][scan]['environment'].value
                scan_label = f['Scans'][compscan][scan].attrs['label']
                scan_comment = f['Scans'][compscan][scan].attrs['comment'] # TODO: do something with this
                
                scanlist.append(Scan(scan_data, False, scan_timestamps, scan_pointing, scan_flags,
                                     scan_label, filename + '/Scans/%s/%s' % (compscan, scan)))
            
            compscanlist.append(CompoundScan(scanlist, compscan_target))
        
        return compscanlist, data_unit, corrconf, antenna, nd_data


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
        f['/'].create_dataset('pointing_model', data=np.zeros(16, dtype=np.float32))
        f['/'].attrs['data_unit'] = dataset.data_unit
        f['/'].attrs['antenna'] = dataset.antenna.name
        f['/'].attrs['comment'] = ''
        
        corrconf_group = f.create_group('CorrelatorConfig')
        corrconf_group.create_dataset('center_freqs', data=dataset.corrconf.freqs, compression='gzip')
        corrconf_group.create_dataset('bandwidths', data=dataset.corrconf.bandwidths, compression='gzip')
        rfi_flags = np.tile(False, len(dataset.corrconf.freqs))
        rfi_flags[dataset.corrconf.rfi_channels] = True
        corrconf_group.create_dataset('rfi_channels', data=rfi_flags)
        corrconf_group.attrs['dump_rate'] = dataset.corrconf.dump_rate
        
        nd_group = f.create_group('NoiseDiodeModel')
        nd_group.create_dataset('temperature_x', data=dataset.noise_diode_data.temperature_x, compression='gzip')
        nd_group.create_dataset('temperature_y', data=dataset.noise_diode_data.temperature_y, compression='gzip')
        
        scans_group = f.create_group('Scans')
        for compscan_ind, compscan in enumerate(dataset.compscans):
            compscan_group = scans_group.create_group('CompoundScan%d' % compscan_ind)
            compscan_group.attrs['target'] = compscan.target.name
            compscan_group.attrs['comment'] = ''
            
            for scan_ind, scan in enumerate(compscan.scans):
                scan_group = compscan_group.create_group('Scan%d' % scan_ind)
                
                coherency_order = ['XX', 'YY', 'XY', 'YX']
                # Always save power data in complex64 form
                complex_data = np.rec.fromarrays([scan.coherency(key) for key in coherency_order],
                                                 names=coherency_order, formats=['complex64'] * 4)
                scan_group.create_dataset('data', data=complex_data, compression='gzip')
                # Convert from float64 seconds to uint64 milliseconds
                scan_group.create_dataset('timestamps', data=np.round(1000.0 * scan.timestamps).astype(np.uint64),
                                          compression='gzip')
                # Convert contents of pointing from radians to degrees, without disturbing original
                pointing_deg = scan.pointing.copy()
                pointing_view = pointing_deg.view(np.float32)
                pointing_view *= 180.0 / np.pi
                scan_group.create_dataset('pointing', data=pointing_deg, compression='gzip')
                scan_group.create_dataset('flags', data=scan.flags, compression='gzip')
                # Dummy environmental data for now
                num_samples = len(scan.timestamps)
                enviro = np.rec.fromarrays(np.zeros((3, num_samples), dtype=np.float32),
                                           names='temperature,pressure,humidity')
                scan_group.create_dataset('environment', data=enviro, compression='gzip')
                
                scan_group.attrs['label'] = scan.label
                scan_group.attrs['comment'] = ''
