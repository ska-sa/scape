"""Read and write HDF5 files."""

from __future__ import with_statement

import os.path

import h5py
import numpy as np

from .gaincal import NoiseDiodeModel
from .scan import Scan
from .compoundscan import SpectralConfig, CompoundScan

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
    spectral : :class:`compoundscan.SpectralConfig` object
        Spectral configuration object
    antenna : string
        Name of antenna that produced the data set
    nd_data : :class:`NoiseDiodeModel` object
        Noise diode model
    
    """
    with h5py.File(filename, 'r') as f:
        # top level attributes
        pointing_model = f['pointing_model'].value  # TODO return this as well
        data_unit = f.attrs['data_unit']
        antenna = f.attrs['antenna']
        comment = f.attrs['comment'] # TODO return this as well
        
        # CorrelatorConfig stuff
        center_freqs = f['CorrelatorConfig']['center_freqs'].value
        bandwidths = f['CorrelatorConfig']['bandwidths'].value
        rfi_channels = f['CorrelatorConfig']['rfi_channels'].value.nonzero()[0].tolist()
        dump_rate = f['CorrelatorConfig'].attrs['dump_rate']
        spectral = SpectralConfig(center_freqs, bandwidths, rfi_channels, [], dump_rate) # TODO remove channels_per_band empty_list
        
        # noise diode model
        temperature_x = f['NoiseDiodeModel']['temperature_x'].value
        temperature_y = f['NoiseDiodeModel']['temperature_y'].value
        nd_data = NoiseDiodeModel(temperature_x, temperature_y)
        
        # compound scans
        compscanlist = []
        for compscan in f['Scans']:
            compscan_target = f['Scans'][compscan].attrs['target']
            compscan_comment = f['Scans'][compscan].attrs['comment'] # TODO: do something with this
            
            scanlist = []
            for scan in f['Scans'][compscan]:
                scan_complex_data = f['Scans'][compscan][scan]['data'].value
                scan_data = np.dstack([scan_complex_data['XX'].real, scan_complex_data['YY'].real,
                                     2.0 * scan_complex_data['XY'].real, 2.0 * scan_complex_data['XY'].imag])
                scan_timestamps = f['Scans'][compscan][scan]['timestamps'].value
                scan_pointing = f['Scans'][compscan][scan]['pointing'].value
                scan_flags = f['Scans'][compscan][scan]['flags'].value
                scan_environment = f['Scans'][compscan][scan]['environment'].value
                scan_label = f['Scans'][compscan][scan].attrs['label']
                scan_comment = f['Scans'][compscan][scan].attrs['comment'] # TODO: do something with this
                
                scanlist.append(Scan(scan_data, False, scan_timestamps, scan_pointing, scan_flags,
                                      scan_label, filename + '/Scans/%s/%s' % (compscan, scan)))
            
            compscanlist.append(CompoundScan(scanlist, compscan_target))
        
        return compscanlist, data_unit, spectral, antenna, nd_data


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
        f['/'].create_dataset('pointing_model', data=np.zeros(16))
        f['/'].attrs['data_unit'] = dataset.data_unit
        f['/'].attrs['antenna'] = dataset.antenna.name
        f['/'].attrs['comment'] = ''
        
        spectral_group = f.create_group('CorrelatorConfig')
        spectral_group.create_dataset('center_freqs', data=dataset.spectral.freqs, compression='gzip')
        spectral_group.create_dataset('bandwidths', data=dataset.spectral.bandwidths, compression='gzip')
        rfi_flags = np.tile(False, len(dataset.spectral.freqs))
        rfi_flags[dataset.spectral.rfi_channels] = True
        spectral_group.create_dataset('rfi_channels', data=rfi_flags)
        spectral_group.attrs['dump_rate'] = dataset.spectral.dump_rate
        
        nd_group = f.create_group('NoiseDiodeModel')
        nd_group.create_dataset('temperature_x', data=dataset.noise_diode_data.table_x, compression='gzip')
        nd_group.create_dataset('temperature_y', data=dataset.noise_diode_data.table_y, compression='gzip')
        
        scans_group = f.create_group('Scans')
        for compscan_ind, compscan in enumerate(dataset.compscans):
            compscan_group = scans_group.create_group('CompoundScan%d' % compscan_ind)
            compscan_group.attrs['target'] = compscan.target.name
            compscan_group.attrs['comment'] = ''
            
            for scan_ind, scan in enumerate(compscan.scans):
                scan_group = compscan_group.create_group('Scan%d' % scan_ind)
                
                coherency_order = ['XX', 'YY', 'XY', 'YX']
                complex_data = np.rec.fromarrays([scan.coherency(key) for key in coherency_order],
                                                 names=coherency_order, formats=['complex64'] * 4)
                scan_group.create_dataset('data', data=complex_data, compression='gzip')
                scan_group.create_dataset('timestamps', data=scan.timestamps, compression='gzip')
                scan_group.create_dataset('pointing', data=scan.pointing, compression='gzip')
                scan_group.create_dataset('flags', data=scan.flags, compression='gzip')
                # Dummy environmental data for now
                num_samples = len(scan.timestamps)
                enviro = np.rec.fromarrays([np.zeros(num_samples), np.zeros(num_samples), np.zeros(num_samples)],
                                           names=['temperature','pressure', 'humidity'])
                scan_group.create_dataset('environment', data=enviro, compression='gzip')
                
                scan_group.attrs['label'] = scan.label
                scan_group.attrs['comment'] = ''
