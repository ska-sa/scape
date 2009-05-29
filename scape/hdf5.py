"""Read and write HDF5 files."""

from __future__ import with_statement

import os.path

import h5py
import numpy as np

from .gaincal import NoiseDiodeModel
from .subscan import SubScan
from .scan import SpectralConfig, Scan

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
    scanlist : list of :class:`scan.Scan` objects
        List of scans
    data_unit : {'raw', 'K', 'Jy'}
        Physical unit of power data
    spectral : :class:`scan.SpectralConfig` object
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

        # scans
        scanlist = []
        for s in f['Scans']:
            scan_target = f['Scans'][s].attrs['target']
            scan_comment = f['Scans'][s].attrs['comment'] # TODO: do something with this
     
            sslist = []
            for ss in f['Scans'][s]:
                ss_complex_data = f['Scans'][s][ss]['data'].value
                ss_data = np.dstack([ss_complex_data['XX'].real, ss_complex_data['YY'].real, 
                                     2.0 * ss_complex_data['XY'].real, 2.0 * ss_complex_data['XY'].imag])
                ss_timestamps = f['Scans'][s][ss]['timestamps'].value
                ss_pointing = f['Scans'][s][ss]['pointing'].value
                ss_flags = f['Scans'][s][ss]['flags'].value
                ss_environment = f['Scans'][s][ss]['environment'].value
                ss_label = f['Scans'][s][ss].attrs['label']
                ss_comment = f['Scans'][s][ss].attrs['comment'] # TODO: do something with this

                sslist.append(SubScan(ss_data, False, ss_timestamps, ss_pointing, ss_flags,
                                      ss_label, filename + '/Scans/%s/%s' % (s, ss)))

            scanlist.append(Scan(sslist,scan_target))

        return scanlist, data_unit, spectral, antenna, nd_data


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
        for scan_ind, s in enumerate(dataset.scans):            
            scan_group = scans_group.create_group('Scan%d' % scan_ind)
            scan_group.attrs['target'] = s.target.name
            scan_group.attrs['comment'] = ''
            
            for subscan_ind, ss in enumerate(s.subscans):
                subscan_group = scan_group.create_group('SubScan%d' % subscan_ind)
                
                coherency_order = ['XX', 'YY', 'XY', 'YX']
                complex_data = np.rec.fromarrays([ss.coherency(key) for key in coherency_order], 
                                                 names=coherency_order, formats=['complex64'] * 4)
                subscan_group.create_dataset('data', data=complex_data, compression='gzip')
                subscan_group.create_dataset('timestamps', data=ss.timestamps, compression='gzip')
                subscan_group.create_dataset('pointing', data=ss.pointing, compression='gzip')
                subscan_group.create_dataset('flags', data=ss.flags, compression='gzip')
                # Dummy environmental data for now
                num_samples = len(ss.timestamps)
                enviro = np.rec.fromarrays([np.zeros(num_samples), np.zeros(num_samples), np.zeros(num_samples)],
                                           names=['temperature','pressure', 'humidity'])
                subscan_group.create_dataset('environment', data=enviro, compression='gzip')
                
                subscan_group.attrs['label'] = ss.label
                subscan_group.attrs['comment'] = ''
