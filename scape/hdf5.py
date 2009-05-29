"""Read and write HDF5 files."""

from __future__ import with_statement

import os.path

import h5py
import numpy as np

from .gaincal import NoiseDiodeModel

#--------------------------------------------------------------------------------------------------
#--- FUNCTIONS
#--------------------------------------------------------------------------------------------------

def load_dataset(data_filename):
    """Load data set from XDM FITS file series.
    
    This loads the XDM data set starting at the given filename and consisting of
    consecutively numbered FITS files. The noise diode model can also be
    overridden. Since this function is usually not called directly, but via the
    :class:`dataset.DataSet` initialiser, the noise diode file should rather be
    assigned to :data:`default_nd_filename`.
    
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
    nd_data : :class:`NoiseDiodeXDM` object
        Noise diode model
    
    """
    pass


def save_dataset(dataset, filename):
    """Save data set to HDF5 file.
    
    Parameters
    ----------
    dataset : :class:`dataset.DataSet` object
        Data set object to save
    data_filename : string
        Name of output HDF5 file
    
    Raises
    ------
    ValueError
        If file already exists
    
    """
    if os.path.exists(filename):
        raise ValueError('File %s already exists - please remove first!' % filename)
    with h5py.File(filename, 'w') as f:
        f['/'].create_dataset('pointing_model', data=np.zeros(16))
        f['/'].attrs['data_unit'] = dataset.data_unit
        f['/'].attrs['antenna'] = dataset.antenna.name
        f['/'].attrs['comment'] = ''
        
        spectral_group = f.create_group('CorrelatorConfig')
        spectral_group.create_dataset('center_freqs', data=dataset.spectral.freqs, compression='gzip')
        spectral_group.create_dataset('bandwidths', data=dataset.spectral.bandwidths, compression='gzip')
        spectral_group.create_dataset('rfi_channels', data=np.array(dataset.spectral.rfi_channels))
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
