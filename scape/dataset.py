"""Container for the data of a single-dish experiment."""

import os.path

from scan import Scan

# Try to import all available formats
try:
    import xdmfits
    xdmfits_found = True
except ImportError:
    xdmfits_found = False
try:
    import hdf5
    hdf5_found = True
except ImportError:
    hdf5_found = False

class DataSet(object):
    """Container for the data of a single-dish experiment.
    
    This is the top-level container for the data of a single-dish experiment.
    Given a data filename, the initialiser determines the appropriate file format
    to use, based on the file extension. If the filename is blank, the
    :class:`DataSet` can also be directly initialised from its constituent
    parts, which is useful for simulations and creating the data sets in the
    first place. The :class:`DataSet` object contains a list of scans as well as
    the noise diode characteristics.
    
    Parameters
    ----------
    filename : string
        Name of data set file, or blank string if the other parameters are given
    scanlist : list of :class:`scan.Scan` objects, optional
        Use this to initialise data set if filename is explicitly blanked
    nd_data : :class:`gaincal.NoiseDiodeBase` object, optional
        Use this to initialise data set if filename is explicitly blanked
    
    Raises
    ------
    ImportError
        If file extension is known, but appropriate module would not import
    ValueError
        If file extension is unknown
    
    """
    def __init__(self, filename, scanlist=None, nd_data=None):
        if filename:
            ext = os.path.splitext(filename)[1]
            if ext == '.fits':
                if not xdmfits_found:
                    raise ImportError('XDM FITS support could not be loaded - please check xdmfits module')
                scanlist, nd_data = xdmfits.load_dataset(filename)
            elif (ext == '.h5') or (ext == '.hdf5'):
                if not hdf5_found:
                    raise ImportError('HDF5 support could not be loaded - please check hdf5 module')
                scanlist, nd_data = hdf5.load_dataset(filename)
            else:
                raise ValueError("File extension '%s' not understood" % ext)
        self.scans = scanlist
        self.noise_diode_data = nd_data
    
    def iter_subscans(self):
        """Iterator over all subscans in data set."""
        subscanlist = []
        for s in self.scans:
            subscanlist.extend(s.subscans)
        return iter(subscanlist)

    def select(self, labelkeep=None, timekeep=None, freqkeep=None, copy=False):
        """Select subset of data set, based on subscan label, time and frequency."""
        scanlist = []
        for s in self.scans:
            subscanlist = []
            for ss in s.subscans:
                if (labelkeep is None) or (ss.label in labelkeep):
                    subscanlist.append(ss.select(timekeep, freqkeep, copy))
            if subscanlist:
                scanlist.append(Scan(subscanlist))
        return DataSet(None, scanlist, self.noise_diode_data)
