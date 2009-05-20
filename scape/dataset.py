"""Container for the data of a single-dish experiment."""

import os.path

import numpy as np

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

    def select(self, labelkeep=None, flagkeep=None, freqkeep=None, copy=False):
        """Select subset of data set, based on subscan label, flags and frequency.
        
        This returns a data set with a possibly reduced number of time samples,
        frequency channels/bands and subscans, based on the selection criteria.
        Since each subscan potentially has a different number of time samples,
        it is less useful to filter directly on sample index. Instead, the
        flags are used to select a subset of time samples in each subscan. The
        list of flags are ANDed together to determine which parts are kept. It
        is also possible to invert flags by prepending a ~ (tilde) character.
        
        Based on the value of *copy*, the new data set contains either a view of
        the original data or a copy. All criteria are optional, and with no
        parameters the returned data set is unchanged. This can be used to make
        a copy of the data set.
        
        Parameters
        ----------
        labelkeep : list of strings, optional
            All subscans with labels in this list will be kept. The default is
            None, which means all labels are kept.
        flagkeep : list of strings, optional
            List of flags used to select time ranges in each subscan. The time
            samples for which all the flags in the list are true are kept.
            Individual flags can be negated by prepending a ~ (tilde) character.
            The default is None, which means all time samples are kept.
        freqkeep : sequence of bools or ints, optional
            Sequence of indicators of which frequency channels/bands to keep
            (either integer indices or booleans that are True for the values to
            be kept). The default is None, which keeps all channels/bands.
        copy : {False, True}, optional
            True if the new subscan is a copy, False if it is a view
        
        Returns
        -------
        dataset : :class:`DataSet` object
            Data set with selection of subscans with possibly smaller data arrays.
        
        Raises
        ------
        KeyError
            If flag in *flagkeep* is unknown
        
        """
        scanlist = []
        for s in self.scans:
            subscanlist = []
            for ss in s.subscans:
                # Convert flag selection to time sample selection
                if flagkeep is None:
                    timekeep = None
                else:
                    # By default keep all time samples
                    timekeep = np.tile(True, len(ss.timestamps))
                    for flag in flagkeep:
                        invert = False
                        # Flags prepended with ~ get inverted
                        if flag[0] == '~':
                            invert = True
                            flag = flag[1:]
                        # Ignore unknown flags
                        try:
                            flag_data = ss.flags[flag]
                        except KeyError:
                            raise KeyError("Unknown flag '%s'" % flag)
                        if invert:
                            timekeep &= ~flag_data
                        else:
                            timekeep &= flag_data
                if (labelkeep is None) or (ss.label in labelkeep):
                    subscanlist.append(ss.select(timekeep, freqkeep, copy))
            if subscanlist:
                scanlist.append(Scan(subscanlist))
        return DataSet(None, scanlist, self.noise_diode_data)
