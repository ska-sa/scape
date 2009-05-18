"""Read XDM FITS files."""

import logging
import cPickle

import pyfits
import numpy as np

# Needed for pickled target and mount objects
import acsm

#import datasetbase
#import scan
import subscan
import coord

logger = logging.getLogger("scape.subscan")


def acsm_target_name(target):
    ref_target = target.get_reference_target()
    if ref_target._name:
        return ref_target._name
    else:
        return ref_target.get_description()

def load_subscan(filename):
    """Load data set from XDM FITS file series."""
    hdu = pyfits.open(filename)
    try:
        hdu.verify(option='exception')
    except pyfits.VerifyError:
        hdu.close()
        logger.error("File '%s' does not comply with FITS standard" % filename)
    header = hdu['PRIMARY'].header
    
    is_stokes = (header['Stokes0'] == 'I')
    start_time = np.double(header['tEpoch'])
    start_time_offset = np.double(header['tStart'])
    dump_rate = np.double(header['DumpRate'])
    sample_period = 1.0 / dump_rate
    num_samples = int(header['Samples'])
    channel_width = np.double(header['ChannelB'])
    exp_seq_num = int(header['ExpSeqN'])
    
    if is_stokes:
        data = np.rec.fromarrays([hdu['MSDATA'].data.field(s) for s in subscan.stokes_order],
                                 names=subscan.stokes_order)
    else:
        data = np.rec.fromarrays([hdu['MSDATA'].data.field(s) for s in subscan.coherency_order],
                                 names=subscan.coherency_order)
    data_unit = 'raw'
    timestamps = np.arange(num_samples) * sample_period + start_time + start_time_offset
    pointing = np.rec.fromarrays([coord.radians(hdu['MSDATA'].data.field(s)) 
                                  for s in ['AzAng', 'ElAng', 'RotAng']],
                                 names=['az', 'el', 'rot'])
    flags = np.rec.fromarrays([np.array(hdu['MSDATA'].data.field(s), 'bool')
                               for s in ['Valid_F', 'ND_ON_F', 'RX_ON_F']],
                              names=['valid', 'nd_on', 'rx_on'])
    freqs = hdu['CHANNELS'].data.field('Freq')
    rfi_channels = [x[0] for x in hdu['RFI'].data.field('Channels')]
    # The FITS file doesn't like empty lists, so an empty list is represented by [-1] (an invalid index)
    # Therefore, remove any invalid indices, as a further safeguard
    rfi_channels = [x for x in rfi_channels if (x >= 0) and (x < len(freqs))]
    channels_per_band = [x.tolist() for x in hdu['BANDS'].data.field('Channels')]

    target = acsm_target_name(cPickle.loads(hdu['OBJECTS'].data.field('Target')[0]))
    mount = cPickle.loads(hdu['OBJECTS'].data.field('Mount')[0])
    antenna = mount.get_decorated_coordinate_system().get_attribute('position').get_description().split()[0]
    
    data_header = hdu['MSDATA'].header
    label = str(data_header['ID'+str(data_header['DATAID'])])
    
    return exp_seq_num, subscan.SubScan(data, data_unit, timestamps, pointing, flags,
                                        freqs, channel_width, rfi_channels, channels_per_band, dump_rate,
                                        target, antenna, label)
