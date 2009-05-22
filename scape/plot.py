"""Plotting routines."""

import time
import logging

import numpy as np
import matplotlib as mpl
import pylab as pl

import stats

logger = logging.getLogger("scape.plots")

def _shrink_axes(axes, shift=0.075):
    """Shrink axes on the left to leave more space for labels.
    
    Parameters
    ----------
    axes : :class:`matplotlib.axes.Axes` object
        Matplotlib Axes object
    shift : float
        Amount to advance left border of axes (axes width will decrease by same
        amount), in axes coords
    
    Returns
    -------
    axes : :class:`matplotlib.axes.Axes` object
        Adjusted axes object
    
    """
    pos = axes.get_position().bounds
    axes.set_position([pos[0] + shift, pos[1], pos[2] - shift, pos[3]])
    return axes

def waterfall(dataset, title='', channel_decimation=16, fig=None):
    """Waterfall plot of power data as a function of time and frequency.
    
    Parameters
    ----------
    dataset : :class:`scape.DataSet` object
        Data set to plot
    title : string, optional
        Title to add to figure
    channel_decimation : int, optional
        Number of channels skipped at a time (i.e. plot every
        channel_decimation'th channel)
    fig : :class:`matplotlib.figure.Figure` object
        Matplotlib Figure object to contain plots (default is current figure)
    
    Returns
    -------
    axes_list : list of :class:`matplotlib.axes.Axes` objects
        List of matplotlib Axes objects, one per plot
    
    """
    if fig is None:
        fig = pl.gcf()
    # Set up axes: one figure with custom subfigures for waterfall and spectrum plots, with shared x and y axes
    axes_list = []
    axes_list.append(_shrink_axes(fig.add_axes([0.125, 6 / 11., 0.6, 4 / 11.])))
    axes_list.append(_shrink_axes(fig.add_axes([0.125, 0.1, 0.6, 4 / 11.], 
                                 sharex=axes_list[0], sharey=axes_list[0])))
    axes_list.append(fig.add_axes([0.74, 6 / 11., 0.16, 4 / 11.], sharey=axes_list[0]))
    axes_list.append(fig.add_axes([0.74, 0.1, 0.16, 4 / 11.],
                                 sharex=axes_list[2], sharey=axes_list[0]))
    
    # Use relative time axis and obtain data limits (of smoothed data) per channel
    blocks = [s for s in dataset.iter_subscans()]
    if not blocks:
        logger.error('Data set is empty')
        return
    channel_freqs_GHz = blocks[0].freqs / 1e9
    channel_bandwidths_GHz = blocks[0].bandwidths / 1e9
    rfi_channels = blocks[0].rfi_channels
    num_channels = len(channel_freqs_GHz)
    data_min = {'XX': np.tile(np.inf, (len(blocks), num_channels)), 
                'YY': np.tile(np.inf, (len(blocks), num_channels))}
    data_max = {'XX': np.zeros((len(blocks), num_channels)),
                'YY': np.zeros((len(blocks), num_channels))}
    time_origin = np.double(np.inf)
    for n, block in enumerate(blocks):
        time_origin = min(time_origin, block.timestamps.min())
        for pol in ['XX', 'YY']:
            smoothed_power = stats.remove_spikes(block.coherency(pol))
            channel_min = smoothed_power.min(axis=0)
            data_min[pol][n] = np.where(channel_min < data_min[pol][n], channel_min, data_min[pol][n])
            channel_max = smoothed_power.max(axis=0)
            data_max[pol][n] = np.where(channel_max > data_max[pol][n], channel_max, data_max[pol][n])
    # Obtain minimum and maximum values in each channel (of smoothed data)
    for pol in ['XX', 'YY']:
        data_min[pol] = data_min[pol].min(axis=0)
        data_max[pol] = data_max[pol].max(axis=0)
    channel_list = np.arange(0, num_channels, channel_decimation, dtype='int')
    offsets = np.column_stack((np.zeros(len(channel_list), dtype='float'), channel_freqs_GHz[channel_list]))
    scale = 0.08 * num_channels
    
    # Plot of raw XX and YY power in all channels
    esn_labels, tLimits, pLimits = [], [], []
    for axis_ind, pol in enumerate(['XX', 'YY']):
        # Time-frequency waterfall plots
        axis = axes_list[axis_ind]
        all_blocks = []
        for block in blocks:
            # Grey out RFI-tagged channels using alpha transparency
            if block.label == 'scan':
                colors = [(0.0, 0.0, 1.0, 1.0 - 0.6 * (chan in block.rfi_channels)) for chan in channel_list]
            else:
                colors = [(0.0, 0.0, 0.0, 1.0 - 0.6 * (chan in block.rfi_channels)) for chan in channel_list]
            time_line = block.timestamps - time_origin
            # Normalise the data in each channel to lie between 0 and (channel bandwidth * scale)
            norm_power = scale * (block.bandwidths[np.newaxis, :] / 1e9) * \
                        (block.coherency(pol) - data_min[pol][np.newaxis, :]) / \
                        (data_max[pol][np.newaxis, :] - data_min[pol][np.newaxis, :])
            segments = [np.vstack((time_line, norm_power[:, chan])).transpose() for chan in channel_list]
            lines = mpl.collections.LineCollection(segments, colors=colors, offsets=offsets)
            lines.set_linewidth(0.5)
            axis.add_collection(lines)
#            esn_labels.append(axis.text(time_line.mean(), -0.08 * num_channels, label[3:label.find('_')], \
#                             ha='center', va='bottom', clip_on=True))
            tLimits += [time_line.min(), time_line.max()]
            all_blocks.append(block.coherency(pol))
        if channel_decimation <= 3:
            nth_str = ['', '2nd', '3rd'][channel_decimation - 1]
        else:
            nth_str = '%dth' % channel_decimation
        waterfall_title = 'Raw %s power in every %s channel' % (pol, nth_str)
        if pol == 'XX':
            if title:
                title_obj = axis.set_title(title + '\n' + waterfall_title + '\n')
            else:
                title_obj = axis.set_title(waterfall_title + '\n')
            extra_title = '\n\nGreyed-out channels are RFI-flagged'
            # This is more elaborate because the subplot axes are shared
            pl.setp(axis.get_xticklabels(), visible=False)
        else:
            title_obj = axis.set_title(waterfall_title + '\n')
            extra_title = '\n(blue = normal scans, black = cal/Tsys scans)'
            axis.set_xlabel('Time (s), since %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_origin)))
        # Shrink the font of the second line of the title to make it fit
        title_pos = title_obj.get_position()
        axis.text(title_pos[0], title_pos[1], extra_title, fontsize='smaller', transform=axis.transAxes, ha='center')
        axis.set_ylabel('Frequency (GHz)')
        # Power spectrum box plots, with bar plot behind it indicating min-to-max data range
        axis = axes_list[axis_ind + 2]
        all_blocks = np.concatenate(all_blocks, axis=0)
        rfi_channel_list = list(set(channel_list) & set(rfi_channels))
        non_rfi_channel_list = list(set(channel_list) - set(rfi_channel_list))
        # Do RFI and non-RFI channels separately, in order to grey out the RFI channels (boxplot constraint)
        for rfi_flag, channels in enumerate([non_rfi_channel_list, rfi_channel_list]):
            if len(channels) == 0:
                continue
            chan_data = all_blocks[:, channels]
            axis.bar(chan_data.min(axis=0), channel_decimation * channel_bandwidths_GHz[channels],
                     chan_data.max(axis=0) - chan_data.min(axis=0), channel_freqs_GHz[channels],
                     color='b', alpha=(0.5 - 0.2 * rfi_flag), linewidth=0, align='center', orientation='horizontal')
            handles = axis.boxplot(chan_data, vert=0, positions=channel_freqs_GHz[channels], sym='',
                                   widths=channel_decimation * channel_bandwidths_GHz[channels])
            pl.setp(handles['whiskers'], linestyle='-')
            if rfi_flag:
                pl.setp([h for h in mpl.cbook.flatten(handles.itervalues())], alpha=0.4)
            # Restore yticks corrupted by boxplot
            axis.yaxis.set_major_locator(mpl.ticker.AutoLocator())
        axis.set_xscale('log')
        # Add extra ticks on the right to indicate channel numbers
        # second_axis = axis.twinx()
        # second_axis.yaxis.tick_right()
        # second_axis.yaxis.set_label_position('right')
        # second_axis.set_ylabel('Channel number')
        # second_axis.set_yticks(channel_freqs_GHz[channel_list])
        # second_axis.set_yticklabels([str(chan) for chan in channel_list])
        pLimits += [all_blocks.min(), all_blocks.max()]
        axis.set_title('%s power spectrum' % pol)
        if pol == 'XX':
            # This is more elaborate because the subplot axes are shared
            pl.setp(axis.get_xticklabels(), visible=False)
            pl.setp(axis.get_yticklabels(), visible=False)
        else:
            pl.setp(axis.get_yticklabels(), visible=False)
            axis.set_xlabel('Raw power')
    
    # Fix limits globally
    tLimits = np.array(tLimits)
    yRange = channel_freqs_GHz.max() - channel_freqs_GHz.min()
    for axis in axes_list[:2]:
        axis.set_xlim(tLimits.min(), tLimits.max())
        axis.set_ylim(channel_freqs_GHz.min() - 0.1 * yRange, channel_freqs_GHz.max() + 0.1 * yRange)
    pLimits = np.array(pLimits)
    for axis in axes_list[2:]:
        axis.set_xlim(pLimits.min(), pLimits.max())
        
    return axes_list
