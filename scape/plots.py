"""Plotting routines."""

import time
import logging

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from .stats import remove_spikes

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

def ordinal_suffix(n):
    """Returns the ordinal suffix of integer *n* as a string."""
    if n % 100 in [11, 12, 13]:
        return 'th'
    else:
        return {1 : 'st', 2 : 'nd', 3 : 'rd'}.get(n % 10, 'th')

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  waterfall
#--------------------------------------------------------------------------------------------------

def waterfall(dataset, title='', channel_skip=None, fig=None):
    """Waterfall plot of power data as a function of time and frequency.
    
    Parameters
    ----------
    dataset : :class:`scape.DataSet` object
        Data set to plot
    title : string, optional
        Title to add to figure
    channel_skip : int, optional
        Number of channels skipped at a time (i.e. plot every channel_skip'th
        channel). The default results in about 32 channels.
    fig : :class:`matplotlib.figure.Figure` object, optional
        Matplotlib Figure object to contain plots (default is current figure)
    
    Returns
    -------
    axes_list : list of :class:`matplotlib.axes.Axes` objects
        List of matplotlib Axes objects, one per plot
    
    """
    if not channel_skip:
        channel_skip = max(len(dataset.freqs) // 32, 1)
    if fig is None:
        fig = plt.gcf()
    # Set up axes: one figure with custom subfigures for waterfall and spectrum plots, with shared x and y axes
    axes_list = []
    axes_list.append(fig.add_axes([0.125, 6 / 11., 0.6, 4 / 11.]))
    axes_list.append(fig.add_axes([0.125, 0.1, 0.6, 4 / 11.], 
                                               sharex=axes_list[0], sharey=axes_list[0]))
    axes_list.append(fig.add_axes([0.74, 6 / 11., 0.16, 4 / 11.], sharey=axes_list[0]))
    axes_list.append(fig.add_axes([0.74, 0.1, 0.16, 4 / 11.],
                                  sharex=axes_list[2], sharey=axes_list[0]))
    
    # Use relative time axis and obtain data limits (of smoothed data) per channel
    subscans = dataset.subscans
    if not subscans:
        logger.error('Data set is empty')
        return
    channel_freqs_GHz = dataset.freqs / 1e9
    channel_bandwidths_GHz = dataset.bandwidths / 1e9
    rfi_channels = dataset.rfi_channels
    num_channels = len(channel_freqs_GHz)
    data_min = {'XX': np.tile(np.inf, (len(subscans), num_channels)), 
                'YY': np.tile(np.inf, (len(subscans), num_channels))}
    data_max = {'XX': np.zeros((len(subscans), num_channels)),
                'YY': np.zeros((len(subscans), num_channels))}
    time_origin = np.double(np.inf)
    for n, ss in enumerate(subscans):
        time_origin = min(time_origin, ss.timestamps.min())
        for pol in ['XX', 'YY']:
            smoothed_power = remove_spikes(ss.coherency(pol))
            channel_min = smoothed_power.min(axis=0)
            data_min[pol][n] = np.where(channel_min < data_min[pol][n], channel_min, data_min[pol][n])
            channel_max = smoothed_power.max(axis=0)
            data_max[pol][n] = np.where(channel_max > data_max[pol][n], channel_max, data_max[pol][n])
    # Obtain minimum and maximum values in each channel (of smoothed data)
    for pol in ['XX', 'YY']:
        data_min[pol] = data_min[pol].min(axis=0)
        data_max[pol] = data_max[pol].max(axis=0)
    channel_list = np.arange(0, num_channels, channel_skip, dtype='int')
    offsets = np.column_stack((np.zeros(len(channel_list), dtype='float'), channel_freqs_GHz[channel_list]))
    scale = 0.08 * num_channels
    
    # Plot of raw XX and YY power in all channels
    t_limits, p_limits = [], []
    for axis_ind, pol in enumerate(['XX', 'YY']):
        # Time-frequency waterfall plots
        axis = axes_list[axis_ind]
        all_subscans = []
        for scan_ind, s in enumerate(dataset.scans):
            for ss in s.subscans:
                # Grey out RFI-tagged channels using alpha transparency
                if ss.label == 'scan':
                    colors = [(0.0, 0.0, 1.0, 1.0 - 0.6 * (chan in dataset.rfi_channels)) for chan in channel_list]
                else:
                    colors = [(0.0, 0.0, 0.0, 1.0 - 0.6 * (chan in dataset.rfi_channels)) for chan in channel_list]
                time_line = ss.timestamps - time_origin
                # Normalise the data in each channel to lie between 0 and (channel bandwidth * scale)
                norm_power = scale * (dataset.bandwidths[np.newaxis, :] / 1e9) * \
                            (ss.coherency(pol) - data_min[pol][np.newaxis, :]) / \
                            (data_max[pol][np.newaxis, :] - data_min[pol][np.newaxis, :])
                segments = [np.vstack((time_line, norm_power[:, chan])).transpose() for chan in channel_list]
                if len(segments) > 1:
                    lines = mpl.collections.LineCollection(segments, colors=colors, offsets=offsets)
                    lines.set_linewidth(0.5)
                    axis.add_collection(lines)
                else:
                    axis.plot(segments[0][:, 0] + offsets.squeeze()[0], 
                              segments[0][:, 1] + offsets.squeeze()[1], color=colors[0], lw=0.5)
                t_limits += [time_line.min(), time_line.max()]
                all_subscans.append(ss.coherency(pol))
            # Add scan target name and partition lines between scans
            if s.subscans:
                start_time_ind = len(t_limits) - 2 * len(s.subscans)
                if scan_ind >= 1:
                    border_time = (t_limits[start_time_ind - 1] + t_limits[start_time_ind]) / 2.0
                    axis.plot([border_time, border_time], [0.0, 10.0 * channel_freqs_GHz.max()], '--k')
                axis.text((t_limits[start_time_ind] + t_limits[-1]) / 2.0,
                          offsets[0, 1] - scale * dataset.bandwidths[0] / 1e9, s.target.name,
                          ha='center', va='bottom', clip_on=True)
        # Set up title and axis labels
        nth_str = ''
        if channel_skip > 1:
            nth_str = '%d%s' % (channel_skip, ordinal_suffix(channel_skip))
        if dataset.data_unit == 'Jy':
            waterfall_title = '%s flux density in every %s channel' % (pol, nth_str)
        if dataset.data_unit == 'K':
            waterfall_title = '%s temperature in every %s channel' % (pol, nth_str)
        else:
            waterfall_title = 'Raw %s power in every %s channel' % (pol, nth_str)
        if pol == 'XX':
            if title:
                title_obj = axis.set_title(title + '\n' + waterfall_title + '\n')
            else:
                title_obj = axis.set_title(waterfall_title + '\n')
            extra_title = '\n\nGreyed-out channels are RFI-flagged'
            # This is more elaborate because the subplot axes are shared
            plt.setp(axis.get_xticklabels(), visible=False)
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
        all_subscans = np.concatenate(all_subscans, axis=0)
        rfi_channel_list = list(set(channel_list) & set(rfi_channels))
        non_rfi_channel_list = list(set(channel_list) - set(rfi_channel_list))
        # Do RFI and non-RFI channels separately, in order to grey out the RFI channels (boxplot constraint)
        for rfi_flag, channels in enumerate([non_rfi_channel_list, rfi_channel_list]):
            if len(channels) == 0:
                continue
            chan_data = all_subscans[:, channels]
            axis.bar(chan_data.min(axis=0), channel_skip * channel_bandwidths_GHz[channels],
                     chan_data.max(axis=0) - chan_data.min(axis=0), channel_freqs_GHz[channels],
                     color='b', alpha=(0.5 - 0.2 * rfi_flag), linewidth=0, align='center', orientation='horizontal')
            handles = axis.boxplot(chan_data, vert=0, positions=channel_freqs_GHz[channels], sym='',
                                   widths=channel_skip * channel_bandwidths_GHz[channels])
            plt.setp(handles['whiskers'], linestyle='-')
            if rfi_flag:
                plt.setp([h for h in mpl.cbook.flatten(handles.itervalues())], alpha=0.4)
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
        p_limits += [all_subscans.min(), all_subscans.max()]
        axis.set_title('%s power spectrum' % pol)
        if pol == 'XX':
            # This is more elaborate because the subplot axes are shared
            plt.setp(axis.get_xticklabels(), visible=False)
            plt.setp(axis.get_yticklabels(), visible=False)
        else:
            plt.setp(axis.get_yticklabels(), visible=False)
            if dataset.data_unit == 'Jy':
                axis.set_xlabel('Flux density (Jy)')
            elif dataset.data_unit == 'K':
                axis.set_xlabel('Temperature (K)')
            else:
                axis.set_xlabel('Raw power')
    # Fix limits globally
    t_limits = np.array(t_limits)
    y_range = channel_freqs_GHz.max() - channel_freqs_GHz.min()
    if y_range < channel_bandwidths_GHz[0]:
        y_range = 10.0 * channel_bandwidths_GHz[0]
    for axis in axes_list[:2]:
        axis.set_xlim(t_limits.min(), t_limits.max())
        axis.set_ylim(channel_freqs_GHz.min() - 0.1 * y_range, channel_freqs_GHz.max() + 0.1 * y_range)
    p_limits = np.array(p_limits)
    for axis in axes_list[2:]:
        axis.set_xlim(p_limits.min(), p_limits.max())
        
    return axes_list


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  plot_marker_3d
#---------------------------------------------------------------------------------------------------------

## Pseudo-3D plot that plots markers at given x-y positions, with marker size determined by z values.
# This is an alternative to pcolor, with the advantage that the x and y values do not need to be on
# a regular grid, and that it is easier to compare the relative size of z values. The disadvantage is
# that the markers may have excessive overlap or very small sizes, which obscures the plot. This can
# be controlled by the maxSize and minSize parameters.
#
# @param axis       Matplotlib axes object associated with a matplotlib Figure
# @param x          Array of x coordinates of markers
# @param y          Array of y coordinates of markers
# @param z          Array of z heights, transformed to marker size
# @param maxSize    The radius of the biggest marker, relative to the average spacing between markers
# @param minSize    The radius of the smallest marker, relative to the average spacing between markers
# @param markerType Type of marker ('circle' [default], or 'asterisk')
# @param numLines   Number of lines in asterisk [8]
# @param kwargs     Dictionary containing extra keyword arguments, which are passed on to plot() or add_patch()
# @return Handle of asterisk line object, or list of circle patches
# pylint: disable-msg=W0142
def plot_marker_3d(axis, x, y, z, maxSize=0.75, minSize=0.05, markerType='circle', numLines=8, **kwargs):
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    assert maxSize >= minSize, "In plot_marker_3d, minSize should not be bigger than maxSize."
    
    # Normalise z to lie between 0 and 1
    zMinInd, zMaxInd = z.argmin(), z.argmax()
    z = (z - z[zMinInd]) / (z[zMaxInd] - z[zMinInd])
    # Threshold z, so that the minimum size will have the desired ratio to the maximum size
    z[z < minSize/maxSize] = minSize/maxSize
    # Determine median spacing between vectors
    minDist = np.zeros(len(x))
    for ind in xrange(len(x)):
        distSq = (x - x[ind]) ** 2 + (y - y[ind]) ** 2
        minDist[ind] = np.sqrt(distSq[distSq > 0].min())
    # Scale z so that maximum value is desired factor of median spacing
    z *= maxSize * np.median(minDist)
    
    if markerType == 'asterisk':
        # Use random initial angles so that asterisks don't overlap in regular pattern, which obscures their size
        ang = np.pi*np.random.random_sample(z.shape)
        xAsterisks = []
        yAsterisks = []
        # pylint: disable-msg=W0612
        for side in range(numLines):
            xDash = np.vstack((x - z*np.cos(ang), x + z*np.cos(ang), np.tile(np.nan, x.shape))).transpose()
            yDash = np.vstack((y - z*np.sin(ang), y + z*np.sin(ang), np.tile(np.nan, y.shape))).transpose()
            xAsterisks += xDash.ravel().tolist()
            yAsterisks += yDash.ravel().tolist()
            ang += np.pi / numLines
        # All asterisks form part of one big line...
        return axis.plot(xAsterisks, yAsterisks, **kwargs)
        
    elif markerType == 'circle':
        # Add a circle patch for each marker
        for ind in xrange(len(x)):
            axis.add_patch(patches.Circle((x[ind], y[ind]), z[ind], **kwargs))
        return axis.patches
        
    else:
        raise ValueError, "Unknown marker type '" + markerType + "'"


## Plot beam pattern fitted to multiple scans through a single point source, in target space.
# This plots contour ellipses of a Gaussian beam function fitted to multiple scans through a point source,
# as well as the power values of the scans themselves as a pseudo-3D plot. It highlights the success of the
# beam fitting procedure.
# @param figColorList   List of matplotlib Figure objects to contain plots (one per frequency band)
# @param calibScanList  List of SingleDishData objects of calibrated main scans (one per scan)
# @param beamFuncList   List of Gaussian beam functions and valid flags (one per band)
# @param expName        Title of experiment
# @return axesColorList List of matplotlib Axes objects, one per plot
# pylint: disable-msg=R0912,R0913,R0914,R0915
def plot_beam_pattern_target(figColorList, calibScanList, beamFuncList, expName):
    # Set up axes
    axesColorList = []
    # Use the frequency bands of the first scan as reference for the rest
    plotFreqs = calibScanList[0].freqs_Hz / 1e9   # Hz to GHz
    numBands = len(plotFreqs)
    # One figure per frequency band
    for band in range(numBands):
        axesColorList.append(figColorList[band].add_subplot(1, 1, 1))
    
    # Extract total power and target coordinates (in degrees) of all scans
    totalPower = []
    targetCoords = []
    for scan in calibScanList:
        totalPower.append(scan.stokes('I'))
        targetCoords.append(rad_to_deg(scan.targetCoords[:, :2]))
    # Also extract beam centres, in order to unwrap them with the rest of angle data
    for band in range(numBands):
        targetCoords.append(rad_to_deg(np.atleast_2d(beamFuncList[band][0].mean)))
    totalPower = np.concatenate(totalPower)
    targetCoords = np.concatenate(targetCoords)
    # Unwrap all angle coordinates in the plot concurrently, to prevent bad plots
    # (this is highly unlikely, as the target coordinate space is typically centred around the origin)
    targetCoords = np.array([misc.unwrap_angles(ang) for ang in targetCoords.transpose()]).transpose()
    beamCentres = targetCoords[-numBands:]
    targetCoords = targetCoords[:-numBands]
    
    # Iterate through figures (one per band)
    for band in range(numBands):
        axis = axesColorList[band]
        power = totalPower[:, band]
        # Show the locations of the scan samples themselves, with marker sizes indicating power values
        vis.plot_marker_3d(axis, targetCoords[:, 0], targetCoords[:, 1], power)
        # Plot the fitted Gaussian beam function as contours
        if beamFuncList[band][1]:
            ellType, centerType = 'r-', 'r+'
        else:
            ellType, centerType = 'y-', 'y+'
        ellipses = misc.gaussian_ellipses(deg_to_rad(beamCentres[band]), np.diag(beamFuncList[band][0].var), \
                                          contour=[0.5, 0.1])
        for ellipse in ellipses:
            axis.plot(rad_to_deg(ellipse[:, 0]), rad_to_deg(ellipse[:, 1]), ellType, lw=2)
        axis.plot([beamCentres[band][0]], [beamCentres[band][1]], centerType, ms=12, aa=False, mew=2)
    
    # Axis settings and labels
    for band in range(numBands):
        axis = axesColorList[band]
        xRange = [targetCoords[:, 0].min(), targetCoords[:, 0].max()]
        yRange = [targetCoords[:, 1].min(), targetCoords[:, 1].max()]
        if not np.any(np.isnan(xRange + yRange)):
            axis.set_xlim(xRange)
            axis.set_ylim(yRange)
        axis.set_aspect('equal')
        axis.set_xlabel('Target coord 1 (deg)')
        axis.set_ylabel('Target coord 2 (deg)')
        axis.set_title(expName + ' : Beam fitted in band %d : %3.3f GHz' % (band, plotFreqs[band]))
    
    return axesColorList
