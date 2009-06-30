"""Plotting routines."""

import time
import logging

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from katpoint import rad2deg, plane_to_sphere
from .stats import remove_spikes, minimise_angle_wrap
from .beam_baseline import fwhm_to_sigma, interpolate_measured_beam

logger = logging.getLogger("scape.plots")

def ordinal_suffix(n):
    """Returns the ordinal suffix of integer *n* as a string."""
    if n % 100 in [11, 12, 13]:
        return 'th'
    else:
        return {1 : 'st', 2 : 'nd', 3 : 'rd'}.get(n % 10, 'th')

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  plot_waterfall
#--------------------------------------------------------------------------------------------------

def plot_waterfall(dataset, title='', channel_skip=None, fig=None):
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
    scans = dataset.scans
    if not scans:
        logger.error('Data set is empty')
        return
    channel_freqs_GHz = dataset.freqs / 1e9
    channel_bandwidths_GHz = dataset.bandwidths / 1e9
    rfi_channels = dataset.rfi_channels
    num_channels = len(channel_freqs_GHz)
    data_min = {'XX': np.tile(np.inf, (len(scans), num_channels)), 
                'YY': np.tile(np.inf, (len(scans), num_channels))}
    data_max = {'XX': np.zeros((len(scans), num_channels)),
                'YY': np.zeros((len(scans), num_channels))}
    time_origin = np.double(np.inf)
    for n, scan in enumerate(scans):
        time_origin = min(time_origin, scan.timestamps.min())
        for pol in ['XX', 'YY']:
            smoothed_power = remove_spikes(scan.coherency(pol))
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
    for ax_ind, pol in enumerate(['XX', 'YY']):
        # Time-frequency waterfall plots
        ax = axes_list[ax_ind]
        all_scans = []
        for compscan_ind, compscan in enumerate(dataset.compscans):
            for scan in compscan.scans:
                # Grey out RFI-tagged channels using alpha transparency
                if scan.label == 'scan':
                    colors = [(0.0, 0.0, 1.0, 1.0 - 0.6 * (chan in dataset.rfi_channels)) for chan in channel_list]
                else:
                    colors = [(0.0, 0.0, 0.0, 1.0 - 0.6 * (chan in dataset.rfi_channels)) for chan in channel_list]
                time_line = scan.timestamps - time_origin
                # Normalise the data in each channel to lie between 0 and (channel bandwidth * scale)
                norm_power = scale * (dataset.bandwidths[np.newaxis, :] / 1e9) * \
                            (scan.coherency(pol) - data_min[pol][np.newaxis, :]) / \
                            (data_max[pol][np.newaxis, :] - data_min[pol][np.newaxis, :])
                segments = [np.vstack((time_line, norm_power[:, chan])).transpose() for chan in channel_list]
                if len(segments) > 1:
                    lines = mpl.collections.LineCollection(segments, colors=colors, offsets=offsets)
                    lines.set_linewidth(0.5)
                    ax.add_collection(lines)
                else:
                    ax.plot(segments[0][:, 0] + offsets.squeeze()[0], 
                            segments[0][:, 1] + offsets.squeeze()[1], color=colors[0], lw=0.5)
                t_limits += [time_line.min(), time_line.max()]
                all_scans.append(scan.coherency(pol))
            # Add compound scan target name and partition lines between compound scans
            if compscan.scans:
                start_time_ind = len(t_limits) - 2 * len(compscan.scans)
                if compscan_ind >= 1:
                    border_time = (t_limits[start_time_ind - 1] + t_limits[start_time_ind]) / 2.0
                    ax.plot([border_time, border_time], [0.0, 10.0 * channel_freqs_GHz.max()], '--k')
                ax.text((t_limits[start_time_ind] + t_limits[-1]) / 2.0,
                        offsets[0, 1] - scale * dataset.bandwidths[0] / 1e9, compscan.target.name,
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
                title_obj = ax.set_title(title + '\n' + waterfall_title + '\n')
            else:
                title_obj = ax.set_title(waterfall_title + '\n')
            extra_title = '\n\nGreyed-out channels are RFI-flagged'
            # This is more elaborate because the subplot axes are shared
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            title_obj = ax.set_title(waterfall_title + '\n')
            extra_title = '\n(blue = normal scans, black = cal/Tsys scans)'
            ax.set_xlabel('Time (s), since %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_origin)))
        # Shrink the font of the second line of the title to make it fit
        title_pos = title_obj.get_position()
        ax.text(title_pos[0], title_pos[1], extra_title, fontsize='smaller', transform=ax.transAxes, ha='center')
        ax.set_ylabel('Frequency (GHz)')
        # Power spectrum box plots, with bar plot behind it indicating min-to-max data range
        ax = axes_list[ax_ind + 2]
        all_scans = np.concatenate(all_scans, axis=0)
        rfi_channel_list = list(set(channel_list) & set(rfi_channels))
        non_rfi_channel_list = list(set(channel_list) - set(rfi_channel_list))
        # Do RFI and non-RFI channels separately, in order to grey out the RFI channels (boxplot constraint)
        for rfi_flag, channels in enumerate([non_rfi_channel_list, rfi_channel_list]):
            if len(channels) == 0:
                continue
            chan_data = all_scans[:, channels]
            ax.bar(chan_data.min(axis=0), channel_skip * channel_bandwidths_GHz[channels],
                   chan_data.max(axis=0) - chan_data.min(axis=0), channel_freqs_GHz[channels],
                   color='b', alpha=(0.5 - 0.2 * rfi_flag), linewidth=0, align='center', orientation='horizontal')
            handles = ax.boxplot(chan_data, vert=0, positions=channel_freqs_GHz[channels], sym='',
                                 widths=channel_skip * channel_bandwidths_GHz[channels])
            plt.setp(handles['whiskers'], linestyle='-')
            if rfi_flag:
                plt.setp([h for h in mpl.cbook.flatten(handles.itervalues())], alpha=0.4)
            # Restore yticks corrupted by boxplot
            ax.yaxis.set_major_locator(mpl.ticker.AutoLocator())
        ax.set_xscale('log')
        # Add extra ticks on the right to indicate channel numbers
        # second_axis = ax.twinx()
        # second_axis.yaxis.tick_right()
        # second_axis.yaxis.set_label_position('right')
        # second_axis.set_ylabel('Channel number')
        # second_axis.set_yticks(channel_freqs_GHz[channel_list])
        # second_axis.set_yticklabels([str(chan) for chan in channel_list])
        p_limits += [all_scans.min(), all_scans.max()]
        ax.set_title('%s power spectrum' % pol)
        if pol == 'XX':
            # This is more elaborate because the subplot axes are shared
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
            if dataset.data_unit == 'Jy':
                ax.set_xlabel('Flux density (Jy)')
            elif dataset.data_unit == 'K':
                ax.set_xlabel('Temperature (K)')
            else:
                ax.set_xlabel('Raw power')
    # Fix limits globally
    t_limits = np.array(t_limits)
    y_range = channel_freqs_GHz.max() - channel_freqs_GHz.min()
    if y_range < channel_bandwidths_GHz[0]:
        y_range = 10.0 * channel_bandwidths_GHz[0]
    for ax in axes_list[:2]:
        ax.set_xlim(t_limits.min(), t_limits.max())
        ax.set_ylim(channel_freqs_GHz.min() - 0.1 * y_range, channel_freqs_GHz.max() + 0.1 * y_range)
    p_limits = np.array(p_limits)
    for ax in axes_list[2:]:
        ax.set_xlim(p_limits.min(), p_limits.max())
        
    return axes_list

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  plot_compacted_segments
#--------------------------------------------------------------------------------------------------

def plot_compacted_segments(segments, labels=None, ax=None, **kwargs):
    """Plot sequence of line segments in compacted form.
    
    This plots a sequence of line segments (of possibly varying length) on a
    single set of axes, with no gaps between the segments along the *x* axis.
    The tick labels on the *x* axis are modified to reflect the original (padded)
    values, and the breaks between segments are indicated by vertical lines.
    
    Parameters
    ----------
    segments : sequence of array-like, shape (N_k, 2)
        Sequence of line segments (*line0*, *line1*, ..., *linek*, ..., *lineK*),
        where the k'th line is given by::
        
            linek = (x0, y0), (x1, y1), ... (x_{N_k - 1}, y_{N_k - 1})
        
        or the equivalent numpy array with two columns (for *x* and *y* values, 
        respectively). Each line segment can be a different length. This is
        identical to the *segments* parameter of
        :class:`matplotlib.collections.LineCollection`.
    labels : sequence of strings, optional
        Corresponding sequence of text labels to add below each segment
    ax : :class:`matplotlib.axes.Axes` object, optional
        Matplotlib axes object to receive plot (default is current axes)
    kwargs : dict, optional
        Extra keyword arguments are passed on to line collection constructor
    
    Returns
    -------
    segment_lines : :class:`matplotlib.collections.LineCollection` object
        Collection of segment lines
    border_lines : :class:`matplotlib.collections.LineCollection` object
        Collection of vertical lines separating the segments
    text_labels : list of :class:`matplotlib.text.Text` objects
        List of added text labels
    
    """
    if ax is None:
        ax = plt.gca()
    if labels is None:
        labels = []
    start = np.array([np.asarray(segm)[:, 0].min() for segm in segments])
    end = np.array([np.asarray(segm)[:, 0].max() for segm in segments])
    compacted_start = [0.0] + np.cumsum(end - start).tolist()
    compacted_segments = [np.column_stack((np.asarray(segm)[:, 0] - start[n] + compacted_start[n],
                                           np.asarray(segm)[:, 1])) for n, segm in enumerate(segments)]
    # Plot the segment lines as a collection
    segment_lines = mpl.collections.LineCollection(compacted_segments, **kwargs)
    ax.add_collection(segment_lines)
    # These border lines have x coordinates fixed to the data and y coordinates fixed to the axes
    transFixedY = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    border_lines = mpl.collections.LineCollection([[(s, 0), (s, 1)] for s in compacted_start[1:-1]], 
                                                  colors='k', linewidths=0.5, linestyles='dotted',
                                                  transform=transFixedY)
    ax.add_collection(border_lines)
    ax.set_xlim(0.0, compacted_start[-1])
    text_labels = []
    for n, label in enumerate(labels):
        text_labels.append(ax.text(np.mean(compacted_start[n:n+2]), 0.02, label, transform=transFixedY,
                                   ha='center', va='bottom', clip_on=True))
    # Redefine x-axis label formatter to display the correct time for each segment
    class SegmentedScalarFormatter(mpl.ticker.ScalarFormatter):
        """Expand x axis value to correct segment before labelling."""
        def __init__(self, useOffset=True, useMathText=False):
            mpl.ticker.ScalarFormatter.__init__(self, useOffset, useMathText)
        def __call__(self, x, pos=None):
            if x > compacted_start[0]:
                segment = (compacted_start[:-1] < x).nonzero()[0][-1]
                x = x - compacted_start[segment] + start[segment]
            return mpl.ticker.ScalarFormatter.__call__(self, x, pos)
    ax.xaxis.set_major_formatter(SegmentedScalarFormatter())
    return segment_lines, border_lines, text_labels

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  plot_compound_scan_in_time
#--------------------------------------------------------------------------------------------------

def plot_compound_scan_in_time(compscan, stokes='I', add_scan_ids=True, band=0, ax=None):
    """Plot total power scans of compound scan with superimposed beam/baseline fit.
    
    This plots time series plots of the total power in the scans comprising a
    compound scan, with the beam and baseline fits superimposed. It highlights
    the success of the beam and baseline fitting procedure.
    
    Parameters
    ----------
    compscan : :class:`compoundscan.CompoundScan` object
        Compound scan object to plot
    stokes : {'I', 'Q', 'U', 'V'}, optional
        The Stokes parameter to display
    add_scan_ids : {True, False}, optional
        True if scan index numbers are to be added to plot
    band : int, optional
        Frequency band to plot
    ax : :class:`matplotlib.axes.Axes` object, optional
        Matplotlib axes object to receive plot (default is current axes)
    
    Returns
    -------
    axes_list : list of :class:`matplotlib.axes.Axes` objects
        List of matplotlib Axes objects, one per plot
    
    """
    if ax is None:
        ax = plt.gca()
    time_origin = np.array([scan.timestamps.min() for scan in compscan.scans]).min()
    power_limits, data_segments = [], []
    baseline_segments, beam_segments, inner_beam_segments = [], [], []
    
    # Construct segments to be plotted
    for scan in compscan.scans:
        timeline = scan.timestamps - time_origin
        measured_power = scan.stokes(stokes)[:, band]
        smooth_power = remove_spikes(measured_power)
        power_limits.extend([smooth_power.min(), smooth_power.max()])
        data_segments.append(np.column_stack((timeline, measured_power)))
        if scan.baseline:
            baseline_power = scan.baseline(scan.timestamps)
            baseline_segments.append(np.column_stack((timeline, baseline_power)))
            if compscan.beam:
                beam_power = compscan.beam(scan.target_coords.transpose()) + baseline_power
                radius = np.sqrt(((scan.target_coords - compscan.beam.center[:, np.newaxis]) ** 2).sum(axis=0))
                inner = radius < 0.6 * np.mean(compscan.beam.width)
                inner_beam_power = beam_power.copy()
                inner_beam_power[~inner] = np.nan
                beam_segments.append(np.column_stack((timeline, beam_power)))
                inner_beam_segments.append(np.column_stack((timeline, inner_beam_power)))
            else:
                beam_segments.append(np.column_stack((timeline, np.tile(np.nan, len(timeline)))))
                inner_beam_segments.append(np.column_stack((timeline, np.tile(np.nan, len(timeline)))))
        else:
            baseline_segments.append(np.column_stack((timeline, np.tile(np.nan, len(timeline)))))
            beam_segments.append(np.column_stack((timeline, np.tile(np.nan, len(timeline)))))
            inner_beam_segments.append(np.column_stack((timeline, np.tile(np.nan, len(timeline)))))
    # Plot segments from back to front
    plot_compacted_segments(baseline_segments, ax=ax, color='r', lw=2)
    plot_compacted_segments(beam_segments, ax=ax, color='r', lw=1, linestyles='dashed')
    plot_compacted_segments(inner_beam_segments, ax=ax, color='r', lw=2)
    labels = [str(n) for n in xrange(len(compscan.scans))] if add_scan_ids else []
    plot_compacted_segments(data_segments, labels, ax=ax, color='b')
    # Format axes
    power_range = max(power_limits) - min(power_limits)
    if power_range == 0.0:
        power_range = 1.0
    ax.set_ylim(min(power_limits) - 0.05 * power_range, max(power_limits) + 0.05 * power_range)
    ax.set_xlabel('Time (s), since %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_origin)))
    ax.set_ylabel('Stokes %s' % stokes)
    return ax

#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  plot_marker_3d
#---------------------------------------------------------------------------------------------------------

def plot_marker_3d(x, y, z, max_size=0.75, min_size=0.05, marker_type='scatter', num_lines=8, ax=None, **kwargs):
    """Pseudo-3D scatter plot using marker size to indicate height.
    
    This plots markers at given ``(x, y)`` positions, with marker size determined
    by *z* values. This is an alternative to :func:`matplotlib.pyplot.pcolor`,
    with the advantage that the *x* and *y* values do not need to be on a regular
    grid, and that it is easier to compare the relative size of *z* values. The
    disadvantage is that the markers may have excessive overlap or very small
    sizes, which obscures the plot. This can be controlled by the max_size and
    min_size parameters.
    
    Parameters
    ----------
    x : sequence
        Sequence of *x* coordinates of markers
    y : sequence
        Sequence of *y* coordinates of markers
    z : sequence
        Sequence of *z* heights, transformed to marker size
    max_size : float, optional
        Radius of biggest marker, relative to average spacing between markers
    min_size : float, optional
        Radius of smallest marker, relative to average spacing between markers
    marker_type : {'scatter', 'circle', 'asterisk'}, optional
        Type of marker
    num_lines : int, optional
        Number of lines in asterisk
    ax : :class:`matplotlib.axes.Axes` object, optional
        Matplotlib axes object to receive plot (default is current axes)
    kwargs : dict, optional
        Extra keyword arguments are passed on to underlying plot function
    
    Returns
    -------
    handle : handle or list
        Handle of asterisk line, list of circle patches, or scatter collection
    
    Raises
    ------
    ValueError
        If marker type is unknown
    
    """
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    assert max_size >= min_size, "In plot_marker_3d, min_size should not be bigger than max_size."
    if ax is None:
        ax = plt.gca()
    
    # Normalise z to lie between 0 and 1
    z = (z - z.min()) / (z.max() - z.min())
    # Threshold z, so that the minimum size will have the desired ratio to the maximum size
    z[z < min_size/max_size] = min_size/max_size
    # Determine median spacing between vectors
    min_dist = np.zeros(len(x))
    for ind in xrange(len(x)):
        dist_sq = (x - x[ind]) ** 2 + (y - y[ind]) ** 2
        min_dist[ind] = np.sqrt(dist_sq[dist_sq > 0].min())
    # Scale z so that maximum value is desired factor of median spacing
    z *= max_size * np.median(min_dist)
    
    if marker_type == 'asterisk':
        # Use random initial angles so that asterisks don't overlap in regular pattern, which obscures their size
        ang = np.pi * np.random.random_sample(z.shape)
        x_asterisks, y_asterisks = [], []
        # pylint: disable-msg=W0612
        for side in range(num_lines):
            x_dash = np.vstack((x - z * np.cos(ang), x + z * np.cos(ang), np.tile(np.nan, x.shape))).transpose()
            y_dash = np.vstack((y - z * np.sin(ang), y + z * np.sin(ang), np.tile(np.nan, y.shape))).transpose()
            x_asterisks += x_dash.ravel().tolist()
            y_asterisks += y_dash.ravel().tolist()
            ang += np.pi / num_lines
        # All asterisks form part of one big line...
        return ax.plot(x_asterisks, y_asterisks, **kwargs)
        
    elif marker_type == 'circle':
        # Add a circle patch for each marker
        for ind in xrange(len(x)):
            ax.add_patch(mpl.patches.Circle((x[ind], y[ind]), z[ind], **kwargs))
        return ax.patches
    
    elif marker_type == 'scatter':
        # Get axes size in points
        points_per_axis = ax.get_position().extents[2:] * ax.get_figure().get_size_inches() * 72.0
        # Get points per data units in x and y directions
        x_range, y_range = 1.1 * (x.max() - x.min()), 1.1 * (y.max() - y.min())
        points_per_data = points_per_axis / np.array((x_range, y_range))
        # Scale according to largest data axis
        z *= points_per_data.min()
        return ax.scatter(x, y, 20.0 * z ** 2, **kwargs)
        
    else:
        raise ValueError("Unknown marker type '" + marker_type + "'")

#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  gaussian_ellipses
#---------------------------------------------------------------------------------------------------------

def gaussian_ellipses(mean, cov, contour=0.5, num_points=200):
    """Contour ellipses of two-dimensional Gaussian function.
    
    Parameters
    ----------
    mean : real array-like, shape (2,)
        Two-dimensional mean vector
    cov : real array-like, shape (2, 2)
        Two-by-two covariance matrix
    contour : float, or real array-like, shape (*K*,), optional
        Contour height of ellipse(s), as a (list of) factor(s) of the peak value.
        For a factor *sigma* of standard deviation, use ``exp(-0.5 * sigma**2)``.
    num_points : int, optional
        Number of points *N* on each ellipse
    
    Returns
    -------
    ellipses : real array, shape (*K*, *N*, 2)
        Array containing 2-D ellipse coordinates
    
    Raises
    ------
    ValueError
        If mean and/or cov has wrong shape
    
    """
    mean = np.asarray(mean)
    cov = np.asarray(cov)
    contour = np.atleast_1d(np.asarray(contour))
    if (mean.shape != (2,)) or (cov.shape != (2, 2)):
        raise ValueError('Mean and covariance should be 2-dimensional, with shapes (2,) and (2, 2) instead of'
                         + str(mean.shape) + ' and ' + str(cov.shape))
    # Create parametric circle
    t = np.linspace(0.0, 2.0 * np.pi, num_points)
    circle = np.vstack((np.cos(t), np.sin(t)))
    # Determine and apply transformation to ellipse
    eig_val, eig_vec = np.linalg.eig(cov)
    circle_to_ellipse = np.dot(eig_vec, np.diag(np.sqrt(eig_val)))
    base_ellipse = np.real(np.dot(circle_to_ellipse, circle))
    ellipses = []
    for cnt in contour:
        ellipse = np.sqrt(-2.0 * np.log(cnt)) * base_ellipse + mean[:, np.newaxis]
        ellipses.append(ellipse.transpose())
    return np.array(ellipses)

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  plot_compound_scan_on_target
#--------------------------------------------------------------------------------------------------

def plot_compound_scan_on_target(compscan, subtract_baseline=True, levels=None, add_scan_ids=True, band=0, ax=None):
    """Plot total power scans of compound scan in target space with beam fit.
    
    This plots contour ellipses of a Gaussian beam function fitted to the scans
    of a compound scan, as well as the total power of the scans as a pseudo-3D
    plot. It highlights the success of the beam fitting procedure.
    
    Parameters
    ----------
    compscan : :class:`compoundscan.CompoundScan` object
        Compound scan object to plot
    subtract_baseline : {True, False}, optional
        True to subtract baselines (only scans with baselines are then shown)
    levels : float, or real array-like, shape (K,), optional
        Contour level (or sequence of levels) to plot for Gaussian beam, as
        factor of beam height. The default is [0.5, 0.1].
    add_scan_ids : {True, False}, optional
        True if scan index numbers are to be added to plot
    band : int, optional
        Frequency band to plot
    ax : :class:`matplotlib.axes.Axes` object, optional
        Matplotlib axes object to receive plot (default is current axes)
    
    Returns
    -------
    ax : :class:`matplotlib.axes.Axes` object
        Matplotlib Axes object representing plot
    
    """
    if ax is None:
        ax = plt.gca()
    if levels is None:
        levels = [0.5, 0.1]
    # Extract total power and target coordinates (in degrees) of all scans (or those with baselines)
    if subtract_baseline:
        total_power = np.hstack([remove_spikes(scan.stokes('I')[:, band]) - scan.baseline(scan.timestamps) 
                                 for scan in compscan.scans if scan.baseline])
        target_coords = rad2deg(np.hstack([scan.target_coords for scan in compscan.scans if scan.baseline]))
    else:
        total_power = np.hstack([remove_spikes(scan.stokes('I')[:, band]) for scan in compscan.scans])
        target_coords = rad2deg(np.hstack([scan.target_coords for scan in compscan.scans]))
    
    # Show the locations of the scan samples themselves, with marker sizes indicating power values
    plot_marker_3d(target_coords[0], target_coords[1], total_power, ax=ax)
    # Plot the fitted Gaussian beam function as contours
    if compscan.beam:
        ell_type, center_type = 'r-', 'r+'
        var = fwhm_to_sigma(compscan.beam.width) ** 2.0
        if np.isscalar(var):
            var = [var, var]
        ellipses = gaussian_ellipses(compscan.beam.center, np.diag(var), contour=levels)
        for ellipse in ellipses:
            ax.plot(rad2deg(ellipse[:, 0]), rad2deg(ellipse[:, 1]), ell_type, lw=2)
        ax.plot([rad2deg(compscan.beam.center[0])], [rad2deg(compscan.beam.center[1])],
                center_type, ms=12, aa=False, mew=2)
    # Add scan number label next to the start of each scan
    if add_scan_ids:
        for n, scan in enumerate(compscan.scans):
            if subtract_baseline and not scan.baseline:
                continue
            start, end = rad2deg(scan.target_coords[:, 0]), rad2deg(scan.target_coords[:, -1])
            start_offset = start - 0.03 * (end - start)
            ax.text(start_offset[0], start_offset[1], str(n), ha='center', va='center')
    # Axis settings and labels
    x_range = [target_coords[0].min(), target_coords[0].max()]
    y_range = [target_coords[1].min(), target_coords[1].max()]
    if not np.any(np.isnan(x_range + y_range)):
        extra_space = 0.1 * max(x_range[1] - x_range[0], y_range[1] - y_range[0])
        ax.set_xlim(x_range + extra_space * np.array([-1.0, 1.0]))
        ax.set_ylim(y_range + extra_space * np.array([-1.0, 1.0]))
    ax.set_aspect('equal')
    ax.set_xlabel('x (deg)')
    ax.set_ylabel('y (deg)')
    return ax

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  plot_data_set_in_mount_space
#--------------------------------------------------------------------------------------------------

def plot_data_set_in_mount_space(dataset, levels=None, band=0, ax=None):
    """Plot total power scans of all compound scans in mount space with beam fits.
    
    This plots the total power of all scans in the data set as a pseudo-3D plot
    in 'instantaneous mount' space. This space has azimuth and elevation
    coordinates like the standard antenna pointing data, but assumes that each
    compound scan occurred instantaneously at the center time of the compound
    scan. This has the advantage that both fixed and moving targets are frozen
    in mount space, which makes the plots easier to interpret. Its advantage
    over normal target space is that it can display multiple compound scans on
    the same plot.
    
    For each compound scan, contour ellipses of the fitted Gaussian beam function
    are added, if it exists. It highlights the success of the beam fitting
    procedure.
    
    Parameters
    ----------
    dataset : :class:`scape.DataSet` object
        Data set to plot
    levels : float, or real array-like, shape (K,), optional
        Contour level (or sequence of levels) to plot for each Gaussian beam, as
        factor of beam height. The default is [0.5, 0.1].
    band : int, optional
        Frequency band to plot
    ax : :class:`matplotlib.axes.Axes` object, optional
        Matplotlib axes object to receive plot (default is current axes)
    
    Returns
    -------
    ax : :class:`matplotlib.axes.Axes` object
        Matplotlib Axes object representing plot
    
    """
    if ax is None:
        ax = plt.gca()
    if levels is None:
        levels = [0.5, 0.1]
    
    for compscan in dataset.compscans:
        total_power = np.hstack([remove_spikes(scan.stokes('I')[:, band]) for scan in compscan.scans])
        target_coords = np.hstack([scan.target_coords for scan in compscan.scans])
        center_time = np.median(np.hstack([scan.timestamps for scan in compscan.scans]))
        # Instantaneous mount coordinates are back on the sphere, but at a single time instant for all points
        mount_coords = list(plane_to_sphere(dataset.antenna, compscan.target,
                                            target_coords[0], target_coords[1], center_time))
        # Obtain ellipses and center, and unwrap az angles for all objects simultaneously to ensure they stay together
        if compscan.beam:
            var = fwhm_to_sigma(compscan.beam.width) ** 2.0
            if np.isscalar(var):
                var = [var, var]
            target_ellipses = gaussian_ellipses(compscan.beam.center, np.diag(var), contour=levels)
            mount_ellipses = list(plane_to_sphere(dataset.antenna, compscan.target,
                                                  target_ellipses[:, :, 0], target_ellipses[:, :, 1], center_time))
            mount_center = list(plane_to_sphere(dataset.antenna, compscan.target,
                                                compscan.beam.center[0], compscan.beam.center[1], center_time))
            all_az = np.concatenate((mount_coords[0], [mount_center[0]], mount_ellipses[0].flatten()))
            all_az = minimise_angle_wrap(all_az)
            mount_coords[0] = all_az[:len(mount_coords[0])]
            mount_center[0] = all_az[len(mount_coords[0])]
            mount_ellipses[0] = all_az[len(mount_coords[0]) + 1:].reshape(mount_ellipses[0].shape[:2])
        else:
            mount_coords[0] = minimise_angle_wrap(mount_coords[0])
            
        # Show the locations of the scan samples themselves, with marker sizes indicating power values
        plot_marker_3d(rad2deg(mount_coords[0]), rad2deg(mount_coords[1]), total_power, ax=ax)
        # Plot the fitted Gaussian beam function as contours
        if compscan.beam:
            ell_type, center_type = 'r-', 'r+'
            for ellipse in np.dstack(mount_ellipses):
                ax.plot(rad2deg(ellipse[:, 0]), rad2deg(ellipse[:, 1]), ell_type, lw=2)
            ax.plot([rad2deg(mount_center[0])], [rad2deg(mount_center[1])], center_type, ms=12, aa=False, mew=2)
    
    # Axis settings and labels
    ax.set_aspect('equal')
    ax.set_xlabel('az (deg)')
    ax.set_ylabel('el (deg)')
    return ax

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  plot_db_contours
#--------------------------------------------------------------------------------------------------

def plot_db_contours(x, y, Z, levels=None, sin_coords=False, add_lines=True, ax=None):
    """Filled contour plot of 2-D spherical function in decibels.
    
    The spherical function ``z = f(x, y)`` is a function of two angles, *x* and
    *y*, given in degrees. The function should be real-valued, but may contain
    negative parts. These are indicated by dashed contours. The contour levels
    are based on the absolute value of *z* in dBs.
    
    Parameters
    ----------
    x : real array-like, shape (N,)
        Vector of x coordinates, in degrees
    y : real array-like, shape (M,)
        Vector of y coordinates, in degrees
    Z : real array-like, shape (M, N)
        Matrix of z values, with rows associated with *y* and columns with *x*
    levels : real array-like, shape (K,), optional
        Sequence of ascending contour levels, in dB (default ranges from -60 to 0)
    sin_coords : {False, True}, optional
        True if coordinates should be converted to projected sine values. This
        is useful if a large portion of the sphere is plotted.
    add_lines : {True, False}, optional
        True if contour lines should be added to plot
    ax : :class:`matplotlib.axes.Axes` object, optional
        Matplotlib axes object to receive plot (default is current axes)
    
    Returns
    -------
    cset : :class:`matplotlib.contour.ContourSet` object
        Set of filled contour regions (useful for setting up color bar)
    
    """
    # pylint: disable-msg=C0103
    if ax is None:
        ax = plt.gca()
    if levels is None:
        levels = np.linspace(-60.0, 0.0, 21)
    levels = np.sort(levels)
    # Crude corner cutouts to indicate region outside spherical projection
    quadrant = np.linspace(0.0, np.pi / 2.0, 401)
    corner_x = np.concatenate([np.cos(quadrant), [1.0, 1.0]])
    corner_y = np.concatenate([np.sin(quadrant), [1.0, 0.0]])
    if sin_coords:
        x, y = np.sin(x * np.pi / 180.0), np.sin(y * np.pi / 180.0)
    else:
        corner_x, corner_y = 90.0 * corner_x, 90.0 * corner_y
    Z_db = 10.0 * np.log10(np.abs(Z))
    # Remove -infs (keep above lowest contour level to prevent white patches in contourf)
    Z_db[Z_db < levels.min() + 0.01] = levels.min() + 0.01
    # Also keep below highest contour level for the same reason
    Z_db[Z_db > levels.max() - 0.01] = levels.max() - 0.01
    
    cset = ax.contourf(x, y, Z_db, levels)
    mpl.rc('contour', negative_linestyle='solid')
    if add_lines:
        # Non-negative function has straightforward contours
        if Z.min() >= 0.0:
            ax.contour(x, y, Z_db, levels, colors='k', linewidths=0.5)
        else:
            # Indicate positive parts with solid contours
            Z_db_pos = Z_db.copy()
            Z_db_pos[Z < 0.0] = levels.min() + 0.01
            ax.contour(x, y, Z_db_pos, levels, colors='k', linewidths=0.5)
            # Indicate negative parts with dashed contours
            Z_db_neg = Z_db.copy()
            Z_db_neg[Z > 0.0] = levels.min() + 0.01
            mpl.rc('contour', negative_linestyle='dashed')
            ax.contour(x, y, Z_db_neg, levels, colors='k', linewidths=0.5)
    if sin_coords:
        ax.set_xlabel(r'sin $\theta$ sin $\phi$')
        ax.set_ylabel(r'sin $\theta$ cos $\phi$')
    else:
        ax.set_xlabel('x (deg)')
        ax.set_ylabel('y (deg)')
    ax.axis('image')
    ax.fill( corner_x,  corner_y, facecolor='w')
    ax.fill(-corner_x,  corner_y, facecolor='w')
    ax.fill(-corner_x, -corner_y, facecolor='w')
    ax.fill( corner_x, -corner_y, facecolor='w')
    return cset

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  plot_measured_beam_pattern
#--------------------------------------------------------------------------------------------------

def plot_measured_beam_pattern(compscan, stokes='I', subtract_baseline=True, add_samples=True, add_colorbar=True,
                               band=0, ax=None, **kwargs):
    """Plot measured beam pattern contained in compound scan.
    
    Parameters
    ----------
    compscan : :class:`compoundscan.CompoundScan` object
        Compound scan object to plot
    stokes : {'I', 'Q', 'U', 'V'}, optional
        The Stokes parameter to display
    subtract_baseline : {True, False}, optional
        True to subtract baselines (only scans with baselines are then shown)
    add_samples : {True, False}, optional
        True if scan sample locations are to be added
    add_colorbar : {True, False}, optional
        True if color bar indicating contour levels is to be added
    band : int, optional
        Frequency band to plot
    ax : :class:`matplotlib.axes.Axes` object, optional
        Matplotlib axes object to receive plot (default is current axes)
    kwargs : dict, optional
        Extra keyword arguments are passed to underlying :func:`plot_db_contours`
        function
    
    Returns
    -------
    ax : :class:`matplotlib.axes.Axes` object
        Matplotlib Axes object representing plot
    cset : :class:`matplotlib.contour.ContourSet` object
        Set of filled contour regions (useful for setting up color bar)
    
    """
    if ax is None:
        ax = plt.gca()
    # If there are no baselines in data set, don't subtract them
    if np.array([scan.baseline is None for scan in compscan.scans]).all():
        subtract_baseline = False
    # Extract Stokes parameter and target coordinates of all scans (or those with baselines)
    if subtract_baseline:
        power = np.hstack([remove_spikes(scan.stokes(stokes)[:, band]) - scan.baseline(scan.timestamps)
                           for scan in compscan.scans if scan.baseline])
        x, y = np.hstack([scan.target_coords for scan in compscan.scans if scan.baseline])
    else:
        power = np.hstack([remove_spikes(scan.stokes(stokes)[:, band]) for scan in compscan.scans])
        x, y = np.hstack([scan.target_coords for scan in compscan.scans])
    if compscan.beam:
        power /= compscan.beam.height
        x -= compscan.beam.center[0]
        y -= compscan.beam.center[1]
    else:
        power /= power.max()
    # Interpolate scattered data onto regular grid
    grid_x, grid_y, smooth_power = interpolate_measured_beam(x, y, power)
    # Plot contours and associated color bar (if requested)
    cset = plot_db_contours(rad2deg(grid_x), rad2deg(grid_y), smooth_power, ax=ax, **kwargs)
    if add_colorbar:
        plt.colorbar(cset, cax=plt.axes([0.9, 0.1, 0.02, 0.8]), format='%d')
        plt.gcf().text(0.96, 0.5, 'dB')
    # Show the locations of the scan samples themselves
    if add_samples:
        ax.plot(rad2deg(x), rad2deg(y), '.k', ms=2)
    # Axis settings and labels
    ax.set_aspect('equal')
    ax.set_title('Stokes %s' % stokes)
    ax.set_xlabel('x (deg)')
    ax.set_ylabel('y (deg)')
    return ax, cset
