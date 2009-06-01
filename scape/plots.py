"""Plotting routines."""

import time
import logging

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from .stats import remove_spikes
from .beam_baseline import fwhm_to_sigma
from .coord import rad2deg

logger = logging.getLogger("scape.plots")

def _shrink_axes(ax, shift=0.075):
    """Shrink axes on the left to leave more space for labels.
    
    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes` object
        Matplotlib Axes object
    shift : float
        Amount to advance left border of axes (axes width will decrease by same
        amount), in axes coords
    
    Returns
    -------
    ax : :class:`matplotlib.axes.Axes` object
        Adjusted axes object
    
    """
    pos = ax.get_position().bounds
    ax.set_position([pos[0] + shift, pos[1], pos[2] - shift, pos[3]])
    return ax

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
    for ax_ind, pol in enumerate(['XX', 'YY']):
        # Time-frequency waterfall plots
        ax = axes_list[ax_ind]
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
                    ax.add_collection(lines)
                else:
                    ax.plot(segments[0][:, 0] + offsets.squeeze()[0], 
                            segments[0][:, 1] + offsets.squeeze()[1], color=colors[0], lw=0.5)
                t_limits += [time_line.min(), time_line.max()]
                all_subscans.append(ss.coherency(pol))
            # Add scan target name and partition lines between scans
            if s.subscans:
                start_time_ind = len(t_limits) - 2 * len(s.subscans)
                if scan_ind >= 1:
                    border_time = (t_limits[start_time_ind - 1] + t_limits[start_time_ind]) / 2.0
                    ax.plot([border_time, border_time], [0.0, 10.0 * channel_freqs_GHz.max()], '--k')
                ax.text((t_limits[start_time_ind] + t_limits[-1]) / 2.0,
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
        all_subscans = np.concatenate(all_subscans, axis=0)
        rfi_channel_list = list(set(channel_list) & set(rfi_channels))
        non_rfi_channel_list = list(set(channel_list) - set(rfi_channel_list))
        # Do RFI and non-RFI channels separately, in order to grey out the RFI channels (boxplot constraint)
        for rfi_flag, channels in enumerate([non_rfi_channel_list, rfi_channel_list]):
            if len(channels) == 0:
                continue
            chan_data = all_subscans[:, channels]
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
        p_limits += [all_subscans.min(), all_subscans.max()]
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
#--- FUNCTION :  fitted_beam_scans
#--------------------------------------------------------------------------------------------------

def fitted_beam_scans(scan, band=0, fig=None):
    """Plot beam pattern fitted to multiple scans through a single point source.
    
    This plots time series plots of the subscans comprising a scan, with the
    beam and baseline fits superimposed. It highlights the success of the beam
    and baseline fitting procedure.
    
    Parameters
    ----------
    scan : :class:`scan.Scan` object
        Scan object to plot
    band : int, optional
        Frequency band to plot
    fig : :class:`matplotlib.figure.Figure` object, optional
        Matplotlib Figure object to contain plots (default is current figure)
    
    Returns
    -------
    axes_list : list of :class:`matplotlib.axes.Axes` objects
        List of matplotlib Axes objects, one per plot
    
    """
    if fig is None:
        fig = plt.gcf()
    # Set up axes: one figure, with one set of axes per subscan
    axes_list = []
    time_origin = np.array([ss.timestamps.min() for ss in scan.subscans]).min()
    power_limits = []
    
    for n, ss in enumerate(scan.subscans):
        measured_power = ss.stokes('I')[:, band]
        time_line = ss.timestamps - time_origin
        smooth_power = remove_spikes(measured_power)
        power_limits.extend([smooth_power.min(), smooth_power.max()])
        ax = plt.subplot(len(scan.subscans), 1, n + 1)
        if scan.fitted_beam:
            fitted_power = scan.fitted_beam(ss.target_coords.transpose())
            baseline_power = scan.fitted_beam.baseline(ss.target_coords.transpose())
            ax.plot(time_line, baseline_power, 'r', lw=2)
            ax.plot(time_line, fitted_power, 'r', lw=2)
        ax.plot(time_line, measured_power, 'b')
        axes_list.append(ax)
    
    ylim = (min(power_limits), 1.1 * max(power_limits) - 0.1 * min(power_limits))
    for n, ax in enumerate(axes_list):
        ax.set_ylim(ylim)
        if n == len(scan.subscans) - 1:
            ax.set_xlabel('Time (s), since %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_origin)))
        ax.set_ylabel('Power')
    return axes_list

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
        raise ValueError('Mean and covariance should be 2-dimensional, with shapes (2,) and (2,2) instead of'
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
#--- FUNCTION :  fitted_beam_target
#--------------------------------------------------------------------------------------------------

def fitted_beam_target(scan, band=0, ax=None):
    """Plot beam pattern fitted to multiple scans through a single point source.
    
    This plots contour ellipses of a Gaussian beam function fitted to multiple
    scans through a point source, as well as the power values of the scans
    themselves as a pseudo-3D plot. It highlights the success of the beam
    fitting procedure.
    
    Parameters
    ----------
    scan : :class:`scan.Scan` object
        Scan object to plot
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
    # Extract total power and target coordinates (in degrees) of all scans
    total_power = np.hstack([ss.stokes('I')[:, band] for ss in scan.subscans])
    target_coords = rad2deg(np.hstack([ss.target_coords for ss in scan.subscans]))
    
    # Show the locations of the scan samples themselves, with marker sizes indicating power values
    plot_marker_3d(target_coords[0], target_coords[1], total_power, ax=ax)
    # Plot the fitted Gaussian beam function as contours
    if scan.fitted_beam:
        ell_type, center_type = 'r-', 'r+'
        ellipses = gaussian_ellipses(scan.fitted_beam.beam_center,
                                     np.diag(fwhm_to_sigma(scan.fitted_beam.beam_width) ** 2.0),
                                     contour=[0.5, 0.1])
        for ellipse in ellipses:
            ax.plot(rad2deg(ellipse[:, 0]), rad2deg(ellipse[:, 1]), ell_type, lw=2)
        ax.plot([rad2deg(scan.fitted_beam.beam_center[0])], [rad2deg(scan.fitted_beam.beam_center[1])],
                center_type, ms=12, aa=False, mew=2)
    
    # Axis settings and labels
    x_range = [target_coords[0].min(), target_coords[0].max()]
    y_range = [target_coords[1].min(), target_coords[1].max()]
    if not np.any(np.isnan(x_range + y_range)):
        extra_space = 0.2 * max(x_range[1] - x_range[0], y_range[1] - y_range[0])
        ax.set_xlim(x_range + extra_space * np.array([-1.0, 1.0]))
        ax.set_ylim(y_range + extra_space * np.array([-1.0, 1.0]))
    ax.set_aspect('equal')
    ax.set_xlabel('x (deg)')
    ax.set_ylabel('y (deg)')
    return ax
