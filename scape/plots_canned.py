"""Canned plots."""

import time
import logging

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from katpoint import rad2deg
from .stats import robust_mu_sigma, remove_spikes, minimise_angle_wrap
from .beam_baseline import fwhm_to_sigma, extract_measured_beam, interpolate_measured_beam
from .plots_basic import plot_line_segments, plot_compacted_images, plot_marker_3d, \
                         gaussian_ellipses, plot_db_contours, ordinal_suffix

logger = logging.getLogger("scape.plots_canned")

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  plot_xy
#--------------------------------------------------------------------------------------------------

def plot_xy(data, x='time', y='amp', z=None, pol='I', sigma=1.0, ax=None, **kwargs):
    """Generic 2-D plotting."""
    if ax is None:
        ax = plt.gca()
    # Create list of scans from whatever input form the data takes (DataSet and CompoundScan have .scans)
    scans = getattr(data, 'scans', [data])
    if len(scans) == 0:
        raise ValueError('No scans found to plot')
    # Extract earliest timestamp, used if one of axes is 'time', and associated dataset
    time_origin = np.min([s.timestamps.min() for s in scans])
    dataset = scans[0].compscan.dataset
    # Create dict of plottable data types and the corresponding functions that will extract them from a Scan object,
    # plus their axis labels
    func = {'time'     : ('Time (s), since %s' % (time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(time_origin)),),
                          lambda scan: scan.timestamps - time_origin),
            'freq'     : ('Frequency (MHz)', lambda scan: scan.compscan.dataset.freqs),
            'chan'     : ('Channel index', lambda scan: range(len(scan.compscan.dataset.freqs))),
            'amp'      : ('%s amplitude (%s)' % (pol, dataset.data_unit), lambda scan: np.abs(scan.pol(pol))),
            'phase'    : ('%s phase (deg)' % (pol,), lambda scan: rad2deg(np.angle(scan.pol(pol)))),
            'real'     : ('Real part of %s (%s)' % (pol, dataset.data_unit), lambda scan: scan.pol(pol).real),
            'imag'     : ('Imaginary part of %s (%s)' % (pol, dataset.data_unit),
                          lambda scan: scan.pol(pol).imag),
            'az'       : ('Azimuth angle (deg)', lambda scan: rad2deg(scan.pointing['az'])),
            'el'       : ('Elevation angle (deg)', lambda scan: rad2deg(scan.pointing['el'])),
            'parangle' : ('Parallactic angle (deg)', lambda scan: rad2deg(scan.parangle))}
    try:
        labelx, fx = func[x] if isinstance(x, basestring) else x
        labely, fy = func[y] if isinstance(y, basestring) else y
    except KeyError:
        raise ValueError("Unknown quantity to plot - choose one of %s" % (func.keys(),))
    xx, yy = [fx(s) for s in scans], [fy(s) for s in scans]
    num_chans = len(dataset.freqs)
    # Plot of correlator data (y) vs frequency or similar (x)
    if np.shape(xx[0]) == (num_chans,) and np.ndim(yy[0]) == 2 and np.shape(yy[0])[1] == num_chans:
        xx, yy = xx[0], np.vstack(yy)
        y_mean, y_stdev = robust_mu_sigma(yy, axis=0)
        plot_segments(xx, np.vstack((y_mean - sigma * y_stdev, y_mean + sigma * y_stdev)).transpose(),
                      monotonic_axis='x', ax=ax, **kwargs)
        plot_segments(xx, y_mean, add_breaks=False, monotonic_axis='x', ax=ax, **kwargs)
        plot_segments(xx, yy.min(axis=0), add_breaks=False, monotonic_axis='x', ax=ax, linestyles='dashed', **kwargs)
        plot_segments(xx, yy.max(axis=0), add_breaks=False, monotonic_axis='x', ax=ax, linestyles='dashed', **kwargs)
    # Plot of frequency or similar (y) vs correlator data (x)
    elif np.shape(yy[0]) == (num_chans,) and np.ndim(xx[0]) == 2 and np.shape(xx[0])[1] == num_chans:
        yy, xx = yy[0], np.vstack(xx)
        x_mean, x_stdev = robust_mu_sigma(xx, axis=0)
        plot_segments(np.vstack((x_mean - sigma * x_stdev, x_mean + sigma * x_stdev)).transpose(), yy,
                      monotonic_axis='y', ax=ax, **kwargs)
        plot_segments(x_mean, yy, add_breaks=False, monotonic_axis='y', ax=ax, **kwargs)
        plot_segments(xx.min(axis=0), yy, add_breaks=False, monotonic_axis='y', ax=ax, linestyles='dashed', **kwargs)
        plot_segments(xx.max(axis=0), yy, add_breaks=False, monotonic_axis='y', ax=ax, linestyles='dashed', **kwargs)
    else:
        plot_segments(xx, yy, labels=range(len(scans)), ax=ax, **kwargs)

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  plot_spectrum
#--------------------------------------------------------------------------------------------------

def plot_spectrum(dataset, pol='I', scan=-1, sigma=1.0, vertical=True, dB=True, ax=None):
    """Spectrum plot of power data as a function of frequency.

    This plots the power spectrum of the given scan (either in Stokes or
    coherency form), with error bars indicating the variation in the data
    (+/- *sigma* times the standard deviation). Robust statistics are used for
    the plot (median and standard deviation derived from interquartile range).

    Parameters
    ----------
    dataset : :class:`scape.DataSet` object
        Data set to plot
    pol : {'I', 'Q', 'U', 'V', 'HH', 'VV', 'XX', 'YY'}, optional
        The coherency / Stokes parameter to display (must be real)
    scan : int, optional
        Index of scan in data set to plot (-1 to plot all scans together)
    sigma : float, optional
        The error bar is this factor of standard deviation above and below mean
    vertical : {True, False}, optional
        True if frequency is on the x-axis and power is on the y-axis, and False
        if it is the other way around
    dB : {True, False}, optional
        True to plot power logarithmically in dB of the underlying unit
    ax : :class:`matplotlib.axes.Axes` object, optional
        Matplotlib axes object to receive plot (default is current axes)

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes` object
        Matplotlib Axes object representing plot
    power_lim : list of 2 floats
        Overall minimum and maximum value of data, useful for setting plot limits

    Raises
    ------
    ValueError
        If *pol* is not one of the allowed names

    """
    if not pol in ('I', 'Q', 'U', 'V', 'HH', 'VV', 'XX', 'YY'):
        raise ValueError("Polarisation key should be one of 'I', 'Q', 'U', 'V', 'HH', 'VV', 'XX' or 'YY' (i.e. real)")
    if ax is None:
        ax = plt.gca()
    if scan >= 0:
        data = np.abs(dataset.scans[scan].pol(pol))
    else:
        data = np.vstack([np.abs(s.pol(pol)) for s in dataset.scans])
    power = robust_mu_sigma(data)
    power_min, power_max = data.min(axis=0), data.max(axis=0)
    del data

    # Form makeshift rectangular patches indicating power variation in each channel
    power_mean = np.repeat(power.mu, 3)
    power_upper = np.repeat(np.clip(power.mu + sigma * power.sigma, -np.inf, power_max), 3)
    power_lower = np.repeat(np.clip(power.mu - sigma * power.sigma, power_min, np.inf), 3)
    power_min, power_max = np.repeat(power_min, 3), np.repeat(power_max, 3)
    if dB:
        power_mean = 10 * np.log10(power_mean)
        power_min, power_max = 10 * np.log10(power_min), 10 * np.log10(power_max)
        power_upper, power_lower = 10 * np.log10(power_upper), 10 * np.log10(power_lower)
    freqs = np.array([dataset.freqs - 0.999 * dataset.bandwidths / 2.0,
                      dataset.freqs + 0.999 * dataset.bandwidths / 2.0,
                      dataset.freqs + dataset.bandwidths / 2.0]).transpose().ravel()
    mask = np.arange(len(power_mean)) % 3 == 2

    # Fill_between (which uses poly path) is much faster than a Rectangle patch collection
    if vertical:
        ax.fill_between(freqs, power_min, power_max, where=~mask, facecolors='0.8', edgecolors='0.8')
        ax.fill_between(freqs, power_lower, power_upper, where=~mask, facecolors='0.6', edgecolors='0.6')
        ax.plot(freqs, np.ma.masked_array(power_mean, mask), color='b', lw=2)
        ax.plot(dataset.freqs, power_mean[::3], 'ob')
        ax.set_xlim(dataset.freqs[0]  - dataset.bandwidths[0] / 2.0,
                    dataset.freqs[-1] + dataset.bandwidths[-1] / 2.0)
        freq_label, power_label = ax.set_xlabel, ax.set_ylabel
    else:
        ax.fill_betweenx(freqs, power_min, power_max, where=~mask, facecolors='0.8', edgecolors='0.8')
        ax.fill_betweenx(freqs, power_lower, power_upper, where=~mask, facecolors='0.6', edgecolors='0.6')
#        ax.plot(np.ma.masked_array(power_mean, mask), freqs, color='b', lw=2)
        ax.plot(power_mean, np.ma.masked_array(freqs, mask), color='b', lw=2)
        ax.plot(power_mean[::3], dataset.freqs, 'ob')
        ax.set_ylim(dataset.freqs[0]  - dataset.bandwidths[0] / 2.0,
                    dataset.freqs[-1] + dataset.bandwidths[-1] / 2.0)
        freq_label, power_label = ax.set_ylabel, ax.set_xlabel
    freq_label('Frequency (MHz)')
    db_str = 'dB ' if dB else ''
    if dataset.data_unit == 'Jy':
        power_label('Flux density (%sJy)' % db_str)
    elif dataset.data_unit == 'K':
        power_label('Temperature (%sK)' % db_str)
    else:
        power_label('Raw power (%scounts)' % db_str)
    return ax, [power_min.min(), power_max.max()]

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
    channel_freqs_MHz = dataset.freqs
    channel_bandwidths_MHz = dataset.bandwidths
    num_channels = len(channel_freqs_MHz)
    data_min = {'HH': np.tile(np.inf, (len(scans), num_channels)),
                'VV': np.tile(np.inf, (len(scans), num_channels))}
    data_max = {'HH': np.zeros((len(scans), num_channels)),
                'VV': np.zeros((len(scans), num_channels))}
    time_origin = np.double(np.inf)
    for n, scan in enumerate(scans):
        time_origin = min(time_origin, scan.timestamps.min())
        for pol in ['HH', 'VV']:
            smoothed_power = remove_spikes(np.abs(scan.pol(pol)))
            channel_min = smoothed_power.min(axis=0)
            data_min[pol][n] = np.where(channel_min < data_min[pol][n], channel_min, data_min[pol][n])
            channel_max = smoothed_power.max(axis=0)
            data_max[pol][n] = np.where(channel_max > data_max[pol][n], channel_max, data_max[pol][n])
    # Obtain minimum and maximum values in each channel (of smoothed data)
    for pol in ['HH', 'VV']:
        data_min[pol] = data_min[pol].min(axis=0)
        data_max[pol] = data_max[pol].max(axis=0)
    channel_list = np.arange(0, num_channels, channel_skip, dtype='int')
    offsets = np.column_stack((np.zeros(len(channel_list), dtype='float'), channel_freqs_MHz[channel_list]))
    scale = 0.08 * num_channels

    # Plot of raw HH and YY power in all channels
    t_limits, p_limits = [], []
    for ax_ind, pol in enumerate(['HH', 'VV']):
        # Time-frequency waterfall plots
        ax = axes_list[ax_ind]
        for compscan_ind, compscan in enumerate(dataset.compscans):
            for scan in compscan.scans:
                # Grey out RFI-tagged channels using alpha transparency
                if scan.label == 'scan':
                    colors = [(0.0, 0.0, 1.0, 1.0 - 0.6 * (chan not in dataset.channel_select))
                              for chan in channel_list]
                else:
                    colors = [(0.0, 0.0, 0.0, 1.0 - 0.6 * (chan not in dataset.channel_select))
                              for chan in channel_list]
                time_line = scan.timestamps - time_origin
                # Normalise the data in each channel to lie between 0 and (channel bandwidth * scale)
                norm_power = scale * channel_bandwidths_MHz[np.newaxis, :] * \
                            (np.abs(scan.pol(pol)) - data_min[pol][np.newaxis, :]) / \
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
            # Add compound scan target name and partition lines between compound scans
            if compscan.scans:
                start_time_ind = len(t_limits) - 2 * len(compscan.scans)
                if compscan_ind >= 1:
                    border_time = (t_limits[start_time_ind - 1] + t_limits[start_time_ind]) / 2.0
                    ax.plot([border_time, border_time], [0.0, 10.0 * channel_freqs_MHz.max()], '--k')
                ax.text((t_limits[start_time_ind] + t_limits[-1]) / 2.0,
                        offsets[0, 1] - scale * channel_bandwidths_MHz[0], compscan.target.name,
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
        if pol == 'HH':
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
        ax.set_ylabel('Frequency (MHz)')
        # Power spectrum box plots
        ax = axes_list[ax_ind + 2]
        ax, power_lim = plot_spectrum(dataset, pol=pol, scan=-1, vertical=False, ax=ax)
        # Add extra ticks on the right to indicate channel numbers
        # second_axis = ax.twinx()
        # second_axis.yaxis.tick_right()
        # second_axis.yaxis.set_label_position('right')
        # second_axis.set_ylabel('Channel number')
        # second_axis.set_yticks(channel_freqs_MHz[channel_list])
        # second_axis.set_yticklabels([str(chan) for chan in channel_list])
        p_limits += power_lim
        ax.set_ylabel('')
        ax.set_title('%s power spectrum' % pol)
        if pol == 'HH':
            # This is more elaborate because the subplot axes are shared
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.set_xlabel('')
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
    # Fix limits globally
    t_limits = np.array(t_limits)
    y_range = channel_freqs_MHz.max() - channel_freqs_MHz.min()
    if y_range < channel_bandwidths_MHz[0]:
        y_range = 10.0 * channel_bandwidths_MHz[0]
    for ax in axes_list[:2]:
        ax.set_xlim(t_limits.min(), t_limits.max())
        ax.set_ylim(channel_freqs_MHz.min() - 0.1 * y_range, channel_freqs_MHz.max() + 0.1 * y_range)
    p_limits = np.array(p_limits)
    for ax in axes_list[2:]:
        ax.set_xlim(p_limits.min(), p_limits.max())

    return axes_list

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  plot_spectrogram
#--------------------------------------------------------------------------------------------------

def plot_spectrogram(dataset, pol='I', add_scan_ids=True, dB=True, ax=None):
    """Plot spectrogram of all scans in data set in compacted form.

    This plots the spectrogram of each scan in the data set on a single set of
    axes, with no gaps between the spectrogram images. This is done for all times
    and all channels. The tick labels on the *x* axis are modified to reflect
    the correct timestamps, and the breaks between scans are indicated by
    vertical lines. RFI-flagged channels are greyed out in the display.

    Parameters
    ----------
    dataset : :class:`scape.DataSet` object
        Data set to plot
    pol : {'I', 'Q', 'U', 'V', 'HH', 'VV', 'HV', 'VH', 'XX', 'YY', 'XY', 'YX'}, optional
        The coherency / Stokes parameter to display (must be real for single-dish)
    add_scan_ids : {True, False}, optional
        True if scan index numbers are to be added to plot
    dB : {True, False}, optional
        True to plot power logarithmically in dB of the underlying unit
    ax : :class:`matplotlib.axes.Axes` object, optional
        Matplotlib axes object to receive plot (default is current axes)

    Returns
    -------
    images : list of :class:`matplotlib.image.AxesImage` objects
        List of spectrogram images
    border_lines : :class:`matplotlib.collections.LineCollection` object
        Collection of vertical lines separating the segments
    text_labels : list of :class:`matplotlib.text.Text` objects
        List of added text labels

    Raises
    ------
    ValueError
        If *pol* is not one of the allowed names

    """
    if ax is None:
        ax = plt.gca()
    db_func = (lambda x: 10.0 * np.log10(x)) if dB else (lambda x: x)
    if dataset.scans[0].has_autocorr:
        if not pol in ('I', 'Q', 'U', 'V', 'HH', 'VV', 'XX', 'YY'):
            raise ValueError("Polarisation key should be one of 'I', 'Q', 'U', 'V', " +
                             "'HH', 'VV', 'XX' or 'YY' (i.e. real) for single-dish data")
        imdata = [db_func(scan.pol(pol)).transpose() for scan in dataset.scans]
    else:
        imdata = [db_func(np.abs(scan.pol(pol))).transpose() for scan in dataset.scans]
    xticks = [scan.timestamps for scan in dataset.scans]
    time_origin = np.min([x.min() for x in xticks])
    labels = [str(n) for n in xrange(len(dataset.scans))] if add_scan_ids else []
    ylim = (dataset.freqs[0], dataset.freqs[-1])
    clim = [np.double(np.inf), np.double(-np.inf)]
    for scan in dataset.scans:
        if dataset.scans[0].has_autocorr:
            smoothed_power = db_func(remove_spikes(scan.pol(pol)))
        else:
            smoothed_power = db_func(remove_spikes(np.abs(scan.pol(pol))))
        clim = [min(clim[0], smoothed_power.min()), max(clim[1], smoothed_power.max())]
    grey_rows = list(set(range(len(dataset.freqs))) - set(dataset.channel_select))
    images, border_lines, text_labels = plot_compacted_images(imdata, xticks, labels, ylim, clim, grey_rows, ax)
    ax.set_xlabel('Time (s), since %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_origin)))
    ax.set_ylabel('Channel frequency (MHz)')
    return images, border_lines, text_labels

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  plot_fringes
#--------------------------------------------------------------------------------------------------

def plot_fringes(dataset, pol='I', add_scan_ids=True, ax=None):
    """Plot fringe phase of all scans in data set in compacted form.

    This plots the *fringe phase* (phase as a function of time and frequency) of
    each scan in the data set on a single set of axes as a set of images, with no
    gaps between the images. This is done for all times and all channels. The
    tick labels on the *x* axis are modified to reflect the correct timestamps,
    and the breaks between scans are indicated by vertical lines. RFI-flagged
    channels are greyed out in the display.

    Parameters
    ----------
    dataset : :class:`scape.DataSet` object
        Data set to plot
    pol : {'I', 'Q', 'U', 'V', 'HH', 'VV', 'HV', 'VH', 'XX', 'YY', 'XY', 'YX'}, optional
        The coherency / Stokes parameter to display
    add_scan_ids : {True, False}, optional
        True if scan index numbers are to be added to plot
    ax : :class:`matplotlib.axes.Axes` object, optional
        Matplotlib axes object to receive plot (default is current axes)

    Returns
    -------
    images : list of :class:`matplotlib.image.AxesImage` objects
        List of fringe phase images
    border_lines : :class:`matplotlib.collections.LineCollection` object
        Collection of vertical lines separating the segments
    text_labels : list of :class:`matplotlib.text.Text` objects
        List of added text labels

    Raises
    ------
    ValueError
        If *pol* is not one of the allowed names

    """
    if ax is None:
        ax = plt.gca()
    imdata = [np.angle(scan.pol(pol)).transpose() for scan in dataset.scans]
    xticks = [scan.timestamps for scan in dataset.scans]
    time_origin = np.min([x.min() for x in xticks])
    labels = [str(n) for n in xrange(len(dataset.scans))] if add_scan_ids else []
    ylim = (dataset.freqs[0], dataset.freqs[-1])
    clim = [-np.pi, np.pi]
    grey_rows = list(set(range(len(dataset.freqs))) - set(dataset.channel_select))
    images, border_lines, text_labels = plot_compacted_images(imdata, xticks, labels, ylim, clim, grey_rows, ax)
    ax.set_xlabel('Time (s), since %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_origin)))
    ax.set_ylabel('Channel frequency (MHz)')
    return images, border_lines, text_labels

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  plot_rfi_segmentation
#--------------------------------------------------------------------------------------------------

def plot_rfi_segmentation(dataset, sigma=8.0, min_bad_scans=0.25, channel_skip=None, add_scan_ids=True, fig=None):
    """Plot separate time series of data classified as RFI and non-RFI."""
    num_channels = len(dataset.freqs)
    if not channel_skip:
        channel_skip = max(num_channels // 32, 1)
    if fig is None:
        fig = plt.gcf()
    # Set up axes: one figure with custom subfigures for signal and RFI plots, with shared x and y axes
    axes_list = []
    axes_list.append(fig.add_axes([0.125, 6 / 11., 0.8, 4 / 11.]))
    axes_list.append(fig.add_axes([0.125, 0.1, 0.8, 4 / 11.], sharex=axes_list[0], sharey=axes_list[0]))

    labels = [str(n) for n in xrange(len(dataset.scans))] if add_scan_ids else []
    start = np.array([scan.timestamps.min() for scan in dataset.scans])
    end = np.array([scan.timestamps.max() for scan in dataset.scans])
    compacted_start = [0.0] + np.cumsum(end - start).tolist()
    time_origin = start.min()
    # Identify RFI channels, and return extra data
    rfi_channels, rfi_count, rfi_data = dataset.identify_rfi_channels(sigma, min_bad_scans, extra_outputs=True)
    channel_list = np.arange(0, num_channels, channel_skip, dtype='int')
    non_rfi_channels = list(set(range(num_channels)) - set(rfi_channels))
    rfi_channels = [n for n in channel_list if n in rfi_channels]
    non_rfi_channels = [n for n in channel_list if n in non_rfi_channels]
    template = [np.column_stack((scan.timestamps - time_origin, rfi_data[s][1]))
                for s, scan in enumerate(dataset.scans)]
    # Do signal (non-RFI) display
    ax = axes_list[0]
    for s, scan in enumerate(dataset.scans):
        timeline = scan.timestamps - start[s] + compacted_start[s]
        average_std = np.sqrt(np.sqrt(2) / len(timeline)) * rfi_data[s][2][:, non_rfi_channels].mean(axis=1)
        lower, upper = rfi_data[s][1] - np.sqrt(sigma) * average_std, rfi_data[s][1] + np.sqrt(sigma) * average_std
        ax.fill_between(timeline, upper, lower, edgecolor='0.7', facecolor='0.7', lw=0)
        data_segments = [np.column_stack((timeline, rfi_data[s][0][:, n])) for n in non_rfi_channels]
        ax.add_collection(mpl.collections.LineCollection(data_segments))
    plot_line_segments(template, labels, ax=ax, lw=2, color='k')
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('Normalised power')
    # do RFI display
    ax = axes_list[1]
    for s, scan in enumerate(dataset.scans):
        timeline = scan.timestamps - start[s] + compacted_start[s]
        average_std = np.sqrt(np.sqrt(2) / len(timeline)) * rfi_data[s][2][:, rfi_channels].mean(axis=1)
        lower, upper = rfi_data[s][1] - np.sqrt(sigma) * average_std, rfi_data[s][1] + np.sqrt(sigma) * average_std
        ax.fill_between(timeline, upper, lower, edgecolor='0.7', facecolor='0.7', lw=0)
        data_segments = [np.column_stack((timeline, rfi_data[s][0][:, n])) for n in rfi_channels]
        ax.add_collection(mpl.collections.LineCollection(data_segments))
    plot_line_segments(template, labels, ax=ax, lw=2, color='k')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Time (s), since %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_origin)))
    ax.set_ylabel('Normalised power')
    return axes_list

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  plot_compound_scan_in_time
#--------------------------------------------------------------------------------------------------

def plot_compound_scan_in_time(compscan, pol='I', add_scan_ids=True, band=0, ax=None):
    """Plot compound scan data in time with superimposed beam/baseline fit.

    This plots time series of the selected polarisation power in all the scans
    comprising a compound scan, with the beam and baseline fits superimposed.
    It highlights the success of the beam and baseline fitting procedure. It is
    assumed that the beam and baseline was fit to the selected polarisation.

    Parameters
    ----------
    compscan : :class:`compoundscan.CompoundScan` object
        Compound scan object to plot
    pol : {'I', 'Q', 'U', 'V', 'XX', 'YY'}, optional
        The coherency / Stokes parameter to display (must be real)
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

    Raises
    ------
    ValueError
        If *pol* is not one of the allowed names

    """
    if not pol in ('I', 'Q', 'U', 'V', 'HH', 'VV', 'XX', 'YY'):
        raise ValueError("Polarisation key should be one of 'I', 'Q', 'U', 'V', 'HH', 'VV', 'XX' or 'YY' (i.e. real)")
    if ax is None:
        ax = plt.gca()
    time_origin = np.array([scan.timestamps.min() for scan in compscan.scans]).min()
    power_limits, data_segments = [], []
    baseline_segments, beam_segments, inner_beam_segments = [], [], []

    # Construct segments to be plotted
    for scan in compscan.scans:
        timeline = scan.timestamps - time_origin
        measured_power = np.abs(scan.pol(pol)[:, band])
        smooth_power = remove_spikes(measured_power)
        power_limits.extend([smooth_power.min(), smooth_power.max()])
        data_segments.append(np.column_stack((timeline, measured_power)))
        if scan.baseline:
            baseline_power = scan.baseline(scan.timestamps)
            baseline_segments.append(np.column_stack((timeline, baseline_power)))
        elif compscan.baseline:
            baseline_power = compscan.baseline(scan.target_coords)
            baseline_segments.append(np.column_stack((timeline, baseline_power)))
        else:
            baseline_segments.append(np.column_stack((timeline, np.tile(np.nan, len(timeline)))))
        beam_power, inner_beam_power = np.tile(np.nan, len(timeline)), np.tile(np.nan, len(timeline))
        if compscan.beam:
            if (compscan.beam.refined and scan.baseline) or (not compscan.beam.refined and compscan.baseline):
                beam_power = compscan.beam(scan.target_coords) + baseline_power
            if scan.baseline:
                radius = np.sqrt(((scan.target_coords - compscan.beam.center[:, np.newaxis]) ** 2).sum(axis=0))
                inner = radius < 0.6 * np.mean(compscan.beam.width)
                inner_beam_power = beam_power.copy()
                inner_beam_power[~inner] = np.nan
        beam_segments.append(np.column_stack((timeline, beam_power)))
        inner_beam_segments.append(np.column_stack((timeline, inner_beam_power)))
    # Get overall y limits
    power_range = max(power_limits) - min(power_limits)
    if power_range == 0.0:
        power_range = 1.0
    # Plot segments from back to front
    labels = [str(n) for n in xrange(len(compscan.scans))] if add_scan_ids else []
    plot_line_segments(data_segments, labels, ax=ax, color='b', lw=1)
    beam_color = ('r' if compscan.beam.refined else 'g') if compscan.beam and compscan.beam.is_valid else 'y'
    baseline_colors = [('r' if scan.baseline else 'g') for scan in compscan.scans]
    plot_line_segments(baseline_segments, ax=ax, color=baseline_colors, lw=2)
    if compscan.beam and compscan.beam.refined:
        plot_line_segments(beam_segments, ax=ax, color=beam_color, lw=2, linestyles='dashed')
        plot_line_segments(inner_beam_segments, ax=ax, color=beam_color, lw=2)
    else:
        plot_line_segments(beam_segments, ax=ax, color=beam_color, lw=2)
    ax.set_ylim(min(power_limits) - 0.05 * power_range, max(power_limits) + 0.05 * power_range)
    # Format axes
    ax.set_xlabel('Time (s), since %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_origin)))
    ax.set_ylabel('Pol %s' % pol)
    return ax

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  plot_compound_scan_on_target
#--------------------------------------------------------------------------------------------------

def plot_compound_scan_on_target(compscan, pol='I', subtract_baseline=True, levels=None,
                                 add_scan_ids=True, band=0, ax=None):
    """Plot compound scan data in target space with beam fit.

    This plots contour ellipses of a Gaussian beam function fitted to the scans
    of a compound scan, as well as the selected power of the scans as a pseudo-3D
    plot. It highlights the success of the beam fitting procedure.

    Parameters
    ----------
    compscan : :class:`compoundscan.CompoundScan` object
        Compound scan object to plot
    pol : {'I', 'HH', 'VV', 'XX', 'YY'}, optional
        The coherency / Stokes parameter to display (must be real and positive)
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

    Raises
    ------
    ValueError
        If *pol* is not one of the allowed names

    """
    if not pol in ('I', 'HH', 'VV', 'XX', 'YY'):
        raise ValueError("Polarisation key should be one of 'I', 'HH', 'VV', 'XX' or 'YY' (i.e. positive)")
    if ax is None:
        ax = plt.gca()
    if levels is None:
        levels = [0.5, 0.1]
    # Check that there are any baselines to plot
    if subtract_baseline and not np.any([scan.baseline for scan in compscan.scans]):
        subtract_baseline = False
        logger.warning('No scans were found with baselines - setting subtract_baseline to False')
    # Extract total power and target coordinates (in degrees) of all scans (or those with baselines)
    if subtract_baseline:
        compscan_power = np.hstack([remove_spikes(np.abs(scan.pol(pol)[:, band])) - scan.baseline(scan.timestamps)
                                 for scan in compscan.scans if scan.baseline])
        target_coords = rad2deg(np.hstack([scan.target_coords for scan in compscan.scans if scan.baseline]))
    else:
        compscan_power = np.hstack([remove_spikes(np.abs(scan.pol(pol)[:, band])) for scan in compscan.scans])
        target_coords = rad2deg(np.hstack([scan.target_coords for scan in compscan.scans]))

    # Show the locations of the scan samples themselves, with marker sizes indicating power values
    plot_marker_3d(target_coords[0], target_coords[1], compscan_power, ax=ax, color='b',  alpha=0.75)
    # Plot the fitted Gaussian beam function as contours
    if compscan.beam:
        if compscan.beam.is_valid:
            ell_type, center_type = 'r-', 'r+'
        else:
            ell_type, center_type = 'y-', 'y+'
        var = fwhm_to_sigma(compscan.beam.width) ** 2.0
        if np.isscalar(var):
            var = [var, var]
        ellipses = gaussian_ellipses(compscan.beam.center, np.diag(var), contour=levels)
        for ellipse in ellipses:
            ax.plot(rad2deg(ellipse[:, 0]), rad2deg(ellipse[:, 1]), ell_type, lw=2)
        expected_var = 2 * [fwhm_to_sigma(compscan.beam.expected_width) ** 2.0]
        expected_ellipses = gaussian_ellipses(compscan.beam.center, np.diag(expected_var), contour=levels)
        for ellipse in expected_ellipses:
            ax.plot(rad2deg(ellipse[:, 0]), rad2deg(ellipse[:, 1]), 'k--', lw=2)
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
        total_power = np.hstack([remove_spikes(np.abs(scan.pol('I')[:, band])) for scan in compscan.scans])
        target_coords = np.hstack([scan.target_coords for scan in compscan.scans])
        center_time = np.median(np.hstack([scan.timestamps for scan in compscan.scans]))
        # Instantaneous mount coordinates are back on the sphere, but at a single time instant for all points
        mount_coords = list(compscan.target.plane_to_sphere(target_coords[0], target_coords[1],
                                                            center_time, dataset.antenna))
        # Obtain ellipses and center, and unwrap az angles for all objects simultaneously to ensure they stay together
        if compscan.beam:
            var = fwhm_to_sigma(compscan.beam.width) ** 2.0
            if np.isscalar(var):
                var = [var, var]
            target_ellipses = gaussian_ellipses(compscan.beam.center, np.diag(var), contour=levels)
            mount_ellipses = list(compscan.target.plane_to_sphere(target_ellipses[:, :, 0], target_ellipses[:, :, 1],
                                                                  center_time, dataset.antenna))
            mount_center = list(compscan.target.plane_to_sphere(compscan.beam.center[0], compscan.beam.center[1],
                                                                center_time, dataset.antenna))
            all_az = np.concatenate((mount_coords[0], [mount_center[0]], mount_ellipses[0].flatten()))
            all_az = minimise_angle_wrap(all_az)
            mount_coords[0] = all_az[:len(mount_coords[0])]
            mount_center[0] = all_az[len(mount_coords[0])]
            mount_ellipses[0] = all_az[len(mount_coords[0]) + 1:].reshape(mount_ellipses[0].shape[:2])
        else:
            mount_coords[0] = minimise_angle_wrap(mount_coords[0])

        # Show the locations of the scan samples themselves, with marker sizes indicating power values
        plot_marker_3d(rad2deg(mount_coords[0]), rad2deg(mount_coords[1]), total_power, ax=ax, alpha=0.75)
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
#--- FUNCTION :  plot_measured_beam_pattern
#--------------------------------------------------------------------------------------------------

def plot_measured_beam_pattern(compscan, pol='I', band=0, subtract_baseline=True,
                               add_samples=True, add_colorbar=True, ax=None, **kwargs):
    """Plot measured beam pattern contained in compound scan.

    Parameters
    ----------
    compscan : :class:`compoundscan.CompoundScan` object
        Compound scan object to plot
    pol : {'I', 'Q', 'U', 'V', 'HH', 'VV', 'XX', 'YY'}, optional
        The coherency / Stokes parameter to display (must be real)
    band : int, optional
        Frequency band to plot
    subtract_baseline : {True, False}, optional
        True to subtract baselines (only scans with baselines are then shown)
    add_samples : {True, False}, optional
        True if scan sample locations are to be added
    add_colorbar : {True, False}, optional
        True if color bar indicating contour levels is to be added
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

    Raises
    ------
    ValueError
        If *pol* is not one of the allowed names

    """
    if ax is None:
        ax = plt.gca()
    # Extract beam pattern as smoothed data on a regular grid
    x, y, power = extract_measured_beam(compscan, pol, band, subtract_baseline)
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
    ax.set_title('Pol %s' % pol)
    return ax, cset
