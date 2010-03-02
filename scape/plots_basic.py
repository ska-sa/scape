"""Basic plotting routines, used to create canned plots at a higher level."""

import logging

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

logger = logging.getLogger("scape.plots_basic")

def ordinal_suffix(n):
    """Returns the ordinal suffix of integer *n* as a string."""
    if n % 100 in [11, 12, 13]:
        return 'th'
    else:
        return {1 : 'st', 2 : 'nd', 3 : 'rd'}.get(n % 10, 'th')

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  plot_line_segments
#--------------------------------------------------------------------------------------------------

def plot_line_segments(segments, labels=None, width=0.0, compact=True, add_breaks=True,
                       monotonic_axis='x', ax=None, **kwargs):
    """Plot sequence of line segments.

    This plots a sequence of line segments (of possibly varying length) on a
    single set of axes. Usually, one of the axes is considered *monotonic*,
    which means that the line segment coordinates along that axis increase
    monotonically through the sequence of segments. The classic example of such
    a plot is when the line segments represent time series data with time on the
    monotonic x-axis.

    Each segment may be labelled by a text string next to it. If *compact* is
    True, there will be no gaps between the segments along the monotonic axis.
    The tick labels on this axis are modified to reflect the original (padded)
    values. If *add_breaks* is True, the breaks between segments along the
    monotonic axis are indicated by dashed lines. If there is no monotonic axis,
    all these features (text labels, compaction and break lines) are disabled.

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
        Corresponding sequence of text labels to add next to each segment along
        monotonic axis (only makes sense if there is such an axis)
    width : float, optional
        If non-zero, replace contiguous line with staircase levels of specified
        width along x-axis
    compact : {True, False}, optional
        Plot with no gaps between segments along monotonic axis (only makes
        sense if there is such an axis)
    add_breaks : {True, False}, optional
        Add vertical (or horizontal) lines to indicate breaks between segments
        along monotonic axis (only makes sense if there is such an axis)
    monotonic_axis : {'x', 'y', None}, optional
        Monotonic axis, along which segment coordinate increases monotonically
    ax : :class:`matplotlib.axes.Axes` object, optional
        Matplotlib axes object to receive plot (default is current axes)
    kwargs : dict, optional
        Extra keyword arguments are passed on to line collection constructor

    Returns
    -------
    segment_lines : :class:`matplotlib.collections.LineCollection` object
        Collection of segment lines
    break_lines : :class:`matplotlib.collections.LineCollection` object, or None
        Collection of break lines separating the segments
    text_labels : list of :class:`matplotlib.text.Text` objects
        List of added text labels

    """
    if ax is None:
        ax = plt.gca()
    if labels is None:
        labels = []
    # Disable features that depend on a monotonic axis
    if monotonic_axis is None:
        labels, compact, add_breaks = [], False, False

    # Get segment startpoints and endpoints along monotonic axis
    if monotonic_axis == 'x':
        start = np.array([np.asarray(segm)[:, 0].min() for segm in segments])
        end = np.array([np.asarray(segm)[:, 0].max() for segm in segments])
    else:
        start = np.array([np.asarray(segm)[:, 1].min() for segm in segments])
        end = np.array([np.asarray(segm)[:, 1].max() for segm in segments])

    if compact:
        # Calculate offset between original and compacted coordinate, and adjust coordinates accordingly
        compacted_end = np.cumsum(end - start)
        compacted_start = np.array([0.0] + compacted_end[:-1].tolist())
        offset = start - compacted_start
        start, end = compacted_start, compacted_end
        # Redefine monotonic axis label formatter to add appropriate offset to label value, depending on segment
        class SegmentedScalarFormatter(mpl.ticker.ScalarFormatter):
            """Expand x axis value to correct segment before labelling."""
            def __init__(self, useOffset=True, useMathText=False):
                mpl.ticker.ScalarFormatter.__init__(self, useOffset, useMathText)
            def __call__(self, x, pos=None):
                segment = max(start.searchsorted(x, side='right') - 1, 0)
                return mpl.ticker.ScalarFormatter.__call__(self, x + offset[segment], pos)
        # Subtract segment offsets from appropriate coordinate
        if monotonic_axis == 'x':
            segments = [np.column_stack((np.asarray(segm)[:, 0] - offset[n], np.asarray(segm)[:, 1]))
                        for n, segm in enumerate(segments)]
            ax.xaxis.set_major_formatter(SegmentedScalarFormatter())
        else:
            segments = [np.column_stack((np.asarray(segm)[:, 0], np.asarray(segm)[:, 1] - offset[n]))
                        for n, segm in enumerate(segments)]
            ax.yaxis.set_major_formatter(SegmentedScalarFormatter())

    # Plot the segment lines as a collection
    segment_lines = mpl.collections.LineCollection(segments, **kwargs)
    ax.add_collection(segment_lines)

    segment_centers, breaks = (start + end) / 2, (start[1:] + end[:-1]) / 2
    break_lines, text_labels = None, []
    if monotonic_axis == 'x':
        # Break lines and labels have x coordinates fixed to data and y coordinates fixed to axes (like axvline)
        transFixedY = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        for n, label in enumerate(labels):
            text_labels.append(ax.text(segment_centers[n], 0.02, label, transform=transFixedY,
                                       ha='center', va='bottom', clip_on=True))
        if add_breaks:
            break_lines = mpl.collections.LineCollection([[(s, 0), (s, 1)] for s in breaks], colors='k',
                                                         linewidths=0.5, linestyles='dotted', transform=transFixedY)
            ax.add_collection(break_lines)
        # Only set monotonic axis limits
        ax.set_xlim(start[0], end[-1])
        ax.autoscale_view(scalex=False)
    elif monotonic_axis == 'y':
        # Break lines and labels have x coordinates fixed to axes and y coordinates fixed to data (like axhline)
        transFixedX = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
        for n, label in enumerate(labels):
            text_labels.append(ax.text(0.02, segment_centers[n], label, transform=transFixedX,
                                       ha='left', va='center', clip_on=True))
        if add_breaks:
            break_lines = mpl.collections.LineCollection([[(0, s), (1, s)] for s in breaks], colors='k',
                                                         linewidths=0.5, linestyles='dotted', transform=transFixedX)
            ax.add_collection(break_lines)
        ax.set_ylim(start[0], end[-1])
        ax.autoscale_view(scaley=False)
    else:
        ax.autoscale_view()

    return segment_lines, break_lines, text_labels

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  plot_segments
#--------------------------------------------------------------------------------------------------

def plot_segments(x, y, z=None, labels=None, width=0.0, compact=True, add_breaks=True,
                  monotonic_axis='auto', ax=None, **kwargs):
    """Plot sequence of line segments, bars or images.

    This plots a sequence of bars. Usually, one of the axes is considered *monotonic*,
    which means that the line segment coordinates along that axis increase
    monotonically through the sequence of segments. The classic example of such
    a plot is when the line segments represent time series data with time on the
    monotonic x-axis.

    Each segment may be labelled by a text string next to it. If *compact* is
    True, there will be no gaps between the segments along the monotonic axis.
    The tick labels on this axis are modified to reflect the original (padded)
    values. If *add_breaks* is True, the breaks between segments along the
    monotonic axis are indicated by dashed lines. If there is no monotonic axis,
    all these features (text labels, compaction and break lines) are disabled.

    Parameters
    ----------
    x, y : sequence of array-like, shape (N_k,)
        Coordinates of bars
    labels : sequence of strings, optional
        Corresponding sequence of text labels to add next to each segment along
        monotonic axis (only makes sense if there is such an axis)
    width : float, optional
        If non-zero, replace contiguous line with staircase levels of specified
        width along x-axis
    compact : {True, False}, optional
        Plot with no gaps between segments along monotonic axis (only makes
        sense if there is such an axis)
    add_breaks : {True, False}, optional
        Add vertical (or horizontal) lines to indicate breaks between segments
        along monotonic axis (only makes sense if there is such an axis)
    monotonic_axis : {'auto', 'x', 'y', None}, optional
        Monotonic axis, along which segment coordinate increases monotonically
        (automatically detected by default)
    ax : :class:`matplotlib.axes.Axes` object, optional
        Matplotlib axes object to receive plot (default is current axes)
    kwargs : dict, optional
        Extra keyword arguments are passed on to line collection constructor

    Returns
    -------
    segments : :class:`matplotlib.collections.LineCollection` object
        Collection of segment lines
    text_labels : list of :class:`matplotlib.text.Text` objects
        List of added text labels
    break_lines : :class:`matplotlib.collections.LineCollection` object, or None
        Collection of break lines separating the segments

    """
    if ax is None:
        ax = plt.gca()
    if labels is None:
        labels = []
    # Ensure that inputs are sequences of coordinate sequences
    x = [x] if np.isscalar(x[0]) else x
    y = [y] if np.isscalar(y[0]) else y
    # Attempt to detect monotonic axis if requested - look for axis with 1-dimensional data that is sorted too
    if monotonic_axis == 'auto':
        if np.all(np.array([np.ndim(xsegm) for xsegm in x]) == 1) and \
           np.abs(np.sign(np.diff(np.hstack(x))).sum()) == np.sum([len(xsegm) for xsegm in x]) - 1:
            monotonic_axis = 'x'
        elif np.all(np.array([np.ndim(ysegm) for ysegm in y]) == 1) and \
           np.abs(np.sign(np.diff(np.hstack(y))).sum()) == np.sum([len(ysegm) for ysegm in y]) - 1:
            monotonic_axis = 'y'
        else:
            monotonic_axis = None
    # Disable features that depend on a monotonic axis and multiple segments
    if monotonic_axis is None or ((len(x) == 1) and (len(y) == 1)):
        compact, add_breaks = False, False

    # Double-check coordinate shapes and select appropriate plot type
    if np.isscalar(x[0][0]) and np.isscalar(y[0][0]):
        plot_type = 'line'
    elif np.shape(x[0][0]) == () and np.shape(y[0][0]) == (2,):
        plot_type = 'barv'
    elif np.shape(x[0][0]) == (2,) and np.shape(y[0][0]) == ():
        plot_type = 'barh'
    if [np.shape(xsegm)[0] for xsegm in x] != [np.shape(ysegm)[0] for ysegm in y]:
        raise ValueError('Shape mismatch between x and y (segment lengths are %s vs %s)' %
                         ([np.shape(xsegm)[0] for xsegm in x], [np.shape(ysegm)[0] for ysegm in y]))

    # Get segment startpoints and endpoints along monotonic axis
    if monotonic_axis == 'x':
        if plot_type == 'barh':
            raise ValueError('X-axis cannot be monotonic when x-y data indicate horizontal bars')
        start = np.array([np.min(xsegm) for xsegm in x]) - width / 2.0
        end = np.array([np.max(xsegm) for xsegm in x]) + width / 2.0
    elif monotonic_axis == 'y':
        if plot_type == 'barv':
            raise ValueError('Y-axis cannot be monotonic when x-y data indicate vertical bars')
        start = np.array([np.min(ysegm) for ysegm in y]) - width / 2.0
        end = np.array([np.max(ysegm) for ysegm in y]) + width / 2.0

    if compact:
        # Calculate offset between original and compacted coordinate, and adjust coordinates accordingly
        compacted_end = np.cumsum(end - start)
        compacted_start = np.array([0.0] + compacted_end[:-1].tolist())
        offset = start - compacted_start
        start, end = compacted_start, compacted_end
        # Redefine monotonic axis label formatter to add appropriate offset to label value, depending on segment
        class SegmentedScalarFormatter(mpl.ticker.ScalarFormatter):
            """Expand x axis value to correct segment before labelling."""
            def __init__(self, useOffset=True, useMathText=False):
                mpl.ticker.ScalarFormatter.__init__(self, useOffset, useMathText)
            def __call__(self, x, pos=None):
                segment = max(start.searchsorted(x, side='right') - 1, 0)
                return mpl.ticker.ScalarFormatter.__call__(self, x + offset[segment], pos)
        # Subtract segment offsets from appropriate coordinate
        if monotonic_axis == 'x':
            x = [np.asarray(xsegm) - offset[n] for n, xsegm in enumerate(x)]
            ax.xaxis.set_major_formatter(SegmentedScalarFormatter())
        else:
            y = [np.asarray(ysegm) - offset[n] for n, ysegm in enumerate(y)]
            ax.yaxis.set_major_formatter(SegmentedScalarFormatter())

    if plot_type == 'line':
        segments = mpl.collections.LineCollection([zip(xsegm, ysegm) for xsegm, ysegm in zip(x, y)], **kwargs)
        ax.add_collection(segments)
    elif plot_type == 'barv':
        x = np.hstack(x)
        y1 = np.hstack([np.asarray(ysegm)[:, 0] for ysegm in y])
        y2 = np.hstack([np.asarray(ysegm)[:, 1] for ysegm in y])
        # Form makeshift rectangular patches
        xxx = np.array([x - 0.999 * width / 2, x + 0.999 * width / 2, x]).transpose().ravel()
        yyy1, yyy2 = np.repeat(y1, 3), np.repeat(y2, 3)
        mask = np.arange(len(xxx)) % 3 == 2
        segments = ax.fill_between(xxx, yyy1, yyy2, where=~mask, facecolors='0.8', edgecolors='0.8', **kwargs)
    elif plot_type == 'barh':
        x1 = np.hstack([np.asarray(xsegm)[:, 0] for xsegm in x])
        x2 = np.hstack([np.asarray(xsegm)[:, 1] for xsegm in x])
        y = np.hstack(y)
        # Form makeshift rectangular patches
        yyy = np.array([y - 0.999 * width / 2, y + 0.999 * width / 2, y]).transpose().ravel()
        xxx1, xxx2 = np.repeat(x1, 3), np.repeat(x2, 3)
        mask = np.arange(len(yyy)) % 3 == 2
        segments = ax.fill_betweenx(yyy, xxx1, xxx2, where=~mask, facecolors='0.8', edgecolors='0.8', **kwargs)

    text_labels, break_lines = [], None
    if monotonic_axis == 'x':
        # Break lines and labels have x coordinates fixed to data and y coordinates fixed to axes (like axvline)
        transFixedY = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        for n, label in enumerate(labels):
            text_labels.append(ax.text((start[n] + end[n]) / 2, 0.02, label, transform=transFixedY,
                                       ha='center', va='bottom', clip_on=True))
        if add_breaks:
            breaks = (start[1:] + end[:-1]) / 2
            break_lines = mpl.collections.LineCollection([[(s, 0), (s, 1)] for s in breaks], colors='k',
                                                         linewidths=0.5, linestyles='dotted', transform=transFixedY)
            ax.add_collection(break_lines)
        # Only set monotonic axis limits
        ax.set_xlim(start[0], end[-1])
        ax.autoscale_view(scalex=False)
    elif monotonic_axis == 'y':
        # Break lines and labels have x coordinates fixed to axes and y coordinates fixed to data (like axhline)
        transFixedX = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
        for n, label in enumerate(labels):
            text_labels.append(ax.text(0.02, (start[n] + end[n]) / 2, label, transform=transFixedX,
                                       ha='left', va='center', clip_on=True))
        if add_breaks:
            breaks = (start[1:] + end[:-1]) / 2
            break_lines = mpl.collections.LineCollection([[(0, s), (1, s)] for s in breaks], colors='k',
                                                         linewidths=0.5, linestyles='dotted', transform=transFixedX)
            ax.add_collection(break_lines)
        ax.set_ylim(start[0], end[-1])
        ax.autoscale_view(scaley=False)
    else:
        for n, label in enumerate(labels):
            text_labels.append(ax.text(x[n][len(x[n]) // 2], y[n][len(y[n]) // 2], label,
                               ha='center', va='center', clip_on=True, backgroundcolor='w'))
        ax.autoscale_view()

    return segments, text_labels, break_lines

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  plot_compacted_images
#--------------------------------------------------------------------------------------------------

def plot_compacted_images(imdata, xticks, labels=None, ylim=None, clim=None, grey_rows=None, ax=None):
    """Plot sequence of images in compacted form.

    This plots a sequence of 2-D arrays (with the same number of rows but
    possibly varying number of columns) as images on a single set of axes, with
    no gaps between the images along the *x* axis. Each image has an associated
    sequence of *x* ticks, with one tick per image column. The tick labels on the
    *x* axis are modified to reflect the original (padded) tick values, and the
    breaks between segments are indicated by vertical lines. Some of the image
    rows may optionally be greyed out (e.g. to indicate RFI-corrupted channels).

    Parameters
    ----------
    imdata : sequence of array-like, shape (M, N_k)
        Sequence of 2-D arrays (*image0*, *image1*, ..., *imagek*, ..., *imageK*)
        to be displayed as images, where each array has the same number of rows,
        *M*, but a potentially unique number of columns, *N_k*
    xticks : sequence of array-like, shape (N_k,)
        Sequence of 1-D arrays (*x_0*, *x_1*, ..., *x_k*, ..., *x_K*) serving as
        *x*-axis ticks for the corresponding images, where *x_k* has length *N_k*
    labels : sequence of strings, optional
        Corresponding sequence of text labels to add below each image
    ylim : sequence of 2 floats, or None, optional
        Shared *y* limit of images, as (*ymin*, *ymax*), based on their common
        rows (default is (1, M))
    clim : sequence of 2 floats, or None, optional
        Shared color limits of images, as (*vmin*, *vmax*). The default uses the
        the global minimum and maximum of all the arrays in *imdata*.
    grey_rows : sequence of integers, optional
        Sequence of indices of rows which will be greyed out in each image
        (default is no greyed-out rows)
    ax : :class:`matplotlib.axes.Axes` object, optional
        Matplotlib axes object to receive plot (default is current axes)

    Returns
    -------
    images : list of :class:`matplotlib.image.AxesImage` objects
        List of images
    border_lines : :class:`matplotlib.collections.LineCollection` object
        Collection of vertical lines separating the images
    text_labels : list of :class:`matplotlib.text.Text` objects
        List of added text labels

    """
    if ax is None:
        ax = plt.gca()
    if clim is None:
        cmin = np.min([im.min() for im in imdata])
        cmax = np.max([im.max() for im in imdata])
        crange = cmax - cmin
        if crange == 0.0:
            crange = 1.0
        clim = (cmin - 0.05 * crange, cmax + 0.05 * crange)
    if ylim is None:
        ylim = (1, imdata[0].shape[0])
    if labels is None:
        labels = []
    start = np.array([x.min() for x in xticks])
    end = np.array([x.max() for x in xticks])
    compacted_start = [0.0] + np.cumsum(end - start).tolist()
    x_origin = start.min()
    images = []
    for k, (x, im) in enumerate(zip(xticks, imdata)):
        colornorm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])
        image_data = mpl.cm.jet(colornorm(im))
        if grey_rows is not None:
            image_data_grey = mpl.cm.gray(colornorm(im))
            image_data[grey_rows, :, :] = image_data_grey[grey_rows, :, :]
        images.append(ax.imshow(np.uint8(np.round(image_data * 255)), aspect='auto',
                                interpolation='nearest', origin='lower',
                                extent=(x[0] - start[k] + compacted_start[k],
                                        x[-1] - start[k] + compacted_start[k], ylim[0], ylim[1])))
    # These border lines have x coordinates fixed to the data and y coordinates fixed to the axes
    transFixedY = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    border_lines = mpl.collections.LineCollection([[(s, 0), (s, 1)] for s in compacted_start[1:-1]],
                                                  colors='k', linewidths=2.0, linestyles='solid',
                                                  transform=transFixedY)
    ax.add_collection(border_lines)
    ax.axis([compacted_start[0], compacted_start[-1], ylim[0], ylim[1]])
    text_labels = []
    for k, label in enumerate(labels):
        text_labels.append(ax.text(np.mean(compacted_start[k:k+2]), 0.02, label, transform=transFixedY,
                                   ha='center', va='bottom', clip_on=True, color='w'))
    # Redefine x-axis label formatter to display the correct time for each segment
    class SegmentedScalarFormatter(mpl.ticker.ScalarFormatter):
        """Expand x axis value to correct segment before labelling."""
        def __init__(self, useOffset=True, useMathText=False):
            mpl.ticker.ScalarFormatter.__init__(self, useOffset, useMathText)
        def __call__(self, x, pos=None):
            if x > compacted_start[0]:
                segment = (compacted_start[:-1] < x).nonzero()[0][-1]
                x = x - compacted_start[segment] + start[segment] - x_origin
            return mpl.ticker.ScalarFormatter.__call__(self, x, pos)
    ax.xaxis.set_major_formatter(SegmentedScalarFormatter())
    return images, border_lines, text_labels

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
