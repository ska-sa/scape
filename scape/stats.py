"""Statistics routines."""

import numpy as np
import scipy.signal as signal
import scipy.stats as stats

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  angle_wrap
#--------------------------------------------------------------------------------------------------

def angle_wrap(angle, period=2.0 * np.pi):
    """Wrap angle into interval centred on zero.

    This wraps the *angle* into the interval -*period* / 2 ... *period* / 2.

    """
    return (angle + 0.5 * period) % period - 0.5 * period

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  minimise_angle_wrap
#--------------------------------------------------------------------------------------------------

def minimise_angle_wrap(angles, axis=0):
    """Minimise wrapping of angles to improve interpretation.

    Move wrapping point as far away as possible from mean angle on given axis.
    The main use of this function is to improve the appearance of angle plots.

    Parameters
    ----------
    angles : array-like
        Array of angles to unwrap, in radians
    axis : int, optional
        Axis along which angle wrap is evaluated. Plots along this axis will
        typically improve in appearance.

    Returns
    -------
    angles : array
        Array of same shape as input array, with angles wrapped around new point

    """
    angles = np.asarray(angles)
    # Calculate a "safe" mean on the unit circle
    mu = np.arctan2(np.sin(angles).mean(axis=axis), np.cos(angles).mean(axis=axis))
    # Wrap angle differences into interval -pi ... pi
    delta_ang = angle_wrap(angles - np.expand_dims(mu, axis))
    return delta_ang + np.expand_dims(mu, axis)

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  mu_sigma
#--------------------------------------------------------------------------------------------------

def mu_sigma(data, axis=0):
    """Determine second-order statistics from data.

    Convenience function to return second-order statistics of data along given
    axis.

    Parameters
    ----------
    data : array-like
        Numpy data array (or equivalent) of arbitrary shape
    axis : int, optional
        Index of axis along which stats are calculated (will be averaged away
        in the process)

    Returns
    -------
    mu, sigma : array
        Mean and standard deviation as arrays of same shape as *data*, but
        without given axis

    """
    return np.mean(data, axis=axis), np.std(data, axis=axis)

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  robust_mu_sigma
#--------------------------------------------------------------------------------------------------

def robust_mu_sigma(data, axis=0):
    """Determine second-order statistics from data, using robust statistics.

    Convenience function to return second-order statistics of data along given
    axis. These are determined via the median and interquartile range.

    Parameters
    ----------
    data : array-like
        Numpy data array (or equivalent) of arbitrary shape
    axis : int, optional
        Index of axis along which stats are calculated (will be averaged away
        in the process)

    Returns
    -------
    mu, sigma : array
        Mean and standard deviation as arrays of same shape as *data*, but
        without given axis

    """
    data = np.asarray(data)
    # Create sequence of axis indices with specified axis at the front, and the rest following it
    move_axis_to_front = range(len(data.shape))
    move_axis_to_front.remove(axis)
    move_axis_to_front = [axis] + move_axis_to_front
    # Create copy of data sorted along specified axis, and reshape so that the specified axis becomes the first one
    sorted_data = np.sort(data, axis=axis).transpose(move_axis_to_front)
    # Obtain quartiles
    perc25 = sorted_data[int(0.25 * len(sorted_data))]
    perc50 = sorted_data[int(0.50 * len(sorted_data))]
    perc75 = sorted_data[int(0.75 * len(sorted_data))]
    # Conversion factor from interquartile range to standard deviation (based on normal pdf)
    iqr_to_std = 0.741301109253
    return perc50, iqr_to_std * (perc75 - perc25)

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  periodic_mu_sigma
#--------------------------------------------------------------------------------------------------

def periodic_mu_sigma(data, axis=0, period=2.0 * np.pi):
    """Determine second-order statistics of periodic (angular, directional) data.

    Convenience function to return second-order statistics of data along given
    axis. This handles periodic variables, which exhibit the problem of
    wrap-around and therefore are unsuited for the normal mu_sigma function. The
    period with which the values repeat can be explicitly specified, otherwise
    the data is assumed to be radians. The mean is in the range -period/2 ...
    period/2, and the maximum standard deviation is about period/4.

    Parameters
    ----------
    data : array-like
        Numpy array (or equivalent) of arbitrary shape, containing angles
    axis : int, optional
        Index of axis along which stats are calculated (will be averaged away
        in the process)
    period : float, optional
        Period with which data values repeat

    Returns
    -------
    mu, sigma : array
        Mean and standard deviation as arrays of same shape as *data*, but
        without given axis

    Notes
    -----
    The approach in [1]_ is used.

    .. [1] R. J. Yamartino, "A Comparison of Several 'Single-Pass' Estimators
       of the Standard Deviation of Wind Direction," Journal of Climate and
       Applied Meteorology, vol. 23, pp. 1362-1366, 1984.

    """
    data = np.asarray(data, dtype='double')
    # Create sequence of axis indices with specified axis at the front, and the rest following it
    move_axis_to_front = range(len(data.shape))
    move_axis_to_front.remove(axis)
    move_axis_to_front = [axis] + move_axis_to_front
    # Create copy of data, and reshape so that the specified axis becomes the first one
    data = data.copy().transpose(move_axis_to_front)
    # Scale data so that one period becomes 2*pi, the natural period for angles
    scale = 2.0 * np.pi / period
    data *= scale
    # Calculate a "safe" mean on the unit circle
    mu = np.arctan2(np.sin(data).mean(axis=0), np.cos(data).mean(axis=0))
    # Wrap angle differences into interval -pi ... pi
    delta_ang = angle_wrap(data - mu)
    # Calculate variance using standard formula with a second correction term
    sigma2 = (delta_ang ** 2.0).mean(axis=0) - (delta_ang.mean(axis=0) ** 2.0)
    # Scale answers back to original data range
    return mu / scale, np.sqrt(sigma2) / scale

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  remove_spikes
#--------------------------------------------------------------------------------------------------

def remove_spikes(data, axis=0, spike_width=3, outlier_sigma=5.0):
    """Remove outliers from data, replacing them with a local median value.

    The data is median-filtered along the specified axis, and any data values
    that deviate significantly from the local median is replaced with the median.

    Parameters
    ----------
    data : array-like
        N-dimensional numpy array containing data to clean
    axis : int, optional
        Axis along which to perform median, between 0 and N-1
    spike_width : int, optional
        Spikes with widths up to this limit (in samples) will be removed. A size
        of <= 0 implies no spike removal. The kernel size for the median filter
        will be 2 * spike_width + 1.
    outlier_sigma : float, optional
        Multiple of standard deviation that indicates an outlier

    Returns
    -------
    cleaned_data : array
        N-dimensional numpy array of same shape as original data, with outliers
        removed

    Notes
    -----
    This is very similar to a *Hampel filter*, also known as a *decision-based
    filter* or three-sigma edit rule combined with a Hampel outlier identifier.

    .. todo::

       TODO: Make this more like a Hampel filter by making MAD time-variable too.

    """
    data = np.atleast_1d(data)
    kernel_size = 2 * max(int(spike_width), 0) + 1
    if kernel_size == 1:
        return data
    # Median filter data along the desired axis, with given kernel size
    kernel = np.ones(data.ndim, dtype='int32')
    kernel[axis] = kernel_size
    # Medfilt now seems to upcast 32-bit floats to doubles - convert it back to floats...
    filtered_data = np.asarray(signal.medfilt(data, kernel), data.dtype)
    # The deviation is measured relative to the local median in the signal
    abs_dev = np.abs(data - filtered_data)
    # Calculate median absolute deviation (MAD)
    med_abs_dev = np.expand_dims(np.median(abs_dev, axis), axis)
#    med_abs_dev = signal.medfilt(abs_dev, kernel)
    # Assuming normally distributed deviations, this is a robust estimator of the standard deviation
    estm_stdev = 1.4826 * med_abs_dev
    # Identify outliers (again based on normal assumption), and replace them with local median
    outliers = (abs_dev > outlier_sigma * estm_stdev)
    cleaned_data = data.copy()
    cleaned_data[outliers] = filtered_data[outliers]
    return cleaned_data

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  chi2_conf_interval
#--------------------------------------------------------------------------------------------------

def chi2_conf_interval(dof, mean=1.0, sigma=3.0):
    """Confidence interval for chi-square distribution.

    Return lower and upper limit of confidence interval of chi-square
    distribution, defined in terms of a normal confidence interval. That is,
    given *sigma*, which is a multiple of the standard deviation, calculate the
    probability mass within the interval [-sigma, sigma] for a standard normal
    distribution, and return the interval with the same probability mass for the
    chi-square distribution with *dof* degrees of freedom. The interval is
    scaled by ``(mean/dof)``, which enforces the given mean and implies a
    standard deviation of ``mean*sqrt(2/dof)``. This represents the distribution
    of the power estimator $P = 1/N \sum_{i=1}^{N} x_i^2$, with N = *dof* and
    zero-mean Gaussian voltages $x_i$ with variance *mean*.

    Parameters
    ----------
    dof : array-like or float
        Degrees of freedom (number of independent samples summed to form chi^2
        variable)
    mean : array-like or float, optional
        Desired mean of chi^2 distribution
    sigma : array-like or float, optional
        Multiple of standard deviation, used to specify size of required
        confidence interval

    Returns
    -------
    lower : array or float
        Lower limit of confidence interval (numpy array if any input is one)
    upper : array or float
        Upper limit of confidence interval (numpy array if any input is one)

    Notes
    -----
    The advantage of this approach is that it uses a well-known concept to
    specify the interval (multiples of standard deviation), while returning
    valid intervals for all values of *dof*. For (very) large values of *dof*,
    (lower, upper) will be close to

    (mean - sigma * mean*sqrt(2/dof), mean + sigma * mean*sqrt(2/dof)),

    as the chi-square distribution will be approximately normal. For small *dof*
    or large *sigma*, however, this approximation breaks down and may lead to
    negative lower values, for example.

    """
    if not np.isscalar(dof):
        dof = np.atleast_1d(np.asarray(dof))
    if not np.isscalar(mean):
        mean = np.atleast_1d(np.asarray(mean))
    if not np.isscalar(sigma):
        sigma = np.atleast_1d(np.asarray(sigma))
    # Ensure degrees of freedom is positive integer >= 1
    dof = np.array(np.clip(np.floor(dof), 1.0, np.inf), dtype=np.int)
    chi2_rv = stats.chi2(dof)
    normal_rv = stats.norm()
    # Translate normal conf interval to chi^2 distribution, maintaining the probability inside interval
    lower = chi2_rv.ppf(normal_rv.cdf(-sigma)) * mean / dof
    upper = chi2_rv.ppf(normal_rv.cdf(sigma)) * mean / dof
    return lower, upper
