"""Statistics routines."""

import copy

import numpy as np
import scipy.signal as signal
import scipy.stats as stats

#--------------------------------------------------------------------------------------------------
#--- CLASS :  MuSigmaArray
#--------------------------------------------------------------------------------------------------

class MuSigmaArray(np.ndarray):
    """Container that bundles mean and standard deviation of N-dimensional data.
    
    This is a subclass of numpy.ndarray, which adds a :attr:`sigma` data member.
    The idea is that the main array is the mean, while :atrr:`sigma` contains
    the standard deviation of each corresponding element of the main array.
    Sigma should therefore have the same shape as the main array (mean). This
    approach allows the object itself to be the mean (e.g. ``x``), while the
    standard deviation can be accessed as ``x.sigma``. This makes the standard
    use of the mean (required in most calculations) cleaner, while still
    allowing the mean and standard deviation to travel together instead of as
    two arrays. The mean is also accessible as ``x.mu``.

    Alternative solutions:
    - A class with .mu and .sigma arrays (cumbersome when using mu)
    - A dict with 'mu' and 'sigma' keys and array values (ditto)
    - A tuple containing two arrays (mean still has to be extracted first)
    - Extending the mu array to contain another dimension (can be misleading)
    
    Parameters
    ----------
    mu : array-like
        The mean array (which becomes the main array of this object)
    sigma : array-like, optional
        The standard deviation array (default=None)
    
    Raises
    ------
    TypeError
        If *sigma* does not have the same shape as *mu*
    
    Notes
    -----
    The class has both a creator (__new__) and initialiser (__init__) method.
    The former casts the *mu* parameter to the :class:`MuSigmaArray` class,
    while the latter stores the *sigma* parameter and verifies that its
    shape is the same as that of *mu*. The :attr:`mu` and :attr:`sigma`
    attributes are actually handled via properties.
    
    """
    def __init__(self, mu, sigma=None):
        # Standard deviation of each element in main array (internal variable set via property).
        self._sigma = None
        # Standard deviation of each element in main array (property).
        self.sigma = sigma
    
    def __new__(cls, mu, sigma=None):
        """Object creation, which casts the mu array to the current subclass."""
        return np.asarray(mu).view(cls)
    
    # pylint: disable-msg=E0211,E0202,W0612,W0142,W0613
    def mu():
        """Class method which creates mean property.
        
        This is a nice way to create Python properties. It prevents clutter of
        the class namespace with getter and setter methods, while being more
        readable than lambda notation. The function below is effectively hidden
        by giving it the same name as the eventual property. Pylint gets queasy
        here, for obvious reasons.
        
        Returns
        -------
        local : dict
            Dictionary with property getter and setter methods, and doc string
        
        """
        doc = 'Mean array.'
        def fget(self):
            return self.view(np.ndarray)
        def fset(self, value):
            self = value
        return locals()
    # Mean array. This is merely for convenience, to restrict the object to be an numpy.ndarray.
    # Normal access to the object also provides the mean.
    mu = property(**mu())
    
    # pylint: disable-msg=E0211,E0202,W0612,W0142,W0212
    def sigma():
        """Class method which creates sigma property.
        
        See the docstring of :meth:`mu` for more details.
        
        Returns
        -------
        local : dict
            Dictionary with property getter and setter methods, and doc string
        
        """
        doc = 'Standard deviation of each element in main array.'
        def fget(self):
            return self._sigma
        def fset(self, value):
            if value != None:
                value = np.asarray(value)
                if value.shape != self.shape:
                    raise TypeError("X.sigma should have the same shape as X (i.e. %s instead of %s )" %
                                    (self.shape, value.shape))
            self._sigma = value
        return locals()
    # Standard deviation of each element in main array (property).
    sigma = property(**sigma())
    
    def __repr__(self):
        """Official string representation."""
        return self.__class__.__name__ + '(' + repr(self.mu) + ',' + repr(self.sigma) + ')'

    def __str__(self):
        """Informal string representation."""
        return 'mu    = ' + str(self.mu) + '\nsigma = ' + str(self.sigma)

    def __getitem__(self, value):
        """Index both arrays at the same time."""
        if self.sigma == None:
            return MuSigmaArray(self.mu[value], None)
        else:
            return MuSigmaArray(self.mu[value], self.sigma[value])

    def __getslice__(self, first, last):
        """Index both arrays at the same time."""
        if self.sigma == None:
            return MuSigmaArray(self.mu[first:last], None)
        else:
            return MuSigmaArray(self.mu[first:last], self.sigma[first:last])
    
    def __copy__(self):
        """Shallow copy operation."""
        return MuSigmaArray(self.mu, self.sigma)
    
    def __deepcopy__(self, memo):
        """Deep copy operation."""
        return MuSigmaArray(copy.deepcopy(self.mu, memo), copy.deepcopy(self.sigma, memo))
    
def ms_concatenate(msa_list):
    """Concatenate MuSigmaArrays.
    
    Parameters
    ----------
    msa_list : list of :class:`MuSigmaArray` objects
        List of MuSigmaArrays to concatenate
        
    Returns
    -------
    msa : :class:`MuSigmaArray` object
        MuSigmaArray that is concatenation of list
    
    """
    mu_list = [msa.mu for msa in msa_list]
    sigma_list = [msa.sigma for msa in msa_list]
    # If any sigma is None, discard the rest
    if None in sigma_list:
        return MuSigmaArray(np.concatenate(mu_list), None)
    else:
        return MuSigmaArray(np.concatenate(mu_list), np.concatenate(sigma_list))

def ms_hstack(msa_list):
    """Stack MuSigmaArrays horizontally.
    
    Parameters
    ----------
    msa_list : list of :class:`MuSigmaArray` objects
        List of MuSigmaArrays to stack
        
    Returns
    -------
    msa : :class:`MuSigmaArray` object
        MuSigmaArray that is horizontal stack of list
    
    """
    mu_list = [msa.mu for msa in msa_list]
    sigma_list = [msa.sigma for msa in msa_list]
    # If any sigma is None, discard the rest
    if None in sigma_list:
        return MuSigmaArray(np.hstack(mu_list), None)
    else:
        return MuSigmaArray(np.hstack(mu_list), np.hstack(sigma_list))

def ms_vstack(msa_list):
    """Stack MuSigmaArrays vertically.
    
    Parameters
    ----------
    msa_list : list of :class:`MuSigmaArray` objects
        List of MuSigmaArrays to stack
        
    Returns
    -------
    msa : :class:`MuSigmaArray` object
        MuSigmaArray that is vertical stack of list
    
    """
    mu_list = [msa.mu for msa in msa_list]
    sigma_list = [msa.sigma for msa in msa_list]
    # If any sigma is None, discard the rest
    if None in sigma_list:
        return MuSigmaArray(np.vstack(mu_list), None)
    else:
        return MuSigmaArray(np.vstack(mu_list), np.vstack(sigma_list))

def mu_sigma(data, axis=0):
    """Determine second-order statistics from data.
    
    Convenience function to return second-order statistics of data along given
    axis as a MuSigmaArray.
    
    Parameters
    ----------
    data : array
        Numpy data array of arbitrary shape
    axis : int, optional
        Index of axis along which stats are calculated (will be averaged away
        in the process) [default=0]
    
    Returns
    -------
    msa : :class:`MuSigmaArray` object
        MuSigmaArray containing data stats, of same dimension as data, but
        without given axis
    
    """
    return MuSigmaArray(data.mean(axis=axis), data.std(axis=axis))

def robust_mu_sigma(data, axis=0):
    """Determine second-order statistics from data, using robust statistics.
    
    Convenience function to return second-order statistics of data along given
    axis as a MuSigmaArray. These are determined via the median and
    interquartile range.
    
    Parameters
    ----------
    data : array
        Numpy data array of arbitrary shape
    axis : int, optional
        Index of axis along which stats are calculated (will be averaged away
        in the process) [default=0]
    
    Returns
    -------
    msa : :class:`MuSigmaArray` object
        MuSigmaArray containing data stats, of same dimension as data, but
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
    return MuSigmaArray(perc50, iqr_to_std * (perc75 - perc25))

def periodic_mu_sigma(data, axis=0, period=2.0 * np.pi):
    """Determine second-order statistics of periodic (angular, directional) data.
    
    Convenience function to return second-order statistics of data along given
    axis as a MuSigmaArray. This handles periodic variables, which exhibit the
    problem of wrap-around and therefore are unsuited for the normal mu_sigma
    function. The period with which the values repeat can be explicitly
    specified, otherwise the data is assumed to be radians. The mean is in the
    range -period/2 ... period/2, and the maximum standard deviation is about
    period/4. The approach in [1]_ is used.

    .. [1] R. J. Yamartino, "A Comparison of Several 'Single-Pass' Estimators
       of the Standard Deviation of Wind Direction," Journal of Climate and
       Applied Meteorology, vol. 23, pp. 1362-1366, 1984.
    
    Parameters
    ----------
    data : array
        Numpy array of arbitrary shape, containing angles (typically in radians)
    axis : int, optional
        Index of axis along which stats are calculated (will be averaged away
        in the process) [default=0]
    period : float, optional
        Period with which data values repeat [default is 2.0 * pi]
    
    Returns
    -------
    msa : :class:`MuSigmaArray` object
        MuSigmaArray containing data stats, of same dimension as data, but
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
    delta_ang = data - mu
    # Wrap angle differences into interval -pi ... pi
    delta_ang = (delta_ang + np.pi) % (2.0 * np.pi) - np.pi
    # Calculate variance using standard formula with a second correction term
    sigma2 = (delta_ang ** 2.0).mean(axis=0) - (delta_ang.mean(axis=0) ** 2.0)
    # Scale answers back to original data range
    return MuSigmaArray(mu / scale, np.sqrt(sigma2) / scale)

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
    delta_ang = angles - np.expand_dims(mu, axis)
    # Wrap angle differences into interval -pi ... pi
    delta_ang = (delta_ang + np.pi) % (2.0 * np.pi) - np.pi
    return delta_ang + np.expand_dims(mu, axis)
    
#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  remove_spikes
#--------------------------------------------------------------------------------------------------

def remove_spikes(data, axis=0, kernel_size=7, outlier_sigma=5.0):
    """Remove outliers from data, replacing them with a local median value.
    
    The data is median-filtered along the specified axis, and any data values
    that deviate significantly from the local median is replaced with the median.
    
    Parameters
    ----------
    data : array-like
        N-dimensional numpy array containing data to clean
    axis : int, optional
        Axis along which to perform median, between 0 and N-1
    kernel_size : int, optional
        Kernel size for median filter, should be an odd integer
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
