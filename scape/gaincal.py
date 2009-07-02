"""Gain calibration via noise injection."""

import numpy as np

from .fitting import Spline1DFit
from .fitting import randomise as fitting_randomise
from .stats import MuSigmaArray, robust_mu_sigma, minimise_angle_wrap

#--------------------------------------------------------------------------------------------------
#--- CLASS :  NoiseDiodeModel
#--------------------------------------------------------------------------------------------------

class NoiseDiodeNotFound(Exception):
    """No noise diode characteristics were found in data file."""
    pass

class NoiseDiodeModel(object):
    """Container for noise diode calibration data.
    
    This allows different noise diode data formats to co-exist in the code.
    
    Parameters
    ----------
    temperature_x : real array, shape (N, 2)
        Table containing frequencies [MHz] in the first column and measured
        temperatures [K] in the second column, for port 1 or V (X polarisation)
    temperature_y : real array, shape (N, 2)
        Table containing frequencies [MHz] in the first column and measured
        temperatures [K] in the second column, for port 2 or H (Y polarisation)
    
    """
    def __init__(self, temperature_x=None, temperature_y=None):
        self.temperature_x = temperature_x
        self.temperature_y = temperature_y
    
    def __eq__(self, other):
        """Equality comparison operator."""
        return np.all(self.temperature_x == other.temperature_x) and \
               np.all(self.temperature_y == other.temperature_y)
    
    def __ne__(self, other):
        """Inequality comparison operator."""
        return not self.__eq__(other)
    
    def temperature(self, freqs, randomise=False):
        """Obtain noise diode temperature at given frequencies.
        
        Obtain interpolated noise diode temperature at desired frequencies.
        Optionally, randomise the smooth fit to the noise diode power spectrum,
        to represent some uncertainty as part of a larger Monte Carlo iteration.
        The function returns an array of shape (2, *F*), where the first
        dimension represents the feed input ports (X and Y polarisations) and
        second dimension is frequency.
        
        Parameters
        ----------
        freqs : float or array-like, shape (*F*,)
            Frequency (or frequencies) at which to evaluate temperature, in MHz
        randomise : {False, True}, optional
            True if noise diode spectrum smoothing should be randomised
        
        Returns
        -------
        temp : real array, shape (*F*, 2)
            Noise diode temperature interpolated to the frequencies in *freqs*,
            for X and Y polarisations
        
        """
        # Fit a spline to noise diode power spectrum measurements, with optional perturbation
        interp_x, interp_y = Spline1DFit(), Spline1DFit()
        interp_x.fit(self.temperature_x[:, 0], self.temperature_x[:, 1])
        interp_y.fit(self.temperature_y[:, 0], self.temperature_y[:, 1])
        if randomise:
            interp_x = fitting_randomise(interp_x, self.temperature_x[:, 0], self.temperature_x[:, 1], 'shuffle')
            interp_y = fitting_randomise(interp_y, self.temperature_y[:, 0], self.temperature_y[:, 1], 'shuffle')
        # Evaluate the smoothed spectrum at the desired frequencies
        return np.dstack((interp_x(freqs), interp_y(freqs))).squeeze()

#--------------------------------------------------------------------------------------------------
#--- FUNCTIONS
#--------------------------------------------------------------------------------------------------

class NoSuitableNoiseDiodeDataFound(Exception):
    """No suitable noise diode on/off blocks were found in data set."""
    pass

def estimate_nd_jumps(dataset, min_duration=1.0, jump_significance=10.0):
    """Estimate jumps in power when noise diode toggles state in data set.
    
    This examines all time instants where the noise diode flag changes state
    (both off -> on and on -> off). The average power is calculated for the time
    segments immediately before and after the jump, for all frequencies and
    polarisations, using robust statistics. All jumps with a significant
    difference between these two power levels are returned.
    
    Parameters
    ----------
    dataset : :class:`dataset.DataSet` object
        Data set to analyse
    min_duration : float, optional
        Minimum duration of each time segment in seconds, to ensure good estimates
    jump_significance : float, optional
        The jump in power level should be at least this number of standard devs
    
    Returns
    -------
    nd_jump_times : list of floats
        Timestamps at which jumps occur
    nd_jump_power : list of :class:`stats.MuSigmaArray` objects, shape (*F*, 4)
        Power level changes at each jump, stored as a :class:`stats.MuSigmaArray`
        object of shape (*F*, 4), where *F* is the number of channels/bands
    
    """
    nd_jump_times, nd_jump_power = [], []
    min_samples = dataset.dump_rate * min_duration
    for scan in dataset.scans:
        # Find indices where noise diode flag changes value
        jumps = (np.diff(scan.flags['nd_on']).nonzero()[0] + 1).tolist()
        if jumps:
            before_jump = [0] + jumps[:-1]
            at_jump = jumps
            after_jump = jumps[1:] + [len(scan.timestamps)]
            # For every jump, obtain segments before and after jump with constant noise diode state
            for start, mid, end in zip(before_jump, at_jump, after_jump):
                # Restrict these segments to indices where data is valid
                before_segment = scan.flags['valid'][start:mid].nonzero()[0] + start
                after_segment = scan.flags['valid'][mid:end].nonzero()[0] + mid
                if (len(before_segment) > min_samples) and (len(after_segment) > min_samples):
                    # Utilise both off -> on and on -> off transitions 
                    # (mid is the first sample of the segment after the jump)
                    if scan.flags['nd_on'][mid]:
                        off_segment, on_segment = before_segment, after_segment
                    else:
                        on_segment, off_segment = before_segment, after_segment
                    # Calculate mean and standard deviation of the *averaged* power data in the two segments.
                    # Use robust estimators to suppress spikes and transients in data. Since the estimated mean
                    # of data is less variable than the data itself, we have to divide the data sigma by sqrt(N).
                    nd_off = robust_mu_sigma(scan.data[off_segment, :, :])
                    nd_off.sigma /= np.sqrt(len(off_segment))
                    nd_on = robust_mu_sigma(scan.data[on_segment, :, :])
                    nd_on.sigma /= np.sqrt(len(on_segment))
                    # Obtain mean and standard deviation of difference between averaged power in the segments
                    nd_delta = MuSigmaArray(nd_on.mu - nd_off.mu,
                                            np.sqrt(nd_on.sigma ** 2 + nd_off.sigma ** 2))
                    # Only keep jumps with significant change in power
                    # This discards segments where noise diode did not fire as expected
                    significance = np.abs(nd_delta.mu / nd_delta.sigma)
                    # Remove NaNs which typically occur with perfect simulated data (zero mu and zero sigma)
                    significance[np.isnan(significance)] = 0.0
                    if np.mean(significance, axis=0).max() > jump_significance:
                        nd_jump_times.append(scan.timestamps[mid])
                        nd_jump_power.append(nd_delta)
    return nd_jump_times, nd_jump_power

def estimate_gain(dataset, **kwargs):
    """Estimate gain and relative phase of both polarisations via injected noise.
    
    Each successful noise diode transition in the data set is used to estimate
    the gain and relative phase in the two receiver chains for the X and Y
    polarisations at the instant of the transition. The X and Y gains are power
    gains (i.e. the square of the voltage gains found in the Jones matrix), and
    the phase is that of the Y chain relative to the X chain.
    
    Parameters
    ----------
    dataset : :class:`dataset.DataSet` object
        Data set to analyse (converted to coherency form in the process)
    kwargs : dict, optional
        Extra keyword arguments are passed to :func:`estimate_nd_jumps`
    
    Returns
    -------
    timestamps : real array, shape (*T*,)
        Timestamp of each gain measurement, in seconds-since-Unix-epoch
    gain_xx : real array, shape (*T*, *F*)
        Power gain of X chain per measurement and channel, in units of raw/K
    gain_yy : real array, shape (*T*, *F*)
        Power gain of Y chain per measurement and channel, in units of raw/K
    phi : real array, shape (*T*, *F*)
        Phase of Y relative to X, per measurement and channel, in radians
    
    """
    dataset.convert_to_coherency()
    nd_jump_times, nd_jump_power = estimate_nd_jumps(dataset, **kwargs)
    if not nd_jump_times:
        return np.zeros((0)), np.zeros((0, len(dataset.freqs))), \
               np.zeros((0, len(dataset.freqs))), np.zeros((0, len(dataset.freqs)))
    timestamps = np.array(nd_jump_times)
    deltas = np.concatenate([p.mu[np.newaxis] for p in nd_jump_power])
    temp_nd = dataset.noise_diode_data.temperature(dataset.freqs)
    gain_xx = deltas[:, :, 0] / temp_nd[np.newaxis, :, 0]
    gain_yy = deltas[:, :, 1] / temp_nd[np.newaxis, :, 1]
    phi = -np.arctan2(deltas[:, :, 3], deltas[:, :, 2])
    return timestamps, gain_xx, gain_yy, minimise_angle_wrap(phi, axis=1)

def calibrate_gain(dataset, randomise=False, **kwargs):
    """Calibrate X and Y gains and relative phase, based on noise injection.
    
    This converts the raw power measurements in the data set to temperatures,
    based on the change in levels caused by switching the noise diode on and off.
    At the same time it corrects for different gains in the X and Y polarisation
    receiver chains and for relative phase shifts between them.
    
    Parameters
    ----------
    dataset : :class:`dataset.DataSet` object
        Data set to calibrate (converted to coherency form in the process)
    randomise : {False, True}, optional
        True if raw data and noise diode spectrum smoothing should be randomised
    kwargs : dict, optional
        Extra keyword arguments are passed to :func:`estimate_nd_jumps`
    
    Raises
    ------
    NoSuitableNoiseDiodeDataFound
        If no suitable noise diode on/off blocks were found in data set
    
    """
    dataset.convert_to_coherency()
    nd_jump_power = estimate_nd_jumps(dataset, **kwargs)[1]
    if not nd_jump_power:
        raise NoSuitableNoiseDiodeDataFound
    gains = np.concatenate([p.mu[np.newaxis] for p in nd_jump_power])
    if randomise:
        gains += np.concatenate([p.sigma[np.newaxis] for p in nd_jump_power]) * \
                 np.random.standard_normal(gains.shape)
    temp_nd = dataset.noise_diode_data.temperature(dataset.freqs, randomise)
    gains[:, :, :2] /= temp_nd[np.newaxis, :, :]
    # Do a very simple zeroth-order fitting for now, as gains are usually very stable
    smooth_gains = np.expand_dims(gains.mean(axis=0), axis=0)
#    interp = fitting.Independent1DFit(fitting.Polynomial1DFit(max_degree=max_degree), axis=0)
#    interp.fit(np.array(nd_jump_times), gains)
    for scan in dataset.scans:
#        smooth_gains = interp(scan.timestamps)
        # Scale XX and YY with respective power gains
        scan.data[:, :, :2] /= smooth_gains[:, :, :2]
        u, v = scan.data[:, :, 2].copy(), scan.data[:, :, 3].copy()
        # Rotate U and V, using K cos(phi) and -K sin(phi) terms
        scan.data[:, :, 2] =  smooth_gains[:, :, 2] * u + smooth_gains[:, :, 3] * v
        scan.data[:, :, 3] = -smooth_gains[:, :, 3] * u + smooth_gains[:, :, 2] * v
        # Divide U and V by g_x g_y, as well as length of sin + cos terms above
        gain_xy = np.sqrt(smooth_gains[:, :, 0] * smooth_gains[:, :, 1] * 
                          (smooth_gains[:, :, 2] ** 2 + smooth_gains[:, :, 3] ** 2))
        scan.data[:, :, 2:] /= gain_xy[:, :, np.newaxis]
    dataset.data_unit = 'K'
    return dataset
