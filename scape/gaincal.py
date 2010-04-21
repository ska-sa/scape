"""Gain calibration via noise injection."""

import numpy as np

from .fitting import Spline1DFit, Polynomial1DFit
from .fitting import randomise as fitting_randomise
from .stats import robust_mu_sigma, minimise_angle_wrap
from .scan import scape_pol

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
    temperature_h : real array-like, shape (N, 2)
        Table containing frequencies [MHz] in the first column and measured
        temperatures [K] in the second column, for H polarisation
    temperature_v : real array-like, shape (N, 2)
        Table containing frequencies [MHz] in the first column and measured
        temperatures [K] in the second column, for V polarisation
    std_temp : float, optional
        Standard deviation of H and V temperatures [K] which determines
        smoothness of interpolation

    """
    def __init__(self, temperature_h, temperature_v, std_temp=1.0):
        self.temperature_h = np.asarray(temperature_h)
        self.temperature_v = np.asarray(temperature_v)
        self.std_temp = std_temp

    def __eq__(self, other):
        """Equality comparison operator."""
        return np.all(self.temperature_h == other.temperature_h) and \
               np.all(self.temperature_v == other.temperature_v) and \
               (self.std_temp == other.std_temp)

    def __ne__(self, other):
        """Inequality comparison operator."""
        return not self.__eq__(other)

    def temperature(self, freqs, randomise=False):
        """Obtain noise diode temperature at given frequencies.

        Obtain interpolated noise diode temperature at desired frequencies.
        Optionally, randomise the smooth fit to the noise diode power spectrum,
        to represent some uncertainty as part of a larger Monte Carlo iteration.
        The function returns an array of shape (*F*, 2), where the first
        dimension is frequency and the second dimension represents the feed
        input ports (H and V polarisations).

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
            for H and V polarisations

        """
        # Fit a spline to power spectrum measurements (or straight line if too few measurements)
        std_temp = lambda freq, temp: np.tile(self.std_temp, len(temp))
        interp_h = Spline1DFit(std_y=std_temp) if self.temperature_h.shape[0] > 4 else Polynomial1DFit(max_degree=1)
        interp_v = Spline1DFit(std_y=std_temp) if self.temperature_v.shape[0] > 4 else Polynomial1DFit(max_degree=1)
        interp_h.fit(self.temperature_h[:, 0], self.temperature_h[:, 1])
        interp_v.fit(self.temperature_v[:, 0], self.temperature_v[:, 1])
        # Optionally perturb the fit
        if randomise:
            interp_h = fitting_randomise(interp_h, self.temperature_h[:, 0], self.temperature_h[:, 1], 'shuffle')
            interp_v = fitting_randomise(interp_v, self.temperature_v[:, 0], self.temperature_v[:, 1], 'shuffle')
        # Evaluate the smoothed spectrum at the desired frequencies
        return np.dstack((interp_h(freqs), interp_v(freqs))).squeeze()

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
    nd_jump_power_mu : list of arrays, shape (*F*, 4)
        Mean power level changes at each jump, stored as an array of shape
        (*F*, 4), where *F* is the number of channels/bands
    nd_jump_power_sigma : list of arrays, shape (*F*, 4)
        Standard deviation of power level changes at each jump, stored as an
        array of shape (*F*, 4), where *F* is the number of channels/bands

    """
    nd_jump_times, nd_jump_power_mu, nd_jump_power_sigma = [], [], []
    min_samples = dataset.dump_rate * min_duration
    for scan in dataset.scans:
        num_times = len(scan.timestamps)
        # In absence of valid flag, all data is valid
        valid_flag = scan.flags['valid'] if 'valid' in scan.flags.dtype.names else np.tile(True, num_times)
        # Find indices where noise diode flag changes value
        jumps = (np.diff(scan.flags['nd_on']).nonzero()[0] + 1).tolist()
        if jumps:
            before_jump = [0] + jumps[:-1]
            at_jump = jumps
            after_jump = jumps[1:] + [num_times]
            # For every jump, obtain segments before and after jump with constant noise diode state
            for start, mid, end in zip(before_jump, at_jump, after_jump):
                # Restrict these segments to indices where data is valid
                before_segment = valid_flag[start:mid].nonzero()[0] + start
                after_segment = valid_flag[mid:end].nonzero()[0] + mid
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
                    nd_off_mu, nd_off_sigma = robust_mu_sigma(scan.data[off_segment, :, :])
                    nd_off_sigma /= np.sqrt(len(off_segment))
                    nd_on_mu, nd_on_sigma = robust_mu_sigma(scan.data[on_segment, :, :])
                    nd_on_sigma /= np.sqrt(len(on_segment))
                    # Obtain mean and standard deviation of difference between averaged power in the segments
                    nd_delta_mu, nd_delta_sigma = nd_on_mu - nd_off_mu, np.sqrt(nd_on_sigma ** 2 + nd_off_sigma ** 2)
                    # Only keep jumps with significant *increase* in power (focus on the positive HH/VV)
                    # This discards segments where noise diode did not fire as expected
                    norm_jump = nd_delta_mu / nd_delta_sigma
                    norm_jump = norm_jump[:, :2]
                    # Remove NaNs which typically occur with perfect simulated data (zero mu and zero sigma)
                    norm_jump[np.isnan(norm_jump)] = 0.0
                    if np.mean(norm_jump, axis=0).max() > jump_significance:
                        nd_jump_times.append(scan.timestamps[mid])
                        nd_jump_power_mu.append(nd_delta_mu)
                        nd_jump_power_sigma.append(nd_delta_sigma)
    return nd_jump_times, nd_jump_power_mu, nd_jump_power_sigma

def estimate_gain(dataset, **kwargs):
    """Estimate gain and relative phase of both polarisations via injected noise.

    Each successful noise diode transition in the data set is used to estimate
    the gain and relative phase in the two receiver chains for the H and V
    polarisations at the instant of the transition. The H and V gains are power
    gains (i.e. the square of the voltage gains found in the Jones matrix), and
    the phase is that of the V chain relative to the H chain.

    Parameters
    ----------
    dataset : :class:`dataset.DataSet` object
        Data set to analyse
    kwargs : dict, optional
        Extra keyword arguments are passed to :func:`estimate_nd_jumps`

    Returns
    -------
    timestamps : real array, shape (*T*,)
        Timestamp of each gain measurement, in seconds-since-Unix-epoch
    gain_hh : real array, shape (*T*, *F*)
        Power gain of H chain per measurement and channel, in units of counts/K
    gain_vv : real array, shape (*T*, *F*)
        Power gain of V chain per measurement and channel, in units of counts/K
    phi : real array, shape (*T*, *F*)
        Phase of V relative to H, per measurement and channel, in radians

    """
    nd_jump_times, nd_jump_power_mu = estimate_nd_jumps(dataset, **kwargs)[:2]
    if not nd_jump_times:
        return np.zeros((0)), np.zeros((0, len(dataset.freqs))), \
               np.zeros((0, len(dataset.freqs))), np.zeros((0, len(dataset.freqs)))
    timestamps = np.array(nd_jump_times)
    deltas = np.concatenate([p[np.newaxis] for p in nd_jump_power_mu])
    temp_nd = dataset.nd_model.temperature(dataset.freqs)
    gain_hh = deltas[:, :, scape_pol.index('HH')] / temp_nd[np.newaxis, :, 0]
    gain_vv = deltas[:, :, scape_pol.index('VV')] / temp_nd[np.newaxis, :, 1]
    # For single-dish, HV == Re{HV} and VH == Im{HV}
    phi = -np.arctan2(deltas[:, :, scape_pol.index('VH')], deltas[:, :, scape_pol.index('HV')])
    return timestamps, gain_hh, gain_vv, minimise_angle_wrap(phi, axis=1)

def calibrate_gain(dataset, randomise=False, **kwargs):
    """Calibrate H and V gains and relative phase, based on noise injection.

    This converts the raw power measurements in the data set to temperatures,
    based on the change in levels caused by switching the noise diode on and off.
    At the same time it corrects for different gains in the H and V polarisation
    receiver chains and for relative phase shifts between them.

    Parameters
    ----------
    dataset : :class:`dataset.DataSet` object
        Data set to calibrate
    randomise : {False, True}, optional
        True if raw data and noise diode spectrum smoothing should be randomised
    kwargs : dict, optional
        Extra keyword arguments are passed to :func:`estimate_nd_jumps`

    Raises
    ------
    NoSuitableNoiseDiodeDataFound
        If no suitable noise diode on/off blocks were found in data set

    """
    nd_jump_power_mu, nd_jump_power_sigma = estimate_nd_jumps(dataset, **kwargs)[1:]
    if not nd_jump_power_mu:
        raise NoSuitableNoiseDiodeDataFound
    gains = np.concatenate([p[np.newaxis] for p in nd_jump_power_mu])
    if randomise:
        gains += np.concatenate([p[np.newaxis] for p in nd_jump_power_sigma]) * \
                 np.random.standard_normal(gains.shape)
    temp_nd = dataset.nd_model.temperature(dataset.freqs, randomise)
    hh, vv, re_hv, im_hv = [scape_pol.index(pol) for pol in ['HH', 'VV', 'HV', 'VH']]
    gains[:, :, hh] /= temp_nd[np.newaxis, :, 0]
    gains[:, :, vv] /= temp_nd[np.newaxis, :, 1]
    # Do a very simple zeroth-order fitting for now, as gains are usually very stable
    smooth_gains = np.expand_dims(gains.mean(axis=0), axis=0)
#    interp = fitting.Independent1DFit(fitting.Polynomial1DFit(max_degree=max_degree), axis=0)
#    interp.fit(np.array(nd_jump_times), gains)
    for scan in dataset.scans:
#        smooth_gains = interp(scan.timestamps)
        # Remove instances of zero gain, which would lead to NaNs or Infs in the data
        # Usually these are associated with missing H or V polarisations (which get filled in with zeros)
        # Replace them with Infs instead, which suppresses the corresponding channels / polarisations
        # Similar to pseudo-inverse, where scale factors of 1/0 associated with zero eigenvalues are replaced by 0
        smooth_gains[:, :, hh][smooth_gains[:, :, hh] == 0.0] = np.inf
        smooth_gains[:, :, vv][smooth_gains[:, :, vv] == 0.0] = np.inf
        # Scale HH and VV with respective power gains
        scan.data[:, :, hh] /= smooth_gains[:, :, hh]
        scan.data[:, :, vv] /= smooth_gains[:, :, vv]
        u, v = scan.data[:, :, re_hv].copy(), scan.data[:, :, im_hv].copy()
        # Rotate U and V, using K cos(phi) and -K sin(phi) terms
        scan.data[:, :, re_hv] =  smooth_gains[:, :, re_hv] * u + smooth_gains[:, :, im_hv] * v
        scan.data[:, :, im_hv] = -smooth_gains[:, :, im_hv] * u + smooth_gains[:, :, re_hv] * v
        # Divide U and V by g_h g_v, as well as length of sin + cos terms above
        gain_hv = np.sqrt(smooth_gains[:, :, hh] * smooth_gains[:, :, vv] *
                          (smooth_gains[:, :, re_hv] ** 2 + smooth_gains[:, :, im_hv] ** 2))
        # Gain_HV is NaN if HH or VV gain is Inf and Re/Im HV gain is zero (typical of the single-pol case)
        gain_hv[np.isnan(gain_hv)] = np.inf
        scan.data[:, :, re_hv] /= gain_hv
        scan.data[:, :, im_hv] /= gain_hv
    dataset.data_unit = 'K'
    return dataset
