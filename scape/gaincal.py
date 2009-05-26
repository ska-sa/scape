"""Gain calibration via noise injection."""

import numpy as np

import fitting
import stats

#--------------------------------------------------------------------------------------------------
#--- CLASS :  NoiseDiodeBase
#--------------------------------------------------------------------------------------------------

class NoiseDiodeNotFound(Exception):
    """No noise diode characteristics were found in data file."""
    pass

class NoiseDiodeBase(object):
    """Base class for containers of noise diode calibration data.
    
    This allows different noise diode data formats to co-exist in the code.
    
    Attributes
    ----------
    table_x : real array, shape (N, 2)
        Table containing frequencies [Hz] in the first column and measured
        temperatures [K] in the second column, for port 1 or V (X polarisation)
    table_y : real array, shape (N, 2)
        Table containing frequencies [Hz] in the first column and measured
        temperatures [K] in the second column, for port 2 or H (Y polarisation)
    
    """
    def __init__(self):
        self.table_x = self.table_y = None
        raise NotImplementedError
    
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
            Frequency (or frequencies) at which to evaluate temperature, in Hz
        randomise : {False, True}, optional
            True if noise diode spectrum smoothing should be randomised
        
        Returns
        -------
        temp : real array, shape (*F*, 2)
            Noise diode temperature interpolated to the frequencies in *freqs*,
            for X and Y polarisations
        
        """
        # Fit a spline to noise diode power spectrum measurements, with optional perturbation
        interp_x, interp_y = fitting.Spline1DFit(), fitting.Spline1DFit()
        interp_x.fit(self.table_x[:, 0], self.table_x[:, 1])
        interp_y.fit(self.table_y[:, 0], self.table_y[:, 1])
        if randomise:
            interp_x = fitting.randomise(interp_x, self.table_x[:, 0], self.table_x[:, 1], 'shuffle')
            interp_y = fitting.randomise(interp_y, self.table_y[:, 0], self.table_y[:, 1], 'shuffle')
        # Evaluate the smoothed spectrum at the desired frequencies
        return np.dstack((interp_x(freqs), interp_y(freqs))).squeeze()

#--------------------------------------------------------------------------------------------------
#--- FUNCTIONS
#--------------------------------------------------------------------------------------------------

def estimate_nd_jumps(dataset, min_samples=10, jump_significance=10.0):
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
    min_samples : int, optional
        Minimum number of samples in a time segment, to ensure good estimates
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
    for ss in dataset.subscans():
        # Find indices where noise diode flag changes value
        jumps = (np.diff(ss.flags['nd_on']).nonzero()[0] + 1).tolist()
        if jumps:
            before_jump = [0] + jumps[:-1]
            at_jump = jumps
            after_jump = jumps[1:] + [len(ss.timestamps)]
            # For every jump, obtain segments before and after jump with constant noise diode state
            for start, mid, end in zip(before_jump, at_jump, after_jump):
                # Restrict these segments to indices where data is valid
                before_segment = ss.flags['valid'][start:mid].nonzero()[0] + start
                after_segment = ss.flags['valid'][mid:end].nonzero()[0] + mid
                if (len(before_segment) > min_samples) and (len(after_segment) > min_samples):
                    # Utilise both off -> on and on -> off transitions 
                    # (mid is the first sample of the segment after the jump)
                    if ss.flags['nd_on'][mid]:
                        off_segment, on_segment = before_segment, after_segment
                    else:
                        on_segment, off_segment = before_segment, after_segment
                    # Calculate mean and standard deviation of the *averaged* power data in the two segments.
                    # Use robust estimators to suppress spikes and transients in data. Since the estimated mean
                    # of data is less variable than the data itself, we have to divide the data sigma by sqrt(N).
                    nd_off = stats.robust_mu_sigma(ss.data[off_segment, :, :])    
                    nd_off.sigma /= np.sqrt(len(off_segment))
                    nd_on = stats.robust_mu_sigma(ss.data[on_segment, :, :])
                    nd_on.sigma /= np.sqrt(len(on_segment))
                    # Obtain mean and standard deviation of difference between averaged power in the segments
                    nd_delta = stats.MuSigmaArray(nd_on.mu - nd_off.mu,
                                                  np.sqrt(nd_on.sigma ** 2 + nd_off.sigma ** 2))
                    # Only keep jumps with significant change in power
                    # This discards segments where noise diode did not fire as expected
                    if np.mean(np.abs(nd_delta.mu / nd_delta.sigma), axis=0).max() > jump_significance:
                        nd_jump_times.append(ss.timestamps[mid])
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
    gain_xx : real array, shape (*F*, *T*)
        Power gain of X chain per channel and measurement, in units of raw/K
    gain_yy : real array, shape (*F*, *T*)
        Power gain of Y chain per channel and measurement, in units of raw/K
    phi : real array, shape (*F*, *T*)
        Phase of Y relative to X, per channel and measurement, in radians
    
    """
    dataset.convert_to_coherency()
    nd_jump_times, nd_jump_power = estimate_nd_jumps(dataset, **kwargs)
    if not nd_jump_times:
        return np.zeros((0)), np.zeros((len(dataset.freqs), 0)), \
               np.zeros((len(dataset.freqs), 0)), np.zeros((len(dataset.freqs), 0))
    timestamps = np.array(nd_jump_times)
    deltas = np.dstack([p.mu for p in nd_jump_power])
    temp_nd = dataset.noise_diode_data.temperature(dataset.freqs)
    gain_xx = deltas[:, 0, :] / temp_nd[:, 0, np.newaxis]
    gain_yy = deltas[:, 1, :] / temp_nd[:, 1, np.newaxis]
    phi = -np.arctan2(deltas[:, 3, :], deltas[:, 2, :])
    return timestamps, gain_xx, gain_yy, stats.minimise_angle_wrap(phi)

def calibrate(dataset, max_degree=1, randomise=False, **kwargs):
    """Calibrate.
    
    Parameters
    ----------
    dataset : :class:`dataset.DataSet` object
        Data set to calibrate (converted to coherency form in the process)
    randomise : {False, True}, optional
        True if raw data and noise diode spectrum smoothing should be randomised
    kwargs : dict, optional
        Extra keyword arguments are passed to :func:`estimate_nd_jumps`
    
    """
    dataset.convert_to_coherency()
    nd_jump_times, nd_jump_power = estimate_nd_jumps(dataset, **kwargs)
    gains = np.dstack([p.mu for p in nd_jump_power])
    if randomise:
        gains += np.dstack([p.sigma for p in nd_jump_power]) * np.random.standard_normal(gains.shape)
    temp_nd = dataset.noise_diode_data.temperature(dataset.freqs, randomise)
    gains[:, :2, :] /= temp_nd[:, :, np.newaxis]
    interp = fitting.Independent1DFit(fitting.Polynomial1DFit(max_degree=max_degree), axis=2)
    interp.fit(np.array(nd_jump_times), gains)
    for ss in dataset.subscans():
        smooth_gains = interp(ss.timestamps)
    
# plots
# import glob
# pl.figure()
# pl.clf()
# for f in glob.glob('xdm/*/*_0000.fits'):
#     d = dataset.DataSet(f)
#     nd_transition, nd_delta_power = gaincal.estimate_nd_jumps(d)
#     delta = nd_delta_power[0]
#     xx = 0.5*(delta[:,0].mu + delta[:,1].mu)
#     yy = 0.5*(delta[:,0].mu - delta[:,1].mu)
#     ss = d.scans[0].subscans[0]
#     tnd = d.noise_diode_data.temperature(ss.freqs)
#     gx2 = xx / tnd[0, :]
#     gy2 = yy / tnd[1, :]
#     phi = -np.arctan2(delta[:,3].mu, delta[:,2].mu)
#     non_rfi = list(set(range(1024)) - set(ss.rfi_channels))
#     pl.subplot(311); pl.plot(ss.freqs / 1e9, 10*np.log10(gx2), 'b'); pl.plot(ss.freqs[non_rfi] / 1e9, 10*np.log10(gx2[non_rfi]), 'ob')
#     pl.subplot(312); pl.plot(ss.freqs / 1e9, 10*np.log10(gy2), 'b'); pl.plot(ss.freqs[non_rfi] / 1e9, 10*np.log10(gy2[non_rfi]), 'ob')
#     angles = coord.rad2deg(phi)
#     angles = angles % 360.0
#     pl.subplot(313); pl.plot(ss.freqs / 1e9, angles, 'b'); pl.plot(ss.freqs[non_rfi] / 1e9, angles[non_rfi], 'ob')
#     pl.subplot(311); pl.xticks([]); pl.ylabel('dB'); pl.title('XX power gain'); pl.axis([1.4, 1.6, 10, 20])
#     pl.subplot(312); pl.xticks([]); pl.ylabel('dB'); pl.title('YY power gain'); pl.axis([1.4, 1.6, 10, 20])
#     pl.subplot(313); pl.xlabel('Frequency (GHz)'); pl.ylabel('degrees'); pl.title('Phase difference of Y relative to X'); pl.axis([1.4, 1.6, 150, 240])
#     raw_input()
#     