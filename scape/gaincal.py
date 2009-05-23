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
        temp : real array, shape (2, *F*)
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
        return np.vstack((interp_x(freqs), interp_y(freqs)))

#--------------------------------------------------------------------------------------------------
#--- FUNCTIONS
#--------------------------------------------------------------------------------------------------

def estimate_nd_jumps(dataset, min_samples=10, jump_significance=10.0):
    nd_jump_times, nd_jump_power = [], []
    for ss in dataset.subscans():
        jumps = (np.diff(ss.flags['nd_on']).nonzero()[0] + 1).tolist()
        if jumps:
            before_jump = [0] + jumps[:-1]
            at_jump = jumps
            after_jump = jumps[1:] + [len(ss.timestamps)]
            for start, mid, end in zip(before_jump, at_jump, after_jump):
                before_region = ss.flags['valid'][start:mid].nonzero()[0] + start
                after_region = ss.flags['valid'][mid:end].nonzero()[0] + mid
                if (len(before_region) > min_samples) and (len(after_region) > min_samples):
                    if ss.flags['nd_on'][mid]:
                        off_region, on_region = before_region, after_region
                    else:
                        on_region, off_region = before_region, after_region
                    nd_off = stats.robust_mu_sigma(ss.data[off_region, :, :])    
                    nd_off.sigma /= np.sqrt(len(off_region))
                    nd_on = stats.robust_mu_sigma(ss.data[on_region, :, :])
                    nd_on.sigma /= np.sqrt(len(on_region))
                    nd_delta = stats.MuSigmaArray(nd_on.mu - nd_off.mu,
                                                  np.sqrt(nd_on.sigma ** 2 + nd_off.sigma ** 2))
                    if np.mean(np.abs(nd_delta.mu / nd_delta.sigma), axis=0).max() > jump_significance:
                        nd_jump_times.append(ss.timestamps[mid])
                        nd_jump_power.append(nd_delta)
    return nd_jump_times, nd_jump_power

def calibrate(dataset, gain):
    pass

# plots
d = dataset.DataSet('/Users/schwardt/xdmsbe/bin/xdm/2009-01-12-13h48/cal_src_scan_2009-01-12-13h48_0000.fits')
nd_transition, nd_delta_power = estimate_gain(d)
delta = nd_delta_power[0]
xx = 0.5*(delta[:,0].mu + delta[:,1].mu)
yy = 0.5*(delta[:,0].mu - delta[:,1].mu)
ss = d.scans[0].subscans[0]
tnd = d.noise_diode_data.temperature(ss.freqs)
gx2 = xx / tnd[0, :]
gy2 = yy / tnd[1, :]
phi = -np.arctan2(delta[:,3].mu, delta[:,2].mu)
non_rfi = list(set(range(1024)) - set(ss.rfi_channels))

clf()
subplot(311); cla(); plot(ss.freqs / 1e9, 10*np.log10(gx2)); plot(ss.freqs[non_rfi] / 1e9, 10*np.log10(gx2[non_rfi]), 'ob')
subplot(312); cla(); plot(ss.freqs / 1e9, 10*np.log10(gy2)); plot(ss.freqs[non_rfi] / 1e9, 10*np.log10(gy2[non_rfi]), 'ob')
angles = coord.degrees(phi)
angles = angles % 360.0
subplot(313); cla(); plot(ss.freqs / 1e9, angles); plot(ss.freqs[non_rfi] / 1e9, angles[non_rfi], 'ob')
subplot(311); xticks([]); ylabel('dB'); title('XX power gain'); axis([1.4, 1.6, 10, 20])
subplot(312); xticks([]); ylabel('dB'); title('YY power gain'); axis([1.4, 1.6, 10, 20])
subplot(313); xlabel('Frequency (GHz)'); ylabel('degrees'); title('Phase difference of Y relative to X'); axis([1.4, 1.6, 150, 240])
savefig('xdm_2009-01-12-13h48_gaincal.pdf')

