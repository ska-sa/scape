"""Gain calibration via noise injection."""

import numpy as np

import fitting

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
        
        Parameters
        ----------
        freqs : array-like or float
            Frequency (or frequencies) at which to evaluate temperature, in Hz
        randomise : {False, True}, optional
            True if noise diode spectrum smoothing should be randomised
        
        Returns
        -------
        temp : array-like or float
            Noise diode temperature interpolated to the frequencies in freqs
        
        """
        # Use average temperature of two polarisations, as they are supposed to be identical under ideal
        # symmetrical injection. Otherwise, gain variations during the hot/cold experiment can skew the data.
        # There should ideally be a loop of T_ND -> gain cal -> T_ND -> gain cal -> etc.
        table = np.mean(np.dstack([self.table_x, self.table_y]), axis=2)
        # Fit a spline to noise diode power spectrum measurements, with optional perturbation
        interp = fitting.Spline1DFit()
        interp.fit(table[:, 0], table[:, 1])
        if randomise:
            interp = fitting.randomise(interp, table[:, 0], table[:, 1], 'shuffle')
        # Evaluate the smoothed spectrum at the desired frequencies
        return interp(freqs)
