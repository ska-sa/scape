"""Container for the data of a single-dish or single-baseline experiment."""

import os.path
import logging

import numpy as np

import katpoint
from .compoundscan import CompoundScan
from .gaincal import calibrate_gain, NoSuitableNoiseDiodeDataFound
from .beam_baseline import fit_beam_and_baselines
from .stats import remove_spikes, chi2_conf_interval

# Try to import all available formats
try:
    from .xdmfits import load_dataset as xdmfits_load
    xdmfits_found = True
except ImportError:
    xdmfits_found = False
try:
    from .hdf5 import load_dataset as hdf5_load
    from .hdf5 import save_dataset as hdf5_save
    hdf5_found = True
except ImportError:
    hdf5_found = False

logger = logging.getLogger("scape.dataset")

#--------------------------------------------------------------------------------------------------
#--- CLASS :  DataSet
#--------------------------------------------------------------------------------------------------

class DataSet(object):
    """Container for the data of an experiment (single-dish or a single baseline).

    This is the top-level container for experimental data, which is either
    autocorrelation data for a single dish or cross-correlation data for a single
    interferometer baseline, combined with the appropriate metadata.

    Given a data filename, the initialiser determines the appropriate file format
    to use, based on the file extension. If the filename is blank, the
    :class:`DataSet` can also be directly initialised from its constituent
    parts, which is useful for simulations and creating the data sets from
    scratch. The :class:`DataSet` object contains a list of compound scans, as
    well as the correlator configuration, antenna details, and noise diode and
    pointing models.

    Parameters
    ----------
    filename : string
        Name of data set file, or blank string if the other parameters are given
    compscanlist : list of :class:`compoundscan.CompoundScan` objects, optional
        List of compound scans
    data_unit : {'counts', 'K', 'Jy'}, optional
        Physical unit of power data
    corrconf : :class:`compoundscan.CorrelatorConfig` object, optional
        Correlator configuration object
    antenna : :class:`katpoint.Antenna` object or string, optional
        Antenna that produced the data set, as object or description string. For
        interferometer data, this is the first antenna.
    nd_data : :class:`gaincal.NoiseDiodeModel` object, optional
        Noise diode model
    pointing_model : :class:`katpoint.PointingModel` object or array of float, optional
        Pointing model in use during experiment (or array of parameters in radians)
    antenna2 : :class:`katpoint.Antenna` object or string or None, optional
        Second antenna of baseline, as object or description string. This is
        *None* for single-dish autocorrelation data.
    kwargs : dict, optional
        Extra keyword arguments are passed to selected :func:`load_dataset` function

    Attributes
    ----------
    freqs : real array, shape (*F*,)
        Centre frequency of each channel/band, in MHz (same as in *corrconf*)
    bandwidths : real array, shape (*F*,)
        Bandwidth of each channel/band, in MHz (same as in *corrconf*)
    rfi_channels : list of ints
        RFI-flagged channel indices (same as in *corrconf*)
    dump_rate : float
        Correlator dump rate, in Hz (same as in *corrconf*)
    scans : list of :class:`scan.Scan` objects
        Flattened list of all scans in data set

    Raises
    ------
    ImportError
        If file extension is known, but appropriate module would not import
    ValueError
        If file extension is unknown or parameter is invalid

    """
    def __init__(self, filename, compscanlist=None, data_unit=None, corrconf=None,
                 antenna=None, nd_data=None, pointing_model=None, antenna2=None, **kwargs):
        if filename:
            ext = os.path.splitext(filename)[1]
            if ext == '.fits':
                if not xdmfits_found:
                    raise ImportError('XDM FITS support could not be loaded - please check xdmfits module')
                compscanlist, data_unit, corrconf, antenna, nd_data = xdmfits_load(filename, **kwargs)
                antenna2 = None
            elif (ext == '.h5') or (ext == '.hdf5'):
                if not hdf5_found:
                    raise ImportError('HDF5 support could not be loaded - please check hdf5 module')
                hdf5_results = hdf5_load(filename, **kwargs)
                compscanlist, data_unit, corrconf, antenna, antenna2, nd_data = hdf5_results[:6]
                if pointing_model is None:
                    pointing_model = hdf5_results[6]
            else:
                raise ValueError("File extension '%s' not understood" % ext)

        self.compscans = compscanlist
        self.data_unit = data_unit
        self.corrconf = corrconf
        if isinstance(antenna, katpoint.Antenna):
            self.antenna = antenna
        else:
            self.antenna = katpoint.construct_antenna(antenna)
        if antenna2 is None or isinstance(antenna2, katpoint.Antenna):
            self.antenna2 = antenna2
        else:
            self.antenna2 = katpoint.construct_antenna(antenna2)
        self.noise_diode_data = nd_data
        if isinstance(pointing_model, katpoint.PointingModel):
            self.pointing_model = pointing_model
        else:
            self.pointing_model = katpoint.PointingModel(pointing_model, strict=False)

        # Create scan list and calculate target coordinates for all scans
        self.scans = []
        for compscan in self.compscans:
            # Set default antenna on the target object to the (first) data set antenna
            compscan.target.antenna = self.antenna
            self.scans.extend(compscan.scans)
        self.calc_cached_coords()

    def __eq__(self, other):
        """Equality comparison operator."""
        if len(self.compscans) != len(other.compscans):
            return False
        for self_compscan, other_compscan in zip(self.compscans, other.compscans):
            if self_compscan != other_compscan:
                return False
        return (self.data_unit == other.data_unit) and (self.corrconf == other.corrconf) and \
               (self.antenna.description == other.antenna.description) and \
               ((self.antenna2 == other.antenna2 == None) or \
                ((self.antenna2 is not None) and (other.antenna2 is not None) and \
                 self.antenna2.description == other.antenna2.description)) and \
               (self.noise_diode_data == other.noise_diode_data) and \
               np.all(self.pointing_model.params == other.pointing_model.params)

    def __ne__(self, other):
        """Inequality comparison operator."""
        return not self.__eq__(other)

    # Provide properties to access the attributes of the correlator configuration directly
    # This uses the same trick as in stats.MuSigmaArray to create the properties, which
    # leads to less class namespace clutter, but more pylint uneasiness (shame).
    # pylint: disable-msg=E0211,E0202,W0612,W0142
    def freqs():
        """Class method which creates freqs property."""
        doc = 'Centre frequency of each channel/band, in MHz.'
        def fget(self):
            return self.corrconf.freqs
        def fset(self, value):
            self.corrconf.freqs = value
        return locals()
    freqs = property(**freqs())

    # pylint: disable-msg=E0211,E0202,W0612,W0142
    def bandwidths():
        """Class method which creates bandwidths property."""
        doc = 'Bandwidth of each channel/band, in MHz.'
        def fget(self):
            return self.corrconf.bandwidths
        def fset(self, value):
            self.corrconf.bandwidths = value
        return locals()
    bandwidths = property(**bandwidths())

    # pylint: disable-msg=E0211,E0202,W0612,W0142
    def rfi_channels():
        """Class method which creates rfi_channels property."""
        doc = 'List of RFI-flagged channels.'
        def fget(self):
            return self.corrconf.rfi_channels
        def fset(self, value):
            self.corrconf.rfi_channels = value
        return locals()
    rfi_channels = property(**rfi_channels())

    # pylint: disable-msg=E0211,E0202,W0612,W0142
    def dump_rate():
        """Class method which creates dump_rate property."""
        doc = 'Correlator dump rate, in Hz.'
        def fget(self):
            return self.corrconf.dump_rate
        def fset(self, value):
            self.corrconf.dump_rate = value
        return locals()
    dump_rate = property(**dump_rate())

    def __str__(self):
        """Verbose human-friendly string representation of data set object."""
        descr = ["%s, data_unit=%s, bands=%d, freqs=%f - %f MHz, total bw=%f MHz, dumprate=%f Hz" %
                 ("antenna='%s'" % self.antenna.name if self.antenna2 is None else \
                  "baseline='%s - %s'" % (self.antenna.name, self.antenna2.name),
                  self.data_unit, len(self.freqs), self.freqs[0], self.freqs[-1],
                  self.bandwidths.sum(), self.dump_rate)]
        for compscan_ind, compscan in enumerate(self.compscans):
            descr.append("%4d: target='%s' [%s]" %
                         (compscan_ind, compscan.target.name, compscan.target.body_type))
            if compscan.baseline:
                descr[-1] += ', initial baseline offset=%f' % (compscan.baseline.poly[-1],)
            if compscan.beam:
                descr[-1] += ', beam height=%f' % (compscan.beam.height,)
            for scan_ind, scan in enumerate(compscan.scans):
                descr.append('      %4d: %s' % (scan_ind, str(scan)))
        return '\n'.join(descr)

    def __repr__(self):
        """Short human-friendly string representation of data set object."""
        return "<scape.DataSet %s compscans=%d at 0x%x>" % \
               ("antenna='%s'" % self.antenna.name if self.antenna2 is None else \
                "baseline='%s - %s'" % (self.antenna.name, self.antenna2.name),
                len(self.compscans), id(self))

    def calc_cached_coords(self):
        """Calculate cached coordinates, using appropriate target and antenna.

        This calculates the target coordinates and parallactic angle as a function
        of time for each scan in the data set, using the compound scan target
        object and the (first) data set antenna. This functionality is here at
        the highest level because it involves interaction between the DataSet,
        CompoundScan and Scan levels, while the results need to be stored at the
        Scan level.

        """
        for compscan in self.compscans:
            for scan in compscan.scans:
                scan.calc_cached_coords(compscan.target, self.antenna)

    def save(self, filename, **kwargs):
        """Save data set object to file.

        This automatically figures out which file format to use based on the
        file extension.

        Parameters
        ----------
        filename : string
            Name of output file
        kwargs : dict, optional
            Extra keyword arguments are passed on to underlying save function

        Raises
        ------
        IOError
            If output file already exists
        ValueError
            If file extension is unknown or unsupported
        ImportError
            If file extension is known, but appropriate module would not import

        """
        if os.path.exists(filename):
            raise IOError('File %s already exists - please remove first!' % filename)
        ext = os.path.splitext(filename)[1]
        if ext == '.fits':
            raise ValueError('XDM FITS writing support not implemented')
        elif (ext == '.h5') or (ext == '.hdf5'):
            if not hdf5_found:
                raise ImportError('HDF5 support could not be loaded - please check hdf5 module')
            hdf5_save(self, filename, **kwargs)
        else:
            raise ValueError("File extension '%s' not understood" % ext)

    def select(self, labelkeep=None, flagkeep=None, freqkeep=None, copy=False):
        """Select subset of data set, based on scan label, flags and frequency.

        This returns a data set with a possibly reduced number of time samples,
        frequency channels/bands and scans, based on the selection criteria.
        Since each scan potentially has a different number of time samples,
        it is less useful to filter directly on sample index. Instead, the
        flags are used to select a subset of time samples in each scan. The
        list of flags are ANDed together to determine which parts are kept. It
        is also possible to invert flags by prepending a ~ (tilde) character.

        Based on the value of *copy*, the new data set contains either a view of
        the original data or a copy. All criteria are optional, and with no
        parameters the returned data set is unchanged. This can be used to make
        a copy of the data set.

        Parameters
        ----------
        labelkeep : string or list of strings, optional
            All scans with labels in this list will be kept. The default is
            None, which means all labels are kept.
        flagkeep : string or list of strings, optional
            List of flags used to select time ranges in each scan. The time
            samples for which all the flags in the list are true are kept.
            Individual flags can be negated by prepending a ~ (tilde) character.
            The default is None, which means all time samples are kept.
        freqkeep : sequence of bools or ints, optional
            Sequence of indicators of which frequency channels/bands to keep
            (either integer indices or booleans that are True for the values to
            be kept). The default is None, which keeps all channels/bands.
        copy : {False, True}, optional
            True if the new scan is a copy, False if it is a view

        Returns
        -------
        dataset : :class:`DataSet` object
            Data set with selection of scans with possibly smaller data arrays.

        Raises
        ------
        KeyError
            If flag in *flagkeep* is unknown

        """
        # Handle the cases of a single input string (not in a list)
        if isinstance(labelkeep, basestring):
            labelkeep = [labelkeep]
        if isinstance(flagkeep, basestring):
            flagkeep = [flagkeep]
        compscanlist = []
        for compscan in self.compscans:
            scanlist = []
            for scan in compscan.scans:
                # Convert flag selection to time sample selection
                if flagkeep is None:
                    timekeep = None
                else:
                    # By default keep all time samples
                    timekeep = np.tile(True, len(scan.timestamps))
                    for flag in flagkeep:
                        invert = False
                        # Flags prepended with ~ get inverted
                        if flag[0] == '~':
                            invert = True
                            flag = flag[1:]
                        # Ignore unknown flags
                        try:
                            flag_data = scan.flags[flag]
                        except KeyError:
                            raise KeyError("Unknown flag '%s'" % flag)
                        if invert:
                            timekeep &= ~flag_data
                        else:
                            timekeep &= flag_data
                if (labelkeep is None) or (scan.label in labelkeep):
                    scanlist.append(scan.select(timekeep, freqkeep, copy))
            if scanlist:
                compscanlist.append(CompoundScan(scanlist, compscan.target))
        return DataSet(None, compscanlist, self.data_unit, self.corrconf.select(freqkeep),
                       self.antenna, self.noise_diode_data, self.pointing_model, self.antenna2)

    def identify_rfi_channels(self, sigma=8.0, min_bad_scans=0.25, extra_outputs=False):
        """Identify potential RFI-corrupted channels.

        This is a simple RFI detection procedure that assumes that there are
        less RFI-corrupted channels than good channels, and that the desired
        signal is broadband/continuum with similar features across the entire
        spectrum.

        Parameters
        ----------
        sigma : float, optional
            Threshold for deviation from signal (non-RFI) template, as a factor
            of the expected standard deviation of the error under the null
            hypothesis. By increasing it, less channels are marked as bad. This
            factor should typically be larger than expected (in the order of 6
            to 8), as the null hypothesis is quite stringent.
        min_bad_scans : float, optional
            The fraction of scans in which a channel has to be marked as bad in
            order to qualify as RFI-corrupted. This allows for some intermittence
            in RFI corruption.
        extra_outputs : {False, True}, optional
            True if extra information should be returned (intended for plots)

        Returns
        -------
        rfi_channels : list of ints
            List of potential RFI-corrupted channel indices

        """
        rfi_count = np.zeros(len(self.freqs))
        dof = 4.0 * (self.bandwidths * 1e6) / self.dump_rate
        rfi_data = []
        for scan in self.scans:
            # Normalise power in scan by removing spikes, offsets and differences in scale
            power = remove_spikes(scan.pol('I'))
            offset = power.min(axis=0)
            scale = power.max(axis=0) - offset
            scale[scale <= 0.0] = 1.0
            norm_power = (power - offset[np.newaxis, :]) / scale[np.newaxis, :]
            # Form a template of the desired signal as a function of time
            template = np.median(norm_power, axis=1)
            # Use this as average power, after adding back scaling and offset
            mean_signal_power = np.outer(template, scale) + offset[np.newaxis, :]
            # Determine expected standard deviation of power data, assuming it has chi-square distribution
            # Also divide by an extra sqrt(template) factor, which allows more leeway where template is small
            # This is useful for absorbing small discrepancies in baseline when scanning across a source
            expected_std = np.sqrt(2.0 / dof[np.newaxis, :]) * mean_signal_power / scale[np.newaxis, :] / \
                           np.sqrt(template[:, np.newaxis])
            channel_sumsq = (((norm_power - template[:, np.newaxis]) / expected_std) ** 2).sum(axis=0)
            # The sum of squares over time is again a chi-square distribution, with different dof
            lower, upper = chi2_conf_interval(power.shape[0], power.shape[0], sigma)
            rfi_count += (channel_sumsq < lower) | (channel_sumsq > upper)
            if extra_outputs:
                rfi_data.append((norm_power, template, expected_std))
        # Count the number of bad scans per channel, and threshold it
        rfi_channels = (rfi_count > max(min_bad_scans * len(self.scans), 1.0)).nonzero()[0]
        self.rfi_channels = rfi_channels
        if extra_outputs:
            return rfi_channels, rfi_count, rfi_data
        else:
            return rfi_channels

    def remove_rfi_channels(self, rfi_channels=None):
        """Remove RFI-flagged channels from data set, returning a copy.

        This is a convenience function that selects all the non-RFI-flagged
        frequency channels in the data set and returns a copy.

        """
        if rfi_channels is None:
            rfi_channels = self.rfi_channels
        non_rfi = sorted(list(set(range(len(self.freqs))) - set(rfi_channels)))
        d = self.select(freqkeep=non_rfi, copy=True)
        DataSet.__init__(self, None, d.compscans, d.data_unit, d.corrconf,
                         d.antenna, d.noise_diode_data, d.pointing_model, d.antenna2)
        return self

    def convert_power_to_temperature(self, randomise=False, **kwargs):
        """Convert raw power into temperature (K) based on noise injection.

        This is a convenience function that converts the raw power measurements
        in the data set to temperatures, based on the change in levels caused by
        switching the noise diode on and off. At the same time it corrects for
        different gains in the X and Y polarisation receiver chains and for
        relative phase shifts between them. It should be called before averaging
        the data, as gain calibration should happen on the finest available
        frequency scale.

        Parameters
        ----------
        randomise : {False, True}, optional
            True if noise diode data and smoothing should be randomised
        kwargs : dict, optional
            Extra keyword arguments are passed to underlying :mod:`gaincal` functions

        Returns
        -------
        dataset : :class:`DataSet` object
            Data set with calibrated power data

        """
        # Only operate on raw data
        if self.data_unit != 'counts':
            logger.error("Expected raw power data to convert to temperature, got data with units '" +
                         self.data_unit + "' instead.")
            return self
        if self.noise_diode_data is None:
            logger.error('No noise diode model found in data set - calibration aborted')
            return  self
        try:
            return calibrate_gain(self, randomise, **kwargs)
        except NoSuitableNoiseDiodeDataFound:
            logger.error('No suitable noise diode on/off blocks were found - calibration aborted')

    def average(self, channels_per_band='all', time_window=1):
        """Average data in time and/or frequency.

        If *channels_per_band* is not `None`, the frequency channels are grouped
        into bands, and the power data is merged and averaged within each band.
        Each band contains the average power of its constituent channels. If
        *time_window* is larger than 1, the power data is averaged in time in
        non-overlapping windows of this length, and the rest of the time-varying
        data is averaged accordingly. The default behaviour is to average all
        channels into one band, in line with the continuum focus of :mod:`scape`.

        Parameters
        ----------
        channels_per_band : List of lists of ints, optional
            List of lists of channel indices (one list per band), indicating
            which channels are averaged together to form each band. If this is
            the string 'all', all channels are averaged together into 1 band.
            If this is None, no averaging is done (each channel becomes a band).
        time_window : int, optional
            Window length in samples, within which to average data in time. If
            this is 1 or None, no averaging is done in time.

        Returns
        -------
        dataset : :class:`DataSet` object
            Data set with averaged power data

        Notes
        -----
        The average power is simpler to use than the total power in each band,
        as total power is dependent on the bandwidth of each band. This method
        should be called *after* :meth:`convert_power_to_temperature`.

        """
        # The string 'all' means average all channels together
        if channels_per_band == 'all':
            channels_per_band = [range(len(self.freqs))]
        # None means no frequency averaging (band == channel)
        if channels_per_band is None:
            channels_per_band = np.expand_dims(range(len(self.freqs)), axis=1).tolist()
        # None means no time averaging too
        if (time_window is None) or (time_window < 1):
            time_window = 1
        # Prune all empty bands
        channels_per_band = [chans for chans in channels_per_band if len(chans) > 0]
        for scan in self.scans:
            # Average power data along frequency axis first
            num_bands = len(channels_per_band)
            band_data = np.zeros((scan.data.shape[0], num_bands, 4), dtype=scan.data.dtype)
            for band_index, band_channels in enumerate(channels_per_band):
                band_data[:, band_index, :] = scan.data[:, band_channels, :].mean(axis=1)
            scan.data = band_data
            # Now average along time axis, if required
            if time_window > 1:
                # Cap the time window length by the scan length, and only keep an integral number of windows
                num_samples = scan.data.shape[0]
                window = min(max(time_window, 1), num_samples)
                new_len = num_samples // window
                cutoff = new_len * window
                scan.data = scan.data[:cutoff, :, :].reshape((new_len, window, num_bands, 4)).mean(axis=1)
                # Also adjust other time-dependent arrays
                scan.timestamps = scan.timestamps[:cutoff].reshape((new_len, window)).mean(axis=1)
                # The record arrays are more involved - form view, average, and reassemble fields
                num_fields = len(scan.pointing.dtype.names)
                view = scan.pointing.view(dtype=np.float32).reshape((num_samples, num_fields))
                view = view[:cutoff, :].reshape((new_len, window, num_fields)).mean(axis=1)
                scan.pointing = view.view(scan.pointing.dtype).squeeze()
                # All flags in a window are OR'ed together to form the 'average'
                num_fields = len(scan.flags.dtype.names)
                view = scan.flags.view(dtype=np.bool).reshape((num_samples, num_fields))
                view = (view[:cutoff, :].reshape((new_len, window, num_fields)).sum(axis=1) > 0)
                scan.flags = view.view(scan.flags.dtype).squeeze()
                if not scan.target_coords is None:
                    scan.target_coords = scan.target_coords[:, :cutoff].reshape((2, new_len, window)).mean(axis=2)
        if time_window > 1:
            self.corrconf.dump_rate /= time_window
        self.corrconf.merge(channels_per_band)
        return self

    def fit_beams_and_baselines(self, pol='I', band=0, circular_beam=True, compscan=-1, **kwargs):
        """Simultaneously fit beams and baselines to all compound scans.

        This fits a beam pattern and baseline to the total power data of all the
        scans comprising each compound scan, and stores the resulting fitted
        functions in each CompoundScan and Scan object. Only one frequency band
        is used.

        Parameters
        ----------
        pol : {'I', 'Q', 'U', 'V', 'HH', 'VV', 'XX', 'YY'}, optional
            Coherency / Stokes parameter which will be fit. Beam fits are not
            advised for 'Q', 'U' and 'V', which typically exhibit non-Gaussian beams.
        band : int, optional
            Frequency band in which to fit beam and baseline(s)
        circular_beam : {True, False}, optional
            True forces beam to be circular; False allows for elliptical beam
        compscan : integer, optional
            Index of compound scan to fit beam to (default is all compound scans)
        kwargs : dict, optional
            Extra keyword arguments are passed to underlying :mod:`beam_baseline`
            functions

        Returns
        -------
        dataset : :class:`DataSet` object
            Data set with fitted beam/baseline functions added

        """
        # Select all or one compscan
        if compscan == -1:
            compscans = self.compscans
        else:
            compscans = [self.compscans[compscan]]
        # FWHM beamwidth for uniformly illuminated circular dish is 1.03 lambda / D
        # FWHM beamwidth for Gaussian-tapered circular dish is 1.22 lambda / D
        # We are somewhere in between (the factor 1.178 is based on measurements of XDM)
        # TODO: this factor needs to be associated with the antenna
        expected_width = 1.178 * katpoint.lightspeed / (self.freqs[band] * 1e6) / self.antenna.diameter
        if not circular_beam:
            expected_width = [expected_width, expected_width]
        # Degrees of freedom is time-bandwidth product (2 * BW * t_dump) of each sample
        # Stokes I would have double this value, as it is the sum of the independent XX and YY samples
        dof = 2.0 * (self.bandwidths[band] * 1e6) / self.dump_rate
        for compscan in compscans:
            compscan.beam, baselines, compscan.baseline = fit_beam_and_baselines(compscan, expected_width, dof,
                                                                                 pol=pol, band=band, **kwargs)
            for scan, bl in zip(compscan.scans, baselines):
                scan.baseline = bl
        return self
