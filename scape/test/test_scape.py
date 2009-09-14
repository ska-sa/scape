"""Tests for the scape package."""
# pylint: disable-msg=C0103

import time
import unittest

import numpy as np

import scape
import katpoint

class PointSourceScanTestCases(unittest.TestCase):
    """Create point source scan and process it."""

    def setUp(self):
        """Create point source scan."""
        #--------------------------------------------------------------------------------------------------
        #--- User-defined parameters for experiment
        #--------------------------------------------------------------------------------------------------

        # Antenna, target, timestamp
        ant = katpoint.construct_antenna('Test, -33, 18, 30, 15')
        target = "J1230+1223 | *Virgo A, radec, 12:30:49.42, 12:23:28.04, (1408.0 10550.0 4.484 -0.603 -0.0280)"
        time_origin = '2009/06/26 20:00:00 SAST'

        target = katpoint.construct_target(target)
        time_origin = time.mktime(time.strptime(time_origin, '%Y/%m/%d %H:%M:%S %Z'))

        # Frequency setup
        num_channels = 8
        center_freq_MHz = 1.5e3
        channelwidth_MHz = 10.
        rfi_channels = [2, 3, num_channels - 4, num_channels - 3]
        dump_rate = 10.

        freqs = center_freq_MHz + np.arange(-num_channels // 2, num_channels // 2) * channelwidth_MHz
        corrconf = scape.CorrelatorConfig(freqs, np.tile(channelwidth_MHz, num_channels), rfi_channels, dump_rate)

        # Source structure setup
        self.peak_flux = target.flux_density(center_freq_MHz)
        self.expected_width = 1.15 * katpoint.lightspeed / (center_freq_MHz * 1e6) / ant.diameter
        sigma = scape.beam_baseline.fwhm_to_sigma(self.expected_width)
        flux = lambda x, y: self.peak_flux * np.exp(-0.5 * (x ** 2 + y ** 2) / (sigma ** 2))

        # Scan setup
        scans = ['H'] * 5 + ['V'] * 5
        scan_separation = 0.25 * self.expected_width
        along_scan_offset = 2.0 * self.expected_width
        samples_per_scan = 101

        #--------------------------------------------------------------------------------------------------
        #--- Construct data set
        #--------------------------------------------------------------------------------------------------

        # Create scans
        scanlist = []
        time_start = time_origin + 0
        along_scan = np.linspace(-along_scan_offset, along_scan_offset, samples_per_scan)
        h_ind, v_ind = 0, 0
        for n, direction in enumerate(scans):
            if direction == 'H':
                x, y = along_scan, np.tile((h_ind - ((scans.count('H') - 1) // 2)) * scan_separation, samples_per_scan)
                h_ind += 1
            else:
                y, x = along_scan, np.tile((v_ind - ((scans.count('V') - 1) // 2)) * scan_separation, samples_per_scan)
                v_ind += 1
            stokes_I = np.tile(flux(x, y), (num_channels, 1)).transpose()
            data = np.dstack([0.5 * stokes_I, 0.5 * stokes_I, np.zeros(stokes_I.shape), np.zeros(stokes_I.shape)])
            timestamps = time_start + np.arange(samples_per_scan) / dump_rate
            az, el = target.plane_to_sphere(x, y, timestamps, ant)
            pointing = np.rec.fromarrays([az, el], names='az,el')
            flags = np.rec.fromarrays([np.tile(True, samples_per_scan),
                                       np.tile(False, samples_per_scan)], names='valid,nd_on')
            enviro = np.rec.array([(timestamps[0], 35.1, 1020.4, 31.0, 2.0, 45.3)],
                                  dtype=[('timestamp', np.float64), ('temperature', np.float32),
                                         ('pressure', np.float32), ('humidity', np.float32),
                                         ('wind_speed', np.float32), ('wind_direction', np.float32)])
            time_start += samples_per_scan / dump_rate + 10.
            scanlist.append(scape.Scan(data, False, timestamps, pointing, flags, enviro, 'scan', ('scan_%d' % (n,))))

        # Construct data set
        nd_data = scape.gaincal.NoiseDiodeModel(np.array([[center_freq_MHz, 10.]]), np.array([[center_freq_MHz, 10.]]))
        pm = np.ones(20)
        self.dataset = scape.DataSet('', [scape.CompoundScan(scanlist, target)], 'Jy', corrconf, ant, nd_data, pm)

    def test_beam_fit(self):
        """Check if beam fitting is successful."""
        self.dataset.average()
        self.dataset.fit_beams_and_baselines()
        compscan = self.dataset.compscans[0]
        self.assertAlmostEqual(compscan.beam.center[0], 0.0, places=8)
        self.assertAlmostEqual(compscan.beam.center[1], 0.0, places=8)
        # Beam height is underestimated, as remove_spikes() flattens beam top - adjust it based on Gaussian beam
        self.assertAlmostEqual(1.0047 * compscan.beam.height, self.peak_flux, places=0)
        self.assertAlmostEqual(compscan.beam.width, self.expected_width, places=4)
