"""Tests for the xdmfits module."""
# pylint: disable-msg=C0103

import unittest
import pkg_resources
import os.path

import numpy as np

import scape
import katpoint


class XDMFitsTestCases(unittest.TestCase):
    """Load existing XDM FITS data set and process it."""

    def setUp(self):
        """Find path to test data set and store expected results."""
        fitsfile = os.path.join('J1959+4044_2009-07-18-21h40', 'scheduled_obs_2009-07-18-21h40_0000.fits')
        self.dataset = pkg_resources.resource_filename('scape.test', fitsfile)
        self.catalogue = katpoint.Catalogue()
        self.catalogue.add('J1959+4044 | *Cygnus A | CygA | 3C405, radec J2000, '
                           '19:59:28.36, 40:44:2.1, (20.0 2000.0 4.695 0.085 -0.178)')
        self.average_flux = 1513.4451
        self.temperature = 2.00
        self.pressure = 875.00
        self.humidity = 56.60
        self.az = 29.5937309
        self.el = 14.1074702
        # Original beam values based on averaged noise diode cal measurements
        # self.beam_height = 108.8561436
        # self.beam_width = 0.9102625
        # self.baseline_height = 151.6516238
        # self.delta_az = -0.1430513
        # self.delta_el = -0.0676434
        # Beam values based on linearly interpolated cal measurements
        # self.beam_height = 108.9559406
        # self.beam_width = 0.9102636
        # self.baseline_height = 151.7917742
        # self.delta_az = -0.1430509
        # self.delta_el = -0.0686143
        # Beam values based on cal measurements averaged in frequency, linearly interpolated in time (900-second bins)
        self.beam_height = 108.7471834
        self.beam_width = 0.9110111
        self.baseline_height = 151.3675153
        self.delta_az = -0.1430157
        self.delta_el = -0.0675574

    def test_beam_fit(self):
        """Load XDM FITS data set and do point source scan analysis on it."""
        # Load dataset, and quit if XDM FITS support is not included or FITS file was not found
        try:
            d = scape.DataSet(self.dataset, catalogue=self.catalogue)
        except (ImportError, IOError):
            try:
                import nose
                raise nose.SkipTest
            except ImportError:
                return

        # Standard continuum reduction
        d = d.select(freqkeep=d.channel_select)
        d.convert_power_to_temperature()
        d = d.select(labelkeep='scan', copy=False)
        # Save original channel frequencies before averaging
        channel_freqs = d.freqs
        d.average()
        d.fit_beams_and_baselines(pol='I', spike_width=3)
        # Quick checks that all went well
        self.assert_(len(d.compscans) > 0)
        compscan = d.compscans[0]
        self.assert_(compscan.beam is not None)

        # Calculate average target flux over entire band
        flux_spectrum = [compscan.target.flux_density(freq) for freq in channel_freqs]
        average_flux = np.mean([flux for flux in flux_spectrum if flux])
        # Obtain middle timestamp of compound scan, where all pointing calculations are done
        middle_time = np.median([scan.timestamps for scan in compscan.scans], axis=None)
        # Obtain average environmental data
        temperature = np.mean(d.enviro['temperature']['value']) if 'temperature' in d.enviro else np.nan
        pressure = np.mean(d.enviro['pressure']['value']) if 'pressure' in d.enviro else np.nan
        humidity = np.mean(d.enviro['humidity']['value']) if 'humidity' in d.enviro else np.nan
        # Calculate pointing offset
        # Start with requested (az, el) coordinates, as they apply at the middle time for a moving target
        requested_azel = compscan.target.azel(middle_time)
        # Correct for refraction, which becomes the requested value at input of pointing model
        rc = katpoint.RefractionCorrection()
        requested_azel = [requested_azel[0], rc.apply(requested_azel[1], temperature, pressure, humidity)]
        requested_azel = katpoint.rad2deg(np.array(requested_azel))
        # Fitted beam center is in (x, y) coordinates, in projection centred on target
        beam_center_xy = compscan.beam.center
        # Convert this offset back to spherical (az, el) coordinates
        beam_center_azel = compscan.target.plane_to_sphere(beam_center_xy[0], beam_center_xy[1], middle_time)
        # Now correct the measured (az, el) for refraction and then apply the old pointing model
        # to get a "raw" measured (az, el) at the output of the pointing model
        beam_center_azel = [beam_center_azel[0], rc.apply(beam_center_azel[1], temperature, pressure, humidity)]
        beam_center_azel = d.antenna.pointing_model.apply(*beam_center_azel)
        beam_center_azel = katpoint.rad2deg(np.array(beam_center_azel))
        # Make sure the offset is a small angle around 0 degrees
        offset_azel = scape.stats.angle_wrap(beam_center_azel - requested_azel, 360.)

        # Compare calculations to expected results (assuming float32 precision)
        self.assertAlmostEqual(compscan.beam.height, self.beam_height, places=4)
        self.assertAlmostEqual(katpoint.rad2deg(compscan.beam.width), self.beam_width, places=7)
        self.assertAlmostEqual(compscan.baseline_height(), self.baseline_height, places=4)
        self.assertEqual(compscan.beam.refined, 3)
        self.assertAlmostEqual(average_flux, self.average_flux, places=3)
        self.assertAlmostEqual(temperature, self.temperature, places=2)
        self.assertAlmostEqual(pressure, self.pressure, places=2)
        self.assertAlmostEqual(humidity, self.humidity, places=2)
        self.assertAlmostEqual(requested_azel[0], self.az, places=5)
        self.assertAlmostEqual(requested_azel[1], self.el, places=5)
        self.assertAlmostEqual(offset_azel[0], self.delta_az, places=7)
        self.assertAlmostEqual(offset_azel[1], self.delta_el, places=7)
