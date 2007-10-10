## @file point_source_plots.py
#
# Routines used for plots of point source data.
#
# @brief Plotting routines for point source data
# @author Ludwig Schwardt <ludwig@ska.ac.za>,
#         Rudolph van der Merwe <rudolph@ska.ac.za>
# @date 2007/09/27
# copyright (c) 2007 SKA/KAT. All rights reserved.
#


#======================================================================================================================
# Imports
#======================================================================================================================

import xdmsbe.xdmsbelib.interpolator as interp
import xdmsbe.xdmsbelib.vis as vis
from conradmisclib.transforms import rad_to_deg
import matplotlib.axes3d as mplot3d
import numpy as np
import logging
import time


#======================================================================================================================
# Logging strategy
#======================================================================================================================

logger = logging.getLogger('xdmsbe.plots.point_source_plots')


#======================================================================================================================
# Plotting Functions
#======================================================================================================================

## Plot raw power data of multiple scans through a point source.
# @param figColor         Matplotlib Figure object to contain plots
# @param rawPowerScanList List of SingleDishData objects, containing copies of all raw data blocks
# @param expName          Title of experiment
# @return axesColorList   List of matplotlib Axes objects, one per plot
# pylint: disable-msg=R0914
def plot_raw_power(figColor, rawPowerScanList, expName):
    # Set up axes
    axesColorList = []
    numScans = len(rawPowerScanList)
    for sub in range(numScans):
        axesColorList.append(figColor.add_subplot(numScans, 1, sub+1))
    
    # Use relative time axis
    timeRef = np.double(np.inf)
    for rawList in rawPowerScanList:
        for data in rawList:
            timeRef = min(timeRef, data.timeSamples.min())
    
    # Plot of (continuum) raw XX power
    minY = maxY = []
    for scanInd, rawList in enumerate(rawPowerScanList):
        axis = axesColorList[scanInd]
        for val in rawList:
            timeLine = val.timeSamples - timeRef
            contPower = val.powerData[0, :, 0]
            axis.plot(timeLine, contPower, lw=2, color='b')
        if scanInd == numScans-1:
            axis.set_xlabel('Time [s], since %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timeRef)))
        axis.set_ylabel('Raw power')
        if scanInd == 0:
            axis.set_title(expName + ' : raw power for band 0 (XX)')
        minY.append(axis.get_ylim()[0])
        maxY.append(axis.get_ylim()[1])
    
    # Set equal y-axis limits
    for axis in axesColorList:
        axis.set_ylim((np.array(minY).min(), np.array(maxY).max()))
    
    return axesColorList


## Plot baseline fits of multiple scans through a point source.
# @param figColor       Matplotlib Figure object to contain plots
# @param stdScanList    List of StandardSourceScan objects, after power-to-temp conversion, with fitted baselines
# @param expName        Title of experiment
# @return axesColorList List of matplotlib Axes objects, one per plot
# pylint: disable-msg=R0914
def plot_baseline_fit(figColor, stdScanList, expName):
    # Set up axes
    axesColorList = []
    numScans = len(stdScanList)
    # One figure, one subfigure per scan
    for sub in range(numScans):
        axesColorList.append(figColor.add_subplot(numScans, 1, sub+1))
    
    # Use relative time axis
    timeRef = np.double(np.inf)
    for stdScan in stdScanList:
        timeRef = min(timeRef, stdScan.mainData.timeSamples.min())
        for data in stdScan.baselineDataList:
            timeRef = min(timeRef, data.timeSamples.min())
    
    # Plot of (continuum) baseline fits
    minY = maxY = []
    for scanInd, stdScan in enumerate(stdScanList):
        axis = axesColorList[scanInd]
        for val in [stdScan.mainData] + stdScan.baselineDataList:
            timeLine = val.timeSamples - timeRef
            contPower = val.powerData[0].mean(axis=1)
            axis.plot(timeLine, contPower, lw=2, color='b')
            if stdScan.baselineUsesElevation:
                baseline = stdScan.baselineFunc(val.elAng_rad)[0].mean(axis=1)
            else:
                baseline = stdScan.baselineFunc(val.azAng_rad)[0].mean(axis=1)
            axis.plot(timeLine, baseline, lw=2, color='r')
        if scanInd == numScans-1:
            axis.set_xlabel('Time [s], since %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timeRef)))
        axis.set_ylabel('Power (K)')
        if scanInd == 0:
            axis.set_title(expName + ' : baseline fits on XX (all bands)')
        minY.append(axis.get_ylim()[0])
        maxY.append(axis.get_ylim()[1])
    
    # Set equal y-axis limits
    for axis in axesColorList:
        axis.set_ylim((np.array(minY).min(), np.array(maxY).max()))
    
    return axesColorList


## Plot multiple calibrated scans through a point source.
# @param figColorList   List of matplotlib Figure objects to contain plots (one per frequency band)
# @param calibScanList  List of SingleDishData objects containing the fully calibrated main segments of each scan
# @param beamFuncList   List of Gaussian beam functions and valid flags, one per band (None if not available)
# @param expName        Title of experiment
# @return axesColorList List of matplotlib Axes objects, one per plot
# pylint: disable-msg=R0914
def plot_calib_scans(figColorList, calibScanList, beamFuncList, expName):
    # Set up axes
    axesColorList = []
    plotFreqs = calibScanList[0].bandFreqs / 1e9   # Hz to GHz
    numBands = len(plotFreqs)
    numScans = len(calibScanList)
    # One figure per frequency band
    for band in range(numBands):
        # One subfigure per scan
        for sub in range(numScans):
            axesColorList.append(figColorList[band].add_subplot(numScans, 1, sub+1))
    
    # Use relative time axis
    timeRef = np.double(np.inf)
    for val in calibScanList:
        timeRef = min(timeRef, val.timeSamples.min())
    
    # Plot one figure per frequency band
    minY = maxY = []
    for scanInd, val in enumerate(calibScanList):
        timeLine = val.timeSamples - timeRef
        for band in range(numBands):
            axesInd = band*numScans + scanInd
            axis = axesColorList[axesInd]
            # Total power in band
            axis.plot(timeLine, val.total_power()[:, band], color='b', lw=2)
            if beamFuncList != None:
                # Slice through fitted beam function along the same coordinates (change colors for invalid beams)
                if beamFuncList[band][1]:
                    beamColor = 'r'
                else:
                    beamColor = 'y'
                axis.plot(timeLine, beamFuncList[band][0](val.targetCoords[:, 0:2]), color=beamColor, lw=2)
            if scanInd == numScans-1:
                axis.set_xlabel('Time [s], since %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timeRef)))
            axis.set_ylabel('Power (K)')
            if scanInd == 0:
                axis.set_title(expName + ' : calibrated total power in band %d : %3.3f GHz' % (band, plotFreqs[band]))
            minY.append(axis.get_ylim()[0])
            maxY.append(axis.get_ylim()[1])
    
    # Set equal y-axis limits
    for axis in axesColorList:
        axis.set_ylim((np.array(minY).min(), np.array(maxY).max()))
    
    return axesColorList


## Plot beam pattern fitted to multiple scans through a single point source, in target space.
# This plots contour ellipses of a Gaussian beam function fitted to multiple scans through a point source,
# as well as the power values of the scans themselves as a pseudo-3D plot. It highlights the success of the
# beam fitting procedure.
# @param figColorList   List of matplotlib Figure objects to contain plots (one per frequency band)
# @param calibScanList  List of SingleDishData objects of calibrated main scans (one per scan)
# @param beamFuncList   List of Gaussian beam functions and valid flags (one per band)
# @param expName        Title of experiment
# @return axesColorList List of matplotlib Axes objects, one per plot
# pylint: disable-msg=R0912,R0913,R0914,R0915
def plot_beam_pattern_target(figColorList, calibScanList, beamFuncList, expName):
    # Set up axes
    axesColorList = []
    # Use the frequency bands of the first scan as reference for the rest
    plotFreqs = calibScanList[0].bandFreqs / 1e9   # Hz to GHz
    numBands = len(plotFreqs)
    # One figure per frequency band
    for band in range(numBands):
        axesColorList.append(figColorList[band].add_subplot(1, 1, 1))
    
    # Extract total power and target coordinates of all scans
    totalPower = []
    targetCoords = []
    for scan in calibScanList:
        totalPower.append(scan.total_power())
        targetCoords.append(scan.targetCoords)
    totalPower = np.concatenate(totalPower)
    targetCoords = np.concatenate(targetCoords)
    
    # Iterate through figures (one per band)
    for band in range(numBands):
        axis = axesColorList[band]
        power = totalPower[:, band]
        # Show the locations of the scan samples themselves, with marker sizes indicating power values
        vis.plot_marker_3d(axis, rad_to_deg(targetCoords[:, 0]), rad_to_deg(targetCoords[:, 1]), power)
        # Plot the fitted Gaussian beam function as contours
        if beamFuncList[band][1]:
            ellType, centerType = 'r-', 'r+'
        else:
            ellType, centerType = 'y-', 'y+'
        gaussEllipses, gaussCenter = vis.plot_gaussian_ellipse(axis, beamFuncList[band][0].mean, \
                                                               np.diag(beamFuncList[band][0].var), \
                                                               contour=[0.5, 0.1], ellType=ellType, \
                                                               centerType=centerType, lineWidth=2)
        # Change from radians to degrees for prettier picture
        for ellipse in gaussEllipses:
            ellipse[0].set_xdata(rad_to_deg(ellipse[0].get_xdata()))
            ellipse[0].set_ydata(rad_to_deg(ellipse[0].get_ydata()))
        gaussCenter[0].set_xdata(rad_to_deg(gaussCenter[0].get_xdata()[0]))
        gaussCenter[0].set_ydata(rad_to_deg(gaussCenter[0].get_ydata()[0]))
    
    # Axis settings and labels
    for band in range(numBands):
        axis = axesColorList[band]
        axis.set_xlim(rad_to_deg(targetCoords[:, 0].min()), rad_to_deg(targetCoords[:, 0].max()))
        axis.set_ylim(rad_to_deg(targetCoords[:, 1].min()), rad_to_deg(targetCoords[:, 1].max()))
        axis.set_aspect('equal')
        axis.set_xlabel('Target coord 1 (deg)')
        axis.set_ylabel('Target coord 2 (deg)')
        axis.set_title(expName + ' : Beam fitted in band %d : %3.3f GHz' % (band, plotFreqs[band]))
    
    return axesColorList


## Plot rough beam pattern derived from multiple raster scans through a single point source, in target space.
# This works with the power values derived from multiple scans through a single point source, typically made
# in a horizontal raster pattern. It interpolates the power values onto a regular uniform grid, and plots
# contours of the interpolated function. It also assumes that the scans themselves follow a uniform grid pattern 
# in target space, which allows a 3D mesh/wireframe plot of the actual power values.
# @param figColorList   List of matplotlib Figure objects to contain plots (two per frequency band)
# @param calibScanList  List of SingleDishData objects of calibrated main scans (one per scan)
# @param expName        Title of experiment
# @return axesColorList List of matplotlib Axes objects, one per plot
# pylint: disable-msg=R0912,R0913,R0914,R0915
def plot_beam_pattern_raster(figColorList, calibScanList, expName):
    # Set up axes
    axesColorList = []
    # Use the frequency bands of the first scan as reference for the rest
    plotFreqs = calibScanList[0].bandFreqs / 1e9   # Hz to GHz
    numBands = len(plotFreqs)
    # Two figures per frequency band
    for band in range(numBands):
        # Normal axes for contour plot
        axesColorList.append(figColorList[band].add_subplot(1, 1, 1))
        # 3D axes for wireframe plot
        axesColorList.append(mplot3d.Axes3D(figColorList[band + numBands]))
    
    # Extract total power and and first two target coordinates (assumed to be (az,el)) of all scans
    azScans = []
    elScans = []
    totalPowerScans = []
    for scan in calibScanList:
        azScan = scan.targetCoords[:, 0]
        # Reverse any scans with descending azimuth angle, so that all scans run in the same direction
        if azScan[0] < azScan[-1]:
            azScans.append(azScan)
            elScans.append(scan.targetCoords[:, 1])
            totalPowerScans.append(scan.total_power()[np.newaxis, :, :])
        else:
            azScans.append(np.flipud(azScan))
            elScans.append(np.flipud(scan.targetCoords[:, 1]))
            totalPowerScans.append(np.flipud(scan.total_power())[np.newaxis, :, :])
    azScans = np.vstack(azScans)
    elScans = np.vstack(elScans)
    totalPowerScans = np.concatenate(totalPowerScans, axis=0)
    # Set up uniform 101x101 mesh grid for contour plot
    targetX = np.linspace(azScans.min(), azScans.max(), 101)
    targetY = np.linspace(elScans.min(), elScans.max(), 101)
    targetMeshX, targetMeshY = np.meshgrid(targetX, targetY)
    meshCoords = np.vstack((targetMeshX.ravel(), targetMeshY.ravel()))
    # Jitter the target coordinates to make degenerate triangles unlikely during Delaunay triangulation for contours
    jitter = np.vstack((azScans.std() * np.random.randn(azScans.size), \
                        elScans.std() * np.random.randn(elScans.size)))
    jitteredCoords = np.vstack((azScans.ravel(), elScans.ravel())) + 0.0001 * jitter
    
    # Iterate through figures (two per band)
    for band in range(numBands):
        power = totalPowerScans[:, :, band]
        # 2D contour plot
        axis = axesColorList[2*band]
        # Show the locations of the scan samples themselves
        axis.plot(rad_to_deg(azScans.ravel()), rad_to_deg(elScans.ravel()), '.b', alpha=0.5)
        # Interpolate the power values onto a 2-D az-el grid for a smoother contour plot
        gridder = interp.Delaunay2DFit(defaultVal = power.min())
        gridder.fit(jitteredCoords, power.ravel())
        meshPower = gridder(meshCoords).reshape(targetY.size, targetX.size)
        # Choose contour levels as fractions of the peak power
        contourLevels = np.arange(0.1, 1.0, 0.1) * meshPower.max()
        # Indicate half-power beamwidth (contourLevel = 0.5) with wider line
        lineWidths = 2*np.ones(len(contourLevels))
        lineWidths[4] = 4
        # Color the 0.5 and 0.1 contours red, to coincide with the scheme followed in the beam fitting plots
        colors = ['b'] * len(contourLevels)
        colors[0] = colors[4] = 'r'
        # Plot contours of interpolated beam pattern
        axis.contour(rad_to_deg(targetMeshX), rad_to_deg(targetMeshY), meshPower, contourLevels, \
                     linewidths=lineWidths, colors=colors)
        # 3D wireframe plot
        axis = axesColorList[2*band + 1]
        axis.plot_wireframe(rad_to_deg(azScans), rad_to_deg(elScans), power)
        
    # Axis settings and labels
    for band in range(2*numBands):
        axis = axesColorList[band]
        axis.set_xlim(rad_to_deg(targetX[0]), rad_to_deg(targetX[-1]))
        axis.set_ylim(rad_to_deg(targetY[0]), rad_to_deg(targetY[-1]))
        axis.set_aspect('equal')
        axis.set_xlabel('Target coord 1 (deg)')
        axis.set_ylabel('Target coord 2 (deg)')
        axis.set_title(expName + ' : Beam pattern in band %d : %3.3f GHz' % (band // 2, plotFreqs[band // 2]))

    return axesColorList


## Plot beam patterns fitted to multiple scans through multiple point sources, in "instantaneous" mount space.
# The purpose of this plot is to examine the quality of beam functions fitted to multiple sources, based on multiple
# scans through each source. The scans are adjusted to appear as if they happened instantaneously for each source,
# which makes the sources appear stationary in mount coordinates (regardless of their actual motion). The beam
# functions are indicated by contour ellipses, while the power values on each scan are illustrated with pseudo-3D
# plots.
# @param figColorList   List of matplotlib Figure objects to contain plots (one per frequency band)
# @param calibListList  List of list of SingleDishData objects of calibrated main scans (per source, and then per scan)
# @param beamListList   List of list of Gaussian beam functions (per source, and then per band)
# @param transformList  List of TargetToInstantMountTransform objects (one per source)
# @param expName        Title of experiment
# @return axesColorList List of matplotlib Axes objects, one per plot
# pylint: disable-msg=R0912,R0913,R0914,R0915
def plot_beam_patterns_mount(figColorList, calibListList, beamListList, transformList, expName):
    # Set up axes
    axesColorList = []
    # Use the frequency bands of the first scan of the first source as reference for the rest
    plotFreqs = calibListList[0][0].bandFreqs / 1e9   # Hz to GHz
    numBands = len(plotFreqs)
    # One figure per frequency band
    for band in range(numBands):
        axesColorList.append(figColorList[band].add_subplot(1, 1, 1))

    azMin_deg = elMin_deg = np.inf
    azMax_deg = elMax_deg = -np.inf
    # Iterate through sources
    for calibScanList, beamFuncList, targetToInstantMount in zip(calibListList, beamListList, transformList):
        # Extract total power and instantaneous mount coordinates of all scans of a specific source
        totalPower, azAng_deg, elAng_deg = [], [], []
        for scan in calibScanList:
            totalPower.append(scan.total_power())
            mountCoords = np.array([targetToInstantMount(targetCoord) for targetCoord in scan.targetCoords])
            azAng_deg.append(mountCoords[:, 0].tolist())
            elAng_deg.append(mountCoords[:, 1].tolist())
        totalPower = np.concatenate(totalPower)
        azAng_deg, elAng_deg = np.concatenate(azAng_deg), np.concatenate(elAng_deg)
        
        # Determine overall axis limits
        azMin_deg = min(azMin_deg, azAng_deg.min())
        azMax_deg = max(azMax_deg, azAng_deg.max())
        elMin_deg = min(elMin_deg, elAng_deg.min())
        elMax_deg = max(elMax_deg, elAng_deg.max())
        # Iterate through figures (one per band)
        for band in range(numBands):
            axis = axesColorList[band]
            power = totalPower[:, band]
            # Show the locations of the scan samples themselves, with marker sizes indicating power values
            vis.plot_marker_3d(axis, azAng_deg, elAng_deg, power)
            # Plot the fitted Gaussian beam function as contours
            if beamFuncList[band][1]:
                ellType, centerType = 'r-', 'r+'
            else:
                ellType, centerType = 'y-', 'y+'
            gaussEllipses, gaussCenter = vis.plot_gaussian_ellipse(axis, beamFuncList[band][0].mean, \
                                                                   np.diag(beamFuncList[band][0].var), \
                                                                   contour=[0.5, 0.1], ellType=ellType, \
                                                                   centerType=centerType, lineWidth=2)
            # Convert Gaussian ellipse and center to mount coordinate system
            # (where needed, append rotator angle = 0.0 to go from 2-D ellipse coords to 3-D target coords)
            for ellipse in gaussEllipses:
                fromData = np.array((ellipse[0].get_xdata(), ellipse[0].get_ydata())).transpose()
                numPoints = len(fromData)
                newEllipse = np.zeros((numPoints, 2), dtype='double')
                for k in xrange(numPoints):
                    newEllipse[k, :] = targetToInstantMount(fromData[k].tolist() + [0.0])
                ellipse[0].set_xdata(newEllipse[:, 0])
                ellipse[0].set_ydata(newEllipse[:, 1])
            newCenter = targetToInstantMount(beamFuncList[band][0].mean.tolist() + [0.0])
            gaussCenter[0].set_xdata([newCenter[0]])
            gaussCenter[0].set_ydata([newCenter[1]])

    # Axis settings and labels
    for band in range(numBands):
        axis = axesColorList[band]
        axis.set_xlim(azMin_deg, azMax_deg)
        axis.set_ylim(elMin_deg, elMax_deg)
        axis.set_aspect('equal')
        axis.set_xlabel('Azimuth (deg)')
        axis.set_ylabel('Elevation (deg)')
        axis.set_title(expName + ' : Beams fitted in band %d : %3.3f GHz' % (band, plotFreqs[band]))
        
    return axesColorList


## Plot antenna gain info derived from multiple scans through a point source.
# @param figColor       Matplotlib Figure object to contain plots
# @param resultList     List of results, of form (bandFreqs, pointSourceSensitivity, effArea, antGain)
# @param expName        Title of experiment
# @return axesColorList List of matplotlib Axes objects, one per plot
def plot_antenna_gain(figColor, resultList, expName):
    # Set up axes
    axesColorList = []
    bandFreqs, pointSourceSensitivity, effArea, antGain = resultList
    plotFreqs = bandFreqs / 1e9   # Hz to GHz
    # One figure, 3 subfigures
    for sub in range(3):
        axesColorList.append(figColor.add_subplot(3, 1, sub+1))
    
    axis = axesColorList[0]
    try:
        vis.draw_std_corridor(axis, plotFreqs, pointSourceSensitivity.mean, pointSourceSensitivity.sigma, \
                              muLineType='b-o')
    except AttributeError:
        axis.plot(plotFreqs, pointSourceSensitivity, '-ob')
    axis.set_ylim(min(0.95*pointSourceSensitivity.min(), axis.get_ylim()[0]),
                  max(1.05*pointSourceSensitivity.max(), axis.get_ylim()[1]))
    axis.grid()
    axis.set_xticklabels([])
    axis.set_ylabel('Sensitivity (Jy/K)')
    axis.set_title(expName + ' : Point source sensitivity')
    axis = axesColorList[1]
    try:
        vis.draw_std_corridor(axis, plotFreqs, effArea.mean, effArea.sigma, muLineType='b-o')
    except AttributeError:
        axis.plot(plotFreqs, effArea, '-ob')
    axis.set_ylim(min(0.95*effArea.min(), axis.get_ylim()[0]), max(1.05*effArea.max(), axis.get_ylim()[1]))
    axis.grid()
    axis.set_xticklabels([])
    axis.set_ylabel('Area (m^2)')
    axis.set_title(expName + ' : Antenna effective area')
    axis = axesColorList[2]
    try:
        vis.draw_std_corridor(axis, plotFreqs, antGain.mean, antGain.sigma, muLineType='b-o')
    except AttributeError:
        axis.plot(plotFreqs, antGain, '-ob')
    axis.set_ylim(min(0.98*antGain.min(), axis.get_ylim()[0]), max(1.02*antGain.max(), axis.get_ylim()[1]))
    axis.grid()
    axis.set_xlabel('Frequency (GHz)')
    axis.set_ylabel('Gain (dB)')
    axis.set_title(expName + ' : Antenna gain')
    
    return axesColorList


## Plot antenna gain info derived from multiple scans through multiple positions of a point source.
# @param figColorList   List of matplotlib Figure objects to contain plots
# @param resultList     List of results, as (bandFreqs, names, sourceAngs, sensitivity, effArea, antGain, pointErr)
# @param expName        Title of experiment
# @return axesColorList List of matplotlib Axes objects, one per plot
def plot_gain_curve(figColorList, resultList, expName):
    # Set up axes
    axesColorList = []
    # pylint: disable-msg=W0612
    bandFreqs, nameList, sourceAngs, pssBlock, effAreaBlock, antGainBlock, pointingErrors = resultList
    sourceElAng_deg = sourceAngs[:, 1]
    plotFreqs = bandFreqs / 1e9   # Hz to GHz
    numBands = len(plotFreqs)
    # One figure per frequency band
    for band in range(numBands):
        # 3 subfigures
        for sub in range(3):
            axesColorList.append(figColorList[band].add_subplot(3, 1, sub+1))
    
    # Plot one figure per frequency band
    for band in range(numBands):
        axesInd = band*3
        axis = axesColorList[axesInd]
        pointSourceSensitivity = pssBlock[:, band]
        try:
            vis.draw_std_corridor(axis, sourceElAng_deg, pointSourceSensitivity.mean, pointSourceSensitivity.sigma, \
                                  muLineType='b-o')
        except AttributeError:
            axis.plot(sourceElAng_deg, pointSourceSensitivity, '-ob')
        axis.set_ylim(min(0.95*pointSourceSensitivity.min(), axis.get_ylim()[0]),
                      max(1.05*pointSourceSensitivity.max(), axis.get_ylim()[1]))
        axis.grid()
        axis.set_xticklabels([])
        axis.set_ylabel('Sensitivity (Jy/K)')
        axis.set_title(expName + ' : Point source sensitivity in band %d : %3.3f GHz' % (band, plotFreqs[band]))
        
        axesInd = band*3 + 1
        axis = axesColorList[axesInd]
        effArea = effAreaBlock[:, band]
        try:
            vis.draw_std_corridor(axis, sourceElAng_deg, effArea.mean, effArea.sigma, muLineType='b-o')
        except AttributeError:
            axis.plot(sourceElAng_deg, effArea, '-ob')
        axis.set_ylim(min(0.95*effArea.min(), axis.get_ylim()[0]), max(1.05*effArea.max(), axis.get_ylim()[1]))
        axis.grid()
        axis.set_xticklabels([])
        axis.set_ylabel('Area (m^2)')
        axis.set_title(expName + ' : Antenna effective area in band %d : %3.3f GHz' % (band, plotFreqs[band]))
        
        axesInd = band*3 + 2
        axis = axesColorList[axesInd]
        antGain = antGainBlock[:, band]
        try:
            vis.draw_std_corridor(axis, sourceElAng_deg, antGain.mean, antGain.sigma, muLineType='b-o')
        except AttributeError:
            axis.plot(sourceElAng_deg, antGain, '-ob')
        axis.set_ylim(min(0.98*antGain.min(), axis.get_ylim()[0]), max(1.02*antGain.max(), axis.get_ylim()[1]))
        axis.grid()
        axis.set_xlabel('Elevation angle (degrees)')
        axis.set_ylabel('Gain (dB)')
        axis.set_title(expName + ' : Antenna gain in band %d : %3.3f GHz' % (band, plotFreqs[band]))
    
    return axesColorList


## Plot pointing errors derived from multiple point sources.
# @param figColor       Matplotlib Figure object to contain plots
# @param resultList     List of results, as (bandFreqs, names, sourceAngs, sensitivity, effArea, antGain, pointErr)
# @param expName        Title of experiment
# @param scale          Scale of pointing errors, used to exaggerate them for visibility [1]
# @return axesColorList List of matplotlib Axes objects, one per plot
def plot_pointing_error(figColor, resultList, expName, scale=1):
    # Set up axes
    axesColorList = []
    # pylint: disable-msg=W0612
    bandFreqs, nameList, sourceAngs, pssBlock, effAreaBlock, antGainBlock, pointingErrors = resultList
    # One figure, 1 subfigure
    axesColorList.append(figColor.add_subplot(1, 1, 1))
    
    # Extract relevant angles from list
    pointErr = np.array([sourceAngs[ind].tolist() + pointingErrors[ind].tolist() \
                         for ind in xrange(len(sourceAngs)) if not np.isnan(pointingErrors[ind][0])])
    steeredAz, steeredEl, azError, elError, azErrorSigma, elErrorSigma = [pointErr[:, ind] for ind in range(6)]
    # Scale errors to make them more visible
    azError *= scale
    elError *= scale
    azErrorSigma *= scale
    elErrorSigma *= scale
    meanAz, meanEl = steeredAz + azError, steeredEl + elError
    # This forms the lines connecting steered and estimated position markers
    arrowsAz = np.vstack((steeredAz, meanAz, np.tile(np.nan, (len(meanAz))))).transpose().ravel()
    arrowsEl = np.vstack((steeredEl, meanEl, np.tile(np.nan, (len(meanEl))))).transpose().ravel()
    
    axis = axesColorList[0]
    # Location of "true" source, indicated by Gaussian center + spread (1=sigma and 2-sigma contours)
    for ind in xrange(len(meanAz)):
        vis.plot_gaussian_ellipse(axis, (meanAz[ind], meanEl[ind]), \
                                  np.diag((azErrorSigma[ind] ** 2, elErrorSigma[ind] ** 2)), \
                                  contour=np.exp(-0.5*np.array([1, 2]) ** 2), \
                                  ellType='r-', centerType='or', lineWidth=1, markerSize=4)
    # Lines connecting markers
    axis.plot(arrowsAz, arrowsEl, 'k')
    # Marker indicating steered position (where the source was thought to be)
    axis.plot(steeredAz, steeredEl, 'sk', markersize=4)
    axis.set_xlim(axis.get_xlim()[0] - 10, axis.get_xlim()[1] + 10)
    axis.set_ylim(axis.get_ylim()[0] - 10, axis.get_ylim()[1] + 10)
    axis.set_aspect('equal')
    axis.set_xlabel('Azimuth (deg)')
    axis.set_ylabel('Elevation (deg)')
    axis.set_title(expName + ' : Pointing errors')
    
    return axesColorList
