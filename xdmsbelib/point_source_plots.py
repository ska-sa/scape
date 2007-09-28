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
import matplotlib.cm as cm
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
                baseline = stdScan.baselineFunc(val.elAng)[0].mean(axis=1)
            else:
                baseline = stdScan.baselineFunc(val.azAng)[0].mean(axis=1)
            axis.plot(timeLine, baseline, lw=2, color='r')
        if scanInd == numScans-1:
            axis.set_xlabel('Time [s], since %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timeRef)))
        axis.set_ylabel('Power [K]')
        if scanInd == 0:
            axis.set_title(expName + ' : baseline fits on XX (all bands)')
        minY.append(axis.get_ylim()[0])
        maxY.append(axis.get_ylim()[1])
    
    # Set equal y-axis limits
    for axis in axesColorList:
        axis.set_ylim((np.array(minY).min(), np.array(maxY).max()))
    
    return axesColorList


## Plot multiple calibrated scans through a point source.
# @param figColorList   List of matplotlib Figure objects to contain plots
# @param calibScanList  List of SingleDishData objects containing the fully calibrated main segments of each scan
# @param beamFuncList   List of Gaussian beam functions, one per band
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
            # Slice through fitted beam pattern function along the same coordinates
            axis.plot(timeLine, beamFuncList[band](val.targetCoords[:, 0:2]), color='r', lw=2)
            if scanInd == numScans-1:
                axis.set_xlabel('Time [s], since %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timeRef)))
            axis.set_ylabel('Power [K]')
            if scanInd == 0:
                axis.set_title(expName + ' : calibrated total power in band %d : %3.3f GHz' % (band, plotFreqs[band]))
            minY.append(axis.get_ylim()[0])
            maxY.append(axis.get_ylim()[1])
    
    # Set equal y-axis limits
    for axis in axesColorList:
        axis.set_ylim((np.array(minY).min(), np.array(maxY).max()))
    
    return axesColorList


## Plot beam pattern fitted to multiple scans through a point source.
# @param figColorList   List of matplotlib Figure objects to contain plots
# @param calibScanList  List of SingleDishData objects containing the fully calibrated main segments of each scan
# @param beamFuncList   List of Gaussian beam functions, one per band
# @param expName        Title of experiment
# @return axesColorList List of matplotlib Axes objects, one per plot
# pylint: disable-msg=R0914
def plot_beam_pattern(figColorList, calibScanList, beamFuncList, expName):
    # Set up axes
    axesColorList = []
    plotFreqs = calibScanList[0].bandFreqs / 1e9   # Hz to GHz
    numBands = len(plotFreqs)
    # One figure per frequency band
    for band in range(numBands):
        axesColorList.append(figColorList[band].add_subplot(1, 1, 1))
    # Extract total power and coordinates
    targetCoords = []
    totalPower = []
    for scan in calibScanList:
        targetCoords.append(scan.targetCoords)
        totalPower.append(scan.total_power())
    targetCoords = np.concatenate(targetCoords)[:, 0:2].transpose()
    totalPower = np.concatenate(totalPower)
    
    # Setup uniform mesh grid for interpolated power plot
    targetX = np.linspace(targetCoords[0].min(), targetCoords[0].max(), 101)
    targetY = np.linspace(targetCoords[1].min(), targetCoords[1].max(), 101)
    targetMeshX, targetMeshY = np.meshgrid(targetX, targetY)
    uniformGridCoords = np.vstack((targetMeshX.ravel(), targetMeshY.ravel()))
    # Setup slightly shifted uniform grid so that the centers of pcolor patches are correctly aligned
    # with target coordinates for the rest of the plots
    targetXstep = targetX[1] - targetX[0]
    targetYstep = targetY[1] - targetY[0]
    pcolorTargetX = np.array(targetX.tolist() + [targetX[-1]+targetXstep]) - targetXstep/2.0
    pcolorTargetY = np.array(targetY.tolist() + [targetY[-1]+targetYstep]) - targetYstep/2.0
    for band in range(numBands):
        axis = axesColorList[band]
        power = totalPower[:, band]
        # Jitter the target coordinates to prevent degenerate triangles during Delaunay triangulation
        jitter = np.vstack((0.001 * targetCoords[0].std() * np.random.randn(targetCoords.shape[1]), \
                            0.001 * targetCoords[1].std() * np.random.randn(targetCoords.shape[1])))
        jitteredCoords = targetCoords + jitter
        # Interpolate the raw power values onto a grid, and display using pcolor
        beamRoughFit = interp.Delaunay2DFit(defaultVal=power.min())
        beamRoughFit.fit(jitteredCoords, power)
        uniformGridRoughPower = beamRoughFit(uniformGridCoords).reshape(targetY.size, targetX.size)
        axis.pcolor(pcolorTargetX, pcolorTargetY, uniformGridRoughPower, shading='flat', cmap=cm.gray_r)
#        axesColorList[axesInd].contour(targetMeshX, targetMeshY, uniformGridRoughPower, 25, colors='b')
        # Show the locations of the scan samples themselves
        axis.plot(targetCoords[0], targetCoords[1], '.b', zorder=1)
        # Plot the fitted Gaussian beam function as contours
        vis.plot_gaussian_ellipse(axis, beamFuncList[band].mean, np.diag(beamFuncList[band].var), \
                                  contour=[0.5, 0.1], ellipseLineType='r-', centerLineType='r+', lineWidth=2)
        axis.set_xlim(targetX.min(), targetX.max())
        axis.set_ylim(targetY.min(), targetY.max())
        axis.set_aspect('equal')
        axis.set_xlabel('Target coord 1')
        axis.set_ylabel('Target coord 2')
        axis.set_title(expName + ' : Beam fitted in band %d : %3.3f GHz' % (band, plotFreqs[band]))
    
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
        vis.draw_std_corridor(pointSourceSensitivity.mean, pointSourceSensitivity.sigma, \
                              plotFreqs, axis, muLineType='b-o')
    except AttributeError:
        axis.plot(plotFreqs, pointSourceSensitivity, '-ob')
    axis.set_ylim(0.95*pointSourceSensitivity.min(), 1.05*pointSourceSensitivity.max())
    axis.grid()
    axis.set_xticklabels([])
    axis.set_ylabel('Sensitivity (Jy/K)')
    axis.set_title(expName + ' : Point source sensitivity')
    axis = axesColorList[1]
    try:
        vis.draw_std_corridor(effArea.mean, effArea.sigma, plotFreqs, axis, muLineType='b-o')
    except AttributeError:
        axis.plot(plotFreqs, effArea, '-ob')
    axis.set_ylim(0.95*effArea.min(), 1.05*effArea.max())
    axis.grid()
    axis.set_xticklabels([])
    axis.set_ylabel('Area (m^2)')
    axis.set_title(expName + ' : Antenna effective area')
    axis = axesColorList[2]
    try:
        vis.draw_std_corridor(antGain.mean, antGain.sigma, plotFreqs, axis, muLineType='b-o')
    except AttributeError:
        axis.plot(plotFreqs, antGain, '-ob')
    axis.set_ylim(0.95*antGain.min(), 1.05*antGain.max())
    axis.grid()
    axis.set_xlabel('Frequency (GHz)')
    axis.set_ylabel('Gain (dB)')
    axis.set_title(expName + ' : Antenna gain')
    
    return axesColorList
