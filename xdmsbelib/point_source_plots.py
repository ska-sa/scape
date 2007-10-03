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
from acsm.coordinate import Coordinate
import acsm.transform
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
            axis.set_ylabel('Power (K)')
            if scanInd == 0:
                axis.set_title(expName + ' : calibrated total power in band %d : %3.3f GHz' % (band, plotFreqs[band]))
            minY.append(axis.get_ylim()[0])
            maxY.append(axis.get_ylim()[1])
    
    # Set equal y-axis limits
    for axis in axesColorList:
        axis.set_ylim((np.array(minY).min(), np.array(maxY).max()))
    
    return axesColorList


## Plot beam patterns fitted to multiple scans through multiple point sources.
# @param figColorList   List of matplotlib Figure objects to contain plots (one per frequency band)
# @param calibListList  List of list of SingleDishData objects of calibrated main scans (per source, and then per scan)
# @param beamListList   List of list of Gaussian beam functions (per source, and then per band)
# @param expName        Title of experiment
# @param refSource      Index of source in list considered to be reference for coordinate system [0]
# @param plotPower      True if calibrated power values should be plotted as pcolor backdrop to beam patterns [True]
# @return axesColorList List of matplotlib Axes objects, one per plot
# pylint: disable-msg=R0912,R0913,R0914,R0915
def plot_beam_patterns(figColorList, calibListList, beamListList, expName, refSource=0, plotPower=True):
    # Set up axes
    axesColorList = []
    # Use the target coordinate system (and frequency bands) of the indicated source as reference for the rest
    referenceScan = calibListList[refSource][0]
    refCoordSystem = referenceScan.targetCoordSystem
    plotFreqs = referenceScan.bandFreqs / 1e9   # Hz to GHz
    numBands = len(plotFreqs)
    # One figure per frequency band
    for band in range(numBands):
        axesColorList.append(figColorList[band].add_subplot(1, 1, 1))
    
    targetMinX = targetMinY = np.inf
    targetMaxX = targetMaxY = -np.inf
    # Iterate through sources
    for calibScanList, beamFuncList in zip(calibListList, beamListList):
        # Extract total power and target coordinates of all scans of a specific source
        totalPower = []
        targetCoords = []
        for scan in calibScanList:
            totalPower.append(scan.total_power())
            # If more than one source in list, convert target coordinates of scan to a common reference system
            if len(calibListList) > 1:
                targetCoordBlock = np.zeros((len(scan.timeSamples), refCoordSystem.get_dimensions()), dtype='double')
                transformer = acsm.transform.get_factory_instance().get_transformer(scan.targetCoordSystem, \
                                                                                    refCoordSystem)
                for k in xrange(len(scan.timeSamples)):
                    fromCoordinate = Coordinate(scan.targetCoordSystem, scan.targetCoords[k])
                    toCoordinate = transformer.transform_coordinate(fromCoordinate, scan.timeSamples[k])
                    targetCoordBlock[k] = toCoordinate.get_vector()
                targetCoords.append(targetCoordBlock)
            else:
                targetCoords.append(scan.targetCoords)
        totalPower = np.concatenate(totalPower)
        targetCoords = np.concatenate(targetCoords)
        
        # Determine overall axis limits
        targetMinX = min(targetMinX, targetCoords[:, 0].min())
        targetMaxX = max(targetMaxX, targetCoords[:, 0].max())
        targetMinY = min(targetMinY, targetCoords[:, 1].min())
        targetMaxY = max(targetMaxY, targetCoords[:, 1].max())
        # Setup uniform mesh grid for interpolated power plot (if desired)
        if plotPower:
            targetX = np.linspace(targetCoords[:, 0].min(), targetCoords[:, 0].max(), 101)
            targetY = np.linspace(targetCoords[:, 1].min(), targetCoords[:, 1].max(), 101)
            targetMeshX, targetMeshY = np.meshgrid(targetX, targetY)
            uniformGridCoords = np.vstack((targetMeshX.ravel(), targetMeshY.ravel()))
            # Setup slightly shifted uniform grid so that the centers of pcolor patches are correctly aligned
            # with target coordinates for the rest of the plots
            targetXstep = targetX[1] - targetX[0]
            targetYstep = targetY[1] - targetY[0]
            pcolorTargetX = np.array(targetX.tolist() + [targetX[-1]+targetXstep]) - targetXstep/2.0
            pcolorTargetY = np.array(targetY.tolist() + [targetY[-1]+targetYstep]) - targetYstep/2.0
        # Iterate through figures (one per band)
        for band in range(numBands):
            axis = axesColorList[band]
            power = totalPower[:, band]
            # If desired, plot the calibrated power via pcolor as a backdrop for beam pattern, to verify fit
            if plotPower:
                # Jitter the target coordinates to make degenerate triangles unlikely during Delaunay triangulation
                jitter = np.vstack((targetCoords[:, 0].std() * np.random.randn(len(targetCoords)), \
                                    targetCoords[:, 1].std() * np.random.randn(len(targetCoords)))).transpose()
                jitteredCoords = targetCoords[:, 0:2] + 0.001 * jitter
                # Interpolate the power values onto a 2-D grid (ignore rotator angle), and display using pcolor
                beamRoughFit = interp.Delaunay2DFit(defaultVal=power.min())
                beamRoughFit.fit(jitteredCoords.transpose(), power)
                uniformGridRoughPower = beamRoughFit(uniformGridCoords).reshape(targetY.size, targetX.size)
                axis.pcolor(pcolorTargetX * 180.0 / np.pi, pcolorTargetY * 180.0 / np.pi, uniformGridRoughPower, \
                            shading='flat', cmap=cm.gray_r)
            # Show the locations of the scan samples themselves
            axis.plot(targetCoords[:, 0] * 180.0 / np.pi, targetCoords[:, 1] * 180.0 / np.pi, '.b', zorder=1)
            # Plot the fitted Gaussian beam function as contours
            gaussEllipses, gaussCenter = vis.plot_gaussian_ellipse(axis, beamFuncList[band].mean, \
                                                                   np.diag(beamFuncList[band].var), \
                                                                   contour=[0.5, 0.1], ellipseLineType='r-', \
                                                                   centerLineType='r+', lineWidth=2)
            # If more than one source, convert Gaussian ellipse and center to reference coordinate system
            # (where needed, append rotator angle = 0.0 to go from 2-D ellipse coords to 3-D target coords)
            if len(calibListList) > 1:
                fromCoordSystem = calibScanList[0].targetCoordSystem
                # Use arbitrary timestamp from start of scan list
                timeStamp = calibScanList[0].timeSamples[0]
                transformer = acsm.transform.get_factory_instance().get_transformer(fromCoordSystem, refCoordSystem)
                for ellipse in gaussEllipses:
                    fromData = np.array((ellipse[0].get_xdata(), ellipse[0].get_ydata())).transpose()
                    numPoints = len(fromData)
                    newEllipse = np.zeros((numPoints, refCoordSystem.get_dimensions()), dtype='double')
                    for k in xrange(numPoints):
                        fromCoordinate = Coordinate(fromCoordSystem, fromData[k].tolist() + [0.0])
                        toCoordinate = transformer.transform_coordinate(fromCoordinate, timeStamp)
                        newEllipse[k, :] = toCoordinate.get_vector()
                    ellipse[0].set_xdata(newEllipse[:, 0])
                    ellipse[0].set_ydata(newEllipse[:, 1])
                fromCoordinate = Coordinate(fromCoordSystem, beamFuncList[band].mean.tolist() + [0.0])
                toCoordinate = transformer.transform_coordinate(fromCoordinate, timeStamp)
                newCenter = toCoordinate.get_vector()
                gaussCenter[0].set_xdata([newCenter[0]])
                gaussCenter[0].set_ydata([newCenter[1]])
            # Change from radians to degrees for prettier picture
            for ellipse in gaussEllipses:
                ellipse[0].set_xdata(ellipse[0].get_xdata() * 180.0 / np.pi)
                ellipse[0].set_ydata(ellipse[0].get_ydata() * 180.0 / np.pi)
            gaussCenter[0].set_xdata(gaussCenter[0].get_xdata()[0] * 180.0 / np.pi)
            gaussCenter[0].set_ydata(gaussCenter[0].get_ydata()[0] * 180.0 / np.pi)
    
    # Axis settings and labels
    for band in range(numBands):
        axis = axesColorList[band]
        axis.set_xlim(targetMinX * 180.0 / np.pi, targetMaxX * 180.0 / np.pi)
        axis.set_ylim(targetMinY * 180.0 / np.pi, targetMaxY * 180.0 / np.pi)
        axis.set_aspect('equal')
        axis.set_xlabel('Target coord 1 (deg)')
        axis.set_ylabel('Target coord 2 (deg)')
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
    axis.set_ylim(min(0.95*pointSourceSensitivity.min(), axis.get_ylim()[0]),
                  max(1.05*pointSourceSensitivity.max(), axis.get_ylim()[1]))
    axis.grid()
    axis.set_xticklabels([])
    axis.set_ylabel('Sensitivity (Jy/K)')
    axis.set_title(expName + ' : Point source sensitivity')
    axis = axesColorList[1]
    try:
        vis.draw_std_corridor(effArea.mean, effArea.sigma, plotFreqs, axis, muLineType='b-o')
    except AttributeError:
        axis.plot(plotFreqs, effArea, '-ob')
    axis.set_ylim(min(0.95*effArea.min(), axis.get_ylim()[0]), max(1.05*effArea.max(), axis.get_ylim()[1]))
    axis.grid()
    axis.set_xticklabels([])
    axis.set_ylabel('Area (m^2)')
    axis.set_title(expName + ' : Antenna effective area')
    axis = axesColorList[2]
    try:
        vis.draw_std_corridor(antGain.mean, antGain.sigma, plotFreqs, axis, muLineType='b-o')
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
# @param resultList     List of results, of form (bandFreqs, sourceAngs, pointSourceSensitivity, effArea, antGain)
# @param expName        Title of experiment
# @return axesColorList List of matplotlib Axes objects, one per plot
def plot_gain_curve(figColorList, resultList, expName):
    # Set up axes
    axesColorList = []
    bandFreqs, sourceAngs, pssBlock, effAreaBlock, antGainBlock = resultList
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
            vis.draw_std_corridor(pointSourceSensitivity.mean, pointSourceSensitivity.sigma, \
                                  sourceElAng_deg, axis, muLineType='b-o')
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
            vis.draw_std_corridor(effArea.mean, effArea.sigma, sourceElAng_deg, axis, muLineType='b-o')
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
            vis.draw_std_corridor(antGain.mean, antGain.sigma, sourceElAng_deg, axis, muLineType='b-o')
        except AttributeError:
            axis.plot(sourceElAng_deg, antGain, '-ob')
        axis.set_ylim(min(0.98*antGain.min(), axis.get_ylim()[0]), max(1.02*antGain.max(), axis.get_ylim()[1]))
        axis.grid()
        axis.set_xlabel('Elevation angle (degrees)')
        axis.set_ylabel('Gain (dB)')
        axis.set_title(expName + ' : Antenna gain in band %d : %3.3f GHz' % (band, plotFreqs[band]))
    
    return axesColorList
