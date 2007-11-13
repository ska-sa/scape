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

import xdmsbe.xdmsbelib.fitting as fitting
import xdmsbe.xdmsbelib.vis as vis
import xdmsbe.xdmsbelib.misc as misc
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
        for block in rawList:
            timeLine = block.timeSamples - timeRef
            contPower = block.powerData[0, :, 0]
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
# pylint: disable-msg=R0912,R0914
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
        if stdScan.baselineDataList != None:
            for data in stdScan.baselineDataList:
                timeRef = min(timeRef, data.timeSamples.min())
    
    # Plot of (continuum) baseline fits
    minY = maxY = []
    for scanInd, stdScan in enumerate(stdScanList):
        axis = axesColorList[scanInd]
        dataBlocks = [stdScan.mainData]
        if stdScan.baselineDataList:
            dataBlocks += stdScan.baselineDataList
        for block in dataBlocks:
            timeLine = block.timeSamples - timeRef
            contPower = block.powerData[0].mean(axis=1)
            axis.plot(timeLine, contPower, lw=2, color='b')
            if stdScan.baselineFunc != None:
                if stdScan.baselineUsesElevation:
                    baseline = stdScan.baselineFunc(block.elAng_rad)[0].mean(axis=1)
                else:
                    baseline = stdScan.baselineFunc(block.azAng_rad)[0].mean(axis=1)
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
    for scanInd, block in enumerate(calibScanList):
        timeLine = block.timeSamples - timeRef
        for band in range(numBands):
            axesInd = band*numScans + scanInd
            axis = axesColorList[axesInd]
            # Total power in band
            axis.plot(timeLine, block.total_power()[:, band], color='b', lw=2)
            if beamFuncList != None:
                # Slice through fitted beam function along the same coordinates (change colors for invalid beams)
                if beamFuncList[band][1]:
                    beamColor = 'r'
                else:
                    beamColor = 'y'
                axis.plot(timeLine, beamFuncList[band][0](block.targetCoords[:, 0:2]), color=beamColor, lw=2)
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
    
    # Extract total power and target coordinates (in degrees) of all scans
    totalPower = []
    targetCoords = []
    for scan in calibScanList:
        totalPower.append(scan.total_power())
        targetCoords.append(rad_to_deg(scan.targetCoords))
    # Also extract beam centres, in order to unwrap them with the rest of angle data
    for band in range(numBands):
        targetCoords.append(rad_to_deg(np.atleast_2d(beamFuncList[band][0].mean)))
    totalPower = np.concatenate(totalPower)
    targetCoords = np.concatenate(targetCoords)
    # Unwrap all angle coordinates in the plot concurrently, to prevent bad plots
    # (this is highly unlikely, as the target coordinate space is typically centred around the origin)
    targetCoords = np.array([misc.unwrap_angles(ang) for ang in targetCoords.transpose()]).transpose()
    beamCentres = targetCoords[-numBands:]
    targetCoords = targetCoords[:-numBands]
    
    # Iterate through figures (one per band)
    for band in range(numBands):
        axis = axesColorList[band]
        power = totalPower[:, band]
        # Show the locations of the scan samples themselves, with marker sizes indicating power values
        vis.plot_marker_3d(axis, targetCoords[:, 0], targetCoords[:, 1], power)
        # Plot the fitted Gaussian beam function as contours
        if beamFuncList[band][1]:
            ellType, centerType = 'r-', 'r+'
        else:
            ellType, centerType = 'y-', 'y+'
        ellipses = misc.gaussian_ellipses(beamCentres[band], np.diag(beamFuncList[band][0].var), contour=[0.5, 0.1])
        for ellipse in ellipses:
            axis.plot(rad_to_deg(ellipse[:, 0]), rad_to_deg(ellipse[:, 1]), ellType, lw=2)
        axis.plot([beamCentres[band][0]], [beamCentres[band][1]], centerType, ms=12, aa=False, mew=2)
    
    # Axis settings and labels
    for band in range(numBands):
        axis = axesColorList[band]
        axis.set_xlim(targetCoords[:, 0].min(), targetCoords[:, 0].max())
        axis.set_ylim(targetCoords[:, 1].min(), targetCoords[:, 1].max())
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
    azScans_deg = []
    elScans_deg = []
    totalPowerScans = []
    for scan in calibScanList:
        # Unwrap angles, so that the direction in which angles change are more accurately determined
        azScan_deg = misc.unwrap_angles(rad_to_deg(scan.targetCoords[:, 0]))
        # Reverse any scans with descending azimuth angle, so that all scans run in the same direction
        if azScan_deg[0] < azScan_deg[-1]:
            azScans_deg.append(azScan_deg)
            elScans_deg.append(misc.unwrap_angles(rad_to_deg(scan.targetCoords[:, 1])))
            totalPowerScans.append(scan.total_power()[np.newaxis, :, :])
        else:
            azScans_deg.append(np.flipud(azScan_deg))
            elScans_deg.append(np.flipud(misc.unwrap_angles(rad_to_deg(scan.targetCoords[:, 1]))))
            totalPowerScans.append(np.flipud(scan.total_power())[np.newaxis, :, :])
    azScans_deg = np.vstack(azScans_deg)
    elScans_deg = np.vstack(elScans_deg)
    totalPowerScans = np.concatenate(totalPowerScans, axis=0)
    # Unwrap angles yet again, to make sure the group of scans stay together
    # (It's highly unlikely that unwrapping is necessary, as the target space is typically centred around the origin)
    azScans_deg = misc.unwrap_angles(azScans_deg.ravel()).reshape(azScans_deg.shape)
    elScans_deg = misc.unwrap_angles(elScans_deg.ravel()).reshape(elScans_deg.shape)
    # Set up uniform 101x101 mesh grid for contour plot
    targetX = np.linspace(azScans_deg.min(), azScans_deg.max(), 101)
    targetY = np.linspace(elScans_deg.min(), elScans_deg.max(), 101)
    targetMeshX, targetMeshY = np.meshgrid(targetX, targetY)
    meshCoords = np.vstack((targetMeshX.ravel(), targetMeshY.ravel()))
    # Jitter the target coordinates to make degenerate triangles unlikely during Delaunay triangulation for contours
    jitter = np.vstack((azScans_deg.std() * np.random.randn(azScans_deg.size), \
                        elScans_deg.std() * np.random.randn(elScans_deg.size)))
    jitteredCoords = np.vstack((azScans_deg.ravel(), elScans_deg.ravel())) + 0.0001 * jitter
    
    # Iterate through figures (two per band)
    for band in range(numBands):
        power = totalPowerScans[:, :, band]
        # 2D contour plot
        axis = axesColorList[2*band]
        # Show the locations of the scan samples themselves
        axis.plot(azScans_deg.ravel(), elScans_deg.ravel(), '.b', alpha=0.5)
        # Interpolate the power values onto a 2-D az-el grid for a smoother contour plot
        gridder = fitting.Delaunay2DFit(defaultVal = power.min())
        gridder.fit(jitteredCoords, power.ravel())
        meshPower = gridder(meshCoords).reshape(targetY.size, targetX.size)
        # Choose contour levels as fractions of the peak power
        contourLevels = np.arange(0.1, 1.0, 0.1) * meshPower.max()
        # Indicate half-power beamwidth (contourLevel = 0.5) with wider line
        lineWidths = 2*np.ones(len(contourLevels))
        lineWidths[4] = 4
        # Color the 0.5 and 0.1 contours red, to coincide with the scheme followed in the beam fitting plots
#        colors = ['b'] * len(contourLevels)
#        colors[0] = colors[4] = 'r'
        # Plot contours of interpolated beam pattern
        axis.contour(targetMeshX, targetMeshY, meshPower, contourLevels, linewidths=lineWidths, colors='b')
        # 3D wireframe plot
        axis = axesColorList[2*band + 1]
        axis.plot_wireframe(azScans_deg, elScans_deg, power)
    
    # Axis settings and labels
    for band in range(2*numBands):
        axis = axesColorList[band]
        axis.set_xlim(targetX[0], targetX[-1])
        axis.set_ylim(targetY[0], targetY[-1])
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
    numSources = len(calibListList)
    # Number of points in each Gaussian ellipse
    numEllPoints = 200
    contours = [0.5, 0.1]
    numEllipses = len(contours)
    # One figure per frequency band
    for band in range(numBands):
        axesColorList.append(figColorList[band].add_subplot(1, 1, 1))
    
    # Iterate through sources and collect data, converting target coordinates to instantaneous mount coordinates
    totalPowerList, azAngList, elAngList = [], [], []
    for calibScanList, targetToInstantMount in zip(calibListList, transformList):
        # Extract total power and instantaneous mount coordinates of all scans of a specific source
        totalPower, azAng_deg, elAng_deg = [], [], []
        for scan in calibScanList:
            totalPower.append(scan.total_power())
            mountCoords = np.array([targetToInstantMount(targetCoord) for targetCoord in scan.targetCoords])
            azAng_deg.append(mountCoords[:, 0].tolist())
            elAng_deg.append(mountCoords[:, 1].tolist())
        totalPowerList.append(np.concatenate(totalPower))
        azAngList.append(np.concatenate(azAng_deg))
        elAngList.append(np.concatenate(elAng_deg))
    # Create Gaussian contour ellipses in instantaneous mount coordinates
    for beamFuncList, targetToInstantMount in zip(beamListList, transformList):
        targetDim = targetToInstantMount.get_target_dimensions()
        for band in range(numBands):
            # Create ellipses in target coordinates
            ellipses = misc.gaussian_ellipses(beamFuncList[band][0].mean, np.diag(beamFuncList[band][0].var), \
                                              contour=contours, numPoints=numEllPoints)
            # Convert Gaussian ellipses and center to instantaneous mount coordinate system
            # This assumes that the beam function is defined on a 2-D coordinate space, while the target
            # coordinate space is at least 2-D, with the first 2 dimensions corresponding with the beam's ones
            targetCoord = np.zeros(targetDim, dtype='double')
            targetCoord[0:2] = beamFuncList[band][0].mean
            mountCoord = targetToInstantMount(targetCoord)
            azAngList.append(np.array([mountCoord[0]]))
            elAngList.append(np.array([mountCoord[1]]))
            for ellipse in ellipses:
                targetCoords = np.zeros((len(ellipse), targetDim), dtype='double')
                targetCoords[:, 0:2] = ellipse
                mountCoords = np.array([targetToInstantMount(targetCoord) for targetCoord in targetCoords])
                azAngList.append(mountCoords[:, 0])
                elAngList.append(mountCoords[:, 1])
    
    # Do global unwrapping of all angles, across all sources, to prevent sources being split around angle boundaries
    angInds = np.cumsum([0] + [ang.size for ang in azAngList])
    azAngMod = misc.unwrap_angles(np.concatenate(azAngList))
    azAngList = [azAngMod[angInds[i-1]:angInds[i]] for i in xrange(1, len(angInds))]
    elAngMod = misc.unwrap_angles(np.concatenate(elAngList))
    elAngList = [elAngMod[angInds[i-1]:angInds[i]] for i in xrange(1, len(angInds))]
    # Determine overall axis limits
    azMin_deg = np.concatenate(azAngList).min()
    azMax_deg = np.concatenate(azAngList).max()
    elMin_deg = np.concatenate(elAngList).min()
    elMax_deg = np.concatenate(elAngList).max()
    # Separate scan points and beam info again
    azBeamInfo = np.concatenate(azAngList[numSources:]).reshape(numSources, numBands, 1+numEllipses*numEllPoints)
    elBeamInfo = np.concatenate(elAngList[numSources:]).reshape(numSources, numBands, 1+numEllipses*numEllPoints)
    azAngList = azAngList[:numSources]
    elAngList = elAngList[:numSources]
    
    # Iterate through figures (one per band)
    for band in range(numBands):
        axis = axesColorList[band]
        # Iterate through sources and plot data
        for sourceData in zip(azAngList, elAngList, totalPowerList, azBeamInfo, elBeamInfo, beamListList):
            azScan, elScan, totalPower, azBeam, elBeam, beamFuncList = sourceData
            power = totalPower[:, band]
            # Show the locations of the scan samples themselves, with marker sizes indicating power values
            vis.plot_marker_3d(axis, azScan, elScan, power)
            # Plot the fitted Gaussian beam function as contours with a centre marker
            if beamFuncList[band][1]:
                ellType, centerType = 'r-', 'r+'
            else:
                ellType, centerType = 'y-', 'y+'
            for azEllipse, elEllipse in zip(azBeam[band, 1:].reshape(numEllipses, numEllPoints), \
                                            elBeam[band, 1:].reshape(numEllipses, numEllPoints)):
                axis.plot(azEllipse, elEllipse, ellType, lw=2)
            axis.plot([azBeam[band, 0]], [elBeam[band, 0]], centerType, ms=12, aa=False, mew=2)
        
        # Axis settings and labels
        axis.set_xlim(azMin_deg, azMax_deg)
        axis.set_ylim(elMin_deg, elMax_deg)
        # Only set axis aspect ratio to equal if it is not too extreme
        if azMin_deg == azMax_deg:
            aspectRatio = 1e20
        else:
            aspectRatio = (elMax_deg - elMin_deg) / (azMax_deg - azMin_deg)
        if (aspectRatio > 0.1) and (aspectRatio < 10):
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
        vis.draw_std_corridor(axis, plotFreqs, pointSourceSensitivity.mu, pointSourceSensitivity.sigma, \
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
        vis.draw_std_corridor(axis, plotFreqs, effArea.mu, effArea.sigma, muLineType='b-o')
    except AttributeError:
        axis.plot(plotFreqs, effArea, '-ob')
    axis.set_ylim(min(0.95*effArea.min(), axis.get_ylim()[0]), max(1.05*effArea.max(), axis.get_ylim()[1]))
    axis.grid()
    axis.set_xticklabels([])
    axis.set_ylabel('Area (m^2)')
    axis.set_title(expName + ' : Antenna effective area')
    axis = axesColorList[2]
    try:
        vis.draw_std_corridor(axis, plotFreqs, antGain.mu, antGain.sigma, muLineType='b-o')
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
            vis.draw_std_corridor(axis, sourceElAng_deg, pointSourceSensitivity.mu, pointSourceSensitivity.sigma, \
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
            vis.draw_std_corridor(axis, sourceElAng_deg, effArea.mu, effArea.sigma, muLineType='b-o')
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
            vis.draw_std_corridor(axis, sourceElAng_deg, antGain.mu, antGain.sigma, muLineType='b-o')
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
    # Unwrap azimuth angles to make sure steered and estimated positions are as close together as possible
    newAz = np.array([misc.unwrap_angles([steeredAz[ind], meanAz[ind]]) for ind in xrange(len(meanAz))])
    steeredAz, meanAz = newAz.transpose()
    # This forms the lines connecting steered and estimated position markers
    arrowsAz = np.vstack((steeredAz, meanAz, np.tile(np.nan, (len(meanAz))))).transpose().ravel()
    arrowsEl = np.vstack((steeredEl, meanEl, np.tile(np.nan, (len(meanEl))))).transpose().ravel()
    
    axis = axesColorList[0]
    # Location of "true" source, indicated by Gaussian center + spread (1=sigma and 2-sigma contours)
    for ind in xrange(len(meanAz)):
        ellipses = misc.gaussian_ellipses((meanAz[ind], meanEl[ind]), \
                                          np.diag((azErrorSigma[ind] ** 2, elErrorSigma[ind] ** 2)), \
                                          contour=np.exp(-0.5*np.array([1, 2]) ** 2))
        for ellipse in ellipses:
            axis.plot(ellipse[:, 0], ellipse[:, 1], 'r-', lw=1)
        axis.plot([meanAz[ind]], [meanEl[ind]], 'or', ms=4, aa=False, mew=1)
    # Lines connecting markers
    axis.plot(arrowsAz, arrowsEl, 'k')
    # Marker indicating steered position (where the source was thought to be)
    axis.plot(steeredAz, steeredEl, 'sk', markersize=4)
    axis.set_xlim(axis.get_xlim()[0] - 10, axis.get_xlim()[1] + 10)
    axis.set_ylim(axis.get_ylim()[0] - 10, axis.get_ylim()[1] + 10)
    axis.set_aspect('equal')
    axis.set_xlabel('Azimuth (deg)')
    axis.set_ylabel('Elevation (deg)')
    axis.set_title(expName + ' : Pointing errors (magnified %4.2fx)' % scale)
    
    return axesColorList
