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
import xdmsbe.xdmsbelib.stats as stats
import xdmsbe.xdmsbelib.single_dish_data as sdd
from conradmisclib.transforms import rad_to_deg, deg_to_rad
import matplotlib.axes3d as mplot3d
import pylab as pl
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
# @param rawPowerDictList List of dicts of SingleDishData objects, containing copies of all raw data blocks
# @param expName          Title of experiment
# @param stdScanList      List of StandardSourceScan objects, used for its channel lists only [None]
# @return axesColorList   List of matplotlib Axes objects, one per plot
# pylint: disable-msg=R0912,R0914
def plot_raw_power(figColor, rawPowerDictList, expName, stdScanList=None):
    # Set up axes
    axesColorList = []
    numScans = len(rawPowerDictList)
    for sub in range(numScans):
        axesColorList.append(figColor.add_subplot(numScans, 1, sub+1))
    
    # Use relative time axis
    timeRef = np.double(np.inf)
    for rawDict in rawPowerDictList:
        for data in rawDict.itervalues():
            timeRef = min(timeRef, data.timeSamples.min())
    
    # Determine appropriate channel to plot (the first channel valid in all scans, or 0 otherwise)
    channel = 0
    valid = '(no bad channel flags used, so some scans could be corrupted)'
    if stdScanList != None:
        flatChanList = []
        allChans = []
        for scan in stdScanList:
            flatChans = []
            for chanList in scan.channelsPerBand:
                flatChans += chanList
                allChans += chanList
            flatChanList.append(flatChans)
        common = [chan for chan in set(allChans) if np.all([(chan in chans) for chans in flatChanList])]
        if len(common) > 0:
            channel = common[0]
            valid = '(first channel for which all scans are uncorrupted)'
        else:
            channel = 0
            valid = '(no channels totally uncorrupted, so this has some corrupt scans)'
    
    # Plot of (continuum) raw XX power
    minY = maxY = []
    for scanInd, rawDict in enumerate(rawPowerDictList):
        axis = axesColorList[scanInd]
        for block in rawDict.itervalues():
            timeLine = block.timeSamples - timeRef
            contPower = block.coherency('XX')[:, channel]
            axis.plot(timeLine, contPower, lw=2, color='b')
        if (scanInd != numScans-1) and (numScans > 6):
            axis.set_xticklabels([])
        if scanInd == numScans-1:
            axis.set_xlabel('Time (s), since %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timeRef)))
        axis.set_ylabel('Raw power')
        if scanInd == 0:
            axis.set_title(expName + ' : raw XX power for channel ' + str(channel) + '\n' + valid)
        minY.append(axis.get_ylim()[0])
        maxY.append(axis.get_ylim()[1])
    
    # Set equal y-axis limits
    for axis in axesColorList:
        axis.set_ylim((np.array(minY).min(), np.array(maxY).max()))
    
    return axesColorList


## Plot Tsys as a function of frequency, for a single pointing.
# @param figColor       Matplotlib Figure object to contain plots
# @param resultList     List of results, of form (bandFreqs, Tsys, timeStamps, elAng_deg, channelFreqs, confIntervals)
# @param expName        Title of experiment
# @return axesColorList List of matplotlib Axes objects, one per plot
def plot_tsys(figColor, resultList, expName):
    # Set up axes
    axesColorList = []
    bandFreqs, tsys = resultList[:2]
    plotFreqs = bandFreqs / 1e9   # Hz to GHz
    # Determine appropriate frequency range and whisker width
    channelFreqs = resultList[4] / 1e9
    if len(channelFreqs) > 0:
        channelSpacing = np.median(np.diff(channelFreqs))
    else:
        channelSpacing = channelFreqs[0]
    freqRange = [min(channelFreqs.min(), plotFreqs.min()) - channelSpacing, \
                 max(channelFreqs.max(), plotFreqs.max()) + channelSpacing]
    whiskWidth = (freqRange[1] - freqRange[0]) / 50.0
    if len(plotFreqs) > 1:
        whiskWidth = min(whiskWidth, np.median(np.diff(channelFreqs)) / 2.0)
    # One figure, 2 subfigures
    axesColorList.append(figColor.add_subplot(2, 1, 1))
    axesColorList.append(figColor.add_subplot(2, 1, 2))
    
    polName = {0 : 'X', 1 : 'Y'}
    minY = maxY = []
    # Iterate through polarisation channels (X and Y)
    for polInd in range(2):
        axis = axesColorList[polInd]
        # Extract Tsys values of current polarisation of first pointing, and plot
        vis.mu_sigma_plot(axis, plotFreqs, tsys[0, polInd, :], whiskWidth=whiskWidth, linewidth=2, color='b')
        if not np.any(np.isnan(freqRange)):
            axis.set_xlim(freqRange)
        axis.grid()
        axis.set_title(expName + ' for polarisation ' + polName[polInd])
        axis.set_ylabel('Temperature (K)')
        if polInd == 0:
            axis.set_xticklabels([])
        else:
            axis.set_xlabel('Band frequency (GHz)')
        minY.append(axis.get_ylim()[0])
        maxY.append(axis.get_ylim()[1])
    
    # Set equal y-axis limits
    yRange = [np.array(minY).min(), np.array(maxY).max()]
    if not np.any(np.isnan(yRange)):
        for axis in axesColorList:
            axis.set_ylim(yRange)
    
    return axesColorList


## Plot Tsys as a function of time or elevation angle, for multiple pointings.
# @param figColorList   List of matplotlib Figure objects to contain plots
# @param resultList     List of results, of form (bandFreqs, Tsys, timeStamps, elAng_deg, channelFreqs, confIntervals)
# @param expName        Title of experiment
# @return axesColorList List of matplotlib Axes objects, one per plot
def plot_tsys_curve(figColorList, resultList, expName):
    # Set up axes
    axesColorList = []
    bandFreqs, tsys, timeStamps, elAng_deg = resultList[:4]
    plotFreqs = bandFreqs / 1e9   # Hz to GHz
    numBands = len(plotFreqs)
    # Determine which x axis to plot against
    if timeStamps == None:
        xVals = elAng_deg
        xLabel = 'Elevation angle (degrees)'
    else:
        timeRef = timeStamps.min()
        xVals = timeStamps - timeRef
        xLabel = 'Time (s), since %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timeRef))
    # One figure per frequency band
    for band in range(numBands):
        # 2 subfigures
        axesColorList.append(figColorList[band].add_subplot(2, 1, 1))
        axesColorList.append(figColorList[band].add_subplot(2, 1, 2))
    
    polName = {0 : 'X', 1 : 'Y'}
    minY = maxY = []
    # Plot one figure per frequency band
    for band in range(numBands):
        # Iterate through polarisation channels (X and Y)
        for polInd in range(2):
            axesInd = band*2 + polInd
            axis = axesColorList[axesInd]
            # Extract Tsys values of current polarisation for the current band, and plot
            vis.mu_sigma_plot(axis, xVals, tsys[:, polInd, band], linewidth=2, color='b')
            axis.plot(xVals, tsys.mu[:, polInd, band], '--b')
            axis.grid()
            axis.set_title(expName + ' (' + polName[polInd] + \
                           ') in band %d : %3.3f GHz' % (band, plotFreqs[band]))
            axis.set_ylabel('Temperature (K)')
            if polInd == 0:
                axis.set_xticklabels([])
            else:
                axis.set_xlabel(xLabel)
            minY.append(axis.get_ylim()[0])
            maxY.append(axis.get_ylim()[1])
    
    # Set equal y-axis limits
    yRange = [np.array(minY).min(), np.array(maxY).max()]
    if not np.any(np.isnan(yRange)):
        for axis in axesColorList:
            axis.set_ylim(yRange)
    
    return axesColorList


## Plot linearity test results.
# @param figColor       Matplotlib Figure object to contain plots
# @param resultList     List of results, of form (bandFreqs, Tsys, timeStamps, elAng_deg, channelFreqs, confIntervals)
# @param expName        Title of experiment
# @return axesColorList List of matplotlib Axes objects, one per plot
def plot_linearity_test(figColor, resultList, expName):
    # Set up axes
    axesColorList = []
    channelFreqs, confIntervals = resultList[4:6]
    plotFreqs = channelFreqs / 1e9   # Hz to GHz
    # One figure, 2 subfigures
    axesColorList.append(figColor.add_subplot(2, 1, 1))
    axesColorList.append(figColor.add_subplot(2, 1, 2))
    
    # Check linearity
    isLinear = stats.check_equality_of_means(confIntervals)
    polName = {0 : 'X', 1 : 'Y'}
    minY = maxY = []
    freqSpacing = np.diff(plotFreqs).mean() / 2.0
    # Iterate through polarisation channels (X and Y)
    for polInd in range(2):
        axis = axesColorList[polInd]
        # Split data into two lines - one for linear channels, and one for non-linear channels
        linearInts = 100.0 * confIntervals[:, polInd, isLinear[polInd]]
        linearFreqs = plotFreqs[isLinear[polInd]]
        vis.mu_sigma_plot(axis, linearFreqs, \
                          stats.MuSigmaArray(linearInts.mean(axis=0), 0.5*np.diff(linearInts, axis=0)[0]), \
                          whiskWidth=freqSpacing, linewidth=2, color='g', marker='')
        nonLinInts = 100.0 * confIntervals[:, polInd, np.invert(isLinear[polInd])]
        nonLinFreqs = plotFreqs[np.invert(isLinear[polInd])]
        vis.mu_sigma_plot(axis, nonLinFreqs, \
                          stats.MuSigmaArray(nonLinInts.mean(axis=0), 0.5*np.diff(nonLinInts, axis=0)[0]), \
                          whiskWidth=freqSpacing, linewidth=2, color='r', marker='')
        axis.plot([plotFreqs[0], plotFreqs[-1]], [0, 0], linewidth=3, color='k')
        axis.grid()
        axis.set_title(expName + ' : Linearity test for polarisation ' + polName[polInd])
        axis.set_ylabel('(Hot - cold) delta, as % of cold delta')
        if polInd == 0:
            axis.set_xticklabels([])
        else:
            axis.set_xlabel('Channel frequency (GHz)')
        minY.append(axis.get_ylim()[0])
        maxY.append(axis.get_ylim()[1])
    
    # Set equal y-axis limits
    yRange = [np.array(minY).min(), np.array(maxY).max()]
    if not np.any(np.isnan(yRange)):
        for axis in axesColorList:
            axis.set_ylim(yRange)
    
    return axesColorList


## Plot baseline fits of multiple scans through a point source.
# @param figColor       Matplotlib Figure object to contain plots
# @param stdScanList    List of StandardSourceScan objects, after power-to-temp and channel-to-band conversion,
#                       with fitted baselines
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
            axis.set_xlabel('Time (s), since %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timeRef)))
        axis.set_ylabel('Power (K)')
        if scanInd == 0:
            axis.set_title(expName + ' : baseline fits on I (averaged over bands)')
        minY.append(axis.get_ylim()[0])
        maxY.append(axis.get_ylim()[1])
    
    # Set equal y-axis limits
    yRange = [np.array(minY).min(), np.array(maxY).max()]
    if not np.any(np.isnan(yRange)):
        for axis in axesColorList:
            axis.set_ylim(yRange)
    
    return axesColorList


## Plot baseline fits for polarisation experiments.
# @param figColor       Matplotlib Figure object to contain plots
# @param stdScanList    List of StandardSourceScan objects, after power-to-temp and channel-to-band conversion,
#                       with fitted baselines
# @param expName        Title of experiment
# @return axesColorList List of matplotlib Axes objects, one per plot
# pylint: disable-msg=R0912,R0914,R0915
def plot_baseline_pol(figColor, stdScanList, expName):
    # Set up axes
    axesColorList = []
    stokesKey = ['I', 'Q', 'U', 'V']
    numParams = len(stokesKey) + 1
    numScans = len(stdScanList)
    # One figure, one subfigure per Stokes parameter/rotator angle and per standard scan
    for sub in range(numParams * numScans):
        axesColorList.append(figColor.add_subplot(numParams, numScans, sub+1))
    
    # Use relative time axis
    timeRef = np.double(np.inf)
    for stdScan in stdScanList:
        timeRef = min(timeRef, stdScan.mainData.timeSamples.min())
        if stdScan.baselineDataList != None:
            for data in stdScan.baselineDataList:
                timeRef = min(timeRef, data.timeSamples.min())
    
    # Plot of (continuum) baseline fits
    yRange = np.zeros((numParams, 2*numScans))
    stokesLookup = sdd.power_index_dict(True)
    # Iterate over parameters (subplot rows)
    for stokesInd, key in enumerate(stokesKey):
        # Iterate over scans (subplot columns)
        for scanInd, stdScan in enumerate(stdScanList):
            axis = axesColorList[numScans * stokesInd + scanInd]
            dataBlocks = [stdScan.mainData]
            if stdScan.baselineDataList:
                dataBlocks += stdScan.baselineDataList
            for block in dataBlocks:
                timeLine = block.timeSamples - timeRef
                contPower = block.stokes(key).mean(axis=1)
                axis.plot(timeLine, contPower, lw=2, color='b')
                if stdScan.baselineFunc != None:
                    if stdScan.baselineUsesElevation:
                        baseline = stdScan.baselineFunc(block.elAng_rad)[stokesLookup[key]].mean(axis=1)
                    else:
                        baseline = stdScan.baselineFunc(block.azAng_rad)[stokesLookup[key]].mean(axis=1)
                    axis.plot(timeLine, baseline, lw=2, color='r')
            axis.set_xticklabels([])
            # Left-most column
            if scanInd == 0:
                axis.set_ylabel(key + ' (K)')
            else:
                axis.set_yticklabels([])
            # Middle of top row
            if (stokesInd == 0) and (scanInd == numScans // 2):
                axis.set_title(expName + '\nBaseline fits on IQUV (averaged over bands, and per scan)')
            yRange[stokesInd, 2*scanInd:2*scanInd+2] = axis.get_ylim()
    # Plot rotator or parallactic angle
    for scanInd, stdScan in enumerate(stdScanList):
        axis = axesColorList[numScans * (numParams-1) + scanInd]
        dataBlocks = [stdScan.mainData]
        if stdScan.baselineDataList:
            dataBlocks += stdScan.baselineDataList
        if not stdScan.parallacticCorrectionApplied:
            for block in dataBlocks:
                axis.plot(block.timeSamples - timeRef, rad_to_deg(block.parallactic_rotation()), lw=2, color='b')
            if scanInd == 0:
                axis.set_ylabel('Parallactic (deg)')
            else:
                axis.set_yticklabels([])
        else:
            for block in dataBlocks:
                axis.plot(block.timeSamples - timeRef, rad_to_deg(block.rotAng_rad), lw=2, color='b')
            if scanInd == 0:
                axis.set_ylabel('Rotator (deg)')
            else:
                axis.set_yticklabels([])
        if scanInd == numScans // 2:
            axis.set_xlabel('Time (s), since %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timeRef)))
        yRange[numParams-1, 2*scanInd:2*scanInd+2] = axis.get_ylim()
    
    # Use the same y-axis limits for Q, U and V, and separate ones for I and rotator angle
    yRange[0, :2] = [0, 1.1*yRange[0, :].max()]
    yRange[1:-1, :2] = [yRange[1:-1, :].min(), yRange[1:-1, :].max()]
    yRange[-1, :2] = [yRange[-1, :].min() - 20, yRange[-1, :].max() + 20]
    for stokesInd in xrange(numParams):
        for scanInd in xrange(numScans):
            axis = axesColorList[numScans * stokesInd + scanInd]
            axis.set_ylim(yRange[stokesInd, :2])
    
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
    plotFreqs = calibScanList[0].freqs_Hz / 1e9   # Hz to GHz
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
            axis.plot(timeLine, block.stokes('I')[:, band], color='b', lw=2)
            if beamFuncList != None:
                # Slice through fitted beam function along the same coordinates (change colors for invalid beams)
                if beamFuncList[band][1]:
                    beamColor = 'r'
                else:
                    beamColor = 'y'
                axis.plot(timeLine, beamFuncList[band][0](block.targetCoords[:, :2]), color=beamColor, lw=2)
            if scanInd == numScans-1:
                axis.set_xlabel('Time (s), since %s' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timeRef)))
            axis.set_ylabel('Power (K)')
            if scanInd == 0:
                axis.set_title(expName + ' : calibrated total power in band %d : %3.3f GHz' % (band, plotFreqs[band]))
            minY.append(axis.get_ylim()[0])
            maxY.append(axis.get_ylim()[1])
    
    # Set equal y-axis limits
    yRange = [np.array(minY).min(), np.array(maxY).max()]
    if not np.any(np.isnan(yRange)):
        for axis in axesColorList:
            axis.set_ylim(yRange)
    
    return axesColorList


## Plot antenna gain info derived from multiple scans through a point source.
# @param figColor       Matplotlib Figure object to contain plots
# @param resultList     List of results, of form (channelFreqs, bandFreqs, pointSourceSensitivity, effArea, antGain)
# @param expName        Title of experiment
# @return axesColorList List of matplotlib Axes objects, one per plot
def plot_antenna_gain(figColor, resultList, expName):
    # Set up axes
    axesColorList = []
    channelFreqs, bandFreqs, pointSourceSensitivity, effArea, antGain = resultList
    plotFreqs = bandFreqs / 1e9   # Hz to GHz
    # Determine appropriate frequency range and whisker width
    channelFreqs /= 1e9
    if len(channelFreqs) > 0:
        channelSpacing = np.median(np.diff(channelFreqs))
    else:
        channelSpacing = channelFreqs[0]
    freqRange = [min(channelFreqs.min(), plotFreqs.min()) - channelSpacing, \
                 max(channelFreqs.max(), plotFreqs.max()) + channelSpacing]
    whiskWidth = (freqRange[1] - freqRange[0]) / 50.0
    if len(plotFreqs) > 1:
        whiskWidth = min(whiskWidth, np.median(np.diff(channelFreqs)) / 2.0)
    # One figure, 3 subfigures
    for sub in range(3):
        axesColorList.append(figColor.add_subplot(3, 1, sub+1))
    
    axis = axesColorList[0]
    vis.mu_sigma_plot(axis, plotFreqs, pointSourceSensitivity, whiskWidth=whiskWidth, linewidth=2, color='b')
    if not np.any(np.isnan(freqRange)):
        axis.set_xlim(freqRange)
    yRange = [0.95*pointSourceSensitivity.min(), 1.05*pointSourceSensitivity.max()]
    if not np.any(np.isnan(yRange)):
        axis.set_ylim(min(yRange[0], axis.get_ylim()[0]), max(yRange[1], axis.get_ylim()[1]))
    axis.grid()
    axis.set_xticklabels([])
    axis.set_ylabel('Sensitivity (Jy/K)')
    axis.set_title(expName + ' : Point source sensitivity')
    
    axis = axesColorList[1]
    vis.mu_sigma_plot(axis, plotFreqs, effArea, whiskWidth=whiskWidth, linewidth=2, color='b')
    if not np.any(np.isnan(freqRange)):
        axis.set_xlim(freqRange)
    yRange = [0.95*effArea.min(), 1.05*effArea.max()]
    if not np.any(np.isnan(yRange)):
        axis.set_ylim(min(yRange[0], axis.get_ylim()[0]), max(yRange[1], axis.get_ylim()[1]))
    axis.grid()
    axis.set_xticklabels([])
    axis.set_ylabel('Area ($m^2$)')
    axis.set_title(expName + ' : Antenna effective area')
    
    axis = axesColorList[2]
    vis.mu_sigma_plot(axis, plotFreqs, antGain, whiskWidth=whiskWidth, linewidth=2, color='b')
    if not np.any(np.isnan(freqRange)):
        axis.set_xlim(freqRange)
    yRange = [0.98*antGain.min(), 1.02*antGain.max()]
    if not np.any(np.isnan(yRange)):
        axis.set_ylim(min(yRange[0], axis.get_ylim()[0]), max(yRange[1], axis.get_ylim()[1]))
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
        vis.mu_sigma_plot(axis, sourceElAng_deg, pointSourceSensitivity, linewidth=2, color='b')
        yRange = [0.95*pointSourceSensitivity.min(), 1.05*pointSourceSensitivity.max()]
        if not np.any(np.isnan(yRange)):
            axis.set_ylim(min(yRange[0], axis.get_ylim()[0]), max(yRange[1], axis.get_ylim()[1]))
        axis.grid()
        axis.set_xticklabels([])
        axis.set_ylabel('Sensitivity (Jy/K)')
        axis.set_title(expName + ' : Point source sensitivity in band %d : %3.3f GHz' % (band, plotFreqs[band]))
        
        axesInd = band*3 + 1
        axis = axesColorList[axesInd]
        effArea = effAreaBlock[:, band]
        vis.mu_sigma_plot(axis, sourceElAng_deg, effArea, linewidth=2, color='b')
        yRange = [0.95*effArea.min(), 1.05*effArea.max()]
        if not np.any(np.isnan(yRange)):
            axis.set_ylim(min(yRange[0], axis.get_ylim()[0]), max(yRange[1], axis.get_ylim()[1]))
        axis.grid()
        axis.set_xticklabels([])
        axis.set_ylabel('Area ($m^2$)')
        axis.set_title(expName + ' : Antenna effective area in band %d : %3.3f GHz' % (band, plotFreqs[band]))
        
        axesInd = band*3 + 2
        axis = axesColorList[axesInd]
        antGain = antGainBlock[:, band]
        vis.mu_sigma_plot(axis, sourceElAng_deg, antGain, linewidth=2, color='b')
        yRange = [0.98*antGain.min(), 1.02*antGain.max()]
        if not np.any(np.isnan(yRange)):
            axis.set_ylim(min(yRange[0], axis.get_ylim()[0]), max(yRange[1], axis.get_ylim()[1]))
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
    if not np.any(np.isnan(axis.get_xlim())) and not np.any(np.isnan(axis.get_ylim())):
        axis.set_xlim(axis.get_xlim()[0] - 10, axis.get_xlim()[1] + 10)
        axis.set_ylim(axis.get_ylim()[0] - 10, axis.get_ylim()[1] + 10)
    axis.set_aspect('equal')
    axis.set_xlabel('Azimuth (deg)')
    axis.set_ylabel('Elevation (deg)')
    axis.set_title(expName + ' : Pointing errors (magnified %4.2fx)' % scale)
    
    return axesColorList
