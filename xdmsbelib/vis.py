## xdmsbelib/vis.py
#
# Visualization routines used by XDM software backend.
#
# @author Rudolph van der Merwe [rudolph@ska.ac.za] on 2007-04-13.
# @copyright (c) 2007 SKA/KAT. All rights reserved.

from __future__ import division
import numpy as np
import numpy.linalg as linalg
import logging
import pylab
import os

logger = logging.getLogger("xdmsbe.xdmsbelib.vis")

# pylint: disable-msg=C0103

#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  disp_interactive_figure
#---------------------------------------------------------------------------------------------------------

## Display a list of matplotlib figures in interactive Tk windows. This function uses the
# TkAgg backend of matplotlib to render the figures.
#
# Note: This function is blocking. At the end of this function the Tk.mainloop() eventhandler
#       is entered to service the GUI requirements of the windows.
#
# @param     figureList   list of matplotlib figure objects created with matplotlib.figure.Figure()
# @param     title        Title to be displayed in each window. The figure number is appended to this title.
#

def disp_interactive_figure(figureList, title = ''):
    
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
    import Tkinter as Tk
    
    windowList = []
    
    def destroy():
        for window in windowList:
            window.destroy()
    
    
    for n, fig in enumerate(figureList):
        
        windowList.append(Tk.Tk())
        
        if title:
            windowList[n].wm_title(title + ' : Fig ' + str(n+1))
        
        # Generate a tk.DrawingArea
        canvas = FigureCanvasTkAgg(fig, master = windowList[n])
        canvas.get_tk_widget().pack(side = Tk.TOP, fill = Tk.BOTH, expand = 1)
        
        toolbar = NavigationToolbar2TkAgg(canvas, windowList[n])
        toolbar.update()
        # pylint: disable-msg=W0212
        canvas._tkcanvas.pack(side = Tk.TOP, fill = Tk.BOTH, expand = 1)
        
        button = Tk.Button(master = windowList[n], text = 'Quit', command = destroy)
        button.pack(side = Tk.RIGHT)
        
        #canvas.show()
    
    Tk.mainloop()


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  figure_to_imagefile
#---------------------------------------------------------------------------------------------------------

## Save a matplotlib Figure to an image file (png, svg, etc.). The filetype is automatically determined based
# on the filename extension.
#
# @param     fig       matplotlib Figure object
# @param     filename  Filename of image file. The filename must include the correct
#                      type extension, i.e. foo.png or foo.svg
# @param     dpi       Image file DPI (dots-per-inch) [80]

def figure_to_imagefile(fig, filename, dpi = 80):
    
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(filename, dpi)


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  figure_to_imagedata
#---------------------------------------------------------------------------------------------------------

## Convert a matplotlib Figure object to a numpy data array
#
# @param     fig         matplotlib Figure object
# @param     fileType    matplotlib image file type ('png','svg','jpg', etc.) ['png']
# @param     dpi         Image file DPI (dots-per-inch) [80]
# @return    imageData   numpy MxNx4 array of 0-1 normalized floats

def figure_to_imagedata(fig, fileType = 'png', dpi = 80):
    
    from tempfile import mktemp
    
    filename = mktemp('.' + fileType)
    figure_to_imagefile(fig, filename, dpi)
    imageData = pylab.imread(filename)
    os.remove(filename)
    
    return imageData


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  figure_to_svg
#---------------------------------------------------------------------------------------------------------

## Convert a matplotlib Figure object to a SVG object
#
# @param     fig         matplotlib Figure object
# @param     dpi         image file DPI (dots-per-inch) [80]
# @return    svgData     list containing SVG text lines

def figure_to_svg(fig, dpi=100):
    
    from tempfile import mktemp
    from matplotlib.backends.backend_svg import FigureCanvasSVG
    
    filename = mktemp('.svg')
    
    canvas = FigureCanvasSVG(fig)
    canvas.print_figure(filename, dpi)
    
    svgFile = open(filename)
    svgData = svgFile.readlines()
    svgFile.close()
    os.remove(filename)
    
    return svgData



#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  draw_std_corridor
#---------------------------------------------------------------------------------------------------------

## Draw mean and standard deviation corridor to a matplotlib axes.
# If the data contains NaNs, the plot is broken up into segments for each valid data range.
#
# @param    axis            Matplotlib axes object associated with a matplotlib Figure, that will receive plot
# @param    xVals           Array of values on x-axis
# @param    mu              Array of mean values
# @param    sigma           Array of standard deviation values (sigma)
# @param    muLabel         Label string for mean plot [None]
# @param    sigmaLabel      Label string for std corridor [None]
# @param    muLineType      Matplotlib linetype string for mean line ['r-']
# @param    sigmaFillColor  Matplotlib color string used to fill std-corridor ['r']
# @param    sigmaAlpha      Transparency (alpha) value for std-corridor [0.5]
# @param    sigmaLineType   Matplotlib linetype string for outline (boundary) of std-corridor [None]
# @return   muHandle        Handle to mean plot
# @return   sigmaCorHandles List of handles to std-corridor objects (one per segment)
# @return   sigmaPosHandle  Handle to positive sigma-line (if sigmaLineType != None)
# @return   sigmaNegHandle  Handle to negative sigma-line (if sigmaLineType != None)

# pylint: disable-msg=R0913,R0914

def draw_std_corridor(axis, xVals, mu, sigma, muLabel = None, sigmaLabel = None, muLineType = 'b-', \
                      sigmaFillColor = 'b', sigmaAlpha = '0.5', sigmaLineType = None):
    
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    xVals = np.asarray(xVals)
    
    # Find all NaNs in input data, whether in mu, sigma or xVals
    nanCheck = np.isnan(mu) + np.isnan(sigma) + np.isnan(xVals)
    # Each NaN, as well as the last element, is considered to be the end of a plot segment
    segmentEndList = filter(lambda ind: nanCheck[ind], range(len(nanCheck))) + [len(nanCheck)]
    sigmaCorHandles = []
    startIndex = 0
    # Iterate through list of segment endpoints
    for endIndex in segmentEndList:
        # In the case of contiguous NaNs, skip to the next (hopefully non-NaN) position
        if endIndex == startIndex:
            startIndex += 1
            continue
        # Extract plot segment (hopefully containing no NaNs)
        xSegment = xVals[startIndex:endIndex]
        muSegment = mu[startIndex:endIndex]
        sigmaSegment = sigma[startIndex:endIndex]
        assert not any(np.isnan(xSegment)), "x values in plot segment of draw_std_corridor contain NaNs."
        assert not any(np.isnan(muSegment)), "mu values in plot segment of draw_std_corridor contain NaNs."
        assert not any(np.isnan(sigmaSegment)), "sigma values in plot segment of draw_std_corridor contain NaNs."
        # reverse xVals and y2 so the polygon fills in order
        x = np.concatenate( (xSegment, xSegment[::-1]) )
        y1 = muSegment + sigmaSegment
        y2 = muSegment - sigmaSegment
        y = np.concatenate( (y1, y2[::-1]) )
        # Plot filled polygon for segment
        sigmaCorHandles.append(axis.fill(x, y, facecolor = sigmaFillColor, alpha = sigmaAlpha, label = sigmaLabel))
        startIndex = endIndex + 1
    
    muHandle = axis.plot(xVals, mu, muLineType, label = muLabel, lw = 2)
    
    if sigmaLineType:
        sigmaPosHandle = axis.plot(xVals, y1, sigmaLineType)
        sigmaNegHandle = axis.plot(xVals, y2, sigmaLineType)
        return muHandle, sigmaCorHandles, sigmaPosHandle, sigmaNegHandle
    else:
        return muHandle, sigmaCorHandles


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  plot_gaussian_ellipse
#---------------------------------------------------------------------------------------------------------

## Plot ellipse(s) representing 2-D Gaussian function.
#
# @param    axis            Matplotlib axes object associated with a matplotlib Figure
# @param    mean            2-dimensional mean vector
# @param    cov             2x2 covariance matrix
# @param    contour         Contour height of ellipse(s), as a (list of) factor(s) of the peak value [0.5]
#                           For a factor sigma of the standard deviation, use e^(-0.5*sigma^2) here
# @param    ellipseLineType Matplotlib linetype string for ellipse line ['b-']
# @param    centerLineType  Matplotlib linetype string for center marker ['b+']
# @param    lineWidth       Line widths for ellipse and center marker [1]
# @return   ellipseHandle   Handle(s) to contour plot
# @return   centerHandle    Handle to center of Gaussian

def plot_gaussian_ellipse(axis, mean, cov, contour=0.5, ellipseLineType='b-', centerLineType='b+', lineWidth=1):
    
    mean = np.asarray(mean)
    cov = np.asarray(cov)
    contour = np.atleast_1d(np.asarray(contour))
    if (mean.shape != (2,)) or (cov.shape != (2, 2)):
        message = 'Mean and covariance should be 2-dimensional, with shapes (2,) and (2,2) instead of' \
                  + str(mean.shape) + ' and ' + str(cov.shape)
        logger.error(message)
        raise ValueError, message
    # Create parametric circle
    t = np.linspace(0, 2*np.pi, 200)
    circle = np.vstack((np.cos(t), np.sin(t)))
    # Determine and apply transformation to ellipse
    eigVal, eigVec = linalg.eig(cov)
    circleToEllipse = np.dot(eigVec, np.diag(np.sqrt(eigVal)))
    baseEllipse = np.real(np.dot(circleToEllipse, circle))
    ellipseHandle = []
    for cnt in contour:
        ellipse = np.sqrt(-2*np.log(cnt)) * baseEllipse + mean[:, np.newaxis]
        ellipseHandle.append(axis.plot(ellipse[0], ellipse[1], ellipseLineType, lw=lineWidth))
    centerHandle = axis.plot([mean[0]], [mean[1]], centerLineType, markersize=12, aa=False, mew=lineWidth)
    return ellipseHandle, centerHandle
