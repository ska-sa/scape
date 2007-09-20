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

## Draw mean and standard deviation corridor to a matplotlib axes 
#
# @param    mu              array of mean values
# @param    sig             array of standard-deviation values (sigma)
# @param    xVals           xaxis values
# @param    ax              matplotlib axes object associated with a matplotlib Figure
# @param    muLabel         label string for mean plot [None]
# @param    sigLabel        label string for std corridor [None]
# @param    muLineType      matplotlib linetype string for mean line ['r-']
# @param    sigFilColor     matplotlib color string used to fill std-corridor ['r']
# @param    sigAlpha        Transparency (alpha) value for std-corridor [0.5]
# @param    sigLineType     matplotlib linetype string for outline (boundary) of std-corridor [None]
# @return   sigCor          handle to std-corridor object
# @return   muPlot          handle to mean plot
# @return   sigPosPlot      handle to positive sigma-line (if sigLineType != None)
# @return   sigNegPlot      handle to negative sigma-line (if sigLineType != None)
          
# pylint: disable-msg=R0913,R0914

def draw_std_corridor(mu,
                        sig, 
                        xVals, 
                        ax, 
                        muLabel = None, 
                        sigLabel = None, 
                        muLineType = 'b-', 
                        sigFilColor = 'b', 
                        sigAlpha = '0.5', 
                        sigLineType = None):
    

    # reverse xVals and y2 so the polygon fills in order
    
    x = np.concatenate( (xVals, xVals[::-1]) )
    y1 = mu + sig
    y2 = mu - sig
    y = np.concatenate( (y1, y2[::-1]) )
                 
    sigCor = ax.fill(x, y, facecolor = sigFilColor, alpha = sigAlpha, label = sigLabel)
    
    muPlot = ax.plot(xVals, mu, muLineType, label = muLabel, lw = 2)
    
    if sigLineType:
        sigPosPlot = ax.plot(xVals, y1, sigLineType)
        sigNegPlot = ax.plot(xVals, y2, sigLineType)
        return sigCor, muPlot, sigPosPlot, sigNegPlot
    else:
        return sigCor, muPlot


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :  plot_gaussian_ellipse
#---------------------------------------------------------------------------------------------------------

## Plot ellipse(s) representing 2-D Gaussian function.
#
# @param    ax              Matplotlib axes object associated with a matplotlib Figure
# @param    mean            2-dimensional mean vector
# @param    cov             2x2 covariance matrix
# @param    contour         Contour height of ellipse(s), as a (list of) factor(s) of the peak value [0.5]
#                           For a factor sigma of the standard deviation, use e^(-0.5*sigma^2) here
# @param    ellipseLineType Matplotlib linetype string for ellipse line ['b-']
# @param    centerLineType  Matplotlib linetype string for center marker ['b+']
# @param    lineWidth       Line widths for ellipse and center marker [1]
# @return   ellipseHandle   Handle(s) to contour plot
# @return   centerHandle    Handle to center of Gaussian

def plot_gaussian_ellipse(ax, mean, cov, contour=0.5, ellipseLineType='b-', centerLineType='b+', lineWidth=1):
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
        ellipseHandle.append(ax.plot(ellipse[0], ellipse[1], ellipseLineType, lw=lineWidth))
    centerHandle = ax.plot([mean[0]], [mean[1]], centerLineType, markersize=12, aa=False, mew=lineWidth)
    return ellipseHandle, centerHandle
