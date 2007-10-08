## @file xdmsbe/xdmsbelib/pcal.py
#
# Data reduction backend for polarization calibration measurements.
#
# copyright (c) 2007 SKA/KAT. All rights reserved.
# @brief Data reduction backend for polarization calibration measurements.
# @author Mattieu de Villiers [mattieu@ska.ac.za] on 2007-09-18.
# pylint: disable-msg=C0103

from __future__ import division
import logging
import misc
import stats
import time
import numpy as np
from xdmsbe.xdmsbelib import fitsreader
import xdmsbe.xdmsbelib.stats as stats
from numpy import sqrt
from scipy.optimize.optimize import fmin

#=========================================================================================================
#===                                       FUNCTIONS                                                   ===
#=========================================================================================================



#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :    hadec_to_azel
#---------------------------------------------------------------------------------------------------------

## Convert hour-angle and declination to azimuth and elevation.
# See http://en.wikipedia.org/wiki/Horizontal_coordinate_system
#
# @param ha_rad      hour angle in radians
# @param dec_rad     declination in radians
# @param lat_rad     latitude in radians
#
# @return az_rad     azimuth in radians
# @return el_rad     elevation in radians

def hadec_to_azel(ha_rad, dec_rad, lat_rad):
    sinAlt = np.sin(lat_rad)*np.sin(dec_rad) + np.cos(lat_rad)*np.cos(dec_rad)*np.cos(ha_rad)
    cosAzCosAlt = np.cos(lat_rad)*np.sin(dec_rad) - np.sin(lat_rad)*np.cos(dec_rad)*np.cos(ha_rad)
    sinAzCosAlt = -np.cos(dec_rad)*np.sin(ha_rad)
    az_rad = np.arctan2(sinAzCosAlt, cosAzCosAlt)
    az_rad = np.mod(az_rad, 2*np.pi)
    rad1 = np.sqrt(sinAzCosAlt*sinAzCosAlt + cosAzCosAlt*cosAzCosAlt)
    el_rad = np.arctan2(sinAlt, rad1)
    rad2 = np.sqrt(sinAlt*sinAlt + rad1*rad1)
    return az_rad, el_rad


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :    azel_to_hadec
#---------------------------------------------------------------------------------------------------------

## Convert azimuth and elevation to hour angle and declination
# See http://en.wikipedia.org/wiki/Horizontal_coordinate_system
#
# @param az_rad      azimuth in radians
# @param el_rad      elevation in radians
# @param lat_rad     latitude in radians
#
# @return ha_rad     hour angle in radians
# @return dec_rad    declination in radians

def azel_to_hadec(az_rad, el_rad, lat_rad):
    sinDec = np.sin(lat_rad)*np.sin(el_rad) + np.cos(lat_rad)*np.cos(el_rad)*np.cos(az_rad)
    cosDecCosHa = np.cos(lat_rad)*np.sin(el_rad) - np.sin(lat_rad)*np.cos(el_rad)*np.cos(az_rad)
    cosDecSinHa = -np.sin(az_rad)*np.cos(el_rad)
    ha_rad = np.arctan2(cosDecSinHa, cosDecCosHa)
    rad1 = np.sqrt(cosDecSinHa*cosDecSinHa + cosDecCosHa*cosDecCosHa)
    dec_rad = np.arctan2(sinDec, rad1)
    rad2 = np.sqrt(sinDec*sinDec + rad1*rad1)
    return ha_rad, dec_rad


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :    parallactic_rotation
#---------------------------------------------------------------------------------------------------------

## Compute the parallactic rotation angle
#
# @param az_rad       azimuth in radians
# @param el_rad       elevation in radians
# @param lat_rad      latitude in radians
#
# @return parAngle    parallacttic rotation angle in radians

def parallactic_rotation(az_rad, el_rad, lat_rad):
    (ha_rad,dec_rad)=azel_to_hadec(az_rad,el_rad,lat_rad);
    parAngle=np.arctan2(np.sin(ha_rad), np.tan(lat_rad)*np.cos(dec_rad) - np.sin(dec_rad)*np.cos(ha_rad));
    return parAngle


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :    time_to_az_el
#---------------------------------------------------------------------------------------------------------

## Compute the az el coordinates given hour angle
#
# @param time         hour angle in seconds 
# @param dec_rad      declination of antenna in radians
# @param lat_rad      latitude of antenna in radians
#
# @return az_rad      azimuth pointing coordinates for antenna
# @return el_rad      elevation pointing coordinates for antenna

def time_to_az_el(time, dec_rad, lat_rad):
    ha_rad=time/60/60/24*(2*np.pi);
   
    return hadec_to_azel(ha_rad, dec_rad, lat_rad); 


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :    analyse_stokes
#---------------------------------------------------------------------------------------------------------

##  Displays an understandable interpretation of the stokes parameters
#
#  @param   stokes     a stokes vector
#

def analyse_stokes(stokes):

    I             =stokes[0];
    Q             =stokes[1];
    U             =stokes[2];
    V             =stokes[3];

    Ex            =np.sqrt((I+Q)/2);       #magnitude of x pol
    Ey            =np.sqrt((I-Q)/2);       #magnitude of y pol
    Etheta        =np.arctan2(V,U);        #phase angle between pol
    fracPol       =np.sqrt(Q*Q+U*U+V*V)/I; #fracion of power that is polarized
    fracLinPol    =np.sqrt(Q*Q+U*U)/I;     #fraction that is linearly polarized
    fracCircPol   =np.abs(V)/I;            #fraction that is circularly polarized
    posAngle      =0.5*np.arctan2(U,Q);    #position angle/ parallactic rotation angle
    
    print ";    ".join(["Magnitude I: %s" % (I)]);
    print ";    ".join(["Position angle: %s, phase angle: %s" % (posAngle*180/np.pi,Etheta*180/np.pi)]);
    print ";    ".join(["Frac lin pol: %s, frac circ pol: %s" % (fracLinPol,fracCircPol)]);


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :    compare_stokes
#---------------------------------------------------------------------------------------------------------

##  Displays an understandable interpretation of the stokes parameters comparing two stokes vectors side by side
#
#  @param   stokes     a stokes vector
#  @param   stokesX    a (second) stokes vector
#

def compare_stokes(stokes,stokesX):
    I             =stokes[0];
    Q             =stokes[1];
    U             =stokes[2];
    V             =stokes[3];

    Ex            =np.sqrt((I+Q)/2);       #magnitude of x pol
    Ey            =np.sqrt((I-Q)/2);       #magnitude of y pol
    Etheta        =np.arctan2(V,U);        #phase angle between pol
    fracPol       =np.sqrt(Q*Q+U*U+V*V)/I; #fracion of power that is polarized
    fracLinPol    =np.sqrt(Q*Q+U*U)/I;     #fraction that is linearly polarized
    fracCircPol   =np.abs(V)/I;            #fraction that is circularly polarized
    posAngle      =0.5*np.arctan2(U,Q);    #position angle/ parallactic rotation angle

    IX             =stokesX[0];
    QX             =stokesX[1];
    UX             =stokesX[2];
    VX             =stokesX[3];

    EthetaX        =np.arctan2(VX,UX);            #phase angle between pol
    fracLinPolX    =np.sqrt(QX*QX+UX*UX)/IX;      #fraction that is linearly polarized
    fracCircPolX   =np.abs(VX)/IX;                #fraction that is circularly polarized
    posAngleX      =0.5*np.arctan2(UX,QX);        #position angle/ parallactic rotation angle

    print '---Comparison of source------------'
    print '              True         Estimated'
    print  ";".join(["I            %8.3f  \t %8.3f" % (I, IX)]);
    print  ";".join(["Q            %8.3f  \t %8.3f" % (Q, QX)]);
    print  ";".join(["U            %8.3f  \t %8.3f" % (U, UX)]);
    print  ";".join(["V            %8.3f  \t %8.3f" % (V, VX)]);
    print  ";".join(["Pos angle    %8.3f  \t %8.3f" % (posAngle*180/np.pi, posAngleX*180/np.pi)]);
    print  ";".join(["Phase angle  %8.3f  \t %8.3f" % (Etheta*180/np.pi, EthetaX*180/np.pi)]);
    print  ";".join(["Frac linpol  %8.3f  \t %8.3f" % (fracLinPol, fracLinPolX)]);
    print  ";".join(["Frac circpol %8.3f  \t %8.3f" % (fracCircPol, fracCircPolX)]);


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :    analyse_feedMx
#---------------------------------------------------------------------------------------------------------

##  Displays the feed matrix in a readable form
#
#  @param   feedMx     complex feed matrix
#

def analyse_feedMx(feedMx):

    print ";".join(["[%s L %s,  %s L %s;" % (np.abs(feedMx[0,0]),np.angle(feedMx[0,0])*180/np.pi,np.abs(feedMx[0,1]),np.angle(feedMx[0,1])*180/np.pi)]);
    print ";".join([" %s L %s, %s L %s]" % (np.abs(feedMx[1,0]),np.angle(feedMx[1,0])*180/np.pi,np.abs(feedMx[1,1]),np.angle(feedMx[1,1])*180/np.pi)]);



#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :    compare_feedMx
#---------------------------------------------------------------------------------------------------------

##  Displays two feed matrices side by side in a readable form
#
#  @param   feedMx     complex feed matrix
#  @param   feedMxX    secondary complex feed matrix to compare
#

def compare_feedMx(feedMx,feedMxX):
    print '---Comparison of feed Matrix------------'
    print '         True                     Estimated'
    print  ";".join(["XX   %8.3f L %8.3f \t %8.3f L %8.3f" % (np.abs(feedMx[0,0]),np.angle(feedMx[0,0])*180/np.pi,np.abs(feedMxX[0,0]),np.angle(feedMxX[0,0])*180/np.pi)]);
    print  ";".join(["XY   %8.3f L %8.3f \t %8.3f L %8.3f" % (np.abs(feedMx[0,1]),np.angle(feedMx[0,1])*180/np.pi,np.abs(feedMxX[0,1]),np.angle(feedMxX[0,1])*180/np.pi)]);
    print  ";".join(["YX   %8.3f L %8.3f \t %8.3f L %8.3f" % (np.abs(feedMx[1,0]),np.angle(feedMx[1,0])*180/np.pi,np.abs(feedMxX[1,0]),np.angle(feedMxX[1,0])*180/np.pi)]);
    print  ";".join(["YY   %8.3f L %8.3f \t %8.3f L %8.3f" % (np.abs(feedMx[1,1]),np.angle(feedMx[1,1])*180/np.pi,np.abs(feedMxX[1,1]),np.angle(feedMxX[1,1])*180/np.pi)]);


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :    construct_stokes
#---------------------------------------------------------------------------------------------------------

##  Converts comprehendable specification of polarized electromagnetic wave into equivalent stokes parameter representation
#
#  @param    positionAngle  for linear and elliptical polarizations, this is the orientation of the electromagnetic
#                           wave in the plane perpendicular to the direction of propagation.
#  @param    fracLinPol     the fraction of the wave's power that is linearly polarized
#  @param    fracCircPol    the fraction of the wave's power that is circularly polarized
#  @param    stokesI        Stokes parameter I.
#
#  @return   stokes         Stokes parameter representation of the specified wave.

def construct_stokes(positionAngle,fracLinPol,fracCircPol,stokesI):
    #Note: TotFracPol.^2=FracLinPol.^2+FracCircPol.^2;
    
    stokesQ=fracLinPol*stokesI*np.cos(2.0*positionAngle);
    stokesU=fracLinPol*stokesI*np.sin(2.0*positionAngle);
    stokesV=fracCircPol*stokesI;
    stokes=np.array([stokesI,stokesQ,stokesU,stokesV]);
 
    return stokes;



#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :    get_covariance_matrix
#---------------------------------------------------------------------------------------------------------

## Determines the covariance matrix of a stokes vector
#
#  @param    stokes      a stokes vector 
#
#  @return   SS   the covariance matrix

def get_covariance_matrix(stokes):
    
    stokes2coherencyMatrix    =0.5*np.array([[1,1,0,0],[0,0,1,1j],[0,0,1,-1j],[1,-1,0,0]]);
    coherency                 =np.dot(stokes2coherencyMatrix,stokes);
    PP                        =np.array([[coherency[0],coherency[1]],[coherency[2],coherency[3]]]);
    SS                        =np.linalg.cholesky(PP);

    return SS;


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :    generate_values
#---------------------------------------------------------------------------------------------------------

## Creates multiple (randomized) stokes parameter values from the average coherency by simulating nAverage random voltage 'measurements'
#
#  @param    stokes      a stokes vector 
#  @param    nAverage    number of correlator samples to average, could be 'inf' for testing perfect noisefree case
#  @param    n           number of output stokes vectors to create alike to input stokes
#
#  @return   outstokes   the resultant series of stokes vectors.

def generate_values(stokes,nAverage,n):

    if (nAverage=='inf'):
        outstokes        =np.dot(stokes,np.ones([1,n]));
    else:
        scale            =np.sqrt(2.0);
        coherency2stokes =np.array([[1,0,0,1],[1,0,0,-1],[0,1,1,0],[0,-1j,1j,0]]);
        SS               =get_covariance_matrix(stokes);
        outstokes        =np.zeros([4,n]);
        
        for ind in range(0,n):
            randval         =scale*misc.randn_complex(2, nAverage);
            val             =np.dot(SS,randval);
        #now back to coherency
            coherencyin     =np.mean(np.array([[val[0]*np.conj(val[0])],[val[0]*np.conj(val[1])],[val[1]*np.conj(val[0])],[val[1]*np.conj(val[1])]]),2);
        #mean is called here only to get the dimensions compatible (this is because python sucks)
            outstokes[:,ind]=np.mean(np.dot(coherency2stokes,coherencyin),1);

    return outstokes


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :    rotate_stokes
#---------------------------------------------------------------------------------------------------------

## Rotate (parallactic rotation) input stokes vector.
#  The equations were converted from a Jones matrix operating on voltage signals, to an equivalent Mueller matrix operating on a Stokes vector, 
#  and is written in a form that can operate on a series of Stokes vector for efficiency.
#
#  @param    stokes      input series of stokes vectors; each column is a seperate stokes vector 
#  @param    parAngle    a vector of angles in radians that each corresponding stokes vector is rotated by
#
#  @return   newstokes   the resultant series of stokes vectors rotated by parAngle

def rotate_stokes(stokes,parAngle):
    
    cosParAngle2    =np.cos(2.0*parAngle);
    sinParAngle2    =np.sin(2.0*parAngle);
    
    newstokes       =np.array(stokes);
    newstokes[1]    =cosParAngle2*(stokes[1])-sinParAngle2*(stokes[2]);
    newstokes[2]    =sinParAngle2*(stokes[1])+cosParAngle2*(stokes[2]);
    
    return newstokes;

#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :    apply_feed_stokes
#---------------------------------------------------------------------------------------------------------

## Matrix multiplies input stoke vector values by feed matrix by first converting 2x2 Jones feed matrix algebraically into a Mueller matrix
#
#  @param    stokes      input series of stokes vectors; each column is a seperate stokes vector 
#  @param    feedMx      2x2 complex Jones matrix that represents the feed response to voltage signals
#
#  @return   newstokes   the resultant series of stokes vectors transformed by the feed.

def apply_feed_stokes(stokes,feedMx):
    #feedMx=[G,B;-C;H] or rather [g1+j*g2,b1+j*b2;-c1-j*c2;h1+j*h2]; notice errors equations in Simons Johnson's paper! Its a good thing I calculated these from first principles myself
    g1    =np.real(feedMx[0][0]);
    g2    =np.imag(feedMx[0][0]);
    b1    =np.real(feedMx[0][1]);
    b2    =np.imag(feedMx[0][1]);
    c1    =-np.real(feedMx[1][0]);
    c2    =-np.imag(feedMx[1][0]);
    h1    =np.real(feedMx[1][1]);
    h2    =np.imag(feedMx[1][1]);
    
    newstokes=np.array(stokes);
    I0=stokes[0];
    Q0=stokes[1];
    U0=stokes[2];
    V0=stokes[3];
    
    newstokes[0]=1.0/2.0*I0*(h1**2+h2**2+b1**2+b2**2+c2**2+c1**2+g1**2+g2**2)+1.0/2.0*Q0*(-h1**2-h2**2-b1**2-b2**2+c2**2+c1**2+g1**2+g2**2)+(g1*b1+g2*b2-c1*h1-c2*h2)*U0+(g1*b2-g2*b1+c2*h1-c1*h2)*V0;
    newstokes[1]=1.0/2.0*I0*(-h2**2-h1**2+b1**2+b2**2+g1**2+g2**2-c1**2-c2**2)+1.0/2.0*Q0*(h2**2+h1**2-b1**2-b2**2+g1**2+g2**2-c1**2-c2**2)+U0*(c2*h2+c1*h1+g1*b1+g2*b2)+V0*(c1*h2-c2*h1+g1*b2-g2*b1);
    newstokes[2]=I0*(b2*h2+b1*h1-g1*c1-g2*c2)+Q0*(-b2*h2-b1*h1-g1*c1-g2*c2)+U0*(-b2*c2-b1*c1+g1*h1+g2*h2)+V0*(-b2*c1+b1*c2+g1*h2-g2*h1);
    newstokes[3]=I0*(g1*c2-g2*c1-b1*h2+b2*h1)+Q0*(g1*c2-g2*c1+b1*h2-b2*h1)+U0*(-g1*h2+g2*h1+b1*c2-b2*c1)+V0*(g1*h1+g2*h2+b1*c1+b2*c2);
  
    return newstokes;

#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :    get_feed_stokes_X
#---------------------------------------------------------------------------------------------------------

## Matrix multiplies input stokes vector values by feed matrix 
#  This function is identical to apply_feed_stokes but uses the unknown vector X directly
#
#  @param    parAngle    a vector of angles in radians that each corresponding stokes vector is rotated by
#  @param    X           the vector of unknown values that are determined by data reduction
#                        where [[X[0],X[1]+j*X[2]],[X[3]+j*X[4],X[5]+j*X[6]]] is feedMx
#                        and  [X[7],X[8],X[9],X[10]] is stokes vector of source 
#
#  @return   newstokes   the resultant series of stokes vectors transformed by the feed and parallactic rotation.

def get_feed_stokes_X(stokes,X):
   
    I0=stokes[0];
    Q0=stokes[1];
    U0=stokes[2];
    V0=stokes[3];
    
    newstokes=np.array(stokes);
    newstokes[0]=1.0/2.0*I0*(X[5]**2+X[6]**2+X[1]**2+X[2]**2+X[4]**2+X[3]**2+X[0]**2)+1.0/2.0*Q0*(-X[5]**2-X[6]**2-X[1]**2-X[2]**2+X[4]**2+X[3]**2+X[0]**2)+(X[0]*X[1]+X[3]*X[5]+X[4]*X[6])*U0+(X[0]*X[2]-X[4]*X[5]+X[3]*X[6])*V0;
    newstokes[1]=1.0/2.0*I0*(-X[6]**2-X[5]**2+X[1]**2+X[2]**2+X[0]**2-X[3]**2-X[4]**2)+1.0/2.0*Q0*(X[6]**2+X[5]**2-X[1]**2-X[2]**2+X[0]**2-X[3]**2-X[4]**2)+U0*(-X[4]*X[6]-X[3]*X[5]+X[0]*X[1])+V0*(-X[3]*X[6]+X[4]*X[5]+X[0]*X[2]);
    newstokes[2]=I0*(X[2]*X[6]+X[1]*X[5]+X[0]*X[3])+Q0*(-X[2]*X[6]-X[1]*X[5]+X[0]*X[3])+U0*(X[2]*X[4]+X[1]*X[3]+X[0]*X[5])+V0*(X[2]*X[3]-X[1]*X[4]+X[0]*X[6]);
    newstokes[3]=I0*(-X[0]*X[4]-X[1]*X[6]+X[2]*X[5])+Q0*(-X[0]*X[4]+X[1]*X[6]-X[2]*X[5])+U0*(-X[0]*X[6]-X[1]*X[4]+X[2]*X[3])+V0*(X[0]*X[5]-X[1]*X[3]-X[2]*X[4]);

    return newstokes;


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :    get_feed_dstokes_dX
#---------------------------------------------------------------------------------------------------------

# Computes the gradient of get_feed_stokes_X
#
#  @param    stokes      a series of stokes vectors
#  @param    X           the vector of unknown values that are determined by data reduction
#                        where [[X[0],X[1]+j*X[2]],[X[3]+j*X[4],X[5]+j*X[6]]] is feedMx
#                        and  [X[7],X[8],X[9],X[10]] is stokes vector of source 
#
#  @return   newstokes   the resultant series of stokes vectors transformed by the feed.
#  @return   dIdX        derivative of stokes I transformed by get_feed_rotate_stoke_X wrt X
#  @return   dQdX        derivative of stokes Q transformed by get_feed_rotate_stoke_X wrt X
#  @return   dUdX        derivative of stokes U transformed by get_feed_rotate_stoke_X wrt X
#  @return   dVdX        derivative of stokes V transformed by get_feed_rotate_stoke_X wrt X

def get_feed_dstokes_dX(stokes,X):
   
    I0=stokes[0];
    Q0=stokes[1];
    U0=stokes[2];
    V0=stokes[3];
    
    newstokes=np.array(stokes);
    newstokes[0]=1.0/2.0*I0*(X[5]**2+X[6]**2+X[1]**2+X[2]**2+X[4]**2+X[3]**2+X[0]**2)+1.0/2.0*Q0*(-X[5]**2-X[6]**2-X[1]**2-X[2]**2+X[4]**2+X[3]**2+X[0]**2)+(X[0]*X[1]+X[3]*X[5]+X[4]*X[6])*U0+(X[0]*X[2]-X[4]*X[5]+X[3]*X[6])*V0;
    newstokes[1]=1.0/2.0*I0*(-X[6]**2-X[5]**2+X[1]**2+X[2]**2+X[0]**2-X[3]**2-X[4]**2)+1.0/2.0*Q0*(X[6]**2+X[5]**2-X[1]**2-X[2]**2+X[0]**2-X[3]**2-X[4]**2)+U0*(-X[4]*X[6]-X[3]*X[5]+X[0]*X[1])+V0*(-X[3]*X[6]+X[4]*X[5]+X[0]*X[2]);
    newstokes[2]=I0*(X[2]*X[6]+X[1]*X[5]+X[0]*X[3])+Q0*(-X[2]*X[6]-X[1]*X[5]+X[0]*X[3])+U0*(X[2]*X[4]+X[1]*X[3]+X[0]*X[5])+V0*(X[2]*X[3]-X[1]*X[4]+X[0]*X[6]);
    newstokes[3]=I0*(-X[0]*X[4]-X[1]*X[6]+X[2]*X[5])+Q0*(-X[0]*X[4]+X[1]*X[6]-X[2]*X[5])+U0*(-X[0]*X[6]-X[1]*X[4]+X[2]*X[3])+V0*(X[0]*X[5]-X[1]*X[3]-X[2]*X[4]);

    if (np.shape(np.shape(stokes))[0]==1):#python is a pile of shit
        nStokes=1;
    else:
        nStokes=np.shape(stokes)[1];
        
    dIdX=np.zeros([11,nStokes]);
    dIdX[0]=I0*X[0]+Q0*X[0]+U0*X[1]+V0*X[2];
    dIdX[1]=I0*X[1]-Q0*X[1]+U0*X[0];
    dIdX[2]=I0*X[2]-Q0*X[2]+V0*X[0];
    dIdX[3]=I0*X[3]+Q0*X[3]+U0*X[5]+V0*X[6];
    dIdX[4]=I0*X[4]+Q0*X[4]+U0*X[6]-V0*X[5];
    dIdX[5]=I0*X[5]-Q0*X[5]+U0*X[3]-V0*X[4];
    dIdX[6]=I0*X[6]-Q0*X[6]+U0*X[4]+V0*X[3];
    
    dQdX=np.zeros([11,nStokes]);
    dQdX[0]= I0*X[0]+Q0*X[0]+U0*X[1]+V0*X[2];
    dQdX[1]= I0*X[1]-Q0*X[1]+U0*X[0];
    dQdX[2]= I0*X[2]-Q0*X[2]+V0*X[0];
    dQdX[3]=-I0*X[3]-Q0*X[3]-U0*X[5]-V0*X[6];
    dQdX[4]=-I0*X[4]-Q0*X[4]-U0*X[6]+V0*X[5];
    dQdX[5]=-I0*X[5]+Q0*X[5]-U0*X[3]+V0*X[4];
    dQdX[6]=-I0*X[6]+Q0*X[6]-U0*X[4]-V0*X[3];
    
    dUdX=np.zeros([11,nStokes]);
    dUdX[0]=I0*X[3]+Q0*X[3]+U0*X[5]+V0*X[6];
    dUdX[1]=I0*X[5]-Q0*X[5]+U0*X[3]-V0*X[4];
    dUdX[2]=I0*X[6]-Q0*X[6]+U0*X[4]+V0*X[3];
    dUdX[3]=I0*X[0]+Q0*X[0]+U0*X[1]+V0*X[2];
    dUdX[4]=U0*X[2]-V0*X[1];
    dUdX[5]=I0*X[1]-Q0*X[1]+U0*X[0];
    dUdX[6]=I0*X[2]-Q0*X[2]+V0*X[0];
    
    dVdX=np.zeros([11,nStokes]);
    dVdX[0]=-I0*X[4]-Q0*X[4]-U0*X[6]+V0*X[5];
    dVdX[1]=-I0*X[6]+Q0*X[6]-U0*X[4]-V0*X[3];
    dVdX[2]= I0*X[5]-Q0*X[5]+U0*X[3]-V0*X[4];
    dVdX[3]= U0*X[2]-V0*X[1];
    dVdX[4]=-I0*X[0]-Q0*X[0]-U0*X[1]-V0*X[2];
    dVdX[5]= I0*X[2]-Q0*X[2]+V0*X[0];
    dVdX[6]=-I0*X[1]+Q0*X[1]-U0*X[0];

    return (newstokes,dIdX,dQdX,dUdX,dVdX);


#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :    get_feed_rotate_stokes_X
#---------------------------------------------------------------------------------------------------------

## Rotates by parallactic angle and matrix multiplies input stokes vector values by feed matrix 
#  This function uses the unknown vector X directly
#
#  @param    parAngle    a vector of angles in radians that each corresponding stokes vector is rotated by
#  @param    X           the vector of unknown values that are determined by data reduction
#                        where [[X[0],X[1]+j*X[2]],[X[3]+j*X[4],X[5]+j*X[6]]] is feedMx
#                        and  [X[7],X[8],X[9],X[10]] is stokes vector of source 
#
#  @return   newstokes   the resultant series of stokes vectors transformed by the feed and parallactic rotation.

def get_feed_rotate_stokes_X(parAngle,X):
    
    cosParAngle2=np.cos(2.0*parAngle);
    sinParAngle2=np.sin(2.0*parAngle);
  
    newstokes=np.zeros([4,(np.shape(parAngle))[0]]);
    newstokes[0]=1.0/2.0*X[7]*(X[5]**2+X[6]**2+X[1]**2+X[2]**2+X[4]**2+X[3]**2+X[0]**2)+1.0/2.0*(cosParAngle2*X[8]-sinParAngle2*X[9])*(-X[5]**2-X[6]**2-X[1]**2-X[2]**2+X[4]**2+X[3]**2+X[0]**2)+(sinParAngle2*X[8]+cosParAngle2*X[9])*(X[0]*X[1]+X[3]*X[5]+X[4]*X[6])+(X[0]*X[2]-X[4]*X[5]+X[3]*X[6])*X[10];
    newstokes[1]=1.0/2.0*X[7]*(-X[6]**2-X[5]**2+X[1]**2+X[2]**2+X[0]**2-X[3]**2-X[4]**2)+1.0/2.0*(cosParAngle2*X[8]-sinParAngle2*X[9])*(X[6]**2+X[5]**2-X[1]**2-X[2]**2+X[0]**2-X[3]**2-X[4]**2)+(sinParAngle2*X[8]+cosParAngle2*X[9])*(-X[4]*X[6]-X[3]*X[5]+X[0]*X[1])+X[10]*(-X[3]*X[6]+X[4]*X[5]+X[0]*X[2]);
    newstokes[2]=X[7]*(X[2]*X[6]+X[1]*X[5]+X[0]*X[3])+(cosParAngle2*X[8]-sinParAngle2*X[9])*(-X[2]*X[6]-X[1]*X[5]+X[0]*X[3])+(sinParAngle2*X[8]+cosParAngle2*X[9])*(X[2]*X[4]+X[1]*X[3]+X[0]*X[5])+X[10]*(X[2]*X[3]-X[1]*X[4]+X[0]*X[6]);
    newstokes[3]=X[7]*(-X[0]*X[4]-X[1]*X[6]+X[2]*X[5])+(cosParAngle2*X[8]-sinParAngle2*X[9])*(-X[0]*X[4]+X[1]*X[6]-X[2]*X[5])+(sinParAngle2*X[8]+cosParAngle2*X[9])*(-X[0]*X[6]-X[1]*X[4]+X[2]*X[3])+X[10]*(X[0]*X[5]-X[1]*X[3]-X[2]*X[4]);
 
    return newstokes;

#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :    get_feed_rotate_stokes_dX
#---------------------------------------------------------------------------------------------------------

# Computes the gradient of get_feed_rotate_stokes_X
#
#  @param    parAngle    a vector of angles in radians that each corresponding stokes vector is rotated by
#  @param    X           the vector of unknown values that are determined by data reduction
#                        where [[X[0],X[1]+j*X[2]],[X[3]+j*X[4],X[5]+j*X[6]]] is feedMx
#                        and  [X[7],X[8],X[9],X[10]] is stokes vector of source 
#
#  @return   newstokes   the resultant series of stokes vectors transformed by the feed.
#  @return   dIdX        derivative of stokes I transformed by get_feed_rotate_stoke_X wrt X
#  @return   dQdX        derivative of stokes Q transformed by get_feed_rotate_stoke_X wrt X
#  @return   dUdX        derivative of stokes U transformed by get_feed_rotate_stoke_X wrt X
#  @return   dVdX        derivative of stokes V transformed by get_feed_rotate_stoke_X wrt X

def get_feed_rotate_dstokes_dX(parAngle,X):
    
    nStokes=(np.shape(parAngle))[0];
    cosParAngle2=np.cos(2.0*parAngle);
    sinParAngle2=np.sin(2.0*parAngle);
  
    newstokes=np.zeros([4,nStokes]);
    newstokes[0]=1.0/2.0*X[7]*(X[5]**2+X[6]**2+X[1]**2+X[2]**2+X[4]**2+X[3]**2+X[0]**2)+1.0/2.0*(cosParAngle2*X[8]-sinParAngle2*X[9])*(-X[5]**2-X[6]**2-X[1]**2-X[2]**2+X[4]**2+X[3]**2+X[0]**2)+(sinParAngle2*X[8]+cosParAngle2*X[9])*(X[0]*X[1]+X[3]*X[5]+X[4]*X[6])+(X[0]*X[2]-X[4]*X[5]+X[3]*X[6])*X[10];
    newstokes[1]=1.0/2.0*X[7]*(-X[6]**2-X[5]**2+X[1]**2+X[2]**2+X[0]**2-X[3]**2-X[4]**2)+1.0/2.0*(cosParAngle2*X[8]-sinParAngle2*X[9])*(X[6]**2+X[5]**2-X[1]**2-X[2]**2+X[0]**2-X[3]**2-X[4]**2)+(sinParAngle2*X[8]+cosParAngle2*X[9])*(-X[4]*X[6]-X[3]*X[5]+X[0]*X[1])+X[10]*(-X[3]*X[6]+X[4]*X[5]+X[0]*X[2]);
    newstokes[2]=X[7]*(X[2]*X[6]+X[1]*X[5]+X[0]*X[3])+(cosParAngle2*X[8]-sinParAngle2*X[9])*(-X[2]*X[6]-X[1]*X[5]+X[0]*X[3])+(sinParAngle2*X[8]+cosParAngle2*X[9])*(X[2]*X[4]+X[1]*X[3]+X[0]*X[5])+X[10]*(X[2]*X[3]-X[1]*X[4]+X[0]*X[6]);
    newstokes[3]=X[7]*(-X[0]*X[4]-X[1]*X[6]+X[2]*X[5])+(cosParAngle2*X[8]-sinParAngle2*X[9])*(-X[0]*X[4]+X[1]*X[6]-X[2]*X[5])+(sinParAngle2*X[8]+cosParAngle2*X[9])*(-X[0]*X[6]-X[1]*X[4]+X[2]*X[3])+X[10]*(X[0]*X[5]-X[1]*X[3]-X[2]*X[4]);

    dIdX=np.zeros([11,nStokes]);
    dIdX[0]=X[7]*X[0]+X[0]*(cosParAngle2*X[8]-sinParAngle2*X[9])+(sinParAngle2*X[8]+cosParAngle2*X[9])*X[1]+X[10]*X[2];
    dIdX[1]=X[7]*X[1]-X[1]*(cosParAngle2*X[8]-sinParAngle2*X[9])+(sinParAngle2*X[8]+cosParAngle2*X[9])*X[0];
    dIdX[2]=X[7]*X[2]-X[2]*(cosParAngle2*X[8]-sinParAngle2*X[9])+X[10]*X[0];
    dIdX[3]=X[7]*X[3]+X[3]*(cosParAngle2*X[8]-sinParAngle2*X[9])+(sinParAngle2*X[8]+cosParAngle2*X[9])*X[5]+X[10]*X[6];
    dIdX[4]=X[7]*X[4]+X[4]*(cosParAngle2*X[8]-sinParAngle2*X[9])+(sinParAngle2*X[8]+cosParAngle2*X[9])*X[6]-X[10]*X[5];
    dIdX[5]=X[7]*X[5]-X[5]*(cosParAngle2*X[8]-sinParAngle2*X[9])+(sinParAngle2*X[8]+cosParAngle2*X[9])*X[3]-X[10]*X[4];
    dIdX[6]=X[7]*X[6]-X[6]*(cosParAngle2*X[8]-sinParAngle2*X[9])+(sinParAngle2*X[8]+cosParAngle2*X[9])*X[4]+X[10]*X[3];
    dIdX[7]=1/2*(X[5]**2+X[6]**2+X[1]**2+X[2]**2+X[4]**2+X[3]**2+X[0]**2);
    dIdX[8]=1/2*(-X[5]**2-X[6]**2-X[1]**2-X[2]**2+X[4]**2+X[3]**2+X[0]**2)*cosParAngle2+sinParAngle2*(X[0]*X[1]+X[3]*X[5]+X[4]*X[6]);
    dIdX[9]=-1/2*(-X[5]**2-X[6]**2-X[1]**2-X[2]**2+X[4]**2+X[3]**2+X[0]**2)*sinParAngle2+cosParAngle2*(X[0]*X[1]+X[3]*X[5]+X[4]*X[6]);
    dIdX[10]=X[0]*X[2]-X[4]*X[5]+X[3]*X[6];
    
    dQdX=np.zeros([11,nStokes]);
    dQdX[0]=X[7]*X[0]+X[0]*(cosParAngle2*X[8]-sinParAngle2*X[9])+(sinParAngle2*X[8]+cosParAngle2*X[9])*X[1]+X[10]*X[2];
    dQdX[1]=X[7]*X[1]-X[1]*(cosParAngle2*X[8]-sinParAngle2*X[9])+(sinParAngle2*X[8]+cosParAngle2*X[9])*X[0];
    dQdX[2]=X[7]*X[2]-X[2]*(cosParAngle2*X[8]-sinParAngle2*X[9])+X[10]*X[0];
    dQdX[3]=-X[7]*X[3]-X[3]*(cosParAngle2*X[8]-sinParAngle2*X[9])-(sinParAngle2*X[8]+cosParAngle2*X[9])*X[5]-X[10]*X[6];
    dQdX[4]=-X[7]*X[4]-X[4]*(cosParAngle2*X[8]-sinParAngle2*X[9])-(sinParAngle2*X[8]+cosParAngle2*X[9])*X[6]+X[10]*X[5];
    dQdX[5]=-X[7]*X[5]+X[5]*(cosParAngle2*X[8]-sinParAngle2*X[9])-(sinParAngle2*X[8]+cosParAngle2*X[9])*X[3]+X[10]*X[4];
    dQdX[6]=-X[7]*X[6]+X[6]*(cosParAngle2*X[8]-sinParAngle2*X[9])-(sinParAngle2*X[8]+cosParAngle2*X[9])*X[4]-X[10]*X[3];
    dQdX[7]=1/2*(-X[6]**2-X[5]**2+X[1]**2+X[2]**2+X[0]**2-X[3]**2-X[4]**2);
    dQdX[8]=1/2*(X[6]**2+X[5]**2-X[1]**2-X[2]**2+X[0]**2-X[3]**2-X[4]**2)*cosParAngle2+sinParAngle2*(-X[4]*X[6]-X[3]*X[5]+X[0]*X[1]);
    dQdX[9]=-1/2*(X[6]**2+X[5]**2-X[1]**2-X[2]**2+X[0]**2-X[3]**2-X[4]**2)*sinParAngle2+cosParAngle2*(-X[4]*X[6]-X[3]*X[5]+X[0]*X[1]);
    dQdX[10]=-X[3]*X[6]+X[4]*X[5]+X[0]*X[2];
    
    dUdX=np.zeros([11,nStokes]);
    dUdX[0]=X[7]*X[3]+(cosParAngle2*X[8]-sinParAngle2*X[9])*X[3]+(sinParAngle2*X[8]+cosParAngle2*X[9])*X[5]+X[10]*X[6];
    dUdX[1]=X[7]*X[5]-(cosParAngle2*X[8]-sinParAngle2*X[9])*X[5]+(sinParAngle2*X[8]+cosParAngle2*X[9])*X[3]-X[10]*X[4];
    dUdX[2]=X[7]*X[6]-(cosParAngle2*X[8]-sinParAngle2*X[9])*X[6]+(sinParAngle2*X[8]+cosParAngle2*X[9])*X[4]+X[10]*X[3];
    dUdX[3]=X[7]*X[0]+(cosParAngle2*X[8]-sinParAngle2*X[9])*X[0]+(sinParAngle2*X[8]+cosParAngle2*X[9])*X[1]+X[10]*X[2];
    dUdX[4]=(sinParAngle2*X[8]+cosParAngle2*X[9])*X[2]-X[10]*X[1];
    dUdX[5]=X[7]*X[1]-(cosParAngle2*X[8]-sinParAngle2*X[9])*X[1]+(sinParAngle2*X[8]+cosParAngle2*X[9])*X[0];
    dUdX[6]=X[7]*X[2]-(cosParAngle2*X[8]-sinParAngle2*X[9])*X[2]+X[10]*X[0];
    dUdX[7]=X[2]*X[6]+X[1]*X[5]+X[0]*X[3];
    dUdX[8]=cosParAngle2*(-X[2]*X[6]-X[1]*X[5]+X[0]*X[3])+sinParAngle2*(X[2]*X[4]+X[1]*X[3]+X[0]*X[5]);
    dUdX[9]=-sinParAngle2*(-X[2]*X[6]-X[1]*X[5]+X[0]*X[3])+cosParAngle2*(X[2]*X[4]+X[1]*X[3]+X[0]*X[5]);
    dUdX[10]=X[2]*X[3]-X[1]*X[4]+X[0]*X[6];
    
    dVdX=np.zeros([11,nStokes]);
    dVdX[0]=-X[7]*X[4]-(cosParAngle2*X[8]-sinParAngle2*X[9])*X[4]-(sinParAngle2*X[8]+cosParAngle2*X[9])*X[6]+X[10]*X[5];
    dVdX[1]=-X[7]*X[6]+(cosParAngle2*X[8]-sinParAngle2*X[9])*X[6]-(sinParAngle2*X[8]+cosParAngle2*X[9])*X[4]-X[10]*X[3];
    dVdX[2]=X[7]*X[5]-(cosParAngle2*X[8]-sinParAngle2*X[9])*X[5]+(sinParAngle2*X[8]+cosParAngle2*X[9])*X[3]-X[10]*X[4];
    dVdX[3]=(sinParAngle2*X[8]+cosParAngle2*X[9])*X[2]-X[10]*X[1];
    dVdX[4]=-X[7]*X[0]-(cosParAngle2*X[8]-sinParAngle2*X[9])*X[0]-(sinParAngle2*X[8]+cosParAngle2*X[9])*X[1]-X[10]*X[2];
    dVdX[5]=X[7]*X[2]-(cosParAngle2*X[8]-sinParAngle2*X[9])*X[2]+X[10]*X[0];
    dVdX[6]=-X[7]*X[1]+(cosParAngle2*X[8]-sinParAngle2*X[9])*X[1]-(sinParAngle2*X[8]+cosParAngle2*X[9])*X[0];
    dVdX[7]=-X[0]*X[4]-X[1]*X[6]+X[2]*X[5];
    dVdX[8]=cosParAngle2*(-X[0]*X[4]+X[1]*X[6]-X[2]*X[5])+sinParAngle2*(-X[0]*X[6]-X[1]*X[4]+X[2]*X[3]);
    dVdX[9]=-sinParAngle2*(-X[0]*X[4]+X[1]*X[6]-X[2]*X[5])+cosParAngle2*(-X[0]*X[6]-X[1]*X[4]+X[2]*X[3]);
    dVdX[10]=X[0]*X[5]-X[1]*X[3]-X[2]*X[4];

    return (newstokes,dIdX,dQdX,dUdX,dVdX);

#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :    get_stokes_cost
#---------------------------------------------------------------------------------------------------------

## determine mean square error between simulated data (due to model) and measurement data 
#
#  @param    X                the vector of unknown values that are determined by data reduction
#                             where [[X[0],X[1]+j*X[2]],[X[3]+j*X[4],X[5]+j*X[6]]] is feedMx
#                             and  [X[7],X[8],X[9],X[10]] is stokes vector of source 
#  @param    packetHeader     header information 
#  @param    packets          measurement data
#
#  @return   cost             the mean square error between measured and simulated stokes parameter values

def get_stokes_cost(X,packetHeader,packets):

    calStokesVal=get_feed_stokes_X(packetHeader.calStokes,X);
    calStokes=np.dot(calStokesVal[:,np.newaxis],np.ones([1,packetHeader.nCalStokes]));
    cost=0;
    for ipacket in range(0,packetHeader.nPackets):
        sourceStokes=get_feed_rotate_stokes_X(packets[ipacket].parRotation,X);
        
        cost+=np.sum((packets[ipacket].preCalStokes-calStokes)**2)/packetHeader.nCalStokes+np.sum((packets[ipacket].sourceStokes-sourceStokes)**2)/packetHeader.nStokes+np.sum((packets[ipacket].postCalStokes-calStokes)**2)/packetHeader.nCalStokes;

    return cost/packetHeader.nPackets/3;

#---------------------------------------------------------------------------------------------------------
#--- FUNCTION :    get_stokes_dcost
#---------------------------------------------------------------------------------------------------------

## determine the gradient of the mean square error between simulated data (due to model) and measurement data 
#
#  @param    X                the vector of unknown values that are determined by data reduction
#                             where [[X[0],X[1]+j*X[2]],[X[3]+j*X[4],X[5]+j*X[6]]] is feedMx
#                             and  [X[7],X[8],X[9],X[10]] is stokes vector of source 
#  @param    packetHeader     header information 
#  @param    packets          measurement data
#
#  @return   dX               the gradient of the mean square error with respect to the parameters X

def get_stokes_dcost(X,packetHeader,packets):

    (calStokesVal,dIdX,dQdX,dUdX,dVdX)=get_feed_dstokes_dX(packetHeader.calStokes,X);
    calStokes=np.dot(calStokesVal[:,np.newaxis],np.ones([1,packetHeader.nCalStokes]));
    dCalStokes=np.zeros([11,4,packetHeader.nCalStokes]);
    dSourceStokes=np.zeros([11,4,packetHeader.nStokes]);
    for ix in range(0,11):
        dCalStokes[ix]=np.dot(np.array([dIdX[ix],dQdX[ix],dUdX[ix],dVdX[ix]]),np.ones([1,packetHeader.nCalStokes]));
    dX=np.zeros([11,1]);
    for ipacket in range(0,packetHeader.nPackets):
        (sourceStokes,dIdX,dQdX,dUdX,dVdX)=get_feed_rotate_dstokes_dX(packets[ipacket].parRotation,X);
        for ix in range(0,11):
            dSourceStokes[ix]=np.array([dIdX[ix],dQdX[ix],dUdX[ix],dVdX[ix]]);        
            dX[ix]+=-np.sum(2*(packets[ipacket].preCalStokes-calStokes)*dCalStokes[ix])/packetHeader.nCalStokes-np.sum(2*(packets[ipacket].sourceStokes-sourceStokes)*dSourceStokes[ix])/packetHeader.nStokes-np.sum(2*(packets[ipacket].postCalStokes-calStokes)*dCalStokes[ix])/packetHeader.nCalStokes;

    return np.mean(dX,1)/packetHeader.nPackets/3;#mean just to remove dimension



