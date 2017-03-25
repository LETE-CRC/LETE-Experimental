#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
===============================================================================

                      Python code for PIV analysis

   Created by Combustion Research Center CRC at LETE - Sao Paulo, Brasil
   Laboratory of Environmental and Thermal Engineering - LETE
   Escola Politecnica da USP - EPUSP
   
===============================================================================
version:1.0 - 08/2016: Helio Villanueva
"""

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import subprocess
from scipy import signal as sig
from progressbar import ProgressBar
from tvtk.api import tvtk, write_data


#******************************************************************************
## -- Classes
#******************************************************************************

class VTKWriter():
    '''
    Class to write PIV results to VTK format to read in ParaView
    '''
    def __init__(self):
        x, y, z = np.mgrid[0:XPixels-1:49j, 0:YPixels-1:49j, 0:1:1j]
        self.shape = x.shape
        pts = np.empty(self.shape+(3,), dtype=float)
        pts[..., 0] = x
        pts[..., 1] = y
        pts[..., 2] = z
        pts.shape = pts.size / 3, 3
        self.meshq = tvtk.StructuredGrid(dimensions=x.shape, points=pts)
        
    def addScalar(self,S,Sname):
        '''Func to add ScalarField\n
        S -> NumPy Array\n
        Sname -> string for ScalarField name
        '''
        S2 = S.copy()
        S2 = np.flipud(S2)
        S2 = S2.T
        self.meshq.point_data.scalars = S2.ravel()
        self.meshq.point_data.scalars.name = Sname
        
    def addVector(self,u,v,Vname):
        '''Func to add VectorField\n
        u,v -> NumPy Array\n
        Vname -> string for VectorField name'''
        vel = np.empty(self.shape + (3,), dtype=float)
        vel[..., 0] = u
        vel[..., 1] = v
        vel[..., 2] = np.zeros_like(u)
        vel = np.flipud(vel)
        vel = vel.transpose(2,1,0,3).copy()
        vel.shape = vel.size / 3, 3
        self.meshq.point_data.vectors = vel
        self.meshq.point_data.vectors.name = Vname
        
    def write(self,FileName):
        '''Func to write .vtk file.\n
        ex: Filename='U.vtk'
        '''
        write_data(self.meshq,FileName)




#******************************************************************************
## -- Functions
#******************************************************************************


def readData(files):
    '''
    Read PIV data and return variables U,V
    '''
    ## -- Create U, V vel components
    U = np.zeros((XPixels,YPixels,Ttot))
    V = np.zeros_like(U)
    
    pbar = ProgressBar()
    ## -- Loop over all files/times
    print('Reading PIV files')
    pbar.start()
    
    for k,names in enumerate(files):
        
        if k==0:
            perc = 0.
        else:
            perc = k/float(Ttot)*100.
            
        cols = (6,7)
        ## -- Read all raw data in array
        data_raw = np.genfromtxt(names,delimiter=';',skip_header=9,usecols=cols)
        
        U[:,:,k] = data_raw[:,0].reshape((XPixels,YPixels))
        V[:,:,k] = data_raw[:,1].reshape((XPixels,YPixels))
                    
        pbar.update(perc)        

    pbar.finish()

    U = np.flipud(U)
    V = np.flipud(V)
    U = np.nan_to_num(U)
    V = np.nan_to_num(V)
    
    return (U,V)
    
#******************************************************************************
def gaussian(x, mu, signal):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(signal, 2.)))

#******************************************************************************
def generate_video(img,foldername,erase='n',cmap='jet'):
    '''img: Array of images along time.\n
    foldername: Directory name to save images/video.\n
    erase: Erase ('y') or not ('n') the images used for video
    '''
    
    ## -- Create result directory if not already present
    if os.path.isdir(foldername):
        pass
    else:
        os.makedirs(foldername)
        
    ## -- move to res dir and get path
    os.chdir(foldername)
    folder = os.getcwd()

    ## -- Start saving images for field img
    print('Saving images for ' + str(foldername) + ': ')
    plt.figure()
    for i in xrange(len(img[0,1])):
        plt.imshow(img[:,:,i],cmap=cmap)
        plt.title('Time: ' + str(i))
        plt.clim(img[:,:,0].min(),img[:,:,0].max())
        plt.xticks()
        plt.colorbar()
        plt.savefig(folder + "/time%02d.png" % i)
        print(str(i*100/len(img[0,1])) + ' %')
        plt.clf()

    ## -- Create movie from images
    subprocess.call(['ffmpeg','-framerate','6','-i','time%02d.png','-c:v',
                     'libx264','-r','30','-pix_fmt','yuv420p','video.mp4'])
    
    ## -- Erase images uPSDsused for movie
    if erase=='n':
        pass
    else:
        for file_name in glob("*.png"):
            os.remove(file_name)

#******************************************************************************
#******************************************************************************


    
#******************************************************************************
## -- Main
#****************************************************************************** 

## -- Obtain list of files in the directory
files = glob("*.csv")
files.sort()

Ttot = len(files)
XPixels = 49
YPixels = XPixels

## -- Read PIV data
U,V = readData(files)

print('Start calculations')
#******************************************************************************
# -- Temporal calculations
#******************************************************************************

# -- Time average
Utmean = np.mean(U,2,keepdims=True)
Vtmean = np.mean(V,2,keepdims=True)

# -- Time fluctuations
utL = U - Utmean
vtL = V - Vtmean

# -- Reynolds Stress Tensor components
print('Calculating Reynolds Stress tensor components')
uu = np.mean(utL*utL,axis=2)
vv = np.mean(vtL*vtL,axis=2)
uv = np.mean(utL*vtL,axis=2)

print('Calculating Turbulent Kinetic Energy and Dissipation Rate (K epsilon)')
K = 0.5*(2.0*uu + vv) #2.0*uu assumes ww = uu

#******************************************************************************
## -- Signal Processing
#******************************************************************************

print('Calculating PSDs')
# -- Power Spectrum Density using Welch's method
freqs, uPSDs = sig.welch(U,fs=7491.0,window='nuttall',nperseg=Ttot/8,
                         noverlap=Ttot/16,nfft=5189,scaling='density')
freqs, vPSDs = sig.welch(V,fs=7491.0,window='nuttall',nperseg=Ttot/8,
                         noverlap=Ttot/16,nfft=5189,scaling='density')

# -- Frequencies
freq = freqs*len(U[0,1])/(2*np.pi)

# -- Mean PSDs
PSDs = (uPSDs + vPSDs)/2.

print('Calculating turbulence integral time scale')
## -- turbulence Integral time scale
tau_u = np.pi*uPSDs[:,:,0]/(2.*uu)
tau_v = np.pi*vPSDs[:,:,0]/(2.*vv)

## -- turbulence Integral lenght scale
Luut = np.zeros_like(utL)
Lvvt = Luut
for t in range(Ttot):
    Luut[:,:,t] = utL[:,:,t]*tau_u
    Lvvt[:,:,t] = vtL[:,:,t]*tau_v

Luu = np.mean(Luut,axis=2)
Lvv = np.mean(Lvvt,axis=2)

print('Calculating velocity PDFs')
bins = 80
uPDFs = np.zeros((XPixels,YPixels,bins))
vPDFs = np.zeros_like(uPDFs)
uPDFranges = np.zeros((XPixels,YPixels,bins+1))
vPDFranges = np.zeros_like(uPDFranges)
for x in range(XPixels):
    for y in range(YPixels):
        uPDFs[x,y],uPDFranges[x,y]= np.histogram(U[x,y],bins=bins,density=True)
        vPDFs[x,y],vPDFranges[x,y]= np.histogram(V[x,y],bins=bins,density=True)


#******************************************************************************
# -- Write VTK file for ParaView
#******************************************************************************

vtk = VTKWriter()

vtk.addScalar(K,'k')
vtk.addVector(Utmean,Vtmean,'Umean')
vtk.write('PIV_3.vtk')


#******************************************************************************
# -- Plots
#******************************************************************************

## -- Plot the Reynolds Stress components
plt.figure(1)
plt.imshow(uu,interpolation='bicubic')
plt.clim(0.3,2.9)
plt.colorbar()
plt.title('$\overline{uu}$',fontsize=25)

plt.figure(2)
plt.imshow(vv,interpolation='bicubic')
plt.clim(0.4,4.5)
plt.colorbar()
plt.title('$\overline{vv}$',fontsize=25)

plt.figure(3)
plt.imshow(uv,interpolation='bicubic')
plt.clim(-0.6,2.3)
plt.colorbar()
plt.title('$\overline{uv}$',fontsize=25)

plt.figure(4)
plt.imshow(K,interpolation='bicubic')
plt.clim(0.5,4.8)
plt.colorbar()
plt.title('$k$',fontsize=25)

plt.figure(59)
#plt.plot(uPDFranges[20,29,1:],uPDFs[20,29],'k',label='u [20,29]')
#plt.plot(uPDFranges[42,2,1:],uPDFs[42,2],'k--',label='u [42,2]')
#plt.plot(vPDFranges[20,29,1:],vPDFs[20,29],'r',label='v [20,29]')
#plt.plot(vPDFranges[42,2,1:],vPDFs[42,2],'r--',label='v [42,2]')
plt.plot(uPDFranges[5,8,1:],uPDFs[5,8],'k',label='u [5,8]')
plt.plot(vPDFranges[5,8,1:],vPDFs[5,8],'k--',label='v [5,8]')
plt.legend()
plt.title('$Velocity \; PDFs$', fontsize=20)

## -- Generate -5/3 line to plot with PSDs
xs = np.array([3e5,freq.max()])
ys = 3e7*xs**(-5./3)

## -- Turbulent Energy Spectrum (space filtering)
plt.figure(6)
#plt.plot(freq,uPSDs[20,10],'k',label='X')
#plt.plot(freq,vPSDs[20,10],'r',label='Y')
plt.plot(freq,PSDs[20,10],'k',label='mean')
plt.plot(xs,ys,'r--')
plt.xscale('log')
plt.yscale('log')
#plt.legend()
plt.title('Power Spectra')
plt.xlabel('$\omega \;[Hz]$')
plt.ylabel('$E\;(\omega)$',rotation='horizontal')

#plt.figure()
#plt.imshow(VtFFT[:,:,0],cmap='jet_r')
#plt.colorbar()
#plt.tick_params(labelbottom='off',bottom='off',top='off',labelleft='off',
#                left='off',right='off')
#plt.clim(VtFFT[:,:,0].min(),VtFFT[:,:,0].max())

## -- Save instantaneous fields and create video
#generate_video(V,'Vres',erase='n')
#generate_video(U,'Ures',erase='n')
#generate_video(UtFFT,'UtFFTres',erase='n')
#generate_video(EE,'Eres',erase='n',cmap='jet')

## -- Print/save time averaged of vel components
#plt.figure()
#plt.imshow(U[:,:,0],cmap='jet_r')
#plt.colorbar()
#plt.title('U mean [m/s]')
##plt.savefig('Umean.png')

#plt.figure()
#plt.imshow(UspaceL[:,:,-1],cmap='jet_r')
#plt.colorbar()
#plt.title('U mean [m/s]')
##plt.savefig('Vmean.png')


## -- Plot space filters
#plt.figure()
#plt.imshow(Gauss3[:,:,0],cmap='gray',interpolation='bicubic')


plt.show()

print('End')