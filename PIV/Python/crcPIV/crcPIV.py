#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
                      Python code for PIV analysis
   Created by Combustion Research Center CRC at LETE - Sao Paulo, Brasil
   Laboratory of Environmental and Thermal Engineering - LETE
   Escola Politecnica da USP - EPUSP

===============================================================================
version:0.0 - 02/2019: Helio Villanueva
version:1.0 - 04/2019: Helio Villanueva
version:1.1 - 08/2019: Helio Villanueva
version:2.0 - 05/2020: Helio Villanueva
"""

#from scipy import signal
import numpy as np

from pathlib import Path
home = str(Path.home())
import sys
installPath = home + '/Desktop/LETE-Experimental/PIV/Python/crcPIV/classes'
sys.path.append(installPath)

from ReadData import ReadData
from Seeding import SiO2
from Turbulence import Turb
from VisualPost import Plots, plt
##from WriteVTK import WVTK
import outFuncs

#==============================================================================
outFuncs.header()
#==============================================================================


#******************************************************************************
## -- Read PIV Raw files
#******************************************************************************
outFuncs.proc('Read PIV Raw files')

## -- Path to the PIV velocity results files
velPath = home + '/Desktop/PIV/flameless/res'

## -- Instance of class with PIV velocity results infos
velRaw = ReadData(velPath)

## -- Read PIV data. If python format already present read it instead
u,v = velRaw.read2VarTimeSeries('U[m/s]','V[m/s]')
uncR = velRaw.read1VarTimeSeries('UncR(m/s)[m/s]')


### -- Print infos about coordinates and size of Field-of-View (FOV)
velRaw.printCoordTimeInfos()


#******************************************************************************
## -- Read CFD data in csv from plotoverline paraview
#******************************************************************************
outFuncs.proc('Read CFD data in csv from plotoverline paraview')

#CFD = np.genfromtxt('/home/helio/Desktop/Res0/CFD/ParaView-RSM_BSL/y013.csv',
#                    skip_header=1,delimiter=',')

#CFD_x, CFD_velMag, CFD_epsilon = CFD[:,14], CFD[:,0], CFD[:,4]
#CFD_U, CFD_V = CFD[:,2], CFD[:,1]
#CFD_uu, CFD_vv, CFD_uv = CFD[:,8], CFD[:,7], CFD[:,10]


#******************************************************************************
## -- Seeding tracers
#******************************************************************************
outFuncs.proc('Seeding tracers')

tracer = SiO2()
##tracer.graphResponse(ymin=0.8)


#******************************************************************************
## -- Turbulence calculations (velocity mean and magnitude)
#******************************************************************************
outFuncs.proc('Turbulence calculations')

turb = Turb(velRaw,u,v)

#gradUx, gradVx, gradUy, gradVy = turb.calcVelGrad()

#tau11, tau22, tau12 = turb.calcTauijSmagorinsky()
#tau11, tau22, tau12 = turb.calcTauijGradient()

epsilongrad = turb.calcEpsilon('gradient')
epsilonsmagorinsky = turb.calcEpsilon('smagorinsky')

#K = turb.calcK2DPIV()

#FFTu = np.fft.rfft(turb.u[50,30,:])
#frequ = np.fft.rfftfreq(velRaw.Ttot,velRaw.dt)

#plt.figure()
#plt.plot(frequ,FFTu)


#******************************************************************************
## -- Uncertainty calculations
#******************************************************************************
outFuncs.proc('Uncertainty calculations')

uncUMean = np.mean(uncR,axis=2, keepdims=True)

##varMag = turb.uu**2 + turb.vv**2
##uncUMeanSq = np.mean(uncR**2,axis=2, keepdims=True)
##uncSigma = varMag + uncUMeanSq
##uncMeanVel = np.sqrt(uncSigma/velRaw.Ttot)
##uncRuu = turb.uu*np.sqrt(2/velRaw.Ttot)
##uncRvv = turb.vv*np.sqrt(2/velRaw.Ttot)
##uncRuv = turb.uv*np.sqrt(2/velRaw.Ttot)


#******************************************************************************
## -- Save result in VTK format
#******************************************************************************
outFuncs.proc('Save result in VTK format')

## - instantaneous
##velVTKres = WVTK(velPath)
##velVTKres.save2DcellVecTransientVTK(raw.resPath,'U',raw.U,raw.V)
#
## - mean
##VelMeanVTKres = WVTK(velPath)
##VelMeanVTKres.save2DcellVecVTK(raw.resPath,'<U>',turb.U,turb.V)
#
##ReTensorVTKres = WVTK(velPath)
##ReTensorVTKres.save2DcellReynoldsVTK(raw.resPath,'ReStress',
##                                     turb.uu,turb.vv,turb.uv)


#******************************************************************************
## -- Plots
#******************************************************************************
outFuncs.proc('Plots')

plts = Plots(velRaw)

# - Plot singleFramePlots
plts.singleFramePlot(turb.magVel,
                     r'$\overline{U}$ $[m/s]$',
                     t=0, grid=0, title='Non reactive Flameless', vlim=[0,120],
                     save=home+'/Desktop/flss-magU2.png')


plt.show()

#==============================================================================
outFuncs.end()
#==============================================================================
