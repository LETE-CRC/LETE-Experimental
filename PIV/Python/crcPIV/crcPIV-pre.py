#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
                      Python code for PIV analysis
   Created by Combustion Research Center CRC at LETE - Sao Paulo, Brasil
   Laboratory of Environmental and Thermal Engineering - LETE
   Escola Politecnica da USP - EPUSP

===============================================================================
version:2.1 - 12/2020: Helio Villanueva
"""

from pathlib import Path
home = str(Path.home())
import sys
installPath = home + '/Desktop/LETE-Experimental/PIV/Python/crcPIV/classes'
sys.path.append(installPath)

from ReadData import ReadData
#from Seeding import SiO2
from Turbulence import Turb
#from Hotwire import Hotwire
#from VisualPost import Plots
##from WriteVTK import WVTK
import outFuncs


#==============================================================================
outFuncs.header()
#==============================================================================


#******************************************************************************
## -- Seeding tracers
#******************************************************************************
outFuncs.proc('Seeding tracers')


#******************************************************************************
## -- Read PIV Raw files
#******************************************************************************
outFuncs.proc('Read PIV Raw files')

## -- Path to the PIV velocity results files
velPath = 'PATH'

## -- Instance of class with PIV velocity results infos
velRaw = ReadData(velPath)

## -- Read PIV data. If python format already present read it instead
u,v,uncR = velRaw.readVarTimeSeries(['U[m/s]','V[m/s]','UncR(m/s)[m/s]'])


### -- Print infos about coordinates and size of Field-of-View (FOV)
velRaw.printCoordTimeInfos()


#******************************************************************************
## -- Processamento de sinais
#******************************************************************************
outFuncs.proc('Proc sinais')


#******************************************************************************
## -- Turbulence calculations (velocity mean and magnitude)
#******************************************************************************
outFuncs.proc('Turbulence calculations')

turb = Turb(velRaw,u,v)

K = turb.calcK2DPIV()


#******************************************************************************
## -- Uncertainty calculations
#******************************************************************************
outFuncs.proc('Uncertainty calculations')

UncR = turb.calcUncMean(uncR)
Uuu = turb.calcUncRe(uncR,turb.uu)
Uvv = turb.calcUncRe(uncR,turb.vv)


#******************************************************************************
## -- Save reduced results 
#******************************************************************************
outFuncs.proc('Save reduced results')

velRaw.saveReduced([turb.U,turb.V,turb.uu,turb.vv])

#******************************************************************************
## -- Save result in VTK format
#******************************************************************************
outFuncs.proc('Save result in VTK format')


#******************************************************************************
## -- External data results
#******************************************************************************
outFuncs.proc('External Data')


#******************************************************************************
## -- Plots
#******************************************************************************
outFuncs.proc('Plots')


#==============================================================================
outFuncs.end()
#==============================================================================
