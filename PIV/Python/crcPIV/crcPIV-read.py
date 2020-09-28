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

#from scipy import stats
#import numpy as np

from pathlib import Path
home = str(Path.home())
import sys
installPath = home + '/Desktop/LETE-Experimental/PIV/Python/crcPIV/classes'
sys.path.append(installPath)

from ReadData import ReadData
#from Seeding import SiO2
#from Turbulence import Turb
#from Hotwire import Hotwire
#from VisualPost import Plots, plt
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
#velPath = home + '/Desktop/PIV/flameless/res'
velPath = '/run/media/bandeiranegra/Docs/PIV/flameless/res'

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
## -- Read CFD data in csv from plotoverline paraview
#******************************************************************************
outFuncs.proc('Read CFD data in csv from plotoverline paraview')


#******************************************************************************
## -- Seeding tracers
#******************************************************************************
outFuncs.proc('Seeding tracers')


#******************************************************************************
## -- Turbulence calculations (velocity mean and magnitude)
#******************************************************************************
outFuncs.proc('Turbulence calculations')


#******************************************************************************
## -- Uncertainty calculations
#******************************************************************************
outFuncs.proc('Uncertainty calculations')


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
