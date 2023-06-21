#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
                      Python code for TGA/DSC analysis
   Created by Combustion Research Center CRC at LETE - Sao Paulo, Brasil
   Laboratory of Environmental and Thermal Engineering - LETE
   Escola Politecnica da USP - EPUSP
===============================================================================
version:0.0 - 03/2023: Helio Villanueva
"""
import sys
import matplotlib.pyplot as plt
from pathlib import Path
home = str(Path.home())
installdir = home + '/Documents/TGADSC/LETE-pyTGADSC/'
sys.path.append(installdir)

from pyTGADSCClasses import Ensaio

# ******************************************************************************
# -- USER
# ******************************************************************************

name = 'Junco_10Cmin_n2'
fdr = '10Cmin_n2/'
exps = {'Junco1':{'TGA':fdr+'TGAI.txt',
                  'DSC':fdr+'DSCI.txt'},
        'Junco2':{'TGA':fdr+'TGAII.txt',
                  'DSC':fdr+'DSCII.txt'},
        'Junco3':{'TGA':fdr+'TGAIII.txt',
                  'DSC':fdr+'DSCIII.txt'}}

# limites otimizador
xlim = {'agua':{'low':[1.5,1e7,60],
                'up':[3.0,1e8,65]},
        'C':{'low':[2.5,1e4,85],
             'up':[3.0,1e7,110]},
        'Cox':{'low':[1.5,1e6,80,0.5],
               'up':[3.0,3e6,110,1.1]},
        'alpha':{'low':[0.9,1e11,110,0.5],
                 'up':[2.5,1e12,200,1.0]},
        'beta':{'low':[0.9,1e2,100,0.9],
                'up':[3.5,1e11,250,2.1]},
        'ph2o':{'low':[0.06],
                'up':[0.08]},
        'ddts':{'low':[0.1,0.1,0.1,0.1],
                'up':[1.0,1.0,0.4,0.4]}}

# ******************************************************************************
# -- MAIN --
# ******************************************************************************

teste = Ensaio(exps,name)
# teste.optimize()
# err = teste.calcMass(x)
# print(teste.optX)
teste.plotOPTTemp(teste.TGA)
# teste.plotTime(teste.TGA)
# teste.plotTemp(teste.TGA)
# teste.plotTime(teste.DSC)
# teste.plotTemp(teste.DSC)

plt.show()
# ******************************************************************************
