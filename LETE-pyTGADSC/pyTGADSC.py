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

name = 'Aguape_10Cmin_n2'
fdr = '10Cmin_n2/'
exps = {'Aguape1':{'TGA':fdr+'TGAI.txt',
                   'DSC':fdr+'DSCI.txt'},
        'Aguape2':{'TGA':fdr+'TGAII.txt',
                   'DSC':fdr+'DSCII.txt'},
        'Aguape3':{'TGA':fdr+'TGAIII.txt',
                   'DSC':fdr+'DSCIII.txt'}}

# limites otimizador
# nk│Ak│Ek│nO2_k
xlim = {'bd':{'low':[1,1e4,30],
              'up':[1.5,1.1e4,50]},
        'bp':{'low':[2,1e5,90],
              'up':[3,1e7,100]},
        'bo':{'low':[1,1e2,90,0.5],
              'up':[2.0,1e7,110,1.1]},
        'alpha':{'low':[0.9,8e12,180,0.5],
                 'up':[1.1,1e13,200,2.0]},
        'beta':{'low':[0.95,1e2,165,0.9],
                'up':[1.1,1e11,200,1.1]},
        'ph2o':{'low':[0.01],
                'up':[0.09]},
        'ddts':{'low':[0.15,0.45,0.2,0.15],
                'up':[0.4,0.7,0.4,0.7]},
        'h_i':{'low':[1e6,1e4,1e2,1e5,1e6],
               'up':[4e6,1e7,1.5e7,1e7,1e8]}}

# ******************************************************************************
# -- MAIN --
# ******************************************************************************

teste = Ensaio(exps,name)
teste.optimize(xlim)
# err = teste.calcMass(x)
# print(teste.optX)
teste.plotTGAOPTTemp(teste.TGA,plotSpecies=True)
teste.plotDSCOPTTemp(teste.DSC)
# teste.plotTime(teste.TGA)
# teste.plotTemp(teste.TGA)
# teste.plotTime(teste.DSC)
# teste.plotTemp(teste.DSC)

plt.show()
# ******************************************************************************
