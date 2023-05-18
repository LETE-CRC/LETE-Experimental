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

import pyTGADSCClasses as lete
import matplotlib.pyplot as plt

# ******************************************************************************
# -- USER
# ******************************************************************************

exps = {'Junco1':{'TGA':'Junco1_10_tg_ar.txt',
                  'DSC':'Junco1_10_ar_dsc.txt'}}#,
        #'Junco2':{'TGA':'Junco2_10_tg_ar.txt',
        #          'DSC':'Junco2_10_ar_dsc.txt'},
        #'Junco3':{'TGA':'Junco2_10_tg_ar.txt',
        #          'DSC':'Junco3_10_ar_dsc.txt'}}

X = [357307174.878322,66.4312552130049,2.09845480010349,22232.3684508937,85.9230235134947,2.10288331374131,2435421.32901346,95.3722793108986,2.66984533482758,12498015987.8924,169.145292265764,1.12643965353996,10364345760.0656,149.477012577210,1.54249160331694,0.462174913628938,0.688519370305353,0.352561526060474,0.362212168002418,214305.665728726,4536549.56089111,5240945.11633852,7644134.03028114,2635902.07304109,0.0769614335106816,0.672778735855666,0.606776603765679,1.61399141733446]
lb = [3e8,65,1.8,20000,80,2.0,2e6,94,2.5,12e9,160,1.0,1e10,140,1.5,0.23,0.22,0.10,0.12,1.8e5,4e6,5e6,6e6,2.5e6,0.07,0.5,0.5,1.2]
ub = [4e8,69,3.0,25000,90,2.3,3e6,98,2.9,13e9,180,1.3,1.1e10,155,1.7,0.47,0.69,0.36,0.39,3e5,5e6,7e6,8e6,2.9e6,0.1,1.0,1.0,2.0]
# ******************************************************************************
# -- MAIN --
# ******************************************************************************

# Loop entre todos os arquivos de resultados de DSC
for exp in exps:
    teste = lete.Ensaio(exps[exp]['TGA'],exps[exp]['DSC'])
    teste.optimize()
    # err = teste.calcMass(x)
    # print(teste.optX)
    teste.plotOPTTemp(teste.TGA)
    # teste.plotTime(teste.TGA)
    # teste.plotTemp(teste.TGA)
    # teste.plotTime(teste.DSC)
    # teste.plotTemp(teste.DSC)

plt.show()
# ******************************************************************************
