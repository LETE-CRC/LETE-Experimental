#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
                      Python code for PIV analysis
   Created by Combustion Research Center CRC at LETE - Sao Paulo, Brasil
   Laboratory of Environmental and Thermal Engineering - LETE
   Escola Politecnica da USP - EPUSP
===============================================================================
version:0.0 - 04/2022: Helio Villanueva
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import glob

# ******************************************************************************
# -- USER
# ******************************************************************************

path = "DF01-NL"
rCAD = 1  # ratio CAD/Time step (1, 2...)
init0 = 0  # initial images discarted (360)

# ******************************************************************************
# -- MAIN
# ******************************************************************************

imgNames = glob.glob(path + '/*[0-9].*')  # List images in Dir 'path'
imgNames.sort()
imgNames = imgNames[init0:]  # discard initial images

img = image.imread(imgNames[0])  # read single img for general infos
lins = img.shape[0]  # y coord
cols = img.shape[1]  # x coord
stepsOrig = len(imgNames)  # tot of all imgs saved by the camera
rCycleStep = int(720 / rCAD)  # ratio steps / cycle
tStepsCycle = int(stepsOrig / rCycleStep)  # total steps / cycle

steps = tStepsCycle * rCycleStep
imgNames = imgNames[:steps]

limgNames = np.array(imgNames).reshape(tStepsCycle, rCycleStep)

# -- For loops for each cycle and timestep
# ******************************************************************************
for cy in range(tStepsCycle):  # loop over cycles
    print("Cycle No: ", cy)
    # process things for the hole cycle steps
    for stp, name in enumerate(limgNames[cy, :]):
        print("File: ", name)
        # process each timestep things
        imgI = image.imread(name)
        imgAvg = np.average(imgI)
        print(imgAvg)
# ******************************************************************************

plt.figure()
plt.imshow(img, cmap='hot')
plt.colorbar()
plt.show()
