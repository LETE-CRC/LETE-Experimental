#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
                      Python code for SCRE analysis
   Created by Combustion Research Center CRC at LETE - Sao Paulo, Brasil
   Laboratory of Environmental and Thermal Engineering - LETE
   Escola Politecnica da USP - EPUSP
===============================================================================
version:0.0 - 04/2022: Helio Villanueva
version:0.1 - 05/2022: Helio Villanueva
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import glob
from tqdm import tqdm

# ******************************************************************************
# -- USER
# ******************************************************************************

path = "DF01-NL"
imFmt = 'tif'
rCAD = 1  # ratio CAD/Time step (1, 2...)
init0 = 0  # number of initial images discarted (eg: 360)


# ******************************************************************************
# -- MAIN
# ******************************************************************************

header = '\n' + 70*"=" + '\n' + '\t\tPython code for SCRE analysis\n'
header += 'Created by Combustion Research Center CRC at LETE - Sao Paulo, Brasil\n'
header += 'Laboratory of Environmental and Thermal Engineering - LETE\n'
header += 'Escola Politecnica da USP - EPUSP\n'
header += 70*"=" + '\n'
print(header)

imgNames = glob.glob(path + '/*[0-9].' + imFmt)  # List images in Dir 'path'
imgNames.sort()
imgNames = imgNames[init0:]  # discard initial images

img = image.imread(imgNames[0])  # read single img for general infos
lins = img.shape[0]  # y coord
cols = img.shape[1]  # x coord
stepsOrig = len(imgNames)  # tot of all imgs saved by the camera
CADs = int(719 / rCAD)  # ratio steps / cycle
cycles = int(stepsOrig / CADs)  # total steps / cycle

steps = cycles * CADs
imgNames = imgNames[:steps]
limgNames = np.array(imgNames).reshape(cycles, CADs)

print('General Infos')
print(14*'-', '\nImage res: ', lins, 'x', cols)
print('CADs/cycle: ', CADs)
print('Cycles: ', cycles)

# -- Background image for removal process
# ******************************************************************************
imsCy = np.zeros((lins, cols, cycles))
for cy in range(cycles):  # loop over cycles
    #print("Cycle No: ", cy)
    imsCy[:, :, cy] = image.imread(limgNames[cy, 0])

imBackground = np.mean(imsCy, 2)

# -- For loops for each cycle and CAD
# ******************************************************************************
imsCycleMean = np.zeros((lins, cols, CADs))
imsCycleStdDev = np.zeros((lins, cols, CADs))

for t in tqdm(range(CADs), desc="CAD calculations: "):
    imsCy = np.zeros((lins, cols, cycles))
    for cy in range(cycles):  # loop over cycles
        imsCy[:, :, cy] = image.imread(limgNames[cy, t]) - imBackground
    # hole cycle calculation
    imM = np.mean(imsCy, 2, keepdims=True)
    imsCycleMean[:, :, t] = imM[:, :, 0]
    imsCyFluct = np.sqrt((imsCy[:, :, :] - imM)**2)
    imsCycleStdDev[:, :, t] = np.mean(imsCyFluct, 2)

# ******************************************************************************

print('END Calculations\nPlots:')

plt.figure()
plt.imshow(imsCycleMean[:,:,10], cmap='hot')
plt.colorbar()
plt.show()

################################################################################
'''
TODOS:
- cortar imagens pra ajustar bordas
- if a prova de numero errado no rCAD
'''