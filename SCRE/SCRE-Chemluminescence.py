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
version:0.2 - 01/2023: Helio Villanueva
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import glob
from tqdm import tqdm
import os

# ******************************************************************************
# -- USER
# ******************************************************************************

paths = ["DF01-NL"]

inputs = {'imFmt':'tif',
          'rMotor':2000,  # rpm rotacao do motor
          'fCAM':24000,  # Hz taxa aquisicao cameras
          'init0':0,  # number of initial images discarted (eg 360)
          'durationCADs':150,  # imgs with combustion to save
          'saveImgs':True  # save mean and stdDev images for each CAD
          }

# *****************************************************************************
# -- FUNCOES
# *****************************************************************************


class Caso():
    '''Classe que organiza o caso
    '''

    def __init__(self,path, inputs):
        self.path = path
        self.inputs = inputs
        self.lins, self.cols, self.CADs, self.cycles, self.limgNames, self.infos = self.baseInfos()

    def baseInfos(self):
        '''Basic informations
        '''
        # List images in Dir 'path'
        imgNames = glob.glob(self.path + '/*[0-9].' + self.inputs['imFmt'])
        imgNames.sort()
        imgNames = imgNames[self.inputs['init0']:]  # discard initial images

        img = image.imread(imgNames[0])  # read single img for general infos
        lins = img.shape[0]  # y coord
        cols = img.shape[1]  # x coord
        stepsOrig = len(imgNames)  # tot of all imgs saved by the camera
        singleCycle = 2*360  # deg
        rCAD = (singleCycle*self.inputs['rMotor']/60)/self.inputs['fCAM']  # CAD / image
        CADs = int(719 / rCAD)  # CADs observed in each cycle
        cycles = int(stepsOrig / CADs)  # total steps / cycle

        steps = cycles * CADs
        imgNames = imgNames[:steps]
        limgNames = np.array(imgNames).reshape(cycles, CADs)

        infos = 'General Infos\n'
        infos += 14*'-' + '\nImage res: %.1f x %.1f\n' %(lins, cols)
        infos += 'CADs/image: %.2f\n' %rCAD
        infos += 'CADs/cycle: %.1f\n' %CADs
        infos += 'Cycles: %.1f\n' %cycles
        infos += 14*'-'

        return lins, cols, CADs, cycles, limgNames, infos

    def imgBackground(self):
        '''Background image for removal process
        '''
        imsCy = np.zeros((self.lins, self.cols, self.cycles))
        for cy in range(self.cycles):  # loop over cycles
            # print("Cycle No: ", cy)
            imsCy[:, :, cy] = image.imread(self.limgNames[cy, 0])

        return np.mean(imsCy, 2)

    def imgProcess(self):
        '''For loops for each cycle and CAD
        '''
        imgBackground = self.imgBackground()

        try:
            print('Trying to read npy files')
            imsCycleMean = np.load(self.path + '/imsCycleMean.npy')
            imsCycleStdDev = np.load(self.path + '/imsCycleStdDev.npy')
            print('done')
        except Exception:
            print('Reading raw img files')
            imsCycleMean = np.zeros((self.lins, self.cols, self.CADs))
            imsCycleStdDev = np.zeros((self.lins, self.cols, self.CADs))

            for t in tqdm(range(self.CADs), desc="CAD calculations: "):
                if t > self.inputs['durationCADs']:
                    break
                imsCy = np.zeros((self.lins, self.cols, self.cycles))
                for cy in range(self.cycles):  # loop over cycles
                    imsCy[:, :, cy] = image.imread(self.limgNames[cy, t])
                    imsCy[:, :, cy] -= imgBackground

                # hole cycle calculation
                imsCy[imsCy<0] = 0  # No negative values after background removal
                imM = np.mean(imsCy, 2, keepdims=True)
                imsCycleMean[:, :, t] = imM[:, :, 0]
                imsCyFluct = np.sqrt((imsCy[:, :, :] - imM)**2)
                imsCycleStdDev[:, :, t] = np.mean(imsCyFluct, 2)
            print('Saving .npy arrays')
            np.save(self.path + '/imsCycleMean', imsCycleMean)
            np.save(self.path + '/imsCycleStdDev', imsCycleStdDev)

        return imsCycleMean, imsCycleStdDev

    def Plots(self,imsCycleMean,imsCycleStdDev):
        '''
        '''
        # min max for plots
        vMeanMin = imsCycleMean.min()
        vMeanMax = imsCycleMean.max()
        vStdDevMin = imsCycleStdDev.min()
        vStdDevMax = imsCycleStdDev.max()

        print('Saving images')
        if not os.path.exists(self.path + '/CADmean'):
            os.makedirs(self.path + '/CADmean')

        if not os.path.exists(self.path + '/CADstdDev'):
            os.makedirs(self.path + '/CADstdDev')

        for t in tqdm(range(self.CADs), desc="Saving CAD imgs: "):
            if t > self.inputs['durationCADs']:
                break
            # Mean
            plt.figure()
            plt.imshow(imsCycleMean[:, :, t], cmap='hot', vmin=vMeanMin, vmax=vMeanMax)
            plt.title('mean CAD %3d' %t)
            plt.colorbar()
            figNameMean = self.path + '/CADmean/CADmean%04d' %t + '.png'
            plt.savefig(figNameMean)
            plt.close()
            # StdDev
            plt.figure()
            plt.imshow(imsCycleStdDev[:, :, t], cmap='hot', vmin=vStdDevMin, vmax=vStdDevMax)
            plt.title('stdDev CAD %3d' %t)
            plt.colorbar()
            figNameStdDev = self.path + '/CADstdDev/CADstdDev%04d' %t + '.png'
            plt.savefig(figNameStdDev)
            plt.close()
        return 0

################################################################################
'''
TODOS:
- cortar imagens pra ajustar bordas
- if a prova de numero errado no rCAD
'''

# *****************************************************************************
# -- MAIN
# *****************************************************************************


def main():
    '''Funcao main caso arquivo seja disparado via terminal
    '''
    header = '\n' + 70*"=" + '\n' + '\t\tPython code for SCRE analysis\n'
    header += 'Created by Combustion Research Center CRC at LETE - Sao Paulo, Brasil\n'
    header += 'Laboratory of Environmental and Thermal Engineering - LETE\n'
    header += 'Escola Politecnica da USP - EPUSP\n'
    header += 70*"=" + '\n'
    print(header)

    for path in paths:
        print('Processando caso %s' %path)
        case = Caso(path,inputs)
        imgMean, imgDev = case.imgProcess()
        case.Plots(imgMean,imgDev)
    return 0


if __name__ == "__main__":
    # execute only if run as a script
    main()
