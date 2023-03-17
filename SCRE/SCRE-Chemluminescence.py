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
version:0.3 - 02/2023: Helio Villanueva
version:0.4 - 03/2023: Helio Villanueva
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from scipy import signal
import glob
from tqdm import tqdm
import os

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.size": 14
})
plt.rcParams['xtick.top'] = plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.minor.top'] = plt.rcParams['ytick.minor.right'] = True

# ******************************************************************************
# -- USER
# ******************************************************************************

paths = ["inferior_DF02P01_221216"]

inputs = {'imFmt':'jpg',
          'rMotor':2000,  # rpm rotacao do motor
          'fCAM':24000,  # Hz taxa aquisicao cameras
          'init0':0,  # number of initial images discarted (eg 360)
          'durationCADs':150,  # imgs with combustion to save
          'combCycles':21,  # numero de ciclos com combustao
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
        self.lins, self.cols, self.rCAD, self.CADs, self.cycles, self.combCycles, self.limgNames, self.infos = self.baseInfos()

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
        infos += 'N tot images: %.2f\n' %stepsOrig
        infos += 'CADs/image: %.2f\n' %rCAD
        infos += 'CADs/cycle: %.1f\n' %(CADs+1)
        infos += 'Cycles: %.1f\n' %cycles
        infos += 'Cycles w/ combustion: %.1f\n' %self.inputs['combCycles']
        infos += 14*'-'

        return lins, cols, rCAD, CADs, cycles, self.inputs['combCycles'], limgNames, infos

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
                imsCy = np.zeros((self.lins, self.cols, self.combCycles))
                for cy in range(self.combCycles):  # loop over cycles w/ comb
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

    def calcDerivada(self,imsCycleMean):
        print('Processing time derivative')
        # central differencing 2nd order
        # central differencing 4th order
        # - Fourth order CDS scheme from Ferziger Computational methods for
        # - fluid dynamics on page 44 eq 3.14
        scheme = np.array([-1,8,0,-8,1]).reshape(1,1,5)
        den = 12
        num = signal.convolve(imsCycleMean,scheme, mode='same')
        dt = 1/self.inputs['fCAM']
        ddt = num/(den*dt)

        return ddt

    def calcFlameArea(self,imsCycleMean):
        print('Processing Flame area')
        # binarization
        mask = imsCycleMean > 50
        flameArea = np.sum(mask,axis=(0,1))
        return flameArea,mask

    def Plots(self,imsCycleMean,imsCycleStdDev):
        '''
        '''
        # min max for plots
        vMeanMin = imsCycleMean.min()
        vMeanMax = imsCycleMean.max()
        vStdDevMin = imsCycleStdDev.min()
        vStdDevMax = imsCycleStdDev.max()

        # print('Saving images')
        if not os.path.exists(self.path + '/CADmean'):
            os.makedirs(self.path + '/CADmean')

        if not os.path.exists(self.path + '/CADstdDev'):
            os.makedirs(self.path + '/CADstdDev')

        for t in tqdm(range(self.CADs), desc="Saving mean/stdDev CAD imgs: "):
            if t > self.inputs['durationCADs']:
                break
            cad = t*self.rCAD
            # Mean
            plt.figure(tight_layout=True)
            plt.imshow(imsCycleMean[:, :, t], cmap='hot', vmin=vMeanMin, vmax=vMeanMax)
            plt.title('mean CAD %3d' %cad)
            plt.colorbar(label='Intensidade luminosa I')
            figNameMean = self.path + '/CADmean/CADmean%04d' %t + '.png'
            plt.savefig(figNameMean)
            plt.close()
            # StdDev
            plt.figure(tight_layout=True)
            plt.imshow(imsCycleStdDev[:, :, t], cmap='hot', vmin=vStdDevMin, vmax=vStdDevMax)
            plt.title('stdDev CAD %3d' %cad)
            plt.colorbar(label='Intensidade luminosa I')
            figNameStdDev = self.path + '/CADstdDev/CADstdDev%04d' %t + '.png'
            plt.savefig(figNameStdDev)
            plt.close()

        return 0

    def PlotDerivada(self,imsCycleddt):
        '''
        '''
        # min max for plots
        vddtMin = imsCycleddt.min()
        vddtMax = imsCycleddt.max()

        # print('Saving images')

        if not os.path.exists(self.path + '/CADddt'):
            os.makedirs(self.path + '/CADddt')

        for t in tqdm(range(self.CADs), desc="Saving ddt CAD imgs: "):
            if t > self.inputs['durationCADs']:
                break
            cad = t*self.rCAD
            # Ddt
            plt.figure(tight_layout=True)
            plt.imshow(imsCycleddt[:, :, t], cmap='jet', vmin=vddtMin, vmax=vddtMax)
            plt.title('ddt CAD %3d' %cad)
            plt.colorbar()
            figNameddt = self.path + '/CADddt/CADddt%04d' %t + '.png'
            plt.savefig(figNameddt)
            plt.close()

        return 0

    def PlotFlameArea(self, flameArea, flameAreaImg):
        '''
        '''
        # min max for plots
        fAMin = flameAreaImg.min()
        fAMax = flameAreaImg.max()

        # print('Saving images')

        if not os.path.exists(self.path + '/CADflameArea'):
            os.makedirs(self.path + '/CADflameArea')

        for t in tqdm(range(self.CADs), desc="Saving flame Area CAD imgs: "):
            if t > self.inputs['durationCADs']:
                break
            cad = t*self.rCAD
            # Flame Area
            fig = plt.figure(tight_layout=True)
            gs = gridspec.GridSpec(1, 1)
            ax = fig.add_subplot(gs[0, 0])
            ax.imshow(flameAreaImg[:, :, t], cmap='gray', vmin=fAMin, vmax=fAMax)
            ax.set_title('Flame Area CAD %3d' %cad)
            bbox = dict(facecolor='w', alpha=0.7,boxstyle='Round')
            text = 'A = %0.2f $px^2$' %flameArea[t]
            ax.text(0.7,0.8,text,bbox=bbox,transform=ax.transAxes)
            # plt.colorbar()
            figNameFA = self.path + '/CADflameArea/CADflameArea%04d' %t + '.png'
            plt.savefig(figNameFA)
            plt.close()

        return 0

    def PlotFlameIntensity(self, flameArea):
        '''
        '''
        # min max for plots
        fIMin = flameArea.min()
        fIMax = flameArea.max()

        fig = plt.figure(tight_layout=True)
        gs = gridspec.GridSpec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(flameArea,'k')
        ax.set_ylim(fIMin,fIMax)
        ax.set_title('case: ')
        ax.set_xlabel('CAD')
        ax.set_ylabel('Flame Intensity')
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.yaxis.set_minor_locator(MultipleLocator(200))
        figNameFA = self.path + '/flameIntensity.png'
        plt.savefig(figNameFA)
        plt.close()

        return 0

################################################################################

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
        print('\nProcessando caso %s' %path)
        case = Caso(path,inputs)
        print(case.infos)
        imgMean, imgDev = case.imgProcess()
        imgddt = case.calcDerivada(imgMean)
        flameArea, flameAreaImg = case.calcFlameArea(imgMean)
        case.Plots(imgMean,imgDev)
        case.PlotDerivada(imgddt)
        case.PlotFlameArea(flameArea, flameAreaImg)
        case.PlotFlameIntensity(flameArea)
        print('Done\n')
    return 0


if __name__ == "__main__":
    # execute only if run as a script
    main()
