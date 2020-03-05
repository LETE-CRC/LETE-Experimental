#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 09:44:01 2020

@author: Helio
"""

import numpy as np
import scipy.signal as scp
import matplotlib.pyplot as plt


plt.rc('figure',figsize=[6.4,4.8])
plt.rc('figure',dpi=200)
plt.rc('lines',linewidth=2)
plt.rc('text',usetex=1)
plt.rc('font',family='sans-serif')
plt.rc('axes',linewidth=2,labelsize=18)
plt.rc('axes.spines',top=0,right=0)
plt.rc('xtick',labelsize='large')
plt.rc('xtick.major',size=5,width=2)
plt.rc('xtick.minor',visible=1)
plt.rc('ytick',labelsize='large')
plt.rc('ytick.major',size=5,width=2)
plt.rc('ytick.minor',visible=1)


## ----------------------------------------------------------------------------
## -------------------- Definicao de Classes
## ----------------------------------------------------------------------------

class oneDim:
    '''Classe para processar dados de anemometro de fio quente de 1D
    '''
    def __init__(self,casos):
        self.U = np.zeros(len(casos))
        self.UstdDev = np.zeros(len(casos))
        self.Uvar = np.zeros(len(casos))

        self.readStats(casos)

    def readStats(self,casos):
        '''Le os resultados de estatistica do anemometro
        Umean = 1/n * Sum(Ui)

        UstdDev = (1/(n-1) * Sum(Ui - Umean)^2)^0.5 -> rms

        Uvar = 1/(n-1) * Sum(Ui - Umean)^2 -> u'u'
        '''
        m=0
        for caso in casos:
            with open(caso,'r') as f:
                raw = f.readlines()

                for i,txt in enumerate(raw):
                    raw[i] = txt.replace(",",".")

            self.StatsHeader = raw[0].split()
            data = np.genfromtxt(raw, delimiter='\t',skip_header=1,
                                 skip_footer=1)

            self.U[m] = data[0]
            self.UstdDev[m] = data[2]
            self.Uvar[m] = data[3]

            m += 1

    def readPSD(self,casosPSD):
        '''Le os resultados de Power Spectral Density
        '''
        self.PSDs = []

        for casoPSD in casosPSD:
            with open(casoPSD,'r') as f:
                raw = f.readlines()

                for i,txt in enumerate(raw):
                    raw[i] = txt.replace(",",".")

            self.PSDs.append(np.genfromtxt(raw,delimiter='\t',skip_header=1,
                                           skip_footer=1))

    def calcTau(self):
        '''
        '''
        self.E0 = np.zeros(len(self.Uvar))

        for i,psd in enumerate(self.PSDs):
            self.E0[i] = psd[0,1]

        self.tau = np.pi/2 * self.E0/self.Uvar

class twoDim(oneDim):
    '''Classe para processar dados de anemometro de fio quente de 2D
    '''
    def __init__(self,casos):
        self.V = np.zeros(len(casos))
        self.VstdDev = np.zeros(len(casos))
        self.Vvar = np.zeros(len(casos))

        oneDim.__init__(self,casos)

    def readStats(self,casos):
        m=0
        for caso in casos:
            with open(caso,'r') as f:
                raw = f.readlines()

                for i,txt in enumerate(raw):
                    raw[i] = txt.replace(",",".")

            self.StatsHeader = raw[0].split()
            data = np.genfromtxt(raw, delimiter='\t',skip_header=1,
                                     skip_footer=1)

            self.U[m] = data[0,0]
            self.UstdDev[m] = data[0,2]
            self.Uvar[m] = data[0,3]
            self.V[m] = data[1,0]
            self.VstdDev[m] = data[1,2]
            self.Vvar[m] = data[1,3]

            m += 1


## ----------------------------------------------------------------------------
## -------------------- Definicao dos casos
## ----------------------------------------------------------------------------

# ---- Convencional
# Estatistica
casesC = ['21-02-2020-P11-convencional/z0r0-stats.txt',
         '21-02-2020-P11-convencional/z10r0-stats.txt',
         '21-02-2020-P11-convencional/z20r0-stats.txt',
         '21-02-2020-P11-convencional/z30r0-stats.txt',
         '21-02-2020-P11-convencional/z40r0-stats.txt',
         '21-02-2020-P11-convencional/z50r0-stats.txt',
         '21-02-2020-P11-convencional/z60r0-stats.txt',
         '21-02-2020-P11-convencional/z70r0-stats.txt',
         '21-02-2020-P11-convencional/z80r0-stats.txt',
         '21-02-2020-P11-convencional/z90r0-stats.txt',
         '21-02-2020-P11-convencional/z100r0-stats.txt',
         '21-02-2020-P11-convencional/z110r0-stats.txt',
         '21-02-2020-P11-convencional/z120r0-stats.txt',
         '21-02-2020-P11-convencional/z130r0-stats.txt',
         '21-02-2020-P11-convencional/z140r0-stats.txt',
         '21-02-2020-P11-convencional/z150r0-stats.txt',
         '21-02-2020-P11-convencional/z200r0-stats.txt',
         '21-02-2020-P11-convencional/z250r0-stats.txt',
         '21-02-2020-P11-convencional/z300r0-stats.txt'
         ]

# - PSDs
casesPSDsC = ['21-02-2020-P11-convencional/z0r0-psd.txt',
              '21-02-2020-P11-convencional/z10r0-psd.txt',
              '21-02-2020-P11-convencional/z20r0-psd.txt',
              '21-02-2020-P11-convencional/z30r0-psd.txt',
              '21-02-2020-P11-convencional/z40r0-psd.txt',
              '21-02-2020-P11-convencional/z50r0-psd.txt',
              '21-02-2020-P11-convencional/z60r0-psd.txt',
              '21-02-2020-P11-convencional/z70r0-psd.txt',
              '21-02-2020-P11-convencional/z80r0-psd.txt',
              '21-02-2020-P11-convencional/z90r0-psd.txt',
              '21-02-2020-P11-convencional/z100r0-psd.txt',
              '21-02-2020-P11-convencional/z110r0-psd.txt',
              '21-02-2020-P11-convencional/z120r0-psd.txt',
              '21-02-2020-P11-convencional/z130r0-psd.txt',
              '21-02-2020-P11-convencional/z140r0-psd.txt',
              '21-02-2020-P11-convencional/z150r0-psd.txt',
              '21-02-2020-P11-convencional/z200r0-psd.txt',
              '21-02-2020-P11-convencional/z250r0-psd.txt',
              '21-02-2020-P11-convencional/z300r0-psd.txt'
              ]

# - Posicoes realizadas as medicoes
zC = np.array([0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,200,250,300])

# ---- Transicao
# Estatistica
casesT = ['03-03-2020-P11-transition/z0r0-stats.txt',
         '03-03-2020-P11-transition/z10r0-stats.txt',
         '03-03-2020-P11-transition/z20r0-stats.txt',
         '03-03-2020-P11-transition/z30r0-stats.txt',
         '03-03-2020-P11-transition/z40r0-stats.txt',
         '03-03-2020-P11-transition/z50r0-stats.txt',
         '03-03-2020-P11-transition/z60r0-stats.txt',
         '03-03-2020-P11-transition/z70r0-stats.txt',
         '03-03-2020-P11-transition/z80r0-stats.txt',
         '03-03-2020-P11-transition/z90r0-stats.txt',
         '03-03-2020-P11-transition/z100r0-stats.txt',
         '03-03-2020-P11-transition/z110r0-stats.txt',
         '03-03-2020-P11-transition/z120r0-stats.txt',
         '03-03-2020-P11-transition/z130r0-stats.txt',
         '03-03-2020-P11-transition/z140r0-stats.txt',
         '03-03-2020-P11-transition/z150r0-stats.txt',
         '03-03-2020-P11-transition/z200r0-stats.txt',
         '03-03-2020-P11-transition/z250r0-stats.txt'
         ]

 # -PSDs
casesPSDsT = ['03-03-2020-P11-transition/z0r0-psd.txt',
         '03-03-2020-P11-transition/z10r0-psd.txt',
         '03-03-2020-P11-transition/z20r0-psd.txt',
         '03-03-2020-P11-transition/z30r0-psd.txt',
         '03-03-2020-P11-transition/z40r0-psd.txt',
         '03-03-2020-P11-transition/z50r0-psd.txt',
         '03-03-2020-P11-transition/z60r0-psd.txt',
         '03-03-2020-P11-transition/z70r0-psd.txt',
         '03-03-2020-P11-transition/z80r0-psd.txt',
         '03-03-2020-P11-transition/z90r0-psd.txt',
         '03-03-2020-P11-transition/z100r0-psd.txt',
         '03-03-2020-P11-transition/z110r0-psd.txt',
         '03-03-2020-P11-transition/z120r0-psd.txt',
         '03-03-2020-P11-transition/z130r0-psd.txt',
         '03-03-2020-P11-transition/z140r0-psd.txt',
         '03-03-2020-P11-transition/z150r0-psd.txt',
         '03-03-2020-P11-transition/z200r0-psd.txt',
         '03-03-2020-P11-transition/z250r0-psd.txt'
         ]

# - Posicoes realizadas as medicoes
zT = np.array([0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,200,250])

# ---- Flameless
# Estatistica
casesF = ['02-03-2020-P11-flameless/z0r0-stats.txt',
         '02-03-2020-P11-flameless/z10r0-stats.txt',
         '02-03-2020-P11-flameless/z20r0-stats.txt',
         '02-03-2020-P11-flameless/z30r0-stats.txt',
         '02-03-2020-P11-flameless/z40r0-stats.txt',
         '02-03-2020-P11-flameless/z50r0-stats.txt',
         '02-03-2020-P11-flameless/z60r0-stats.txt',
         '02-03-2020-P11-flameless/z70r0-stats.txt',
         '02-03-2020-P11-flameless/z80r0-stats.txt',
         '02-03-2020-P11-flameless/z90r0-stats.txt',
         '02-03-2020-P11-flameless/z100r0-stats.txt',
         '02-03-2020-P11-flameless/z110r0-stats.txt',
         '02-03-2020-P11-flameless/z120r0-stats.txt',
         '02-03-2020-P11-flameless/z130r0-stats.txt',
         '02-03-2020-P11-flameless/z140r0-stats.txt',
         '02-03-2020-P11-flameless/z150r0-stats.txt',
         '02-03-2020-P11-flameless/z200r0-stats.txt',
         '02-03-2020-P11-flameless/z250r0-stats.txt'
         ]

 # -PSDs
casesPSDsF = ['02-03-2020-P11-flameless/z0r0-psd.txt',
         '02-03-2020-P11-flameless/z10r0-psd.txt',
         '02-03-2020-P11-flameless/z20r0-psd.txt',
         '02-03-2020-P11-flameless/z30r0-psd.txt',
         '02-03-2020-P11-flameless/z40r0-psd.txt',
         '02-03-2020-P11-flameless/z50r0-psd.txt',
         '02-03-2020-P11-flameless/z60r0-psd.txt',
         '02-03-2020-P11-flameless/z70r0-psd.txt',
         '02-03-2020-P11-flameless/z80r0-psd.txt',
         '02-03-2020-P11-flameless/z90r0-psd.txt',
         '02-03-2020-P11-flameless/z100r0-psd.txt',
         '02-03-2020-P11-flameless/z110r0-psd.txt',
         '02-03-2020-P11-flameless/z120r0-psd.txt',
         '02-03-2020-P11-flameless/z130r0-psd.txt',
         '02-03-2020-P11-flameless/z140r0-psd.txt',
         '02-03-2020-P11-flameless/z150r0-psd.txt',
         '02-03-2020-P11-flameless/z200r0-psd.txt',
         '02-03-2020-P11-flameless/z250r0-psd.txt'
         ]

# - Posicoes realizadas as medicoes
zF = np.array([0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,200,250])

## ----------------------------------------------------------------------------
## ---------------------- Main
## ----------------------------------------------------------------------------

Conv = oneDim(casesC)
Conv.readPSD(casesPSDsC)

Trans = oneDim(casesT)
Trans.readPSD(casesPSDsT)

Flameless = oneDim(casesF)
Flameless.readPSD(casesPSDsF)

mu = 1e-5 # [m/s]

## ---------------------- CALCULO TURBULENCIA
Conv.calcTau()
Trans.calcTau()
Flameless.calcTau()

Conv.L = Conv.tau*Conv.UstdDev
Trans.L = Trans.tau*Trans.UstdDev
Flameless.L = Flameless.tau*Flameless.UstdDev

## ---------------------- PROCESSAMENTO
a = np.arange(5300,5400)
b = np.arange(8500,12000)

psdFl = Flameless.PSDs[0][:-500,1]
psdFl[a] = scp.medfilt(psdFl[a],19)
psdFl[b] = scp.medfilt(psdFl[b],19)

psdTr = Trans.PSDs[0][:-500,1]
psdTr[a] = scp.medfilt(psdTr[a],19)
psdTr[b] = scp.medfilt(psdTr[b],19)

psdCnv = Conv.PSDs[0][:-500,1]
psdCnv[a] = scp.medfilt(psdCnv[a],19)
psdCnv[b] = scp.medfilt(psdCnv[b],19)

psdFl6 = Flameless.PSDs[6][:-500,1]
psdFl6[a] = scp.medfilt(psdFl6[a],19)
psdFl6[b] = scp.medfilt(psdFl6[b],19)

## ---------------------- PLOTS

# - Plot fig stats
# - media
plt.figure()
plt.plot(zF,Flameless.U,'k*',label='Fio-quente - Flameless')
plt.plot(zT,Trans.U,'b*',label='Fio-quente - Tansicao')
plt.plot(zC,Conv.U,'r*',label='Fio-quente - Convencional')
plt.xlim([0,251])
plt.ylim([0,80])
plt.xlabel(r'$z$ $[mm]$')
plt.ylabel(r'$\overline{U}$ $[m/s]$')
plt.legend()
plt.savefig('Umean.png')
plt.show()

# - variancia
plt.figure()
plt.plot(zF,Flameless.Uvar,'k*',label='Fio-quente - Flameless')
plt.plot(zT,Trans.Uvar,'b*',label='Fio-quente - Tansicao')
plt.plot(zC,Conv.Uvar,'r*',label='Fio-quente - Convencional')
plt.xlim([0,251])
plt.ylim([0,180])
plt.xlabel(r'$z$ $[mm]$')
plt.ylabel(r"$\overline{u^{'}u^{'}}$ $[m^2/s^2]$")
plt.legend()
plt.savefig('Uprimemean.png')
plt.show()

# - tau
plt.figure()
plt.plot(zF,Flameless.tau*1e3,'k*',label='Fio-quente - Flameless')
plt.plot(zT,Trans.tau*1e3,'b*',label='Fio-quente - Tansicao')
plt.plot(zC,Conv.tau*1e3,'r*',label='Fio-quente - Convencional')
plt.xlim([0,251])
plt.ylim([0,25])
plt.xlabel(r'$z$ $[mm]$')
plt.ylabel(r"$\overline{\tau}$ $[ms]$")
plt.legend()
plt.savefig('tau.png')
plt.show()

# - L comprimento integral
plt.figure()
plt.plot(zF,Flameless.L,'k*',label='Fio-quente - Flameless')
plt.plot(zT,Trans.L,'b*',label='Fio-quente - Tansicao')
plt.plot(zC,Conv.L,'r*',label='Fio-quente - Convencional')
plt.xlim([0,251])
#plt.ylim([0,25])
plt.xlabel(r'$z$ $[mm]$')
plt.ylabel(r"$L$ $[m]$")
plt.legend()
plt.savefig('L.png')
plt.show()

# - Plot PSDs
x53 = np.array([2e3,5e4])
y53 = 1e4*x53**(-5/3.)

# - comparacao posicoes
plt.figure()
plt.loglog(x53,y53,'k--')
plt.loglog(Flameless.PSDs[0][:-500,0],Flameless.PSDs[0][:-500,1],'k',label='z0r0')
#plt.loglog(Flameless.PSDs[4][:-500,0],Flameless.PSDs[4][:-500,1],'b',label='z40r0')
plt.loglog(Flameless.PSDs[6][:-500,0],Flameless.PSDs[6][:-500,1],'r',label='z60r0')
#plt.loglog(Flameless.PSDs[10][:-500,0],Flameless.PSDs[10][:-500,1],'b',label='z100r0')
plt.ylim(1e-7,1e0)
plt.xlabel(r'$freq$ [Hz]')
plt.ylabel(r'$PSD$')
plt.legend()
plt.text(3e3,0.1,'$-5/3$',bbox=dict(boxstyle='round',fc='w',ec='k'))
plt.rc('lines',linewidth=1)
#plt.vlines(5e4,1e-7,1e-4,linestyle='dashed')
plt.savefig('psd-posicoes.png')
plt.show()

# - comparacao casos
plt.figure()
plt.loglog(Conv.PSDs[0][:-500,0],psdCnv,'k',label='Convencional')
plt.loglog(Trans.PSDs[0][:-500,0],psdTr,'b',label='Transicao')
plt.loglog(Flameless.PSDs[0][:-500,0],psdFl,'r',label='Flameless')
plt.rc('lines',linewidth=2)
plt.loglog(x53,y53,'k--')
plt.ylim(1e-7,1e-1)
plt.xlim(0,50000)
plt.xlabel(r'$freq \, [Hz]$')
plt.ylabel(r'$PSD$')
plt.legend()
plt.text(3e3,0.03,'-5/3',bbox=dict(boxstyle='round',fc='w',ec='k'))
#plt.rc('lines',linewidth=1)
#plt.vlines(5e4,1e-7,1e-4,linestyle='dashed')
plt.savefig('psd-casos.png')
plt.show()


## END
