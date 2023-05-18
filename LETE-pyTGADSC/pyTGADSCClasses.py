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

import numpy as np
import pandas as pd
import re
import io
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy import signal

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize

# Parametros de estilo dos plots
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.size": 22
})
plt.rcParams['xtick.top'] = plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.minor.top'] = plt.rcParams['ytick.minor.right'] = True
px = 1/plt.rcParams['figure.dpi']  # pixel in inches

# ******************************************************************************


class BaseExp(object):
    def __init__(self):
        self.sName = 'Nome do Sinal (TGA/DSC)'
        self.sunit = 'unidade do sinal'
        self.dunit = 'unidade da derivada'

    def readData(self, fileName, Tempinit, tga=False):
        '''Funcao para ler dados TGA/DSC
        '''
        with open(fileName,'r',encoding='latin1') as f:
            content = f.read()

        noHeader = re.search('\[Data\]\n(.*)',content,flags=re.DOTALL)
        raw = io.StringIO(noHeader.group(1))

        dataPD = pd.read_csv(raw,delimiter='\t',skiprows=[1])
        temp = np.array(dataPD['Temp'].values)
        # Remove dados iniciais antes da rampa de aquecimento
        mask = temp > Tempinit
        self.temp = temp[mask]
        self.tempK = self.temp + 273.15
        time = np.array(dataPD['Time'].values)
        self.time = time[mask]
        self.dt = np.gradient(self.time)
        if tga:
            sinal = np.array(dataPD['TGA'].values)*1e-6  # [kg]
            self.sinal = sinal[mask]
            self.m0 = self.sinal[0]
            self.sinal[self.sinal<0] = 0
            self.sinalNorm = self.sinal/self.m0  # normaliza com o valor de massa inicial
            self.dta = np.array(dataPD['DTA'].values)[mask]
        else:
            self.sinal = np.array(dataPD['DSC'].values)[mask]

# ******************************************************************************


class TGA(BaseExp):
    '''Classe para organizar dados do TGA
    '''

    def __init__(self, tgafile, Tempinit):
        self.fileName = tgafile
        self.readData(self.fileName, Tempinit, tga=True)
        self.sName = 'TGA'
        self.sunit = '-'
        self.dunit = 'kg'


# ******************************************************************************

class DSC(BaseExp):
    '''Classe para organizar dados do DSC
    '''

    def __init__(self, dscfile, Tempinit):
        self.fileName = dscfile
        self.readData(self.fileName, Tempinit, tga=False)
        self.sName = 'DSC'
        self.sunit = 'mW/mg'
        self.dunit = 'mW/mg'


# ******************************************************************************

class Problem(ElementwiseProblem):

    def __init__(self,m0, TGA, DSC, **kwargs):
        self.xl_agua = [1.5,1e7,60]  # 0,1,2
        self.xu_agua = [3.0,1e8,65]  # 0,1,2
        self.xl_C = [2.5,1e4,85]  # 3,4,5
        self.xu_C = [3.0,1e7,110]  # 3,4,5
        self.xl_Cox = [1.5,1e6,80,0.5]  # 6,7,8,9
        self.xu_Cox = [3.0,3e6,110,1.1]  # 6,7,8,9
        self.xl_alpha = [0.9,1e11,110,0.5]  # 10,11,12,13
        self.xu_alpha = [2.5,1e12,200,1.0]  # 10,11,12,13
        self.xl_beta = [0.9,1e2,100,0.9]  # 14,15,16,17
        self.xu_beta = [3.5,1e11,250,2.1]  # 14,15,16,17
        self.xl_ph2o = [0.06]  # 18
        self.xu_ph2o = [0.08]  # 18
        self.xl_ddts = [0.1,0.1,0.1,0.1]  # 19,20,21,22
        self.xu_ddts = [1.0,1.0,0.4,0.4]  # 19,20,21,22
        super().__init__(n_var=23,
                         n_obj=1,  # 3
                         n_ieq_constr=0,
                         xl=np.array(self.xl_agua+self.xl_C+self.xl_Cox
                                     +self.xl_alpha+self.xl_beta+self.xl_ph2o+self.xl_ddts),
                         xu=np.array(self.xu_agua+self.xu_C+self.xu_Cox
                                     +self.xu_alpha+self.xu_beta+self.xu_ph2o+self.xu_ddts))
        # self.m0 = m0
        self.TGA = TGA
        self.DSC = DSC
        # self.dt = np.gradient(self.TGA.time)
        # self.cumdt = np.cumsum(self.dt)-self.dt[0]

        self.R = 8.31446261815324e-3  # kJ/K mol
        self.cp_p = 1840  # J/kg K
        self.cp_agua = 4186  # J/kg K
        self.cp_a = 1260  # J/kg K
        self.cp_ash = 880  # J/kg K
        self.cp_b = 1260  # J/kg K
        self.Y_O2 = 0.232
        self.n_O2 = 1

        self.m_agua = 0.07*self.TGA.sinal
        self.m_C = (1-0.07)*self.TGA.sinal
        self.m_a = np.zeros(len(self.TGA.time))
        self.m_b = np.zeros(len(self.TGA.time))
        self.m_ash = np.zeros(len(self.TGA.time))

    def _evaluate(self, x, out, *args, **kwargs):
        '''Funcao para calcular massa e derivada da massa dos componentes:
        agua, C, a, b, ash, tot
        X: array com variaveis para otimizacao
        '''
        # plt.figure()
        # plt.plot(self.TGA.temp,m_agua,label='agua')
        # plt.plot(self.TGA.temp,m_C,label='C')
        # plt.plot(self.TGA.temp,m_ash,label='ash')
        # # plt.plot(self.TGA.temp,ddt_agua2,'k',label='ddt agua2')
        # # plt.plot(self.TGA.temp,self.m_C,label='C')
        # plt.plot(self.TGA.temp,self.TGA.sinal,label='TGA')
        # # plt.plot(self.TGA.temp[:-1],self.TGA.dta[:-1]*1e-1,label='DTA')
        # plt.plot(self.TGA.temp,-self.TGA.ddt*1e9,label='TGA ddt')
        # plt.plot(self.TGA.temp,ddt_opt*1e1,label='ddt opt')
        # plt.legend()
        # plt.show()
        m_opt, ddt_opt, m_agua, m_C, m_a, m_b, m_ash = self.calcMassDdt(x)
        # den = self.TGA.sinal
        # den[den<1e-10] = 1e-10
        erro2a = sum(abs(m_opt - self.TGA.sinal))
        erro3a = sum(abs(ddt_opt + self.TGA.ddt))

        out["F"] = (erro3a/2 + erro2a/2)

    def calcMassDdt(self,x):
        '''Funcao para calcular massa e derivada da massa dos componentes:
        agua, C, a, b, ash, tot
        X: array com variaveis para otimizacao
        '''
        # nk_agua, Ak_agua, Ek_agua
        x_agua = np.array([x[0],x[1],x[2]])

        # nk_C, Ak_C, Ek_C
        x_C = np.array([x[3],x[4],x[5]])

        # nk_Cox, Ak_Cox, Ek_Cox, expYO2_Cox
        x_Cox = np.array([x[6],x[7],x[8],x[9]])

        # nk_alpha, Ak_alpha, Ek_alpha, expYO2_alpha
        x_alpha = np.array([x[10],x[11],x[12],x[13]])

        # nk_beta, Ak_beta, Ek_beta, expYO2_beta
        x_beta = np.array([x[14],x[15],x[16],x[17]])

        # pddt_a = x[15]
        # pddt_b = x[16]
        # pddt_ash_beta = x[17]
        # pddt_ash_alpha = x[18]

        # fE_C = x[19]
        # fE_Cox = x[20]
        # fE_alpha = x[21]
        # fE_beta = x[22]
        # fE_agua = x[23]

        pH2O = x[18]  # fracao de agua

        # massa no instante inicial
        m_agua = pH2O*self.TGA.sinal  # kg
        m_agua[0] = pH2O*self.TGA.m0  # kg
        m_C = (1-pH2O)*self.TGA.sinal  # kg
        m_C[0] = (1-pH2O)*self.TGA.m0  # kg

        m_a = np.zeros(len(self.TGA.time))
        m_b = np.zeros(len(self.TGA.time))
        m_ash = np.zeros(len(self.TGA.time))
        ddt_opt = np.zeros(len(self.TGA.time))

        for i in range(len(self.TGA.time)-1):
            T = self.TGA.tempK[i]
            # Agua
            ddt_agua = self.ddtFunc(x_agua, m_agua[i], m_agua[0],T)
            m_agua[i+1] = max(0, m_agua[i] - ddt_agua*self.TGA.dt[i])
            # C
            ddt_C = self.ddtFunc(x_C, m_C[i], m_C[0],T)
            ddt_Cox = self.ddtFuncOx(x_Cox, m_C[i], m_C[0],T)
            m_C[i+1] = max(0, m_C[i] - (ddt_C + ddt_Cox)*self.TGA.dt[i])
            # char alpha
            ddt_alpha = self.ddtFuncOx(x_alpha, m_a[i], m_C[0],T)
            ddt_a = x[19]*ddt_C - ddt_alpha
            m_a[i+1] = max(0, m_a[i] + ddt_a*self.TGA.dt[i])
            # char Beta
            ddt_beta = self.ddtFuncOx(x_beta, m_b[i], m_C[0],T)
            ddt_b = x[20]*ddt_Cox - ddt_beta
            m_b[i+1] = max(0, m_b[i] + ddt_b*self.TGA.dt[i])
            # Ash
            ddt_ash = x[22]*ddt_beta - x[21]*ddt_alpha
            m_ash[i+1] = min(1,max(0, m_ash[i] + ddt_ash*self.TGA.dt[i]))
            ddt_opt[i] = ddt_agua + ddt_C + ddt_Cox - ddt_b - ddt_a - ddt_ash

        m_opt = m_agua + m_C + m_a + m_b + m_ash

        return m_opt, ddt_opt, m_agua, m_C, m_a, m_b, m_ash

    def ddtFunc(self, x, m, m0,T):
        '''Function to calculate time derivative in the form:
        ddt = m0*( (m/m0)**x[0] )*x[1]*exp(-x[2]/RT)
        ddt = m0*( (m/m0)**nk )*Ak*exp(-Ek/RT)
        '''
        ddt = m0*((m/m0)**x[0])*x[1]*np.exp(-x[2]/(self.R*T))
        return ddt

    def ddtFuncOx(self, x, m, m0, T):
        '''Function to calculate time derivative in the form:
        ddt = m0*( (m/m0)**x[0] )*x[1]*exp(-x[2]/RT)*Y_O2**x[3]
        ddt = m0*( (m/m0)**nk )*Ak*exp(-Ek/RT)*Y_O2**
        '''
        ddt = self.ddtFunc(x, m, m0, T)
        ddt *= self.Y_O2**x[3]
        return ddt

    '''def massOPTfunc(self,x):
        ddt_opt = self.ddtFunc(x)
        return self.target[0] - ddt_opt*self.cumdt'''

# ******************************************************************************


class BaseEnsaio(object):
    '''Classe base com propriedades
    '''

    def __init__(self):
        # self.R = 8.31446261815324  # J/K mol 0.008314
        # self.cp_p = 1840  # J/kg K
        # self.cp_agua = 4186  # J/kg K
        # self.cp_a = 1260  # J/kg K
        # self.cp_ash = 880  # J/kg K
        # self.cp_b = 1260  # J/kg K
        # self.Y_O2 = 0.232
        # self.n_O2 = 1
        # self.mcp = 0
        # self.X = [357307174.878322,66.4312552130049,2.09845480010349,22232.3684508937,85.9230235134947,2.10288331374131,2435421.32901346,95.3722793108986,2.66984533482758,12498015987.8924,169.145292265764,1.12643965353996,10364345760.0656,149.477012577210,1.54249160331694,0.462174913628938,0.688519370305353,0.352561526060474,0.362212168002418,214305.665728726,4536549.56089111,5240945.11633852,7644134.03028114,2635902.07304109,0.0769614335106816,0.672778735855666,0.606776603765679,1.61399141733446]
        # self.lb = [199148996,60,1.6,2000,80,2.0,1334277,94,2.5,1206847781,145,1.0,10008129676,135,1.4,0.23,0.22,0.10,0.12,167502,4085573,1004695,3344009,2260000,0.07,0.2,0.2,0.2]
        # self.ub = [499148996,69,3.6,25000,90,2.3,3534277,109,2.9,19968477818,192,1.4,16608129676,162,1.7,0.47,0.69,0.36,0.39,967502,9985573,90504695,90944009,2960000,0.09,2.0,2.0,2.0]
        self.lb = [0]
        self.ub = [0]


class Ensaio(BaseEnsaio):
    '''Classe para organizar dados dos ensaios TGA/DSC conjuntamente
    Chamada no arquivo principal. Utiliza as classes anteriores.
    '''

    def __init__(self,tgafile,dscfile,Tempinit=30):
        BaseEnsaio.__init__(self)

        self.TGA = TGA(tgafile, Tempinit)
        self.DSC = DSC(dscfile, Tempinit)
        self.signalProcess(self.TGA, fc=0.005)
        self.signalProcess(self.DSC, fc=0.01)
        self.m0 = self.TGA.m0
        self.DSC.sinal /= self.m0
        self.TGA.ddt = np.gradient(self.TGA.sinalfilt,self.TGA.time)  # *1e-6  # kg/s
        self.DSC.ddt = np.gradient(self.DSC.sinalfilt,self.DSC.time)  # mW/mg/s *1e-3  # W/kg/s
        # init massa
        self.m_tot = np.zeros(len(self.TGA.time))
        self.m_agua = np.zeros(len(self.TGA.time))
        self.m_C = np.zeros(len(self.TGA.time))
        self.m_a = np.zeros(len(self.TGA.time))
        self.m_b = np.zeros(len(self.TGA.time))
        self.m_ash = np.zeros(len(self.TGA.time))
        self.m_aguaDSC = np.zeros(len(self.DSC.time))
        self.m_CDSC = np.zeros(len(self.DSC.time))
        self.m_aDSC = np.zeros(len(self.DSC.time))
        self.m_bDSC = np.zeros(len(self.DSC.time))
        self.m_ashDSC = np.zeros(len(self.DSC.time))
        self.TGA.ddt_opt = np.zeros(len(self.TGA.time))
        self.DSC.sinal_opt = np.zeros(len(self.DSC.time))

    def signalProcess(self,obj,fc=0.005,plot=False):
        '''Funcao para processamento do sinal
        fc = Frequencia de corte para filtro passa baixa [Hz]
        '''
        # Passo de tempo conforme taxa de aquisicao
        dt = obj.time[1]-obj.time[0]

        # Filtro passa baixa
        sos = signal.butter(4,fc,'low', output='sos',fs=1/dt)
        # Sinal filtrado
        obj.sinalfilt = signal.sosfiltfilt(sos,obj.sinal)

        if plot:
            self.plotFilter(fc,dt,obj)

        return 0

    def optimize(self, X=None, lb=None, ub=None):
        if lb is None:
            lb = self.lb
        if ub is None:
            ub = self.ub
        # bounds = Bounds(lb,ub)
        # cntrT = ({'type': 'eq', 'fun': self.calcMass})
        # self.optX = minimize(self.calcMass,x0=X,method='SLSQP',
        #                      tol=1e-6,bounds=bounds,constraints=cntrT)

        problem = Problem(self.m0, self.TGA, self.DSC)

        algorithm = NSGA2(
            pop_size=3,
            n_offsprings=20,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=30),
            eliminate_duplicates=True
        )

        termination = get_termination("n_gen", 50)

        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=1,
                       save_history=True,
                       verbose=True)

        Xres = res.X
        Fres = res.F
        print('X optimized (agua): ', Xres[0:3])
        print('X optimized (C): ', Xres[3:6])
        print('X optimized (Cox): ', Xres[6:10])
        print('X optimized (alpha): ', Xres[10:14])
        print('X optimized (beta): ', Xres[14:18])
        print('X optimized (ash): ', Xres[18:-1])
        print('Sum error: ', Fres)
        self.TGA.sinal_opt, self.TGA.ddt_opt, self.TGA.m_agua, self.TGA.m_C, self.TGA.m_a, self.TGA.m_b, self.TGA.m_ash = problem.calcMassDdt(res.X)
        print('sinal OPT: ',self.TGA.sinal_opt)
        print('ddt OPT: ',self.TGA.ddt_opt)

        '''xl, xu = problem.bounds()
        plt.figure(figsize=(7, 5))
        plt.scatter(X[:, 0], X[:, 1], s=30, facecolors='none', edgecolors='r')
        plt.xlim(xl[0], xu[0])
        plt.ylim(xl[1], xu[1])
        plt.title("Design Space")
        plt.show()

        plt.figure(figsize=(7, 5))
        plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
        plt.title("Objective Space")
        plt.show()'''
        return 0

    def plotFilter(self,fc,dt,obj):
        '''Funcao para plotar grafico do filtro
        '''
        # plot do filtro
        b,a = signal.butter(4,fc,'low', output='ba',analog=True)
        w, h = signal.freqs(b, a)
        plt.figure()
        plt.semilogx(w, 20 * np.log10(abs(h)))
        plt.title('Butterworth filter freq response')
        plt.xlabel('Frequency [radians / second]')
        plt.ylabel('Amplitude [dB]')
        plt.margins(0, fc)
        plt.grid(which='both', axis='both')
        plt.axvline(fc, color='green')  # cutoff frequency
        plt.tight_layout()

        # Plot da FFT do sinal para verificacao da freq de corte do filtro
        # FFT para verificacao do filtro
        sinalFFT = np.fft.rfft(obj.sinal)
        binsfft = np.fft.rfftfreq(len(obj.sinal),d=dt)
        sinalFFTfilt = np.fft.rfft(obj.sinalfilt)
        binsfftfilt = np.fft.rfftfreq(len(obj.sinalfilt),d=dt)

        plt.figure()
        ax = plt.gca()

        plt.plot(binsfft,np.real(sinalFFT),'k',label='Original')
        plt.plot(binsfftfilt,np.real(sinalFFTfilt),'r',label='Filtro LP %.2fHz' %fc)
        plt.ylim(-2,5)
        plt.xlim(-0.05,1.5)
        plt.ylabel('FFT')
        plt.xlabel('freq [Hz]')
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        # plt.title(file.replace('_','\_'))
        plt.legend()
        plt.tight_layout()
        plt.savefig(obj.sName+'FFT.png')

        # Plot do sinal original x filtrado para verificacao do filtro
        plt.figure()
        ax = plt.gca()
        plt.plot(obj.time,obj.sinal,'k',label='Original')
        plt.plot(obj.time,obj.sinalfilt,'r',label='Filtro LP %.2fHz' %fc)
        plt.ylabel(obj.sName)
        plt.xlabel('Tempo [s]')
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        # plt.title(file.replace('_','\_'))
        plt.legend()
        plt.tight_layout()
        plt.savefig(obj.sName+'filt.png')

        return 0

    def plotTime(self,expObj):
        # Plot em funcao do tempo
        fig,ax0 = plt.subplots(figsize=(2*720*px,2*460*px))
        fig.subplots_adjust(right=0.79,left=0.22)

        ax1 = ax0.twinx()
        ax2 = ax0.twinx()
        ax3 = ax0.twiny()
        ax4 = ax0.twinx()

        ax1.spines['left'].set_position(("axes",-0.18))
        ax1.spines['left'].set_visible(True)
        ax1.yaxis.set_label_position('left')
        ax1.yaxis.set_ticks_position('left')
        ax4.spines['right'].set_position(("axes",1.2))

        ax0.plot(expObj.time,expObj.temp, '--k', label='Temperature')
        ax1.plot(expObj.time,expObj.tempK, '--k', label='Temperature')
        ax2.plot(expObj.time,expObj.sinal, 'k', label=expObj.sName)
        ax3.plot(expObj.time/60,expObj.temp, '--k', label='Time')
        # ax4.plot(expObj.time[:-1],expObj.dta[:-1], 'r', label='DTA')
        ax4.plot(expObj.time,expObj.ddt, 'r', label=r'$\frac{d}{dt} (%s)$' %expObj.sName)

        h0,l0 = ax0.get_legend_handles_labels()
        h2,l2 = ax2.get_legend_handles_labels()
        h4,l4 = ax4.get_legend_handles_labels()

        ax0.set_xlabel('Time [s]')
        ax0.set_ylabel('Temperature [C]')
        ax1.set_ylabel('Temperature [K]')
        ax2.set_ylabel(expObj.sName+' ['+expObj.sunit+']')
        ax3.set_xlabel('Time [min]')
        # ax4.set_ylabel('DTA [?]')
        txt = r'$\frac{d}{dt} (%s)\;[%s/s]$'
        ax4.set_ylabel(txt %(expObj.sName,expObj.dunit))

        ax0.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax0.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax1.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax2.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax3.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax4.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax4.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        ax4.get_yaxis().get_offset_text().set_position((1.33,0))

        plt.legend(h0+h2+h4,l0+l2+l4, loc='center right')
        plt.savefig(expObj.fileName+'.png')

        return 0

    def plotTemp(self,expObj):
        # Plot em funcao da Temperatura
        fig,ax0 = plt.subplots(figsize=(2*550*px,2*460*px))
        fig.subplots_adjust(right=0.87,left=0.12)

        ax1 = ax0.twiny()
        ax2 = ax0.twinx()

        # ax2.spines['right'].set_position(("axes",1.2))

        ax0.plot(expObj.temp,expObj.sinal, 'k', label=expObj.sName)
        ax1.plot(expObj.tempK,expObj.sinal, 'k', label=expObj.sName)
        ax2.plot(expObj.temp,expObj.ddt, 'r', label=r'$\frac{d}{dt} (%s)$' %expObj.sName)

        h0,l0 = ax0.get_legend_handles_labels()
        h2,l2 = ax2.get_legend_handles_labels()

        ax0.set_xlabel('Temperature [C]')
        ax0.set_ylabel(expObj.sName+' ['+expObj.sunit+']')
        ax1.set_xlabel('Temperature [K]')
        txt = r'$\frac{d}{dt} (%s)\;[%s/s]$'
        ax2.set_ylabel(txt %(expObj.sName,expObj.dunit))
        # ax2.set_ylim(-0.1,0.1)

        ax0.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax0.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax2.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax2.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        ax2.get_yaxis().get_offset_text().set_position((1.13,0))

        plt.legend(h0+h2,l0+l2)
        plt.savefig(expObj.fileName+'Temp.png')

    def plotOPTTemp(self,expObj):
        # Plot em funcao da Temperatura
        fig,ax0 = plt.subplots(figsize=(2*550*px,2*460*px))
        fig.subplots_adjust(right=0.87,left=0.12)

        ax1 = ax0.twiny()
        ax2 = ax0.twinx()

        # ax2.spines['right'].set_position(("axes",1.2))

        ax0.plot(expObj.temp,expObj.sinalfilt/expObj.m0, 'k', label=expObj.sName)
        ax0.plot(expObj.temp,expObj.sinal_opt/expObj.m0, 'dimgray', ls=(0, (5, 10)),label=expObj.sName+' - GA')
        # ax0.plot(expObj.temp,expObj.m_agua, 'b', label='agua')
        # ax0.plot(expObj.temp,expObj.m_C, label='C')
        # ax0.plot(expObj.temp,expObj.m_a, label='alpha')
        # ax0.plot(expObj.temp,expObj.m_b, label='beta')
        # ax0.plot(expObj.temp,expObj.m_ash, label='ash')
        ax1.plot(expObj.tempK,expObj.sinalfilt/expObj.m0, 'k', label=expObj.sName)
        ax2.plot(expObj.temp,-expObj.ddt, 'r', label=r'$\frac{d}{dt} (%s)$' %expObj.sName)
        ax2.plot(expObj.temp,expObj.ddt_opt, 'crimson', ls=(0, (5, 10)),label=r'$\frac{d}{dt} (%s) - GA$' %expObj.sName)

        h0,l0 = ax0.get_legend_handles_labels()
        h2,l2 = ax2.get_legend_handles_labels()

        ax0.set_xlabel('Temperature [C]')
        ax0.set_ylabel(expObj.sName+' ['+expObj.sunit+']')
        ax1.set_xlabel('Temperature [K]')
        txt = r'$\frac{d}{dt} (%s)\;[%s/s]$'
        ax2.set_ylabel(txt %(expObj.sName,expObj.dunit))
        # ax2.set_ylim(-0.1,0.1)

        ax0.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax0.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax2.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax2.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        ax2.get_yaxis().get_offset_text().set_position((1.13,0))

        plt.legend(h0+h2,l0+l2)
        plt.savefig(expObj.fileName+'Temp-OPT.png')
