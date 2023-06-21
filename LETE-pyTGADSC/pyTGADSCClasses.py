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
from tabulate import tabulate

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


class Problem(ElementwiseProblem):

    def __init__(self, TGA, DSC, xlim, **kwargs):
        XL = xlim['bd']['low']+xlim['bp']['low']+xlim['bo']['low']
        XL += xlim['alpha']['low']+xlim['beta']['low']+xlim['ph2o']['low']
        XL += xlim['ddts']['low']+xlim['h_i']['low']
        XU = xlim['bd']['up']+xlim['bp']['up']+xlim['bo']['up']
        XU += xlim['alpha']['up']+xlim['beta']['up']+xlim['ph2o']['up']
        XU += xlim['ddts']['up']+xlim['h_i']['up']

        super().__init__(n_var=28,
                         n_obj=1,
                         n_ieq_constr=0,xl=np.array(XL),xu=np.array(XU))

        self.TGA = TGA
        self.DSC = DSC

        self.R = 8.31446261815324e-3  # kJ/K mol
        self.cp_p = 1840  # J/kg K
        self.cp_agua = 4186  # J/kg K
        self.cp_a = 1260  # J/kg K
        self.cp_ash = 880  # J/kg K
        self.cp_b = 1260  # J/kg K
        self.Y_O2 = 0.232
#        self.n_O2 = 1

    def _evaluate(self, x, out, *args, **kwargs):
        '''Funcao para calcular massa e derivada da massa dos componentes:
        agua, C, a, b, ash, tot
        X: array com variaveis para otimizacao
        '''
        opts, mSpec, ddts = self.calcMassDdt(x)
        m_opt, ddt_opt, en_opt = opts
        m_agua, m_C, m_a, m_b, m_ash = mSpec

        erro1a = sum(abs(m_opt - self.TGA.sinalfilt))/sum(abs(self.TGA.sinalfilt))
        erro2a = sum(abs(ddt_opt + self.TGA.ddt))/sum(abs(self.TGA.ddt))

        # self.DSC.sinalNew = self.DSC.sinalfilt  # dscnew
        erro3a = sum(abs(en_opt - self.DSC.sinalfilt))/sum(abs(self.DSC.sinalfilt))

#        out["F"] = (erro1a/2 + erro2a/2)
        out["F"] = (erro1a/3 + erro2a/3 + erro3a/3)

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

        # fracao de agua
        pH2O = x[18]

        # x derivadas
        x_ddt = np.array([x[19],x[20],x[21],x[22]])

        # fatores DSC opt
        x_dsc = np.array([x[23],x[24],x[25],x[26],x[27]])

        # massa no instante inicial
        m_agua = pH2O*self.TGA.sinalfilt  # kg
        m_agua[0] = pH2O*self.TGA.m0  # kg
        m_C = (1-pH2O)*self.TGA.sinalfilt  # kg
        m_C[0] = (1-pH2O)*self.TGA.m0  # kg

        m_a = np.zeros(len(self.TGA.time))
        m_b = np.zeros(len(self.TGA.time))
        m_ash = np.zeros(len(self.TGA.time))
        ddt_agua = np.zeros(len(self.TGA.time))
        ddt_C = np.zeros(len(self.TGA.time))
        ddt_Cox = np.zeros(len(self.TGA.time))
        ddt_alpha = np.zeros(len(self.TGA.time))
        ddt_a = np.zeros(len(self.TGA.time))
        ddt_beta = np.zeros(len(self.TGA.time))
        ddt_b = np.zeros(len(self.TGA.time))
        ddt_ash = np.zeros(len(self.TGA.time))
        ddt_opt = np.zeros(len(self.TGA.time))
        en_opt = np.zeros(len(self.TGA.time))

        # loop ao longo do tempo
        for i in range(len(self.TGA.time)-1):
            T = self.TGA.tempK[i]
            # Agua
            ddt_agua[i] = self.ddtFunc(x_agua, m_agua[i], m_agua[0],T)
            m_agua[i+1] = max(0, m_agua[i] - ddt_agua[i]*self.TGA.dt[i])
            # C
            ddt_C[i] = self.ddtFunc(x_C, m_C[i], m_C[0],T)
            ddt_Cox[i] = self.ddtFuncOx(x_Cox, m_C[i], m_C[0],T)
            m_C[i+1] = max(0, m_C[i] - (ddt_C[i] + ddt_Cox[i])*self.TGA.dt[i])
            # char alpha
            ddt_alpha[i] = self.ddtFuncOx(x_alpha, m_a[i], m_C[0],T)
            ddt_a[i] = x_ddt[0]*ddt_C[i] - ddt_alpha[i]
            m_a[i+1] = max(0, m_a[i] + ddt_a[i]*self.TGA.dt[i])
            # char Beta
            ddt_beta[i] = self.ddtFuncOx(x_beta, m_b[i], m_C[0],T)
            ddt_b[i] = x_ddt[1]*ddt_Cox[i] - ddt_beta[i]
            m_b[i+1] = max(0, m_b[i] + ddt_b[i]*self.TGA.dt[i])
            # Ash
            ddt_ash[i] = x_ddt[3]*ddt_beta[i] + x_ddt[2]*ddt_alpha[i]
            m_ash[i+1] = max(0, m_ash[i] + ddt_ash[i]*self.TGA.dt[i])  # min(1,)
            ddt_opt[i] = ddt_agua[i] + ddt_C[i] + ddt_Cox[i] - ddt_b[i] - ddt_a[i] - ddt_ash[i]
        # Energia [mW]
#       mcp = 0
        mcp = m_C*self.cp_p + m_a*self.cp_a + m_b*self.cp_b+m_ash*self.cp_ash
        mcp *= self.TGA.dTdt  # taxaT
        mcp = np.array(mcp, dtype=float)
        en_opt = -ddt_agua*x_dsc[0] - ddt_C*x_dsc[1] + ddt_Cox*x_dsc[2]
        en_opt += ddt_alpha*x_dsc[3] + ddt_beta*x_dsc[4]
        en_opt -= mcp
        en_opt *= 1e3

        m_opt = m_agua + m_C + m_a + m_b + m_ash

        opts = [m_opt, ddt_opt, en_opt]
        mSpec = [m_agua, m_C, m_a, m_b, m_ash]
        ddts = [ddt_agua,ddt_C,ddt_Cox,ddt_alpha,ddt_a,ddt_beta,ddt_b,ddt_ash]

        return opts, mSpec, ddts

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

# ******************************************************************************


class BaseExp(object):
    def __init__(self):
        self.sName = 'Nome do Sinal (TGA/DSC)'
        self.sunit = 'unidade do sinal'
        self.dunit = 'unidade da derivada'

    def readData(self, fileName, Tempinit, AqTime, tga=False):
        '''Funcao para ler dados TGA/DSC
        '''
        with open(fileName,'r',encoding='latin1') as f:
            content = f.read().strip()
        content = content.replace('\n\n','\n')

        # Sample Weight
        m0raw = re.search(r'Sample Weight:\s+(.*)\[mg\]',content,flags=0)
        m0 = float(m0raw.group(1))*1e-6  # [kg]

        # Sample aquisition time
        timeraw = re.search(r'Sampling Time \[ sec \]:\s+(.*)',content,flags=0)
        time = float(timeraw.group(1))  # [s]

        noHeader = re.search(r'\[Data\]\n(.*)',content,flags=re.DOTALL)
        raw = io.StringIO(noHeader.group(1))
        dataPD = pd.read_csv(raw,delimiter='\t',skiprows=[1])
        dataPD = dataPD[::int(AqTime/time)]

        temp = np.array(dataPD['Temp'].values)

        # Remove dados iniciais antes da rampa de aquecimento
        maskLow = temp > Tempinit
        temp = temp[maskLow]
        # Remove dados com temperaturas maiores que 600 C
        maskUp = temp < 600
        temp = temp[maskUp]

        time = np.array(dataPD['Time'].values)[maskLow]
        time = time[maskUp]

        tempK = temp + 273.15
        dt = np.gradient(time)

        if tga:
            sinal = np.array(dataPD['TGA'].values)*1e-6  # [kg]
            sinal = sinal[maskLow]
            sinal = sinal[maskUp]
            sinal[sinal<0] = 0
            sinalNorm = sinal/m0  # normaliza com o valor de massa inicial
            dta = np.array(dataPD['DTA'].values)[maskLow]
            dta = dta[maskUp]
            # print('Tamanho do vetor dados: ', len(sinalNorm))

            return [temp, time, tempK, dt, sinal, sinalNorm, dta, m0]
        else:
            sinal = np.array(dataPD['DSC'].values)[maskLow]
            sinal = sinal[maskUp]
            sinalNorm = sinal*1e-6/m0  # [mW/mg]
            # print('Tamanho do vetor dados: ', len(sinalNorm))

            return [temp, time, tempK, dt, sinal, sinalNorm, m0]

    def _process(self, exps, Tempinit, exName, AqTime):
        tga=False
        if exName=='TGA':
            tga = True
        n_exps = len(exps.keys())
        szs = np.zeros(n_exps)
        m0s = np.zeros(n_exps)
        data = []
        fNames = []
        for i,k in enumerate(exps.keys()):
            fName = exps[k][exName]
            expData = self.readData(fName,Tempinit, AqTime,tga=tga)
            szs[i] = len(expData[0])
            m0s[i] = expData[-1]
            data.append(expData)
            fNames.append(fName)

        minsz = int(min(szs))
        Exps = []
        for i,k in enumerate(exps.keys()):
            exp = np.array(data[i][:-1])
            Exps.append(exp[:,:minsz])

        Exps = np.array(Exps,dtype=object)
        Mean = np.mean(Exps,axis=0)

        return Exps, fNames, Mean, m0s


# ******************************************************************************


class TGA(BaseExp):
    '''Classe para organizar dados do TGA
    sName = 'TGA'
    sunit = '-'
    dunit = 'kg'
    temp, tempK, time, dt, sinal, sinalNorm, m0, dta
    Exps, fNames
    '''

    def __init__(self, exps, Tempinit, AqTime):
        self.sName = 'TGA'
        self.sunit = '-'
        self.dunit = 'kg'
        self.Exps, self.fNames, mean, self.m0s = self._process(exps, Tempinit, self.sName, AqTime)
        self.temp,self.time,self.tempK,self.dt,self.sinal,self.sinalNorm,self.dta = mean
        self.m0 = np.mean(self.m0s)

# ******************************************************************************


class DSC(BaseExp):
    '''Classe para organizar dados do DSC
    sName = 'DSC'
    sunit = 'mW/mg'
    dunit = 'mW/mg'
    temp, tempK, time, dt, sinal
    Exps, fNames
    '''

    def __init__(self, exps, Tempinit, AqTime):
        self.fileName = 'DSCmean'
        self.sName = 'DSC'
        self.sunit = 'mW/mg'
        self.dunit = 'mW/mg'
        self.Exps, self.fNames, mean, self.m0s = self._process(exps, Tempinit, self.sName, AqTime)
        self.temp,self.time,self.tempK,self.dt,self.sinal,self.sinalNorm = mean
        self.m0 = np.mean(self.m0s)

# ******************************************************************************


class Ensaio(object):
    '''Classe para organizar dados dos ensaios TGA/DSC conjuntamente
    Chamada no arquivo principal. Utiliza as classes anteriores.
    '''

    def __init__(self,exps,name,Tempinit=30):
        # BaseEnsaio.__init__(self)
        self.Name = name
        self.lb = []
        self.ub = []

        self.AqTime = self.readAqTime(exps)
        self.TGA = TGA(exps, Tempinit, self.AqTime)
        self.DSC = DSC(exps, Tempinit, self.AqTime)
        self.correctSize()
        self.TGA.sinalfilt = self.signalProcess(self.TGA, fc=0.005)
        self.DSC.sinalfilt = self.signalProcess(self.DSC, fc=0.01)
        self.TGA.ddt = np.gradient(self.TGA.sinalfilt,self.TGA.time)  # kg/s
        self.DSC.ddt = np.gradient(self.DSC.sinalfilt,self.DSC.time)  # mW/mg/s *1e-3  # W/kg/s
        self.TGA.dTdt = np.gradient(self.TGA.tempK,self.TGA.time)  # K/s

    def readAqTime(self, exps):
        '''Funcao para ler dados TGA/DSC
        '''
        time = []
        for i,k in enumerate(exps.keys()):
            fNameTGA = exps[k]['TGA']
            fNameDSC = exps[k]['DSC']
            # TGA
            with open(fNameTGA,'r',encoding='latin1') as fTGA:
                contentTGA = fTGA.read()
            # Sample aquisition time
            timerawTGA = re.search(r'Sampling Time \[ sec \]:\s+(.*)',contentTGA,flags=0)
            timeTGA = float(timerawTGA.group(1))  # [s]
            time.append(timeTGA)
            # DSC
            with open(fNameDSC,'r',encoding='latin1') as fDSC:
                contentDSC = fDSC.read()
            # Sample aquisition time
            timerawDSC = re.search(r'Sampling Time \[ sec \]:\s+(.*)',contentDSC,flags=0)
            timeDSC = float(timerawDSC.group(1))  # [s]
            time.append(timeDSC)

        return max(time)

    def correctSize(self):
        '''Funcao para corrigir e igualar tamanho dos vetores TGA e DSC
        '''
        szTGA = len(self.TGA.temp)
        szDSC = len(self.DSC.temp)
        szMin = min(szTGA, szDSC)
        szDiff = max(szTGA, szDSC) - szMin
        if szTGA is not szMin:
            self.TGA.temp = self.TGA.temp[:-szDiff]
            self.TGA.time = self.TGA.time[:-szDiff]
            self.TGA.tempK = self.TGA.tempK[:-szDiff]
            self.TGA.dt = self.TGA.dt[:-szDiff]
            self.TGA.sinal = self.TGA.sinal[:-szDiff]
            self.TGA.sinalNorm = self.TGA.sinalNorm[:-szDiff]
            self.TGA.dta = self.TGA.dta[:-szDiff]
        if szDSC is not szMin:
            self.DSC.temp = self.DSC.temp[:-szDiff]
            self.DSC.time = self.DSC.time[:-szDiff]
            self.DSC.tempK = self.DSC.tempK[:-szDiff]
            self.DSC.dt = self.DSC.dt[:-szDiff]
            self.DSC.sinal = self.DSC.sinal[:-szDiff]
            self.DSC.sinalNorm = self.DSC.sinalNorm[:-szDiff]
        return 0

    def signalProcess(self,obj,fc=0.005,plot=False):
        '''Funcao para processamento do sinal
        fc = Frequencia de corte para filtro passa baixa [Hz]
        '''
        # Passo de tempo conforme taxa de aquisicao
        dt = obj.time[1]-obj.time[0]

        # Filtro passa baixa
        sos = signal.butter(4,fc,'low', output='sos',fs=1/dt)
        # Sinal filtrado
        sinalfilt = signal.sosfiltfilt(sos,obj.sinal)

        if plot:
            self.plotFilter(fc,dt,obj)

        return sinalfilt

    def optimize(self, xlim=None):
        if xlim is None:
            xlim = {'bd':{'low':[1.5,1e7,60],  # 0,1,2
                          'up':[3.0,1e8,65]},  # 0,1,2
                    'bp':{'low':[2.5,1e4,85],  # 3,4,5
                          'up':[3.0,1e7,110]},  # 3,4,5
                    'bo':{'low':[1.5,1e6,80,0.5],  # 6,7,8,9
                          'up':[3.0,3e6,110,1.1]},  # 6,7,8,9
                    'alpha':{'low':[0.9,1e11,110,0.5],  # 10,11,12,13
                             'up':[2.5,1e12,200,1.0]},  # 10,11,12,13
                    'beta':{'low':[0.9,1e2,100,0.9],  # 14,15,16,17
                            'up':[3.5,1e11,250,2.1]},  # 14,15,16,17
                    'ph2o':{'low':[0.06],  # 18
                            'up':[0.08]},  # 18
                    'ddts':{'low':[0.1,0.1,0.1,0.1],  # 19,20,21,22
                            'up':[1.0,1.0,0.4,0.4]},  # 19,20,21,22
                    'h_i':{'low':[1e5,1e5,1e5,1e5,1e5],  # 23,24,25,26,27
                           'up':[1e7,1e7,1e7,1e7,1e7]}}  # 23,24,25,26,27

        problem = Problem(self.TGA, self.DSC, xlim)

        algorithm = NSGA2(
            pop_size=3,
            n_offsprings=3,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.95, eta=15),
            mutation=PM(eta=30),
            eliminate_duplicates=True
        )

        termination = get_termination("n_gen", 1000)

        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=1,
                       save_history=True,
                       verbose=True)

        Xres = res.X
        Fres = res.F
        header1 = ['MASSA','nk','Ak','Ek','nO2_k']
        tagua = ['bd',Xres[0],Xres[1],Xres[2],'-']
        tC = ['bp',Xres[3],Xres[4],Xres[5],'-']
        tCox = ['bo',Xres[6],Xres[7],Xres[8],Xres[9]]
        talpha = ['alpha_o',Xres[10],Xres[11],Xres[12],Xres[13]]
        tbeta = ['beta_o',Xres[14],Xres[15],Xres[16],Xres[17]]
        table1 = [header1,tagua,tC,tCox,talpha,tbeta]

        txt = '\nRESULTADOS\n%%H2O: %.5f' %Xres[18]
        txt += '\n' + tabulate(table1,headers='firstrow',tablefmt='fancy_grid')

        header2 = ['Omega','nu_a','nu_b','nu_ash alpha','nu_ash beta']
        tddt = ['nu',Xres[19],Xres[20],Xres[21],Xres[22]]
        table2 = [header2,tddt]
        txt += '\n' + tabulate(table2,headers='firstrow',tablefmt='fancy_grid')

        header3 = ['DSC','h_bd','h_bp','h_bo','h_alpha_o','h_beta_o']
        tdsc = ['h_i',Xres[23],Xres[24],Xres[25],Xres[26],Xres[27]]
        table3 = [header3,tdsc]
        txt += '\n' + tabulate(table3,headers='firstrow',tablefmt='fancy_grid')

        txt += '\nSum error: %.5f' %Fres[0]
        print(txt)
        with open(self.Name+'-RES.txt','w') as log:
            log.write(txt)
        #
        opts, mSpec, ddts = problem.calcMassDdt(res.X)
        self.TGA.sinal_opt, self.TGA.ddt_opt, self.DSC.sinal_opt = opts
        self.TGA.m_agua, self.TGA.m_C, self.TGA.m_a, self.TGA.m_b, self.TGA.m_ash = mSpec
        self.TGA.sinal_optNorm = self.TGA.sinal_opt/self.TGA.m0
        self.DSC.sinal_optNorm = self.DSC.sinal_opt*1e-6/self.DSC.m0
        # derivadas
        self.TGA.ddt_agua,self.TGA.ddt_C,self.TGA.ddt_Cox,self.TGA.ddt_alpha,self.TGA.ddt_a,self.TGA.ddt_beta,self.TGA.ddt_b,self.TGA.ddt_ash = ddts

        # print(max(abs(self.DSC.time-self.TGA.time)))
        # print('sinal OPT: ',self.TGA.sinal_opt)
        # print('ddt OPT: ',self.TGA.ddt_opt)
        # print('DSC OPT: ',self.DSC.sinal_opt)

        self.TGA.sinalNorm = self.TGA.sinalfilt/self.TGA.m0
        self.DSC.sinalNorm = self.DSC.sinalfilt*1e-6/self.DSC.m0

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
        sinalFFT = np.fft.rfft(obj.sinalNorm)
        binsfft = np.fft.rfftfreq(len(obj.sinalNorm),d=dt)
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
        plt.plot(obj.time,obj.sinalNorm,'k',label='Original')
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

    def plotTime(self,obj):
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

        ax0.plot(obj.time,obj.temp, '--k', label='Temperature')
        ax1.plot(obj.time,obj.tempK, '--k', label='Temperature')
        ax3.plot(obj.time/60,obj.temp, '--k', label='Time')
        ax2.plot(obj.time,obj.sinalNorm, 'k', label=obj.sName)
        for i in range(np.shape(obj.Exps)[0]):
            ax2.plot(obj.Exps[i,1],obj.Exps[i,5])
        # ax4.plot(obj.time[:-1],obj.dta[:-1], 'r', label='DTA')
        ax4.plot(obj.time,obj.ddt, 'r', label=r'$\frac{d}{dt} (%s)$' %obj.sName)

        h0,l0 = ax0.get_legend_handles_labels()
        h2,l2 = ax2.get_legend_handles_labels()
        h4,l4 = ax4.get_legend_handles_labels()

        ax0.set_xlabel('Time [s]')
        ax0.set_ylabel('Temperature [C]')
        ax1.set_ylabel('Temperature [K]')
        ax2.set_ylabel(obj.sName+' ['+obj.sunit+']')
        ax3.set_xlabel('Time [min]')
        # ax4.set_ylabel('DTA [?]')
        txt = r'$\frac{d}{dt} (%s)\;[%s/s]$'
        ax4.set_ylabel(txt %(obj.sName,obj.dunit))

        ax0.set_xlim(0,max(obj.time))
        ax3.set_xlim(0,max(obj.time)/60)

        ax0.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax0.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax1.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax2.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax3.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax4.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax4.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        ax4.get_yaxis().get_offset_text().set_position((1.33,0))

        title = str(self.Name).replace('_',r'\_')
        plt.title(title)
        plt.legend(h0+h2+h4,l0+l2+l4, loc='center right')
        plt.savefig(self.Name+'-'+obj.sName+'.png')

        return 0

    def plotTemp(self,obj):
        # Plot em funcao da Temperatura
        fig,ax0 = plt.subplots(figsize=(2*550*px,2*460*px))
        fig.subplots_adjust(right=0.87,left=0.12)

        ax1 = ax0.twiny()
        ax2 = ax0.twinx()

        # ax2.spines['right'].set_position(("axes",1.2))

        ax0.plot(obj.temp,obj.sinalNorm, 'k', label=obj.sName)
        ax1.plot(obj.tempK,obj.sinalNorm, 'k', label=obj.sName)
        ax2.plot(obj.temp,obj.ddt, 'r', label=r'$\frac{d}{dt} (%s)$' %obj.sName)

        h0,l0 = ax0.get_legend_handles_labels()
        h2,l2 = ax2.get_legend_handles_labels()

        ax0.set_xlabel('Temperature [C]')
        ax0.set_ylabel(obj.sName+' ['+obj.sunit+']')
        ax1.set_xlabel('Temperature [K]')
        txt = r'$\frac{d}{dt} (%s)\;[%s/s]$'
        ax2.set_ylabel(txt %(obj.sName,obj.dunit))
        # ax2.set_ylim(-0.1,0.1)

        ax0.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax0.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax2.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax2.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        ax2.get_yaxis().get_offset_text().set_position((1.13,0))

        title = str(self.Name).replace('_',r'\_')
        plt.title(title)
        plt.legend(h0+h2,l0+l2)
        plt.savefig(self.Name+'-'+obj.sName+'Temp.png')

    def plotTGAOPTTemp(self,obj,plotSpecies=False):
        # Plot em funcao da Temperatura
        fig,ax0 = plt.subplots(figsize=(2*550*px,2*460*px))
        fig.subplots_adjust(right=0.87,left=0.12)

        ax1 = ax0.twiny()
        ax2 = ax0.twinx()

        # ax2.spines['right'].set_position(("axes",1.2))

        ax0.plot(obj.temp,obj.sinalNorm, 'k', label=obj.sName)
        ax0.plot(obj.temp,obj.sinal_optNorm, 'dimgray', ls=(0, (5, 10)),label=obj.sName+' - GA')
        if plotSpecies:
            ax0.plot(obj.temp,obj.m_agua/obj.m0, 'b', label='$H_2O$')
            ax0.plot(obj.temp,obj.m_C/obj.m0, label='DWF')
            ax0.plot(obj.temp,obj.m_a/obj.m0, label='alpha-char')
            ax0.plot(obj.temp,obj.m_b/obj.m0, label='beta-char')
            ax0.plot(obj.temp,obj.m_ash/obj.m0, label='Ash')
        ax1.plot(obj.tempK,obj.sinalNorm, 'k', label=obj.sName)
        ax2.plot(obj.temp,-obj.ddt, 'r', label=r'$\frac{d}{dt} (%s)$' %obj.sName)
        label = r'$\frac{d}{dt} (%s) - GA$' %obj.sName
        ax2.plot(obj.temp,obj.ddt_opt, 'crimson', ls=(0, (5, 10)),label=label)

        h0,l0 = ax0.get_legend_handles_labels()
        h2,l2 = ax2.get_legend_handles_labels()

        ax0.set_xlabel('Temperature [C]')
        ax0.set_ylabel(obj.sName+' ['+obj.sunit+']')
        ax1.set_xlabel('Temperature [K]')
        txt = r'$\frac{d}{dt} (%s)\;[%s/s]$'
        ax2.set_ylabel(txt %(obj.sName,obj.dunit))
        # ax2.set_ylim(-0.1,0.1)

        ax0.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax0.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax2.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax2.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        ax2.get_yaxis().get_offset_text().set_position((1.13,0))

        title = str(self.Name).replace('_',r'\_')
        plt.title(title)
        plt.legend(h0+h2,l0+l2)
        plt.savefig(self.Name+'-'+obj.sName+'Temp-TGAOPT.png')

        # -------------- DDTs --------------
        # Plot em funcao da Temperatura
        fig,ax0 = plt.subplots(figsize=(2*550*px,2*460*px))
        fig.subplots_adjust(right=0.87,left=0.12)

        ax1 = ax0.twiny()

        ax0.plot(obj.temp,-obj.ddt_agua, 'b', label='$H_2O$')
        ax0.plot(obj.temp,-obj.ddt_C, label='DWF')
        ax0.plot(obj.temp,obj.ddt_a, label='alpha-char')
        ax0.plot(obj.temp,obj.ddt_b, label='beta-char')
        ax0.plot(obj.temp,obj.ddt_ash, label='Ash')

        ax1.plot(obj.tempK,-obj.ddt_agua, 'b', label='$H_2O$')

        h0,l0 = ax0.get_legend_handles_labels()

        ax0.set_xlabel('Temperature [C]')
        # txt = r'$\frac{d}{dt} (%s)\;[%s/s]$'
        txt = r'$\dot{\omega} (%s)\;[%s/s]$'
        ax0.set_ylabel(txt %(obj.sName,obj.dunit))
        ax1.set_xlabel('Temperature [K]')

        ax0.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax0.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax1.xaxis.set_minor_locator(AutoMinorLocator(4))

        title = str(self.Name).replace('_',r'\_')
        plt.title(title)
        plt.legend(h0,l0)
        plt.savefig(self.Name+'-'+obj.sName+'Temp-ddtsOPT.png')

    def plotDSCOPTTemp(self,obj):
        # Plot em funcao da Temperatura
        fig,ax0 = plt.subplots(figsize=(2*550*px,2*460*px))
        fig.subplots_adjust(right=0.87,left=0.12)

        ax1 = ax0.twiny()

        ax0.plot(obj.temp,obj.sinalNorm, 'k', label=obj.sName)
        ax0.plot(obj.temp,obj.sinal_optNorm, 'dimgray', ls=(0, (5, 10)),label=obj.sName+' - GA')

        ax1.plot(obj.tempK, obj.sinalNorm, 'k', label=obj.sName)

        h0,l0 = ax0.get_legend_handles_labels()

        ax0.set_xlabel('Temperature [C]')
        ax0.set_ylabel(obj.sName+' ['+obj.sunit+']')
        ax1.set_xlabel('Temperature [K]')

        ax0.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax0.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
        # ax2.yaxis.set_minor_locator(AutoMinorLocator(4))
        # ax2.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        # ax2.get_yaxis().get_offset_text().set_position((1.13,0))

        title = str(self.Name).replace('_',r'\_')
        plt.title(title)
        plt.legend(h0,l0)
        plt.savefig(self.Name+'-'+obj.sName+'Temp-DSCOPT.png')
