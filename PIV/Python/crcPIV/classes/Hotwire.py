"""
===============================================================================
                      Python code for PIV analysis
   Created by Combustion Research Center CRC at LETE - Sao Paulo, Brasil
   Laboratory of Environmental and Thermal Engineering - LETE
   Escola Politecnica da USP - EPUSP
   
===============================================================================
version:0.0 - 07/2020: Helio Villanueva
"""

from termcolor import colored
import numpy as np
from scipy import signal,fft
from glob import glob
import scipy as scp

class Hotwire(object):
    '''class containing Hotwire methods/data
    
    u, v -> instantaneous velocity data
    
    U, V -> time average velocities
    
    uL, vL -> velocity fluctuations (as function to use less ram)
    
    uu, vv, uv -> Reynolds Stress components
    '''
    def __init__(self,path):
        print(colored(' -> ','magenta')+'Initializing Hotwire class')
        self.path = path
        self.files = glob(self.path + '/*.txt')
        names = [n.strip(self.path) for n in self.files]
        self.names = [n.strip('.txt') for n in names]
        
        self.initVars()
        
        
    def initVars(self):
        self.dt = 0
        self.fs = 0
        self.tTot = 0
        
        self.Umean, self.uu, self.ul = [],[],[]
        self.Tmean, self.Tvar, self.Tstddev = [],[],[]
        
        self.corr = []
        
        self.t0 = []

        self.freq,self.psd = [],[]
        
        return 0
        
    def readRaw(self,i):
        '''Method to read time, Temperature and Velocity variables [s,C,m/s]
        '''
        try:
            t,T,U = np.load(self.path + '/' + self.names[i] + '.npy')
            print(colored(' -> ','magenta') + 
                  'Hotwire data of %s read from python files\n' %self.names[i])
        except:
            txt = 'Reading Hotwire Raw data from file: '
            print(colored(' -> ','magenta')+ txt + self.files[i])
            with open(self.files[i],'r') as f:
                raw = f.readlines()
            raw = raw[191:-1]
            t = np.zeros(len(raw))
            T = np.zeros(len(raw))
            U = np.zeros(len(raw))
        
            for d,data in enumerate(raw):
                draw = data.strip().split(',')
                t[d],T[d],U[d] = draw[0],draw[1],draw[2].strip('[\x17]')
            
            print(colored(' -> ','magenta') + 
                  'Saving Hotwire data in python format')
            np.save(self.path + '/' + self.names[i],np.stack((t,T,U)))
            
            print(colored('  --> ','magenta') + 'Done')
            
        return t, T, U
        
    def stats(self,y):
        '''Method to calculate statistics of y variable
        returns mean, variance and standard deviation
        '''
        Y = np.mean(y)
        yvar = np.mean((y - Y)**2)
        ystddev = np.sqrt(yvar)
        
        return Y, yvar, ystddev
    
    def integralTime(self,corr,dt):
        '''Method to calculate integral time scale based on correlation
        '''
        for i,v in enumerate(corr):
            if v<0:
                idx = i-1
                break
        corrI = corr[:idx]
        t0 = np.trapz(corrI,dx=dt)
        
        return t0
    
    def calc(self):
        '''Main method to calculate hotwire variables
        '''
        try:
            print(colored(' -> ','magenta') + 
                  'Trying to read Hotwire data from python file\n')
            
            hwStats = np.load(self.path+'/hotwireStats.npy')
            hwProp = np.load(self.path+'/hotwireProperties.npy')
            hwPSD = np.load(self.path+'/hotwirePSD.npy')
            hwCorr = np.load(self.path+'/hotwireCorr.npy')
            
            self.Umean,self.uu,self.ul,self.Tmean,self.Tvar,self.Tstddev,self.t0 = hwStats
            
            self.dt,self.fs,self.tTot = hwProp
            self.freq,self.psd = hwPSD
            
            self.corr = hwCorr
            
            print(colored('  --> ','magenta') + 'Done')
            
        except:
            # Inicializa as variaveis para caso tenha lido algum arquivo dentro
            # do try acima
            self.initVars()
            # loop entre todos os arquivos de resultados
            for i,name in enumerate(self.names):
                # le os dados do arquivo
                t,T,U = self.readRaw(i)
                # calculo da media, variancia e desvio padrao para:
                # velocidade
                Umean, uu, ul = self.stats(U)
                # temperatura
                Tmean, Tvar, Tstddev = self.stats(T)
                # calculo do passo de tempo e frequencia de aquisicao
                dt = t[1]
                fs = 1/dt
                # calculo da FFT
                ufft = fft(U)
                # calculo da autocorrelacao
                autocov = 0
                autocov = signal.correlate(U-Umean,U-Umean,mode='full')
                corr = autocov[len(U):]/(uu*len(U))
                # calculo do tempo integral
                t0 = self.integralTime(corr,dt)
                # calculo da densidade espectral de potencia PSD
                freq,psd = signal.welch(U,fs=fs,window='nuttall',
                                        scaling='density',
                                        nperseg=len(t[t<0.33/2])) #
                freq,psd = self.PSDFilt(freq,psd)
                # guarda resultados de cada medicao em uma lista com todos
                self.Umean.append(Umean)
                self.uu.append(uu)
                self.ul.append(ul)
                
                self.Tmean.append(Tmean)
                self.Tvar.append(Tvar)
                self.Tstddev.append(Tstddev)
                
                self.corr.append(corr)
                self.t0.append(t0)
                
                self.freq.append(freq)
                self.psd.append(psd)
    
            self.dt = dt
            self.fs = 1/self.dt
            self.tTot = t[-1]
            
            
            print(colored(' -> ','magenta') + 
                  'Saving Hotwire processed data in python format')
            # arranja resultados em unico array para salvar 
            hwVars = np.stack((self.Umean,self.uu,self.ul,self.Tmean,self.Tvar,
                               self.Tstddev,self.t0))
            hwProp = np.stack((self.dt,self.fs,self.tTot))
            hwPSD = np.stack((self.freq,self.psd))
            # salva resultados como *.npy
            np.save(self.path + '/' + 'hotwireStats',hwVars)
            np.save(self.path + '/' + 'hotwireProperties',hwProp)
            np.save(self.path + '/' + 'hotwireCorr',self.corr)
            np.save(self.path + '/' + 'hotwirePSD',hwPSD)
        
        return 0


    def PSDFilt(self,freq,psd):
        '''Filtra as PSDs 
        '''
        a = np.arange(5300,5400)
        #b = np.arange(8500,10000)
        
        psdFilt = psd[:-7000]
        freqFilt = freq[:-7000]
        
        psdFilt[a] = signal.medfilt(psdFilt[a],19)
        #psdFilt[b] = signal.medfilt(psdFilt[b],19)
            
        return freqFilt,psdFilt

    # path = '/run/media/bandeiranegra/Docs/Doutorado/Hotwire/jul-2020-Flameless/Raw'
#        self.Umean = np.mean(U)
#        self.uu = np.mean((U - self.Umean)**2)
#        self.ul = np.sqrt(self.uu)