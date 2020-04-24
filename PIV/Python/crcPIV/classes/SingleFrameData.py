"""
===============================================================================
                      Python code for PIV analysis
   Created by Combustion Research Center CRC at LETE - Sao Paulo, Brasil
   Laboratory of Environmental and Thermal Engineering - LETE
   Escola Politecnica da USP - EPUSP
   
===============================================================================
version:0.0 - 02/2019: Helio Villanueva
version:1.0 - 04/2019: Helio Villanueva
"""

import re
import numpy as np
from glob import glob
from termcolor import colored

class SingleFrameData(object):
    '''
    Class to read data at each timestep
    
    Methods are: readFrame(time), readFrameCoordinates(time),
    readFrameVelocities(time), printCoordInfos().
    '''
    def __init__(self,resPath):
        self.resPath = resPath
        print(colored('Reading files from: ','magenta') + str(self.resPath))
        self.files = glob(resPath + '/*.dat')
        self.files.sort()
        self.Ttot = len(self.files)

        self.getInfos()
        self.calcCoordProps()
        
    def getInfos(self):
        '''Initialization function for initial infos from files
        '''
        with open(self.files[0]) as f:
            content0 = f.readlines()
            # - get size of data as n pixels (lins,cols)
            size = re.findall(r'I=(.*) J=(.*) ',content0[2])
            self.cols = int(size[0][0])
            self.lins = int(size[0][1])
            # - get timestamp on frame 0
            t0 = re.findall(r'#(.*), (.*).+s',content0[-1])
            tN0 = int(t0[0][0])
            tS0 = float(t0[0][1])
            # - get variables available from file
            self.variables = re.findall(r'"(.*)"',content0[1])
            self.variables = self.variables[0].split('" "')
            
            # - get single frame x and y coordinates
            self.xcoord = np.zeros((self.lins,self.cols,1))
            self.ycoord = np.zeros((self.lins,self.cols,1))
            self.xcoord[:,:,0],self.ycoord[:,:,0] = self.readFrameVariable(0,
                       "x (mm)[mm]","y (mm)[mm]")
            
        with open(self.files[-1]) as f:
            content1 = f.readlines()
            t1 = re.findall(r'#(.*), (.*).+s',content1[-1])
            tN1 = int(t1[0][0])
            tS1 = float(t1[0][1])
        
        if tS1==tS0: #single measurement case (vector statistics) 
            # - timestamp number from dantec
            self.timeNumber = np.array([tN0,tN1])
            # - aquisition frequency
            self.freq = 1.
            # - timestep between measurements
            self.dt = 1
            # - time stamps in secs with correction (Dantec truncate time info)
            self.timeStamp = np.array([0,0])
        else:
            # - timestamp number from dantec
            self.timeNumber = np.linspace(tN0,tN1,num=self.Ttot)
            # - aquisition frequency
            self.freq = np.round(self.Ttot/(tS1-tS0))
            # - timestep between measurements
            self.dt = 1/self.freq
            # - time stamps in secs with correction (Dantec truncate time info)
            tS1c = tS0+self.Ttot*self.dt
            self.timeStamp = np.linspace(tS0,tS1c,num=self.Ttot)    

        return 0
    
    def _readFrame_(self,time,usecols):
        '''Function to read each frame for coordinates or velocities
        '''            
        # - Read data
        data_tecplot = np.genfromtxt(self.files[time],skip_header=3,
                                    skip_footer=6,usecols=usecols)
        
        fxt = np.nan_to_num(np.flipud(data_tecplot[:,0].reshape((self.lins,
                                      self.cols))))
        fyt = np.nan_to_num(np.flipud(data_tecplot[:,1].reshape((self.lins,
                                      self.cols))))
        
        return fxt,fyt
         
    def readFrame1Variable(self,time,varXname):
        '''readFrame1Variable method
        Reads a specified variable from the .dat file for a specific timestep
        ex: varXname = "Rms U[pix]"
        '''
        varxidx = self.variables.index(varXname)
        
        usecols = (varxidx)
        
        # - Read data
        data_tecplot = np.genfromtxt(self.files[time],skip_header=3,
                                    skip_footer=6,usecols=usecols)
        
        varXt = np.nan_to_num(np.flipud(data_tecplot.reshape((self.lins,
                                                              self.cols))))
        
        return varXt
    
    def readFrameVariable(self,time,varXname,varYname):
        '''readFrameVariable method
        Reads two specified variable from the .dat file for a specific timestep
        ex: varXname = "Rms U[pix]"
        '''
        varxidx = self.variables.index(varXname)
        varyidx = self.variables.index(varYname)
        
        usecols = (varxidx,varyidx)
        
        varXt, varYt = self._readFrame_(time,usecols)
        
        return varXt,varYt
    
    def calcCoordProps(self):
        '''Function to calculate properties of the coordinates as object props
        '''
        self.dx = (self.xcoord.max() - self.xcoord.min())*0.001/self.cols
        self.dy = (self.ycoord.max() - self.ycoord.min())*0.001/self.lins
        self.xmin = self.xcoord.min()
        self.xmax = self.xcoord.max()
        self.ymin = self.ycoord.min()
        self.ymax = self.ycoord.max()
        self.Lx = self.xmax-self.xmin
        self.Ly = self.ymax-self.ymin
        return 0
    
    def printCoordTimeInfos(self):        
        domain = '----------------\n| Domain infos |\n----------------'
        time = '----------------\n|  Time infos  |\n----------------'
        xy = colored('X x Y: ','magenta')
        xcoord = colored('X coordinates [mm]: ','magenta')
        ycoord = colored('Y coordinates [mm]: ','magenta')
        Lxc = colored('Lx: ','magenta')
        Lyc = colored('Ly: ','magenta')
        xscl = colored('dX: ','magenta')
        yscl = colored('dY: ','magenta')
        ntstep = colored('Number of time steps: ','magenta')
        tstep = colored('dt: ','magenta')
        afreq = colored('Aquisition Frequency: ','magenta')
        inlast = colored('Initial x last timeStamp: ','magenta')
        Ttime = colored('Delta t: ','magenta')
        
        print(colored(domain,'blue'))
        print(xy + '%d x %d vectors' %(self.cols,self.lins))
        print(xcoord + '(%4.3f, %4.3f) '%(self.xmin,self.xmax) + Lxc + 
              '%4.3f [mm]' %self.Lx)
        print(xscl + '%4.4f mm\n' %(self.dx*1e3))
        print(ycoord + '(%4.3f, %4.3f) '%(self.ymin,self.ymax) + Lyc + 
              '%4.3f [mm]' %self.Ly)
        print(yscl + '%4.4f mm\n' %(self.dy*1e3))
        print(colored(time,'blue'))
        print(ntstep + '%d' %self.Ttot)
        print(afreq + '%5.1f kHz' %(self.freq/1000))
        print(tstep + '%8.4e s' %self.dt)
        print(inlast + '%2.4f x %2.4f s' %(self.timeStamp[0],
                                           self.timeStamp[-1]))
        print(Ttime + '%2.4f s'%(self.timeStamp[-1] - self.timeStamp[0]))
        
        return 0
    
    def calcPIVres(self, LIC, LCCD):
        '''Function to calculate PIV resolution and the smallest scale
        Lr -> smallest length scale available
        LIC -> dimension of the interrogation cell (eg 32 pixels)
        LCCD -> dimension of the CCD array (eg 1028 pixels)
        Lv -> length scale of the viewing area
        '''
        Lv = np.mean(self.Lx, self.Ly)
        Lr = (LIC/LCCD)*Lv
        return Lr