"""
===============================================================================
                      Python code for PIV analysis
   Created by Combustion Research Center CRC at LETE - Sao Paulo, Brasil
   Laboratory of Environmental and Thermal Engineering - LETE
   Escola Politecnica da USP - EPUSP
   
===============================================================================
version:0.0 - 02/2019: Helio Villanueva
version:1.0 - 04/2019: Helio Villanueva
version:1.1 - 08/2019: Helio Villanueva
"""

from classes.ReadData import ReadData
import matplotlib.pyplot as plt
import numpy as np

class Plots(ReadData):
    '''Class for ploting results
    '''
    def __init__(self,resPath):
        ReadData.__init__(self,resPath)
        self.extent = [self.xmin,self.xmax,self.ymin,self.ymax]
        self.xlabel = 'Radius [mm]'
        self.ylabel = r'y [mm]'
        self.interpolation = 'bicubic'
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        
        
    def singleFramePlot(self,data,dataName,t=0,grid='n',vlim=[],tstamp='n'):
        '''method to plot data map
        '''
        if vlim!=[]:
            vmin = vlim[0]
            vmax = vlim[1]
        else:
            vmin = data[:,:,t].min()
            vmax = data[:,:,t].max()
            
        plt.figure(figsize=(5.5,6),dpi=150)
        ax = plt.gca()
        im = ax.imshow(data[:,:,t],cmap='jet',
                       interpolation=self.interpolation,extent=self.extent,
                       vmin=vmin,vmax=vmax)
        if tstamp!='n':
            ax.set_title('Time: %8.3f s' %self.timeStamp[t], fontsize=16)
            
        plt.xlabel(self.xlabel, fontsize=16)
        plt.ylabel(self.ylabel, fontsize=16)
        ax.set_xticks(np.arange(self.extent[0],self.extent[1]), minor=True)
        ax.set_yticks(np.arange(self.extent[2],self.extent[3]), minor=True)
        ax.tick_params(which='minor', bottom=False, left=False)
        plt.xticks(size=16)
        plt.yticks(size=16)
        
        if grid!='n':
            plt.grid(which='minor',color='k') 
        
        cbar = ax.figure.colorbar(im)
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label(dataName,size=16,labelpad=15) #,rotation=0,y=1.05
        
        
    def plothLine(self,data,y,name,err=np.array([0]),CFD=0,xcorr=0):
        '''method to plot horizontal lines
        CFD = [CFD_x*-1000,CFD_velMag]
        '''
        dl = self.gethline(data,y)
        
        if err.any()!=0:
            yerr = self.gethline(err[:,:,0],y)
        else:
            yerr = 0
        
        plt.figure(figsize=(6,6),dpi=150)
        
        plt.errorbar((self.xcoord[0,:,0]+xcorr)/50.,dl,yerr=yerr,fmt='o',
                     ecolor='k',c='k',ms=3,capsize=2,lw=1,label='PIV')
        
        if CFD!=0:
            plt.plot(CFD[0]/50.,CFD[1],'k',label='CFD')
            plt.legend()
            
        plt.xlabel('r/R', size=16)
        plt.ylabel(name, size=16)
        plt.xticks(size=16)
        plt.yticks(size=16)
        plt.title('$Y = 0.13 [m]$', size=18)
        plt.xlim(0,1)
        #plt.ylim(0,30)
        return 0
    
    def plothLineMultiple(self,data,y,name,err=[],CFD=[],xcorr=0,ylim=0):
        '''method to plot horizontal lines
        CFD = [CFD_x*-1000,CFD_velMag]
        '''
        plt.figure(figsize=(6,6),dpi=150)
        
        for i,d in enumerate(data):
            if CFD!=[]:
                plt.plot(CFD[i][0]/50.,CFD[i][1],CFD[i][3],
                         label=CFD[i][2],markersize=2)
            
            if err!=[]:
                yerr = self.gethline(err[i][:,:,0],y)
            else:
                yerr = 0
            
            dl = self.gethline(d[0],y)
            plt.errorbar((self.xcoord[0,:,0]+xcorr)/50.,dl,yerr=yerr,fmt=d[2],
                         ecolor='k',c='k',ms=3,capsize=2,lw=1,label=d[1])
            
        plt.legend()
        plt.xlabel('r/R', size=16)
        plt.ylabel(name, size=16)
        plt.xticks(size=16)
        plt.yticks(size=16)
        plt.title('$Y = 0.13 [m]$', size=18)
        plt.xlim(0,1)
        if ylim!=0:
            plt.ylim(0,70)
        return 0
    
    def gethline(self,data,y):
        '''friend function to interpolate horizontal line values from y\n
        y unit same as dantec saved\n
        hline -> linear interpolated horizontal line using 2 closest values of y
        '''
        idxd, idxu, yd, yu = self.gety_idx(y)
        
        M = (y-yd)/(yu-yd)
        
        hline = data[idxd,:] + M*(data[idxu,:] - data[idxd,:])
        
        return hline
    
    def getvline(self,data,x):
        '''friend function to interpolate vertical line values from x\n
        x unit same as dantec saved
        vline -> linear interpolated vertical line using 2 closest values of x
        '''
        idxd, idxu, xd, xu = self.getx_idx(x)
        
        M = (x-xd)/(xu-xd)
        
        vline = data[:,idxd] + M*(data[:,idxu] - data[:,idxd])

        return vline
    
    def gety_idx(self,value):
        '''friend method to get closest indexes from y value (horizontal line)
        also return y values for interpolation
        '''
        d = np.abs(self.ycoord[:,0,0] - value)
        idxd = np.argsort(d)[0]
        idxu = np.argsort(d)[1]
        yd = self.ycoord[idxd,0,0]
        yu = self.ycoord[idxu,0,0]
        
        return idxd, idxu, yd, yu
    
    def getx_idx(self,value):
        '''friend method to get closest indexes from x value (vertical line)
        also return x values for interpolation
        '''
        d = np.abs(self.xcoord[0,:,0] - value)
        idxd = np.argsort(d)[0]
        idxu = np.argsort(d)[1]
        xd = self.xcoord[0,idxd,0]
        xu = self.xcoord[0,idxu,0]
        
        return idxd, idxu, xd, xu