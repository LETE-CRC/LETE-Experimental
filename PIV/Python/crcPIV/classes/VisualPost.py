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
version:2.0 - 05/2020: Helio Villanueva
"""

from termcolor import colored
import matplotlib.pyplot as plt
import numpy as np

class Plots(object):
    '''Class for ploting results
    '''
    def __init__(self,velObj):
        self.xcoord = velObj.xcoord
        self.ycoord = velObj.ycoord
        self.extent = [velObj.xmin,velObj.xmax,velObj.ymin,velObj.ymax]
        self.xlabel = 'Radius [mm]'
        self.ylabel = r'$z$ [mm]'
        self.interpolation = 'bicubic'
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rc('axes',linewidth=2,labelsize=18)
        plt.rc('axes.spines',top=0,right=0)
        
        plt.rc('xtick',labelsize=16)
        plt.rc('xtick.major',size=5,width=2)
        plt.rc('xtick.minor',visible=1)
        plt.rc('ytick',labelsize=16)
        plt.rc('ytick.major',size=5,width=2)
        plt.rc('ytick.minor',visible=1)
        
    def singleFramePlot(self,data,dataName,t=0,grid=False,vlim=None,
                        tstamp=False,title=None,save=None):
        '''method to plot data map
        '''
        print(colored('singleFramePlot: ','magenta') + dataName)
        if vlim:
            vmin = vlim[0]
            vmax = vlim[1]
        else:
            vmin = data[:,:,t].min()
            vmax = data[:,:,t].max()
            
        plt.figure(figsize=(4.4,6),dpi=160)
        ax = plt.gca()
        im = ax.imshow(data[:,:,t],cmap='jet',
                       interpolation=self.interpolation,extent=self.extent,
                       vmin=vmin,vmax=vmax)
            
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        ax.set_title(title, fontsize=16)
        ax.set_xticks(np.arange(self.extent[0],self.extent[1]), minor=True)
        ax.set_yticks(np.arange(self.extent[2],self.extent[3]), minor=True)
        
        if tstamp:
            ax.set_title('Time: %8.3f s' %self.timeStamp[t], fontsize=16)
        
        if grid:
            plt.grid(which='minor',color='k')
        
        cbar = ax.figure.colorbar(im)
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label(dataName,size=16,labelpad=5) #,rotation=0,y=1.05
        plt.tight_layout(pad=0.2)
        
        if save:
            print(colored(' -> saving: ','magenta') + save)
            plt.savefig(save)
        
        return 0
        
    def plotvLine(self,data,x,yname='y',xname='$z [mm]$',title=None,
                  err=None,yerr=None,CFD=None,ycorr=0,
                  R=1.,hlim=(None,None),vlim=(None,None),save=None):
        '''method to plot vertical lines
        CFD = [CFD_x*-1000,CFD_velMag]
        '''
        print(colored('plotvLine: ','magenta') + yname)
        if x > self.xcoord.max() or x < self.xcoord.min():
            print(colored(' -> plotvLine error: x out of measured bounds','red',
                          attrs=['bold']))
        
        else:                
            dl = self.getvline(data,x)
            
            if err:
                yerr = self.getvline(err[:,:,0],x)
            
            plt.figure(figsize=(6.4,5),dpi=200)
            plt.errorbar((self.ycoord[:,0,0]+ycorr)/R,dl,yerr=yerr,fmt='o',
                         ecolor='k',c='k',ms=3,capsize=2,lw=1,label='PIV')
                
            plt.xlabel(xname)
            plt.ylabel(yname)
            plt.title(title, size=18)
            plt.xlim(hlim)
            plt.ylim(vlim)
            plt.tight_layout(pad=0.5)
                    
            if CFD:
                plt.plot(CFD[0]/R,CFD[1],'k',label='CFD')
                plt.legend()
                
            if save:
                print(colored(' -> saving: ','magenta') + save)
                plt.savefig(save)
            
        return 0
    
    def plothLine(self,data,y,yname='y',xname='$r [mm]$',title=None,
                  err=None,yerr=None,CFD=None,xcorr=0,
                  R=1.,hlim=(None,None),vlim=(None,None),save=None):
        '''method to plot horizontal lines
        CFD = [CFD_x*-1000,CFD_velMag]
        '''
        print(colored('plothLine: ','magenta') + yname)
        if y > self.ycoord.max() or y < self.ycoord.min():
            print(colored(' -> plotvLine error: y out of measured bounds','red',
                          attrs=['bold']))
        
        else:
            dl = self.gethline(data,y)
            
            if err:
                yerr = self.gethline(err[:,:,0],y)
            
            plt.figure(figsize=(6.4,5),dpi=200)
            plt.errorbar((self.xcoord[0,:,0]+xcorr)/R,dl,yerr=yerr,fmt='o',
                         ecolor='k',c='k',ms=3,capsize=2,lw=1,label='PIV')
                
            plt.xlabel(xname)
            plt.ylabel(yname)
            plt.title(title, size=18)
            plt.xlim(hlim)
            plt.ylim(vlim)
            plt.tight_layout(pad=0.5)
            
            if CFD:
                plt.plot(CFD[0]/R,CFD[1],'k',label='CFD')
                plt.legend()
                
            if save:
                print(colored(' -> saving: ','magenta') + save)
                plt.savefig(save)
            
        return 0
    
    def plothLineMultiple(self,data,y,yname='y',xname='$r [mm]$',title=None,
                          err=None,yerr=None,CFD=None,R=1.,xcorr=0,
                          hlim=(None,None),vlim=(None,None),save=None):
        '''method to plot horizontal lines
        CFD = [CFD_x*-1000,CFD_velMag]
        '''
        print(colored('plothLineMultiple: ','magenta') + yname)
        plt.figure(figsize=(6.4,5),dpi=200)
        
        for i,d in enumerate(data):
            if CFD:
                plt.plot(CFD[i][0]/R,CFD[i][1],CFD[i][3],
                         label=CFD[i][2],markersize=2)
            
            if err:
                yerr = self.gethline(err[i][:,:,0],y)
            
            dl = self.gethline(d[0],y)
            plt.errorbar((self.xcoord[0,:,0]+xcorr)/R,dl,yerr=yerr,fmt=d[2],
                         ecolor='k',c='k',ms=3,capsize=2,lw=1,label=d[1])
            
        plt.legend()
        plt.xlabel(xname)
        plt.ylabel(yname)
        plt.title(title, size=18)
        plt.xlim(hlim)
        plt.ylim(vlim)
        plt.tight_layout(pad=0.5)
        
        if save:
            print(colored(' -> saving: ','magenta') + save)
            plt.savefig(save)
            
        return 0
    
    def plotvLineMultiple(self,data,x,yname='y',xname='$z [mm]$',title=None,
                          err=None,yerr=None,CFD=None,R=1.,ycorr=0,
                          hlim=(None,None),vlim=(None,None),save=None):
        '''method to plot horizontal lines
        CFD = [CFD_x*-1000,CFD_velMag]
        '''
        print(colored('plotvLineMultiple: ','magenta') + yname)
        plt.figure(figsize=(6.4,5),dpi=200)
        
        for i,d in enumerate(data):
            
            if CFD:
                plt.plot(CFD[i][0]/R,CFD[i][1],CFD[i][3],
                         label=CFD[i][2],markersize=2)
            
            if err:
                yerr = self.getvline(err[i][:,:,0],x)
            
            dl = self.getvline(d[0],x)
            plt.errorbar((self.ycoord[1:-2,0,0]+ycorr)/R,dl,yerr=yerr,fmt=d[2],
                         ecolor='k',c='k',ms=3,capsize=2,lw=1,label=d[1])
            
        plt.legend()
        plt.xlabel(xname)
        plt.ylabel(yname)
        plt.title(title, size=18)
        plt.xlim(hlim)
        plt.ylim(vlim)
        plt.tight_layout(pad=0.5)
        
        if save:
            print(colored(' -> saving: ','magenta') + save)
            plt.savefig(save)
            
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