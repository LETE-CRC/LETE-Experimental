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
import matplotlib.gridspec as gridspec
import numpy as np
import subprocess
import glob
import os

class Plots(object):
    '''Class for ploting results
    '''
    def __init__(self,velObj):
        self.xcoord = velObj.xcoord
        self.ycoord = velObj.ycoord
        self.timeStamp = velObj.timeStamp
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
        
    def singleFramePlot(self,data,dataName,t=0,grid=False,vlim=None,cmap='jet',
                        streaml=False,glyph=False,objs=False,legend=True,
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
        im = ax.imshow(data[:,:,t],cmap=cmap,
                       interpolation=self.interpolation,extent=self.extent,
                       vmin=vmin,vmax=vmax)
            
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_title(title, fontsize=16)
        ax.set_xticks(np.arange(self.extent[0],self.extent[1]), minor=True)
        ax.set_yticks(np.arange(self.extent[2],self.extent[3]), minor=True)
        
        if tstamp:
            ax.set_title('Time: %8.4f s' %(self.timeStamp[t]-self.timeStamp[0]),
                         fontsize=16)
        
        if grid:
            ax.grid(which='minor',color='k')
        
        if streaml:
            lw = objs[1].magVel/objs[1].magVel.max()
            lw[lw<0.35] = 0.35
            ax.streamplot(objs[0].xcoord[:,:,0],
                          objs[0].ycoord[:,:,0],
                          objs[1].U[:,:,0],
                          objs[1].V[:,:,0],
                          density=0.8,linewidth=lw[:,:,0],color='k',
                          arrowstyle='->')
        
        if glyph:
            ax.quiver(objs[0].xcoord, objs[0].ycoord,
                      objs[1].U[:,:,0],objs[1].V[:,:,0])
        
        if legend:
            cbar = ax.figure.colorbar(im)
            cbar.ax.tick_params(labelsize=16)
            cbar.set_label(dataName,size=16,labelpad=5) #,rotation=0,y=1.05
            
        plt.tight_layout(pad=0.2)
        
        if save:
            print(colored(' -> saving: ','magenta') + save)
            plt.savefig(save)
        
        return 0
        
    def multiplePlots(self,pos,data,dataName,t=0,grid=False,streaml=False,
                      glyph=False,objs=False,vlim=None,cmap=None,legend=True,
                      tstamp=False,title=None,save=None):
        '''method to plot data map
        '''
        print(colored('multiplePlots: ','magenta') + str(dataName))
            
        fig = plt.figure(figsize=(4.4*pos[1],6*pos[0]),dpi=160)
        gs = gridspec.GridSpec(nrows=pos[0], ncols=pos[1])#, height_ratios=[1, 1, 2])
        
        for i,d in enumerate(gs):
            if vlim:
                vmin = vlim[i][0]
                vmax = vlim[i][1]
            else:
                vmin = data[i][:,:,t].min()
                vmax = data[i][:,:,t].max()
                
            ax = fig.add_subplot(d)
            im = ax.imshow(data[i][:,:,t],cmap=cmap[i],
                           interpolation=self.interpolation,extent=self.extent,
                           vmin=vmin,vmax=vmax)
                
            if i==0:
                ax.set_xlabel(self.xlabel)
                ax.set_ylabel(self.ylabel)
            ax.set_title(title[i], fontsize=16)
            ax.set_xticks(np.arange(self.extent[0],self.extent[1]), minor=True)
            ax.set_yticks(np.arange(self.extent[2],self.extent[3]), minor=True)
            
            if tstamp:
                ax.set_title('Time: %8.3f s' %self.timeStamp[t], fontsize=16)
            
            if grid[i]:
                ax.grid(which='minor',color='k')
            
            if streaml[i]:
                lw = objs[1].magVel/objs[1].magVel.max()
                lw[lw<0.4] = 0.4
                ax.streamplot(objs[0].xcoord[:,:,0],
                              objs[0].ycoord[:,:,0],
                              objs[1].U[:,:,0],
                              objs[1].V[:,:,0],
                              density=0.9,linewidth=lw[:,:,0],color='k',
                              arrowstyle='->')
            if glyph[i]:
                N = glyph[i]
                X = objs[0].xcoord[::N,::N,0]
                Y = objs[0].ycoord[::N,::N,0]
                U = objs[1].U[::N,::N,0]
                V = objs[1].V[::N,::N,0]
                ax.quiver(X, Y, U, V,width=0.01,headwidth=3,headlength=7)
            
            if legend[i]:
                cbar = ax.figure.colorbar(im)
                cbar.ax.tick_params(labelsize=16)
                cbar.set_label(dataName[i],size=16,labelpad=5) #,rotation=0,y=1.05
            
        #plt.tight_layout(pad=0.2)
        
        if save:
            print(colored(' -> saving: ','magenta') + save)
            plt.savefig(save)
        
        return 0
    
    def PIVvideo(self,var,dirVideo,ntStep,videoName,varName,fps=10,vlim=None,
                 cmap='jet'):
        '''method to generate video from PIV arrays
        ntStep: number of time steps to save
        '''
        if not os.path.exists(dirVideo):
            os.makedirs(dirVideo)
        
        for t in range(ntStep):
            self.singleFramePlot(var,varName,cmap=cmap,legend=1,
                             t=t, grid=0, title=' ', tstamp=1, vlim=vlim,
                             save=dirVideo + '%05d.png' %t)
            plt.close()
        
        os.chdir(dirVideo)
        subprocess.call(['ffmpeg','-framerate',str(fps),'-i', '%05d.png', '-r',
                         '60', '-pix_fmt', 'yuv420p',videoName, '-y'])
        
        for file_name in glob.glob(dirVideo+"*.png"):
                os.remove(file_name)    
        
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