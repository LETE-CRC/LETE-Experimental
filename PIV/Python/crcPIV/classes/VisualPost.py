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
version:2.1 - 09/2020: Helio Villanueva
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
        self.resPath = velObj.resPath
        self.xlabel = 'Radius [mm]'
        self.ylabel = r'$z$ [mm]'
        self.interpolation = 'bicubic'
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rc('axes',linewidth=3,labelsize=24)
        plt.rc('axes.spines',top=0,right=0)
        
        plt.rc('xtick',labelsize=20)
        plt.rc('xtick.major',size=5,width=2)
        plt.rc('xtick.minor',visible=1,size=3,width=1)
        plt.rc('ytick',labelsize=20)
        plt.rc('ytick.major',size=5,width=2)
        plt.rc('ytick.minor',visible=1,size=3,width=1)
        
    def singleFramePlot(self,data,dataName,t=0,grid=False,vlim=None,cmap='jet',
                        streaml=False,glyph=False,glyphcolor='k',contour=False,
                        velComp=None,legend=True,tstamp=False,title=None,
                        save=None):
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
        ax.set_title(title, fontsize=20)
        ax.set_xticks(np.arange(self.extent[0],self.extent[1]), minor=True)
        ax.set_yticks(np.arange(self.extent[2],self.extent[3]), minor=True)
        
        if tstamp:
            ax.set_title('Time: %8.4f s' %(self.timeStamp[t]-self.timeStamp[0]),
                         fontsize=20)
        
        if grid:
            ax.grid(which='minor',color='k')

        # define X,Y,U,V variables for use in different plots if required
        if streaml or glyph:
            if velComp==None:
                txt = 'ERROR: velComp=[U,V] needed to use streaml or glyph'
                print(colored(txt,'red'))
                
            X = self.xcoord[:,:,0]
            Y = self.ycoord[:,:,0]
            magVel = np.sqrt(velComp[0]**2 + velComp[1]**2)
            
            try:
                U = velComp[0][:,:,t]
                V = velComp[1][:,:,t]
            except:
                U = velComp[0][:,:,0]
                V = velComp[1][:,:,0]
            
        # streamlines
        if streaml:
            lw = magVel/magVel.max()
            lw[lw<0.35] = 0.35
            ax.streamplot(X,Y,U,V,
                          density=0.8,linewidth=lw[:,:,0],color='k',
                          arrowstyle='->')
        # vector field
        if glyph:
            N = glyph
            Xq = X[::N,::N]
            Yq = Y[::N,::N]
            Uq = U[::N,::N]
            Vq = V[::N,::N]
            Q = ax.quiver(Xq,Yq,Uq,Vq,color=glyphcolor,width=0.01,headwidth=3,
                          headlength=7)
            ax.quiverkey(Q, 0.7, 0.98, magVel.max(),
                         r'$%3.0f \frac{m}{s}$' %magVel.max(),
                         color=glyphcolor, labelpos='E', coordinates='figure')
        
        # contour lines
        if contour:
            T = contour[0][:,:,t]
            origin = 'upper'
            levels = contour[1]
            CS = ax.contour(T,levels,colors=('k',),linewidths=(2,),
                  origin=origin,extent=self.extent)
            ax.clabel(CS, fmt='%2.1f', colors='w', fontsize=14)
        
        if legend:
            cbar = ax.figure.colorbar(im)
            cbar.ax.tick_params(labelsize=16)
            cbar.set_label(dataName,size=20,labelpad=5) #,rotation=0,y=1.05
            
        plt.tight_layout(pad=0.2)
        
        if save:
            print(colored(' -> saving: ','magenta') + save)
            plt.savefig(save)
        
        return 0
        
    def multiplePlots(self,pos,data,dataName,t=0,grid=False,streaml=False,
                      glyph=False,glyphcolor='k',velComp=False,vlim=None,
                      cmap=None,legend=True,tstamp=False,title=None,save=None):
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
            ax.set_title(title[i], fontsize=20)
            ax.set_xticks(np.arange(self.extent[0],self.extent[1]), minor=True)
            ax.set_yticks(np.arange(self.extent[2],self.extent[3]), minor=True)
            
            if tstamp:
                ax.set_title('Time: %8.3f s' %self.timeStamp[t], fontsize=20)
            
            if grid[i]:
                ax.grid(which='minor',color='k')
            
            # define X,Y,U,V variables for use in different plots if required
            if streaml or glyph:
                if velComp==None:
                    txt = 'ERROR: velComp=[U,V] needed to use streaml or glyph'
                    print(colored(txt,'red'))
                    
                X = self.xcoord[:,:,0]
                Y = self.ycoord[:,:,0]
                magVel = np.sqrt(velComp[0]**2 + velComp[1]**2)
                    
                try:
                    U = velComp[0][:,:,t]
                    V = velComp[1][:,:,t]
                except:
                    U = velComp[0][:,:,0]
                    V = velComp[1][:,:,0]
            
            # streamlines
            if streaml[i]:
                lw = magVel/magVel.max()
                lw[lw<0.4] = 0.4
                ax.streamplot(X,Y,U,V,density=0.9,linewidth=lw[:,:,0],
                              color='k',arrowstyle='->')
            # vector field
            if glyph[i]:
                N = glyph[i]
                Xq = X[::N,::N]
                Yq = Y[::N,::N]
                Uq = U[::N,::N]
                Vq = V[::N,::N]
                Q = ax.quiver(Xq, Yq, Uq, Vq,color=glyphcolor,width=0.01,
                              headwidth=3,headlength=7)
                ax.quiverkey(Q, 0.7, 0.98, magVel.max(),
                             r'$%3.0f \frac{m}{s}$' %magVel.max(),labelpos='E',
                             color=glyphcolor, coordinates='figure')
                
            if legend[i]:
                cbar = ax.figure.colorbar(im)
                cbar.ax.tick_params(labelsize=16)
                cbar.set_label(dataName[i],size=20,labelpad=5) #,rotation=0,y=1.05
            
        plt.tight_layout(pad=0.2)
        
        if save:
            print(colored(' -> saving: ','magenta') + save)
            plt.savefig(save)
        
        return 0
    
    def PIVvideo(self,var,dirVideo,ntStep,videoName,varName,fps=10,vlim=None,
                 glyph=False,glyphcolor='k',objs=False,cmap='jet'):
        '''method to generate video from PIV arrays
        ntStep: number of time steps to save
        '''
        if not os.path.exists(dirVideo):
            os.makedirs(dirVideo)
        
        for t in range(ntStep):
            self.singleFramePlot(var,varName,cmap=cmap,legend=1,
                             t=t, grid=0, title=' ', tstamp=1, vlim=vlim,
                             glyph=glyph,glyphcolor=glyphcolor,objs=objs,
                             save=dirVideo + '%05d.png' %t)
            plt.close()
        
        os.chdir(dirVideo)
        subprocess.call(['ffmpeg','-framerate',str(fps),'-i', '%05d.png', '-r',
                         '60', '-pix_fmt', 'yuv420p',videoName, '-y'])
        
        for file_name in glob.glob(dirVideo+"*.png"):
                os.remove(file_name)    
        
        return 0
    
    def plotvLine(self,data,x,yname='y',xname='$z [mm]$',title=None,err=None,
                  xout=(None,None),yerr=None,CFD=None,ycorr=0,Unorm=None,R=1.,
                  U0=1.,hlim=(None,None),vlim=(None,None),
                  expFluent=None,save=None):
        '''method to plot vertical lines
        CFD = [CFD_x*-1000,CFD_velMag]
        '''
        print(colored('plotvLine: ','magenta') + yname)
        if x > self.xcoord.max() or x < self.xcoord.min():
            print(colored(' -> plotvLine error: x out of measured bounds','red',
                          attrs=['bold']))
        
        else:                
            dl = self.getvline(data,x)
            
            if xout:
                start = xout[0]
                stop = xout[1]
            
            if err.any():
                yerr = self.getvline(err[:,:,0],x)[start:stop]
            
            if Unorm:
                U0 = dl[0]
                yname='<U>/U0'
            
            _x_ = (self.ycoord[start:stop,0,0]+ycorr)/R
            _y_ = dl[start:stop]/U0
            
            plt.figure(figsize=(6.4,5),dpi=200)
            plt.errorbar(_x_,_y_,yerr=yerr,fmt='o',ecolor='k',c='k',ms=3,lw=1,
                         capsize=2,label='PIV')
                
            plt.xlabel(xname)
            plt.ylabel(yname)
            plt.title(title, size=18)
            plt.xlim(hlim)
            plt.ylim(vlim)
            plt.tight_layout(pad=0.5)
                    
            if CFD:
                plt.plot(CFD[0],CFD[1],'k',label='CFD')
                plt.legend()
                
            if save:
                print(colored(' -> saving: ','magenta') + save)
                plt.savefig(save)
            
            if expFluent:
                self.saveFluentXY(_x_/1000,_y_,yname)
                
        return 0
    
    def plothLine(self,data,y,yname='y',xname='$r [mm]$',title=None,
                  err=None,yerr=None,CFD=None,xcorr=0,Unorm=None,expFluent=None,
                  R=1.,U0=1.,hlim=(None,None),vlim=(None,None),save=None):
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
            
            if Unorm:
                U0 = dl.max()
                yname=r'$<U>/U_0$'
                
            _x_ = (self.xcoord[0,:,0]+xcorr)/R
            _y_ = dl/U0
            
            plt.figure(figsize=(6.4,5),dpi=200)
            plt.errorbar(_x_,_y_,yerr=yerr,fmt='o',ecolor='k',c='k',ms=3,lw=1,
                         capsize=2,label='PIV')
                
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
                
            if expFluent:
                self.saveFluentXY(_x_,_y_,yname)
            
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
        '''friend function to get closest indexes from y value (horizontal line)
        also return y values for interpolation
        '''
        d = np.abs(self.ycoord[:,0,0] - value)
        idxd = np.argsort(d)[0]
        idxu = np.argsort(d)[1]
        yd = self.ycoord[idxd,0,0]
        yu = self.ycoord[idxu,0,0]
        
        return idxd, idxu, yd, yu
    
    def getx_idx(self,value):
        '''friend function to get closest indexes from x value (vertical line)
        also return x values for interpolation
        '''
        d = np.abs(self.xcoord[0,:,0] - value)
        idxd = np.argsort(d)[0]
        idxu = np.argsort(d)[1]
        xd = self.xcoord[0,idxd,0]
        xu = self.xcoord[0,idxu,0]
        
        return idxd, idxu, xd, xu
    
    def saveFluentXY(self,x,y,var):
        '''friend function to save XY table file for fluent plot
        '''
        file = self.resPath + '/' + var + '.xy'
        HEADER = '(title \"%s\")\n' %(var)
        HEADER += '(labels \"Position\" \"%s\")\n\n' %(var)
        HEADER += '((xy/key/label \"PIV\")'
        
        table = np.column_stack((x, y))
        
        np.savetxt(file,table,header=HEADER,footer=')\n',delimiter='\t',
                  fmt=['%.5e','%.5e'],comments='')
        
        return 0