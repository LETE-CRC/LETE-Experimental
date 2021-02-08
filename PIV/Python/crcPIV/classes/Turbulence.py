"""
===============================================================================
                      Python code for PIV analysis
   Created by Combustion Research Center CRC at LETE - Sao Paulo, Brasil
   Laboratory of Environmental and Thermal Engineering - LETE
   Escola Politecnica da USP - EPUSP
   
===============================================================================
version:2.1 - 12/2020: Helio Villanueva
"""

from termcolor import colored
import numpy as np
from scipy import signal


class Turb(object):
    '''class containing turbulence/statistics methods/data
    
    u, v -> instantaneous velocity data
    
    U, V -> time average velocities
    
    uL, vL -> velocity fluctuations (as function to use less ram)
    
    uu, vv, uv -> Reynolds Stress components
    '''
    def __init__(self,velObj,u=None,v=None,U=0,V=0,uu=0,vv=0,uv=0,
                 gradScheme='4thCDS'):
        print(colored(' -> ','magenta')+'Initializing turbulence calculations')
        self.dx = velObj.dx
        self.dy = velObj.dy
        self.dt = velObj.dt
        
        self.U, self.V, self.uu, self.vv, self.uv = self.readVars(u,v,U,V,
                                                                  uu,vv,uv)

        self.magVel = np.sqrt(self.U**2 + self.V**2)
        
        self.stdDevU = np.sqrt(self.uu)
        self.stdDevV = np.sqrt(self.vv)
        
        self.calcVelGrad(gradScheme=gradScheme)
        self.calcSij()
        self.ProdK = self.calcProductionK()
        print(colored(' --> ','magenta') + 'Done\n')

    def readVars(self,u,v,U,V,uu,vv,uv):
        '''
        read variables needed for calculaions

        Parameters
        ----------
        
        u, v - instantaneous velocity components
        
        U, V - time average velocity components
        
        uu, vv, uv - Reynolds Stress Tensor components

        '''
        if u is not None:
            txt = 'Processing instantaneous velocity components'
            print(colored('  - ','magenta') + txt)
            U = np.mean(u,axis=2, keepdims=True)
            V = np.mean(v,axis=2, keepdims=True)
            print(colored('  - ','magenta') + 'Avg Velocity & Avg Magnitude')
            uu, vv, uv = self.calcReStress(u,U,v,V)
            print(colored('  - ','magenta') + 'Reynolds Stress tensor')
        else:
            txt = 'Using reduced variables'
            print(colored('  - ','magenta') + txt)
            
        return U,V,uu,vv,uv
    
    def calcReStress(self,u,U,v,V):
        '''Method to calculate Reynolds stress components
        '''
        uu = np.mean((u-U)**2,axis=2, keepdims=True)
        vv = np.mean((v-V)**2,axis=2, keepdims=True)
        uv = np.mean((u-U)*(v-V),axis=2, keepdims=True)
        
        return uu, vv, uv
    
    def calcK2DPIV(self):
        '''calculates the turbulent kinetic energy for 2D PIV by assuming the
        flow to be locally isotropic
        '''
        K = 3./4 *(self.uu + self.vv)
        
        return K
    
    def calcKPIV(self):
        '''calculates the turbulent kinetic energy for 2D3C PIV.
        ****not implemented****
        '''
        return 0
    
    def calcPSD(self,u,v,scale='density'):
        '''Calculates de Power Spectrum Density in the time coordinate
        '''
        print(colored(' -> ','magenta') + 'calc Power Spectrum Density')
        
        print(colored('  - ','magenta') + 'u component')
        freqU, psdU = signal.welch(u,1/self.dt, window='nuttall',
                                   nfft=1024, average='median',scaling=scale)
        
        print(colored('  --> ','magenta') + 'Done')
        
        print(colored('  - ','magenta') + 'v component')
        
        freqV, psdV = signal.welch(v,1/self.dt, window='nuttall',
                                   nfft=1024, average='median',scaling=scale)
        print(colored('  --> ','magenta') + 'Done')
        
        print(colored(' --> ','magenta') + 'Done\n')
        
        return freqU, psdU, freqV, psdV
    
    def integralTime(self,corr):
        '''Method to calculate integral time scale based on correlation
        '''
        for i,v in enumerate(corr):
            if v<0:
                idx = i-1
                break
        corrI = corr[:idx]
        t0 = np.trapz(corrI,dx=self.dt)
        
        return t0
    
    def calcVelGrad(self,gradScheme):
        '''calculates the velocity gradient tensor
        '''
        print(colored('  - ','magenta') + 'calc Velocity gradient tensor')
        if gradScheme=='4thCDS':
            print(colored('   - ','magenta') + 'using ' + gradScheme)
            # - Fourth order CDS scheme from Ferziger Computational methods for
            # - fluid dynamics on page 44 eq 3.14
            scheme = np.array([[0,0,0,0,0],
                               [-1,8,0,-8,1],
                               [0,0,0,0,0]]).reshape(3,5,1)
            den = 12
        elif gradScheme=='3rdBDS':
            print(colored('   - ','magenta') + 'using ' + gradScheme)
            # 3rd order BDS
            scheme = np.array([[0,0,0,0,0],
                              [-1,6,-3,-2,0],
                              [0,0,0,0,0]]).reshape(3,5,1)
            den = 6
        else:
            print("ERROR: Unrecognized gradient scheme. '4thCDS' or '3rdBDS'")
            

        # - gradients on x direction
        numUx = signal.convolve(self.U,scheme, mode='same')
        numVx = signal.convolve(self.V,scheme, mode='same')
        # dU/dx
        self.grad11 = numUx/(den*self.dx)
        # dV/dx
        self.grad21 = numVx/(den*self.dx)
        
        # - gradients on y direction (BUG o menos tinha que ser um flip vertical pra funcionar o 3rdBDS)
        numUy = signal.convolve(self.U,-scheme.transpose(1,0,2), mode='same')
        numVy = signal.convolve(self.V,-scheme.transpose(1,0,2), mode='same')
        # dU/dy
        self.grad12 = numUy/(den*self.dy)
        # dV/dy
        self.grad22 = numVy/(den*self.dy)
        
        return 0
    
    def calcMagTij(self,T11,T22,T12,T21):
        '''calculate magnitude of Tij tensor for 2D2C
        '''
        TijTij = 2*(T11)**2. + 2*(T22)**2.
        TijTij += 2*(T11*T22) 
        TijTij += 3./2*(T12 + T21)**2
        
        return np.sqrt(TijTij)
    
    def calcSij(self):
        '''calculate Sij tensor for 2D2C - S11, S22, S12
        '''
        print(colored('  - ','magenta') + 'calc Sij tensor')
        self.S11 = self.grad11
        self.S22 = self.grad22
        self.S12 = 0.5*(self.grad21 + self.grad12)
        
        self.magSij = self.calcMagTij(self.grad11, self.grad22,
                                      self.grad12, self.grad21)
        
        return 0
    
    def calcProductionK(self):
        '''
        calculate the turbulence production term.

        '''
        Prod = -(self.uu*self.S11 + self.vv*self.S22 + 2*(self.uv*self.S12))
        
        return Prod
    
    def calcTauijSmagorinsky(self):
        '''calculate the modeled tauij tensor - SGS tensor based on smagorinsky
        method
        
        Smagorinsky (1963)
        '''
        txt = 'calc modeled tauij SGS tensor based on Smagorinsky method'
        txt+= ' | Smagorinsky (1963)'
        print(colored('  -> ','magenta') + txt)
        
        Cs = 0.17 # Lilly 1967 | Cheng 1997 Cs = 0.12
        delta = self.dx # window size which the vel field is spatially avg
        const = -(Cs**2.)*(delta**2.)*self.magSij
        tau11 = const*self.S11
        tau22 = const*self.S22
        tau12 = const*self.S12
        print(colored('  --> ','magenta') + 'Done')
        
        return tau11, tau22, tau12
    
    def calcTauijGradient(self):
        '''calculate the modeled tauij tensor - SGS tensor based on gradient
        method
        
        Clark et al. (1979)
        '''
        txt = 'calc modeled tauij SGS tensor based on Gradient method'
        txt += ' | Clark et al. (1979)'
        print(colored('  -> ','magenta') + txt)
        delta = self.dx
        const = (delta**2.)/12.
        tau11 = const*(self.grad11**2. + self.grad12**2.)
        tau22 = const*(self.grad21**2. + self.grad22**2.)
        tau12 = const*(self.grad21*self.grad11 + self.grad22*self.grad12)
        print(colored('  --> ','magenta') + 'Done')
        
        return tau11, tau22, tau12
    
    def calcEpsilon(self, method='smagorinsky'):
        '''calcluate the dissipation rate using smagorinsky or gradient methods
        '''
        txt = 'calc modeled Dissipation Rate - Epsilon'
        print(colored(' -> ','magenta') + txt)
        
        tau11, tau22, tau12 = ( self.calcTauijGradient() if method=='gradient' 
        else self.calcTauijSmagorinsky() )
            
        epsilon = tau11*self.S11 + tau11*self.S12 + tau12*self.S22
        epsilon += tau12*self.S11 + tau22*self.S12 + tau22*self.S22
        epsilon += 2*tau12*self.S12
        epsilon = np.abs(epsilon)
        print(colored(' --> ','magenta') + 'Done\n')
        
        return epsilon
    
    # Calc uncertainties Mean Vel, Reynolds Stress components
    def calcUncMean(self,uncR,k=3):
        '''calculate uncertainty of mean velocity field
        Parameters
        ----------
        uncR : array with uncertainties in m/s from dantec.
        k : coverage factor -> 1=68% ; 2=95%; 3=99% for gaussian distribution
        *check dantec help
        Returns
        -------
        Uncertainty of mean velocity field.

        '''
        print(colored(' -> ','magenta') + 'calc uncertainty of mean velocity')
        
        varFluct = self.calcMagTij(self.uu, self.vv,
                                   np.abs(self.uv), np.abs(self.uv))
        sigmaU = varFluct + np.mean(k*uncR**2,axis=2,keepdims=True)
        print(colored(' --> ','magenta') + 'Done\n')
        
        return np.sqrt(sigmaU/len(uncR[0,0,:]))
    
    def calcUncRe(self,uncR,Ruu,k=3):
        '''calculate uncertainty for Reynolds Stresses
        Parameters
        ----------
        uncR : array with uncertainties in m/s from dantec.
        Ruu : Reynolds Stress component
        k : coverage factor -> 1=68% ; 2=95%; 3=99% for gaussian distribution
        *check dantec help
        '''
        txt = 'calc uncertainty of Reynolds Stress component'
        print(colored(' -> ','magenta') + txt)
        
        uncRmean = np.mean(k*uncR,axis=2,keepdims=True)
        sigmaUu = np.sqrt(np.mean((k*uncR-uncRmean)**2,axis=2,keepdims=True))
        C = np.sqrt( 1 + ( sigmaUu**2/(2*uncRmean**2) ) )
        URuu = np.sqrt(Ruu**2 + ( np.sqrt(2)*sigmaUu*uncRmean*C )**2 )
        URuu *= np.sqrt(2/len(uncR[0,0,:]))
        print(colored(' --> ','magenta') + 'Done\n')
        
        return URuu
