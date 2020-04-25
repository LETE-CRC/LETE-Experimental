"""
===============================================================================
                      Python code for PIV analysis
   Created by Combustion Research Center CRC at LETE - Sao Paulo, Brasil
   Laboratory of Environmental and Thermal Engineering - LETE
   Escola Politecnica da USP - EPUSP
   
===============================================================================
version:0.0 - 02/2019: Helio Villanueva
version:1.0 - 04/2019: Helio Villanueva
version:2.0 - 05/2020: Helio Villanueva
"""

from termcolor import colored
from SingleFrameData import SingleFrameData, np
from progressbar import ProgressBar

class ReadData(SingleFrameData):
    '''
    Class to read all timesteps from Dantec Data\n
    resPath: Path of raw PIV files from Dantec\n
    '''
    def __init__(self,resPath):
        SingleFrameData.__init__(self,resPath)
        print(colored('\nAvailable variables:','magenta'))
        print(colored(self.variables,'white'))

        
    def read1VarTimeSeries(self,varName):
        '''Method to read one variable\n
        example:
            u = velObject.read1VarTimeSeries('U[m/s]')\n
            uncR = velObject.read1VarTimeSeries('UncR(m/s)[m/s]')
        '''
        
        varXnametratado = varName.replace("/","_")
        
        print(colored('\nReading variable: ','magenta')+'%s' %varName)
        
        try:
            varX = np.load(self.resPath + '/' + varXnametratado + '.npy')
            print(colored(' -> ','magenta') + 
                  'PIV data of %s read from python file\n' %varName)
            
        except:
            print(colored(' -> ','magenta') + 
                  'Generating variable matrices for 1st time')
            varX = np.zeros((self.lins, self.cols, self.Ttot))
            
            print(colored(' -> ','magenta') + 
                  'Reading PIV Frames - Variable components')
            pbar = ProgressBar()
            pbar.start()
            
            ## -- Loop over all files/times
            for time,name in enumerate(self.files):
                if time==0:
                    perc = 0.
                else:
                    perc = time/float(self.Ttot)*100.
                    
                varX[:,:,time] = self.readFrame1Variable(time,varName)
                
                pbar.update(perc)
                
            pbar.finish()
            
            print(colored(' -> ','magenta') + 
                  'Saving PIV data in python format')
            np.save(self.resPath + '/' + varXnametratado,varX)
            print(colored(' -> ','magenta') + 'Done saving')
        
        return varX
    
    def read2VarTimeSeries(self,varXname,varYname):
        '''Method to read two variables\n
        example:
            u,v = velObject.read2VarTimeSeries('U[m/s]','V[m/s]')
        '''
        
        varXnametratado = varXname.replace("/","_")
        varYnametratado = varYname.replace("/","_")
        
        print(colored('\nReading variables: ','magenta') +
              '%s & %s' %(varXname,varYname))
        
        try:
            varX = np.load(self.resPath + '/' + varXnametratado + '.npy')
            varY = np.load(self.resPath + '/' + varYnametratado + '.npy')
            print(colored(' -> ','magenta') + 
                  'PIV data of %s & %s read from python files\n' %(varXname,
                                                                   varYname))
            
        except:
            print(colored(' -> ','magenta') + 
                  'Generating variable matrices for 1st time')
            varX = np.zeros((self.lins, self.cols, self.Ttot))
            varY = np.zeros((self.lins, self.cols, self.Ttot))
            
            print(colored(' -> ','magenta') + 
                  'Reading PIV Frames - Variable components')
            pbar = ProgressBar()
            pbar.start()
            
            ## -- Loop over all files/times
            for time,name in enumerate(self.files):
                if time==0:
                    perc = 0.
                else:
                    perc = time/float(self.Ttot)*100.
                    
                varX[:,:,time],varY[:,:,time] = self.readFrameVariable(time,
                    varXname,varYname)
                
                pbar.update(perc)
                
            pbar.finish()
            
            print(colored(' -> ','magenta') + 
                  'Saving PIV data in python format')
            np.save(self.resPath + '/' + varXnametratado,varX)
            np.save(self.resPath + '/' + varYnametratado,varY)
            print(colored(' -> ','magenta') + 'Done saving')
        
        return varX,varY