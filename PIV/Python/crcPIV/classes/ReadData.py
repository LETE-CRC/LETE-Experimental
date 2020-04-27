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
    
    
    def readVarTimeSeries(self,varNames=[],ReadDat=False):
        '''Method to read N variables\nReadDat force to read .dat files
        example:
            u,v,U = velObject.readVarTimeSeries(['U[m/s]','V[m/s]','length'])
        '''
        
        print(colored('\nReading variables: ','magenta') + '%s' %varNames)
        varNameCorr = []
        varList = []
        #varidxs = []
        
        for vName in varNames:
            vNCorr = vName.replace("/","_")
            varNameCorr.append(vNCorr)
            #varidxs.append(self.variables.index(vName))
        
            try:
                varList.append(np.load(self.resPath + '/' + vNCorr + '.npy'))
                print(colored(' -> ','magenta') + 
                      'PIV data of %s read from python files\n' %vName)
                
            except:
                print(colored(' -> ','magenta') + 
                      'Generating variable matrices for 1st time')
                ReadDat=True
        
        
        if ReadDat:
            print(colored(' -> ','magenta') + 
                  'Reading PIV Frames - Variable components')
            
            varS = np.zeros((self.lins, self.cols, len(varNames), self.Ttot))
        
            pbar = ProgressBar()
            pbar.start()
            
            ## -- Loop over all files/times
            for time,name in enumerate(self.files):
                if time==0:
                    perc = 0.
                else:
                    perc = time/float(self.Ttot)*100.
                    
                varS[:,:,:,time] = self.readFrameNVariables(time,varNames)
                
                pbar.update(perc)
                
            varS = varS.transpose(1,0,2,3)
            pbar.finish()
            
            print(colored(' -> ','magenta') + 
                  'Saving PIV data in python format')
            
            for i,name in enumerate(varNameCorr):
                np.save(self.resPath + '/' + name,varS[i,...])
    
            print(colored(' -> ','magenta') + 'Done saving\n')
        else:
            varS = np.array(varList)
        
        return varS