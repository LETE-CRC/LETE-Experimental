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

from classes.SingleFrameData import SingleFrameData, np, re
from progressbar import ProgressBar

class ReadData(SingleFrameData):
    '''
    Class to read all timesteps from Dantec Data\n
    resPath: Path of raw PIV files from Dantec\n
    '''
    def __init__(self,resPath):
        SingleFrameData.__init__(self,resPath)
        print(self.variables)

    
    def readVar(self,varXname,varYname):
        '''Method to read the raw data if there is no "var".npy file
        Return varX,varY
        '''
        varXnametratado = re.sub('\[.*\]','',varXname).replace(" ","")
        varYnametratado = re.sub('\[.*\]','',varYname).replace(" ","")
        
        try:
            varX = np.load(self.resPath + '/' + varXnametratado + '.npy')
            print('Reading PIV data of %s in python format\n' %varXname)
            varY = np.load(self.resPath + '/' + varYnametratado + '.npy')
            
        except:
            print('Reading PIV raw files for ' + varXname)
        
            varX,varY = self.readFrameVariable(0,varXname,varYname)
            
            print('Saving PIV data in python format')
            np.save(self.resPath + '/' + varXnametratado,varX)
            np.save(self.resPath + '/' + varYnametratado,varY)
            
        return varX,varY
        
    def read1VarTimeSeries(self,varName):
        
        varXnametratado = re.sub('\[.*\]','',varName).replace(" ","")
        
        print('Reading variable: %s' %varXnametratado)
        
        try:
            varX = np.load(self.resPath + '/' + varXnametratado + '.npy')
            print('Reading PIV data of %s in python format\n' %varName)
            
        except:
            print('Generating variable matrices')
            varX = np.zeros((self.lins, self.cols, self.Ttot))
            
            print('Reading PIV Frames - Variable components')
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
            
            print('Saving PIV data in python format')
            np.save(self.resPath + '/' + varXnametratado,varX)
            print('Done saving')
        
        return varX
    
    def read2VarTimeSeries(self,varXname,varYname):
        
        varXnametratado = re.sub('\[.*\]','',varXname).replace(" ","")
        varYnametratado = re.sub('\[.*\]','',varYname).replace(" ","")
        
        print('Reading variables: %s & %s' %(varXnametratado,varYnametratado))
        
        try:
            varX = np.load(self.resPath + '/' + varXnametratado + '.npy')
            print('Reading PIV data of %s in python format\n' %varXname)
            varY = np.load(self.resPath + '/' + varYnametratado + '.npy')
            
        except:
            print('Generating variable matrices')
            varX = np.zeros((self.lins, self.cols, self.Ttot))
            varY = np.zeros((self.lins, self.cols, self.Ttot))
            
            print('Reading PIV Frames - Variable components')
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
            
            print('Saving PIV data in python format')
            np.save(self.resPath + '/' + varXnametratado,varX)
            np.save(self.resPath + '/' + varYnametratado,varY)
            print('Done saving')
        
        return varX,varY
    
    def read1UncTimeSeries(self,varXname):
        
        varXnametratado = re.sub('\[.*\]','',varXname).replace(" ","")
        
        print('Reading variables: %s' %varXnametratado)
        
        try:
            varX = np.load(self.resPath + '/' + 'uncR' + '.npy')
            print('Reading PIV data of %s in python format\n' %varXname)
            
        except:
            print('Generating variable matrices')
            varX = np.zeros((self.lins, self.cols, self.Ttot))
            
            print('Reading PIV Frames - Variable components')
            pbar = ProgressBar()
            pbar.start()
            
            ## -- Loop over all files/times
            for time,name in enumerate(self.files):
                if time==0:
                    perc = 0.
                else:
                    perc = time/float(self.Ttot)*100.
                    
                varX[:,:,time] = self.readFrame1Variable(time,varXname)
                
                pbar.update(perc)
                
            pbar.finish()
            
            print('Saving PIV data in python format')
            np.save(self.resPath + '/' + 'uncR',varX)
            print('Done saving')
        
        return varX
    
    def readUncTimeSeries(self,varXname,varYname):
        
        varXnametratado = re.sub('\[.*\]','',varXname).replace(" ","")
        varYnametratado = re.sub('\[.*\]','',varYname).replace(" ","")
        
        print('Reading variables: %s & %s' %(varXnametratado,varYnametratado))
        
        try:
            varX = np.load(self.resPath + '/' + 'uncR' + '.npy')
            print('Reading PIV data of %s in python format\n' %varXname)
            varY = np.load(self.resPath + '/' + 'uncRpix' + '.npy')
            
        except:
            print('Generating variable matrices')
            varX = np.zeros((self.lins, self.cols, self.Ttot))
            varY = np.zeros((self.lins, self.cols, self.Ttot))
            
            print('Reading PIV Frames - Variable components')
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
            
            print('Saving PIV data in python format')
            np.save(self.resPath + '/' + 'uncR',varX)
            np.save(self.resPath + '/' + 'uncRpix',varY)
            print('Done saving')
        
        return varX,varY