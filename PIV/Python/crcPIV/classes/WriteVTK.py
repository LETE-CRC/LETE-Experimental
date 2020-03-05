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

from classes.SingleFrameData import SingleFrameData, np
from evtk.vtk import VtkFile, VtkImageData


class WVTK(SingleFrameData):
    '''
    Class to write results in VTK format for paraview
    '''
    def __init__(self,fRes):
        SingleFrameData.__init__(self,fRes)
        
    def save2DcellVecTransientVTK(self, resPath, fileName, U, V):   
        '''function to save transient data for 2D vector field in VTK format
        '''
        nx, ny, nz = self.cols, self.lins, 1
        origin, spacing = (0.0,0.0,0.0), (self.xscale,self.yscale,0.0001)
        start, end = (0,0,0), (nx, ny, nz)
        
        print('Saving transient data into paraview format')
        for time,name in enumerate(self.fRes):
            Ut = U[:,:,time]
            Vt = V[:,:,time]
            Uvtk = np.ascontiguousarray(np.rot90(Ut,k=1, axes=(1,0)))
            Vvtk = np.ascontiguousarray(np.rot90(Vt,k=1, axes=(1,0)))
        
            w = VtkFile(resPath + '/' + fileName + str(time), VtkImageData)
            w.openGrid(start = start, end = end, origin = origin, spacing = spacing)
            w.openPiece( start = start, end = end)
            
            # Cell data
            zeroScalar = np.zeros([nx, ny, nz], dtype="float64", order='C')
            w.openData("Cell", vectors = "Velocities")
            w.addData("Velocities", (Uvtk, Vvtk, zeroScalar))
            w.closeData("Cell")
            
            w.closePiece()
            w.closeGrid()
            
            w.appendData(data = (Uvtk,Vvtk,zeroScalar))
            w.save()
        
        return 0
        
    def save2DcellVecVTK(self, resPath, fileName, U, V):
        '''Function to save single frame data for 2D vector field in VTK format
        '''
        nx, ny, nz = self.cols, self.lins, 1
        origin, spacing = (0.0,0.0,0.0), (self.xscale,self.yscale,0.0001)
        start, end = (0,0,0), (nx, ny, nz)
        
        Uvtk = np.ascontiguousarray(np.rot90(U,k=1, axes=(1,0)))
        Vvtk = np.ascontiguousarray(np.rot90(V,k=1, axes=(1,0)))
        
        w = VtkFile(resPath + '/' + fileName, VtkImageData)
        w.openGrid(start = start, end = end, origin = origin, spacing = spacing)
        w.openPiece( start = start, end = end)
        
        # Cell data
        zeroScalar = np.zeros([nx, ny, nz], dtype="float64", order='C')
        w.openData("Cell", vectors = "Velocities")
        w.addData("Velocities", (Uvtk, Vvtk, zeroScalar))
        w.closeData("Cell")
        
        w.closePiece()
        w.closeGrid()
        
        w.appendData(data = (Uvtk,Vvtk,zeroScalar))
        w.save()
        
        return 0
    
    def save2DcellReynoldsVTK(self, resPath, fileName, uu, vv, uv):
        '''Function to save single frame data for 3D vector field in VTK format
        now working for Reynolds Stress tensor 3 components
        '''
        nx, ny, nz = self.cols, self.lins, 1
        origin, spacing = (0.0,0.0,0.0), (self.xscale,self.yscale,0.0001)
        start, end = (0,0,0), (nx, ny, nz)
        
        uuvtk = np.ascontiguousarray(np.rot90(uu,k=1, axes=(1,0)))
        vvvtk = np.ascontiguousarray(np.rot90(vv,k=1, axes=(1,0)))
        uvvtk = np.ascontiguousarray(np.rot90(uv,k=1, axes=(1,0)))
        
        w = VtkFile(resPath + '/' + fileName, VtkImageData)
        w.openGrid(start = start, end = end, origin = origin, spacing = spacing)
        w.openPiece( start = start, end = end)
        
        # Cell data
        #zeroScalar = np.zeros([nx, ny, nz], dtype="float64", order='C')
        w.openData("Cell", vectors = fileName)
        w.addData(fileName, (uuvtk, vvvtk, uvvtk))
        w.closeData("Cell")
        
        w.closePiece()
        w.closeGrid()
        
        w.appendData(data = (uuvtk,vvvtk,uvvtk))
        w.save()
        
        return 0