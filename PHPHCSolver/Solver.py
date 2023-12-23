import math
import numpy as np
from PHPHCSolver.Queue import Queue
from PHPHCSolver.LocalStateSpace import LocalStateSpace
from PHPHCSolver.SubMatrices import SubMatrices


class Solver:
    #evaluate the state distribution of the QBD and various
    #performance metrics of the queue 

    def __init__(self,queue,eps=1e-9):
        self.queue = queue
        self.eps=eps
        self.initialize()
        
    def initialize(self):
        self.ls = LocalStateSpace(self.queue)
        self.ls.generateStateSpace(self.queue.servers)
        self.subMats = SubMatrices(self.queue)
        self.subMats.createForwardMatrix(self.ls)
        self.subMats.createBackwardMatrix(self.ls)
        self.subMats.createLocalMatrix(self.ls)
        

    def solveBoundary(self):
        
        boundaryMat = self.createBoundaryMatrix()
        
        x = self.powerMethod(boundaryMat)
        x = np.transpose(x)
        #normalize
        scaler = np.sum(x[0,0:x.shape[1]-self.subMats.localMat.shape[0]]) + np.sum(np.matmul(x[0,x.shape[1]-self.subMats.localMat.shape[0]:x.shape[1]],np.linalg.inv(np.subtract(np.identity(self.subMats.neutsMat.shape[1]),self.subMats.neutsMat))))
        self.boundaryProb = (1/scaler)*x            
        
    
    def createBoundaryMatrix(self):
        
        self.subMats.createNeutsMatrix(self.eps)
        
        templs = LocalStateSpace(self.queue)
        dim=0
        for i in range(self.queue.servers+1):
            templs.generateStateSpace(i)
            dim += len(templs.stateSpace)
        boundMat = np.zeros((dim,dim))
        
        dim_i=0
        dim_j=0
        for i in range(self.queue.servers+1):
            if i==0:
                fMat = self.subMats.createForwardInhomMatrix(i,i+1)
                lMat = self.queue.arrivalGenerator
                boundMat[0:lMat.shape[0],0:lMat.shape[1]] = lMat
                boundMat[0:fMat.shape[0],lMat.shape[1]:(lMat.shape[1]+fMat.shape[1])] = fMat
                dim_i += fMat.shape[0]
            elif i==self.queue.servers:
                bMat = self.subMats.createBackwardInhomMatrix(i,i-1)
                lMat = self.subMats.createCornerMatrix()
                boundMat[dim_i:(dim_i+bMat.shape[0]),dim_j:(dim_j+bMat.shape[1])] = bMat
                boundMat[dim_i:(dim_i+lMat.shape[0]),(dim_j+bMat.shape[1]):(dim_j+bMat.shape[1]+lMat.shape[1])] = lMat
            else:    
                fMat = self.subMats.createForwardInhomMatrix(i,i+1)
                bMat = self.subMats.createBackwardInhomMatrix(i,i-1)    
                lMat = self.subMats.createLocalInhomMatrix(i,fMat,bMat)

                boundMat[dim_i:(dim_i+bMat.shape[0]),dim_j:(dim_j+bMat.shape[1])] = bMat
                boundMat[dim_i:(dim_i+lMat.shape[0]),(dim_j+bMat.shape[1]):(dim_j+bMat.shape[1]+lMat.shape[1])] = lMat
                boundMat[dim_i:(dim_i+fMat.shape[0]),(dim_j+bMat.shape[1]+lMat.shape[1]):(dim_j+bMat.shape[1]+lMat.shape[1]+fMat.shape[1])] = fMat
                
                dim_i += bMat.shape[0]
                dim_j += bMat.shape[1]
            
        return(boundMat)            
            

    def powerMethod(self,Q):
        #numerically derive the stationary
        #distribution using the power method
        
        qdiag = np.diag(Q)

        offset = 1e-3
        deltat = 1 / (np.max(np.abs(qdiag)) + offset)
        P = Q * deltat + np.identity(Q.shape[0])
        Pt = np.transpose(P)
        
        pi = np.random.rand(Q.shape[0],1)
        sm = np.sum(pi,axis=0)
        pi_new = pi / sm
            
        m=20    
        diff = 1
        while diff > self.eps:
            pi_old = np.copy(pi_new)

            for i in range(m):
                pi_new = np.matmul(Pt,pi_old)
                sm = np.sum(pi_new,axis=0)
                pi_new = pi_new/sm

            #evaluate convergence
            diff = np.max(np.abs(pi_new - pi_old) / pi_new)
            
        return(pi_new)            