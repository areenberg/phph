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
        

    def solveBoundary(self,method="gauss"):
        
        boundaryMat = self.createBoundaryMatrix()
        
        if method=="gauss":
            #solve using gaussian elimination
            x = self.gaussianElim(boundaryMat) 
        elif method=="power":
            #solve using the power method
            x = self.powerMethod(boundaryMat)  
        x = np.transpose(x)
        
        #normalize
        scaler = np.sum(x[0,0:x.shape[1]-self.subMats.localMat.shape[0]]) + np.sum(np.matmul(x[0,x.shape[1]-self.subMats.localMat.shape[0]:x.shape[1]],np.linalg.inv(np.subtract(np.identity(self.subMats.neutsMat.shape[1]),self.subMats.neutsMat))))
        self.boundaryProb = (1/scaler)*x            
        
        
    def createBoundaryMatrix(self):
        
        self.subMats.createNeutsMatrix(self.eps,method="logred")
        
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
        #derive a numerical solution to the
        #stationary distribution using the power method
        
        #create the P matrix
        qdiag = np.diag(Q)
        offset = 1e-3
        deltat = 1 / (np.max(np.abs(qdiag)) + offset)
        P = Q * deltat + np.identity(Q.shape[0])
        Pt = np.transpose(P)
        
        #initialize with random solution
        pi = np.random.rand(Q.shape[0],1)
        sm = np.sum(pi,axis=0)
        pi_new = pi / sm
            
        diff = 1
        while diff > self.eps:   
            pi_old = np.copy(pi_new)
            #update
            pi_new = np.matmul(Pt,pi_old)
            #normalize
            sm = np.sum(pi_new,axis=0)
            pi_new = pi_new/sm
            #evaluate convergence
            diff = np.max(np.abs(pi_new - pi_old) / pi_new)
            
        return(pi_new)


    def gaussianElim(self,Q):
        #derive the exact solution to the stationary
        #distribution using Guassian elimination
        
        A = np.transpose(Q)
        
        #reduction
        for i in range(A.shape[0]-1):
            for j in range(i+1,A.shape[0]):
                mult = -A[j,i]/A[i,i]
                A[j,i] = 0
                for k in range(i+1,A.shape[1]):    
                    A[j,k] += mult*A[i,k]

        #backsubstitution
        pi = np.zeros((A.shape[0],1))
        pi[-1,0] = 1
        for i in range(A.shape[0]-2,-1,-1):
            sm=0
            for j in range(i+1,A.shape[0]):
                sm+=A[i,j]*pi[j,0]
            pi[i] = -sm/A[i,i]
        
        #normalization
        sm = np.sum(pi,axis=0)
        pi = pi/sm
            
        return(pi)
    
    
    
    
    