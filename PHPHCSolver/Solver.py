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
        
        #create matrices
        self.ls = LocalStateSpace(self.queue)
        self.ls.generateStateSpace(self.queue.servers)
        self.subMats = SubMatrices(self.queue)
        self.subMats.createForwardMatrix(self.ls)
        self.subMats.createBackwardMatrix(self.ls)
        self.subMats.createLocalMatrix(self.ls)
        
        #solve the boundary probabilities
        self.__solveBoundary(method="gauss")

    def meanQueueLength(self):
        #returns the mean number of
        #customers that are waiting
        l = len(self.ls.stateSpace)
        meanQueue = np.sum(np.matmul(self.localStateDist(self.queue.servers+1),np.linalg.matrix_power(np.subtract(np.identity(l),self.subMats.neutsMat),-2)))
        return(meanQueue)
    
    def meanWaitingTime(self):
        #returns the mean waiting time
        #as observed by the customers
        meanWait = self.meanQueueLength()*self.queue.meanInterArrivalTime()
        return(meanWait)

    def meanOccupancy(self):
        #returns the mean number
        #of customers in the system
        meanOcc = self.meanQueueLength()+(self.queue.meanInterServiceTime()/self.queue.meanInterArrivalTime())
        return(meanOcc)

    def meanResponse(self):
        #returns the mean response time
        resp = self.meanOccupancy()*self.queue.meanInterArrivalTime()
        return(resp)
    
    def probWait(self,type="actual"):
        #returns the probability of waiting
        if type=="actual": #observed by the customers
            p=0
            for k in range(self.queue.servers):
                p+=self.__probKArrivals(k)
            pw=p.item()    
            return(1-pw)
        elif type=="virtual": #observed by Poisson arrivals
            l = len(self.ls.stateSpace)
            return(1-np.sum(self.boundaryProb[0,0:(self.boundaryProb.shape[1]-l)]))
    
    def probK(self,k,type="actual"):
        #returns the probability of k customers
        #in the system
        if type=="actual": #observed by the customers
            return(self.__probKArrivals(k))
        elif type=="virtual": #observed by Poisson arrivals
            return(np.sum(self.localStateDist(k)))
        
    def __probKArrivals(self,k):
        #returns the probability that an
        #*arriving customer* observes
        #k customers in the system
        pr=0
        for i in range(self.queue.nPhasesArrival()):
            pr+=self.__probKPhase(k,i)*self.__probExitPhase(i)
        return(pr)

    def __probKPhase(self,k,i):
        #returns the conditional probability that
        #the process is in level k when the process
        #is also in phase i of the arrival process  
        s = self.localState(k)
        ids = [m for m, sublist in enumerate(s) if sublist[1]==i]
        lk = np.zeros((len(s),1))
        for idx in ids:
            lk[idx] = 1
        numer = np.matmul(self.localStateDist(k),lk)
        return (numer/self.__probPhase(i))

    def __probPhase(self,i):
        #returns the (unconditional) probability that
        #the arrival process is in phase i  
        
        #calculate inhomogeneous part of state space
        beta=0
        for j in range(self.queue.servers):
            s = self.localState(j)
            ids = [m for m, sublist in enumerate(s) if sublist[1]==i]
            lk = np.zeros((len(s),1))
            for idx in ids:
                lk[idx] = 1
            beta += np.matmul(self.localStateDist(j),lk)

        #add to the inhomogeneous part
        s = self.localState(self.queue.servers)
        ids = [m for m, sublist in enumerate(s) if sublist[1]==i]
        lk = np.zeros((len(s),1))
        for idx in ids:
            lk[idx] = 1    
        pk = beta+np.matmul(np.matmul(self.localStateDist(self.queue.servers),np.linalg.inv(np.subtract(np.identity(self.subMats.neutsMat.shape[1]),self.subMats.neutsMat))),lk)
        return(pk)        

    def __probPhaseExit(self,i):
        #returns the conditional probability
        #of exiting given that the arrival
        #process is currently in phase i
        return self.queue.arrivalExitRates[i,0]/-self.queue.arrivalGenerator[i,i]

    def __probExitPhase(self,i):
        #given that an arrival exits, returns
        #the probability that the arrival came from
        #phase i
        d=0
        for j in range(self.queue.nPhasesArrival()):
            d += self.__probPhaseExit(j)*self.__probPhase(j)
        return((self.__probPhaseExit(i)*self.__probPhase(i))/d)

    def localStateDist(self,k):
        #returns the local state distribution
        #of level k
        
        if k>=self.queue.servers:
            l = len(self.ls.stateSpace)
            xc = self.boundaryProb[0,(self.boundaryProb.shape[1]-l):self.boundaryProb.shape[1]]
            xk = np.matmul(xc,np.linalg.matrix_power(self.subMats.neutsMat,(k-self.queue.servers)))               
        elif k>0:
            ls = LocalStateSpace(self.queue)
            a = 0
            for i in range(k+1):
                ls.generateStateSpace(i)
                a += len(ls.stateSpace)
            ls.generateStateSpace(k) 
            b = len(ls.stateSpace)
            xk = self.boundaryProb[0,(a-b):a]
        elif k==0:
            ls = LocalStateSpace(self.queue)
            ls.generateStateSpace(0)
            l = len(ls.stateSpace)
            xk = self.boundaryProb[0,0:l]
        else:    
            print("Error: Level i cannot be a negative number.")
        return(xk)

    def localState(self,k):
        #returns the definition of the
        #local state space of level k
        
        ls = LocalStateSpace(self.queue)
        ls.generateStateSpace(k)
        return(ls.stateSpace)

    def __solveBoundary(self,method="gauss"):
        
        boundaryMat = self.__createBoundaryMatrix()
        
        if method=="gauss":
            #solve using gaussian elimination
            x = self.__gaussianElim(boundaryMat) 
        elif method=="power":
            #solve using the power method
            x = self.__powerMethod(boundaryMat)  
        x = np.transpose(x)
        
        #normalize
        scaler = np.sum(x[0,0:x.shape[1]-self.subMats.localMat.shape[0]]) + np.sum(np.matmul(x[0,x.shape[1]-self.subMats.localMat.shape[0]:x.shape[1]],np.linalg.inv(np.subtract(np.identity(self.subMats.neutsMat.shape[1]),self.subMats.neutsMat))))
        self.boundaryProb = (1/scaler)*x            
        
        
    def __createBoundaryMatrix(self):
        
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
            

    def __powerMethod(self,Q):
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


    def __gaussianElim(self,Q):
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
    
    
    
    
    