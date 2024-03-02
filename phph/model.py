import math
import numpy as np
from phph.Queue import Queue
from phph.LocalStateSpace import LocalStateSpace
from phph.SubMatrices import SubMatrices
from phph.BlockUniformization import BlockUniformization


class model:
    #evaluate the state distribution of the QBD and various
    #performance metrics of the queue 

    def __init__(self,arrivalInitDistribution,arrivalGenerator,
    serviceInitDistribution,serviceGenerator,
    servers,eps=1e-9):
        
        self.queue = Queue(arrivalInitDistribution,arrivalGenerator,
                           serviceInitDistribution,serviceGenerator,
                           servers)
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
        #precalculations
        self.__storeAllExitPhase()
        self.__storeAllProbPhasei()
        
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
        
    def waitDist(self,t,type="actual"):    
        #returns the probability of waiting
        #longer than t units of time
        if type=="actual":
            return self.__actualWaitDist(t)
        elif type=="virtual":
            return self.__virtualWaitDist(t)
            
    def probK(self,k,type="actual"):
        #returns the probability of observing k customers
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
            pr+=self.__probKPhase(k,i)*self.allExitPhase[0,i]
        return(pr)

    def __probServiceStateArrivals(self,k,j):
        #returns the probability that an
        #*arriving customer* observes
        #level k and local 'service-state' j
        pr=0
        for i in range(self.queue.nPhasesArrival()):
            pr+=self.__probServiceStatePhase(k,i,j)*self.allExitPhase[0,i]
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

    def __probServiceStatePhase(self,k,i,j):
        #returns the conditional probability that
        #the process is in level k and service in
        #local 'service-state' j when the process
        #is also in phase i of the arrival process  
        s = self.localState(k)
        ids = [m for m, sublist in enumerate(s) if sublist[1]==i]
        ids_j = ids[j] 
        lk = np.zeros((len(s),1))
        lk[ids_j] = 1
        numer = np.matmul(self.localStateDist(k),lk)
        return (numer/self.allProbPhasei[0,i])

    def __vectorProbServiceStatePhase(self,i,nSerStates,state,locStateDist):
        #returns a vector of conditional probabilities
        #for each local 'service-state' evaluating if the
        #process is in level k when the process is also in
        #phase i of the arrival process
        prvec = np.zeros((1,nSerStates))
        ids = [m for m, sublist in enumerate(state) if sublist[1]==i]
        for j in range(nSerStates):
            lk = np.zeros((len(state),1))
            lk[ids[j],0] = 1
            numer = np.matmul(locStateDist,lk)
            prvec[0,j] = numer/self.allProbPhasei[0,i]
        return (prvec)

    def __nServiceStates(self,k):
        #returns the number of 'service-states'
        #on level k
        s = self.localState(k)
        ids = [m for m, sublist in enumerate(s) if sublist[1]==0]        
        return len(ids)

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

    def __storeAllExitPhase(self):
        #returns the distribution (as a column vector) 
        #of phases that the arrival exits from.
        #Thus, each element in the distribution reflects
        #the conditional probability that an arrival
        #exited through phase i.  
        
        #create the embedded Markov chain of the PH generator
        diag= abs(np.diag(self.queue.arrivalGenerator))
        pmat = self.queue.arrivalGenerator / diag[:, np.newaxis]
        pmat = pmat + np.identity(self.queue.nPhasesArrival())
        #create a diagonal matrix of exit probabilities 
        evec = 1-np.sum(pmat,axis=1)
        pmat2 = np.zeros((self.queue.nPhasesArrival(),self.queue.nPhasesArrival()))
        np.fill_diagonal(pmat2,evec)
        
        #evaluate (y,allExitPhase) = (y,allExitPhase)*[P,Pdiag] until initial distribution (y) is zero
        y = np.copy(self.queue.arrivalInitDistribution)
        while (np.sum(y)>self.eps):
            self.allExitPhase = np.matmul(y,pmat2)
            y = np.matmul(y,pmat)
    
    def __storeAllProbPhasei(self):        
        self.allProbPhasei = np.zeros((1,self.queue.nPhasesArrival()))
        for i in range(self.queue.nPhasesArrival()):
            self.allProbPhasei[0,i] = self.__probPhase(i)

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
    
    
    def __actualWaitDist(self,t):
        
        #fundamental parameters
        l = len(self.ls.stateSpace)
        state = self.localState(self.queue.servers)
        xc = self.boundaryProb[0,(self.boundaryProb.shape[1]-l):self.boundaryProb.shape[1]]
        pw = self.probWait(type="actual")

        #construct QBD observed by arriving customers that have to wait
        gamma_s = np.matrix([[1.0]])
        T_s = np.matrix([[-1]])
        queue_s = Queue(gamma_s,T_s,self.queue.serviceInitDistribution,
                        self.queue.serviceGenerator,self.queue.servers)
        subMats_s = SubMatrices(queue_s)
        ls = LocalStateSpace(queue_s)
        ls.generateStateSpace(queue_s.servers)        
        subMats_s.createForwardMatrix(ls)
        subMats_s.createBackwardMatrix(ls)
        subMats_s.createLocalMatrix(ls)
        lMat = subMats_s.localMat+np.diag(np.sum(subMats_s.forwardMat, axis=1))
        bMat = subMats_s.backwardMat
        nSerStates = lMat.shape[0]

        #compute initial distribution
        cm=0
        ct = self.queue.servers-1
        while (pw-cm)>self.eps:
            ct += 1
            locStateDist = np.matmul(xc,np.linalg.matrix_power(self.subMats.neutsMat,(ct-self.queue.servers)))
            
            pr=np.zeros((1,nSerStates))
            for i in range(self.queue.nPhasesArrival()):
                prvec = self.__vectorProbServiceStatePhase(i,nSerStates,state,locStateDist)
                for j in range(nSerStates):
                    pr[0,j]+=prvec[0,j]*self.allExitPhase[0,i]    
            cm+=np.sum(pr)
        
        y=np.zeros((1,(ct-self.queue.servers+1)*nSerStates))
        z=0
        for k in range(self.queue.servers,ct+1):
            locStateDist = np.matmul(xc,np.linalg.matrix_power(self.subMats.neutsMat,(k-self.queue.servers)))    
            pr=np.zeros((1,nSerStates))
            for i in range(self.queue.nPhasesArrival()):
                prvec = self.__vectorProbServiceStatePhase(i,nSerStates,state,locStateDist)
                for j in range(nSerStates):
                    pr[0,j]+=prvec[0,j]*self.allExitPhase[0,i]    
            y[0,z:z+nSerStates] = pr
            z+=nSerStates
        
        #employ uniformization
        uni = BlockUniformization(bMat,lMat)
        return uni.run(y,t)
    
    
    def __virtualWaitDist(self,t):
        
        #fundamental parameters
        l = len(self.ls.stateSpace)
        state = self.localState(self.queue.servers)
        xc = self.boundaryProb[0,(self.boundaryProb.shape[1]-l):self.boundaryProb.shape[1]]
        pw = self.probWait(type="virtual")

        #construct QBD observed by arriving customers that have to wait
        gamma_s = np.matrix([[1.0]])
        T_s = np.matrix([[-1]])
        queue_s = Queue(gamma_s,T_s,self.queue.serviceInitDistribution,
                        self.queue.serviceGenerator,self.queue.servers)
        subMats_s = SubMatrices(queue_s)
        ls = LocalStateSpace(queue_s)
        ls.generateStateSpace(queue_s.servers)        
        subMats_s.createForwardMatrix(ls)
        subMats_s.createBackwardMatrix(ls)
        subMats_s.createLocalMatrix(ls)
        lMat = subMats_s.localMat+np.diag(np.sum(subMats_s.forwardMat, axis=1))
        bMat = subMats_s.backwardMat
        nSerStates = lMat.shape[0]

        #compute initial distribution
        cm=0
        ct = self.queue.servers-1
        while (pw-cm)>self.eps:
            ct += 1
            locStateDist = np.matmul(xc,np.linalg.matrix_power(self.subMats.neutsMat,(ct-self.queue.servers)))
            
            pr=np.zeros((1,nSerStates))
            for i in range(self.queue.nPhasesArrival()):
                ids = [m for m, sublist in enumerate(state) if sublist[1]==i]
                for j in range(nSerStates):
                    lk = np.zeros((len(state),1))
                    lk[ids[j],0] = 1
                    pr[0,j]+=np.matmul(locStateDist,lk)    
            cm+=np.sum(pr)
        
        y=np.zeros((1,(ct-self.queue.servers+1)*nSerStates))
        z=0
        for k in range(self.queue.servers,ct+1):
            locStateDist = np.matmul(xc,np.linalg.matrix_power(self.subMats.neutsMat,(k-self.queue.servers)))    
            pr=np.zeros((1,nSerStates))
            for i in range(self.queue.nPhasesArrival()):
                ids = [m for m, sublist in enumerate(state) if sublist[1]==i]
                for j in range(nSerStates):
                    lk = np.zeros((len(state),1))
                    lk[ids[j],0] = 1
                    pr[0,j]+=np.matmul(locStateDist,lk)
            y[0,z:z+nSerStates] = pr
            z+=nSerStates
        
        #employ uniformization
        uni = BlockUniformization(bMat,lMat)
        return uni.run(y,t)    