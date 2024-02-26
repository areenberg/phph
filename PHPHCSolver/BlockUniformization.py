import math
import numpy as np

class BlockUniformization:
    #conducts uniformization exploiting
    #the block matrix structure

    def __init__(self,bMat,lMat,eps=1e-9):
        self.eps=eps
        self.bMat=bMat
        self.lMat=lMat
        
        #set the uniformization rate
        self.uniRate = self.__uniformRate()
          
        #create the stochastic blocks 
        self.p_bmat,self.p_lmat = self.__stochMat()
         
    def __uniformRate(self):
        #returns the uniformization rate
        return np.max(np.abs(np.diag(self.lMat)))
    
    def __stochMat(self):
        return self.bMat*(1.0/self.uniRate), np.add(self.lMat*(1.0/self.uniRate),np.identity(self.lMat.shape[0]))
        
    def __numbIter(self,t):
        #returns the required number of
        #iterations
        sigma = 1
        si = 1
        K = 0
        unit = self.uniRate*t
        tol = (1-self.eps)*math.exp(unit)
        while sigma<tol:
            si = si*((unit)/(K+1))
            sigma = sigma + si
            K += 1
        return K;        
        
    def run(self,initDist,t):
        #evaluate cumulative probability
        #using uniformization
                
        #evaluate risk that self.uniRate*t will cause underflow
        tUnderflow = 70.0/self.uniRate
        steps = 1
        tvec = np.array([t]) 
        if t>tUnderflow:
            steps = math.ceil(t/tUnderflow)
            tvec = np.zeros(steps)
            if steps>1:
                for i in range(steps-1):
                    tvec[i] = tUnderflow
            tvec[steps-1] = t - tUnderflow*(steps-1)
        
        if steps>1:
            return self.__evalInParts(initDist,steps,tvec)    
        else:
            return self.__evalDirect(initDist,t) 
            
    def __evalInParts(self,initDist,steps,tvec):
        #Applies the uniformization algorithm
        #*in parts* to avoid underflow.
        #Exploits the block structure returning the
        #cumulative probability over the states where the
        #process is *not* absorbed after t units
        #of time
        
        #initialize
        newDist = np.copy(initDist)
        y = np.copy(initDist)
        l = self.bMat.shape[0]
        nb = int(initDist.shape[1]/l)
        
        for stp in range(steps):
            #get number of iterations
            unit = self.uniRate*tvec[stp]
            K = self.__numbIter(tvec[stp])
                    
            #iterate
            for k in range(1,K+1):
                
                s_p_mat = np.block([[(self.p_lmat*(unit/k))],
                                    [(self.p_bmat*(unit/k))]])

                for b in range(nb):
                    if b==(nb-1):
                        y[0,-l:] = np.matmul(y[0,-l:],s_p_mat[:l,:])
                    else:
                        y[0,(l*b):(l*(b+1))] = np.matmul(y[0,(l*b):(l*(b+2))],s_p_mat)
                
                newDist = newDist+y            
            #finalize
            newDist *= math.exp(-unit)        
            
            if stp<(steps-1):
                y = np.copy(newDist)
                
        return np.sum(newDist)
                
    def __evalDirect(self,initDist,t):
        #Applies the uniformization algorithm.
        #Exploits the block structure returning the
        #cumulative probability over the states where the
        #process is *not* absorbed after t units
        #of time
        
        #initialize
        cmp = np.sum(initDist) #cumulated probability
        l = self.bMat.shape[0]
        nb = int(initDist.shape[1]/l)
        
        #get number of iterations
        unit = self.uniRate*t
        K = self.__numbIter(t)        
        #iterate        
        for k in range(1,K+1):
                
            s_p_mat = np.block([[(self.p_lmat*(unit/k))],
                                [(self.p_bmat*(unit/k))]])
                
            for b in range(nb):
                if b==(nb-1):
                    initDist[0,-l:] = np.matmul(initDist[0,-l:],s_p_mat[:l,:])
                else:
                    initDist[0,(l*b):(l*(b+1))] = np.matmul(initDist[0,(l*b):(l*(b+2))],s_p_mat)
            cmp += np.sum(initDist)            
            
        #finalize
        cmp *= math.exp(-unit)        
        return cmp
