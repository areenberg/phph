import math
import numpy as np
from PHPHCSolver.Queue import Queue
from PHPHCSolver.LocalStateSpace import LocalStateSpace


class SubMatrices:
    #derive sub-matrices related to the inhomogenuous and homogenuous
    #parts of the state space

    def __init__(self,queue):
        self.queue = queue


    def createFundamentalMatrices(self):
        #create the fundamental sub-matrix
        #associated with transitions in the 
        #homogenuous part of the state space
        
        #create the local state space
        ls = LocalStateSpace(self.queue)
        ls.generateStateSpace(self.queue.servers)
        
        self.createForwardMatrix(ls)
        self.createBackwardMatrix(ls)
        self.createLocalMatrix(ls)
        

    def createForwardMatrix(self,ls):
        #create the sub-matrix associated with
        #*forward* transitions in the homogenuous
        #part of the state space
        
        l = len(ls.stateSpace)
        d = (l,l)
        self.forwardMat = np.zeros(d) 
        
        for sidx in range(l):
            for jidx in range(l):
                if sidx==jidx: 
                    self.forwardMat[sidx,jidx] = self.queue.arrivalExitRates[ls.stateSpace[sidx][1]]*self.queue.arrivalInitDistribution[0,ls.stateSpace[sidx][1]]
                elif np.array_equal(ls.stateSpace[sidx][0],ls.stateSpace[jidx][0]):
                    self.forwardMat[sidx,jidx] = self.queue.arrivalExitRates[ls.stateSpace[sidx][1]]*self.queue.arrivalInitDistribution[0,ls.stateSpace[jidx][1]]


    def createBackwardMatrix(self,ls):
        #create the sub-matrix associated with
        #*backward* transitions in the homogenuous
        #part of the state space
        
        l = len(ls.stateSpace)
        d = (l,l)
        self.backwardMat = np.zeros(d) 
        
        for sidx in range(l):
            for jidx in range(l):
                sj = ls.serviceJumpOne(ls.stateSpace[sidx],ls.stateSpace[jidx])
                if sidx==jidx:
                    sp = ls.stateSpace[sidx][0]
                    nz = np.nonzero(sp)[0]
                    for i in nz:
                        self.backwardMat[sidx,jidx] += sp[i]*self.queue.serviceExitRates[i]*self.queue.serviceInitDistribution[0,i]
                elif sj[0]!=-1 and sj[1]!=-1:
                    sp = ls.stateSpace[sidx][0]
                    self.backwardMat[sidx,jidx] = sp[sj[0]]*self.queue.serviceExitRates[sj[0]]*self.queue.serviceInitDistribution[0,sj[1]] 


    def createLocalMatrix(self,ls):
        #create the sub-matrix associated with
        #*local* transitions in the homogenuous
        #part of the state space
        
        l = len(ls.stateSpace)
        d = (l,l)
        self.localMat = np.zeros(d) 
        
        #off-diagonal changes
        for sidx in range(l):
            for jidx in range(l):
                if sidx!=jidx:
                    if np.array_equal(ls.stateSpace[sidx][0],ls.stateSpace[jidx][0]): #arrival change
                        aj = ls.arrivalJumpOne(ls.stateSpace[sidx],ls.stateSpace[jidx])
                        self.localMat[sidx,jidx] = self.queue.arrivalGenerator[aj[0],aj[1]]
                    elif ls.stateSpace[sidx][1]==ls.stateSpace[jidx][1]: #service change
                        sj = ls.serviceJumpOne(ls.stateSpace[sidx],ls.stateSpace[jidx])
                        if sj[0]!=-1 and sj[1]!=-1: #feasible jump
                            sp = ls.stateSpace[sidx][0]
                            self.localMat[sidx,jidx] = sp[sj[0]]*self.queue.serviceGenerator[sj[0],sj[1]]                
        #diagonal changes
        sm = np.sum(self.localMat,axis=1) + np.sum(self.forwardMat,axis=1) + np.sum(self.backwardMat,axis=1)
        for sidx in range(l):
            self.localMat[sidx,sidx] = -sm[sidx]


    def createForwardInhomMatrix(self,i,j):
        #create a sub-matrix associated with
        #*forward* transitions from level i
        #to level j in the inhomogenuous
        #part of the state space
        
        #assumes i<j

        ls_i = LocalStateSpace(self.queue)
        ls_j = LocalStateSpace(self.queue)
        ls_i.generateStateSpace(i)
        ls_j.generateStateSpace(j)
        
        l_i = len(ls_i.stateSpace)
        l_j = len(ls_j.stateSpace)
        d = (l_i,l_j)
        mat = np.zeros(d) 
        
        for sidx in range(l_i):
            for jidx in range(l_j):
                ns = ls_i.serviceIncreaseOne(ls_i.stateSpace[sidx],ls_j.stateSpace[jidx])
                if ns!=-1:
                    mat[sidx,jidx] = self.queue.arrivalExitRates[ls_i.stateSpace[sidx][1]]*self.queue.arrivalInitDistribution[0,ls_j.stateSpace[jidx][1]]*self.queue.serviceInitDistribution[0,ns]        
        return(mat)


    def createBackwardInhomMatrix(self,i,j):
        #create a sub-matrix associated with
        #*backward* transitions from level i
        #to level j in the inhomogenuous
        #part of the state space
        
        #assumes i>j

        ls_i = LocalStateSpace(self.queue)
        ls_j = LocalStateSpace(self.queue)
        ls_i.generateStateSpace(i)
        ls_j.generateStateSpace(j)
        
        l_i = len(ls_i.stateSpace)
        l_j = len(ls_j.stateSpace)
        d = (l_i,l_j)
        mat = np.zeros(d) 
        
        for sidx in range(l_i):
            for jidx in range(l_j):
                ns = ls_i.serviceReduceOne(ls_i.stateSpace[sidx],ls_j.stateSpace[jidx])
                if ns!=-1:
                    sp = ls_i.stateSpace[sidx][0]
                    mat[sidx,jidx] = self.queue.serviceExitRates[ns]*sp[ns]        
        return(mat)


    def createLocalInhomMatrix(self,i,forwardInhomMat,backwardInhomMat):
        #create a sub-matrix associated with
        #*local* transitions in level i
        #in the inhomogenuous part of the
        #state space
        
        #assumes i>0

        ls_i = LocalStateSpace(self.queue)
        ls_i.generateStateSpace(i)

        l = len(ls_i.stateSpace)
        d = (l,l)
        mat = np.zeros(d) 

        #off-diagonal changes
        for sidx in range(l):
            for jidx in range(l):
                if sidx!=jidx:
                    if np.array_equal(ls_i.stateSpace[sidx][0],ls_i.stateSpace[jidx][0]): #arrival change
                        aj = ls_i.arrivalJumpOne(ls_i.stateSpace[sidx],ls_i.stateSpace[jidx])
                        mat[sidx,jidx] = self.queue.arrivalGenerator[aj[0],aj[1]]
                    elif ls_i.stateSpace[sidx][1]==ls_i.stateSpace[jidx][1]: #service change
                        sj = ls_i.serviceJumpOne(ls_i.stateSpace[sidx],ls_i.stateSpace[jidx])
                        if sj[0]!=-1 and sj[1]!=-1: #feasible jump
                            sp = ls_i.stateSpace[sidx][0]
                            mat[sidx,jidx] = sp[sj[0]]*self.queue.serviceGenerator[sj[0],sj[1]]                
        #diagonal changes
        sm = np.sum(mat,axis=1) + np.sum(forwardInhomMat,axis=1) + np.sum(backwardInhomMat,axis=1)
        for sidx in range(l):
            mat[sidx,sidx] = -sm[sidx]
        return(mat)
