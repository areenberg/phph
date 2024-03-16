import numpy as np
from phph.LocalStateSpace import LocalStateSpace

class SubMatrices:
    #derive sub-matrices related to the inhomogenuous and homogenuous
    #parts of the state space

    def __init__(self,queue):
        self.queue = queue


    def createNeutsMatrix(self,eps,method="logred"):
        #Derive Neut's R matrix (in continuous time) 
        
        #assumes the backward, local, and forward
        #matrices have been derived
        
        if method=="logred":
            self.logRed(eps)
        elif method=="stewart":
            self.stewart(eps)
        
        
    def logRed(self,eps):
        #derive Neut's matrix using
        #logarithmic reduction (original implementation in Matlab by B. F. Nielsen)

        #initialization
        l = self.forwardMat.shape[0]
        Iden = np.identity(l)
        B0 = np.matmul(np.linalg.inv(-self.localMat),self.forwardMat)
        B2 = np.matmul(np.linalg.inv(-self.localMat),self.backwardMat)
        S = B2
        Pi = B0
        #run
        tol = 1
        while tol>eps:
            localMatstar = Iden-np.matmul(B0,B2)-np.matmul(B2,B0)
            forwardMatstar = np.linalg.matrix_power(B0,2)
            backwardMatstar = np.linalg.matrix_power(B2,2)
            B0 = np.matmul(np.linalg.inv(localMatstar),forwardMatstar)
            B2 = np.matmul(np.linalg.inv(localMatstar),backwardMatstar)
            S = np.add(S,np.matmul(Pi,B2))
            Pi = np.matmul(Pi,B0)
            #check convergence
            xtol = np.absolute(np.subtract(np.ones((l,1)),S.sum(axis=1)))
            tol = xtol.max()
        #derive the fundamental matrices
        Gmat = S
        Umat = np.add(self.localMat,np.matmul(self.forwardMat,Gmat))
        self.neutsMat = np.matmul(self.forwardMat,np.linalg.inv(-Umat))
    
    def stewart(self,eps):
        #derive Neut's matrix using the method in:
        #W. J. Stewart (2009), "Probability, Markov Chains, Queues, and Simulation", Princeton University Press
        
        V = np.matmul(self.forwardMat,np.linalg.inv(self.localMat))
        W = np.matmul(self.backwardMat,np.linalg.inv(self.localMat))

        l = self.forwardMat.shape[0]
        self.neutsMat = np.zeros((l,l))
        Rbis = np.subtract(-V,np.matmul(np.matmul(self.neutsMat,self.neutsMat),W))
        
        while np.linalg.norm(np.subtract(self.neutsMat,Rbis),1)>eps:
            self.neutsMat = np.copy(Rbis)
            Rbis = np.subtract(-V,np.matmul(np.matmul(self.neutsMat,self.neutsMat),W))
        self.neutsMat = np.copy(Rbis)
        
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


    def createCornerMatrix(self):
        #create the sub-matrix in the
        #lower right corner of the inhomogenuous part
        #of the major transition rate matrix.
        #the corner matrix corresponds to the sub-matrix
        #at the levels that is equal to the number of 
        #servers.

        #assumes the local, backward, and Neut's
        #matrices have been derived.   

        return(np.add(self.localMat,np.matmul(self.neutsMat,self.backwardMat)))
