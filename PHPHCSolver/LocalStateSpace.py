import numpy as np
from scipy.special import comb


class LocalStateSpace:
    #Class for generating a local state space at a specific level in the
    #QBD.

    def __init__(self,queue):
        self.queue = queue


    def generateStateSpace(self,l):
        #generate state space at level l

        #the local state space
        self.stateSpace = []
        #service level state space
        s = self.serviceStateSpace(l)

        #combine arrival state with service level
        #state space
        for i in range(self.queue.nPhasesArrival()):
            for j in range(self.serviceSpaceSize(l)):
                self.stateSpace.append([s[j,],i])


    def serviceSpaceSize(self,l):
        #calculate the size of the state space
        #accounting for the servers only

        if l>=self.queue.servers:
            z = comb(self.queue.servers + self.queue.nPhasesService()-1,
             self.queue.nPhasesService()-1, exact=True)
        else:
            z = comb(l + self.queue.nPhasesService()-1,
             self.queue.nPhasesService()-1, exact=True)

        return(z)

    def serviceStateSpace(self,l):
        #generate the state space accounting
        #for servers only

        if l>=self.queue.servers:
            x=self.queue.servers
        else:
            x=l

        size = self.serviceSpaceSize(l)
        d = (size,self.queue.nPhasesService())
        s = np.zeros(d)
        s[0,0] = x
        if size>1:
            smAll=0
            for i in range(1,size):
                s[i,] = s[i-1,]
                sw=1
                for j in reversed(range(1,self.queue.nPhasesService())):
                    if sw==1 and smAll<x:
                        s[i,j] = s[i-1,j]+1
                        smAll+=1
                        sw=0
                    elif sw==1:
                        smAll-=s[i-1,j]
                        s[i,j]=0
                        sw=1
                s[i,0] = x-smAll
        return(s)


    def serviceJumpOne(self,s1,s2):
        #returns start and end phases of the jumping server.
        #returns [-1,-1] if jump is infeasible.
        #s1 and s2 indicates the states (as lists) that are compared.

        if s1[1]-s2[1]==0:
            diff = s2[0]-s1[0]
            nn=0
            nneg=0
            npos=0
            nneg_idx = -1
            npos_idx = -1
            for i in range(len(diff)):
                if diff[i]==0:
                    nn+=1
                elif diff[i]==-1:
                    nneg+=1
                    nneg_idx = i
                elif diff[i]==1:
                    npos+=1
                    npos_idx = i

            if npos==1 and nneg==1 and (nn+npos+nneg)==len(diff):
                return([nneg_idx,npos_idx])
            else:
                return([-1,-1])
        else:
            return([-1,-1])

    def serviceIncreaseOne(self,s1,s2):
        #returns phase of the newly occupied server.
        #returns -1 if jump is infeasible.
        #s1 and s2 indicates the states (as lists) that are compared.

        diff = s2[0]-s1[0]
        nn=0
        npos=0
        npos_idx = -1
        for i in range(len(diff)):
            if diff[i]==0:
                nn+=1
            elif diff[i]==1:
                npos+=1
                npos_idx = i

        if npos==1 and (nn+npos)==len(diff):
            return(npos_idx)
        else:
            return(-1)
        
    def serviceReduceOne(self,s1,s2):
        #returns phase of the newly idle server.
        #returns -1 if jump is infeasible.
        #s1 and s2 indicates the states (as lists) that are compared.

        if s1[1]-s2[1]==0:
            diff = s2[0]-s1[0]
            nn=0
            nneg=0
            nneg_idx = -1
            for i in range(len(diff)):
                if diff[i]==0:
                    nn+=1
                elif diff[i]==-1:
                    nneg+=1
                    nneg_idx = i

            if nneg==1 and (nn+nneg)==len(diff):
                return(nneg_idx)
            else:
                return(-1)
        else:
            return(-1)


    def arrivalJumpOne(self,s1,s2):
        #returns the start and end phase of the jumping
        #arrival.
        #returns [-1,-1] if jump is infeasible.
        #s1 and s2 indicates the states that are compared.

        diff = s2[0]-s1[0]
        nn = 0
        for i in range(len(diff)):
            if diff[i]==0:
                nn+=1
        if nn==len(diff):
            return([s1[1],s2[1]])
        else:
            return([-1,-1])

    def noChange(self,s1,s2):
        #returns true if neither the servers nor the arrival
        #change phase
        diff = s2[0]-s1[0]
        nn = 0
        for i in range(len(diff)):
            if diff[i]==0:
                nn+=1
        if nn==len(diff) and (s1[1]-s2[1])==0:
            return(True)
        else:
            return(False)
