import numpy as np
from scipy.special import comb


class LocalStateSpace:
    """
    Class for generating and analyzing the local state space at a given level
    in a Quasi-Birth-Death (QBD) process.
    """

    def __init__(self,queue):
        """
        Initialize the LocalStateSpace object.

        Parameters
        ----------
        queue : object
            Queueing system object providing methods such as number of
            arrival/service phases and number of servers.
        """
        self.queue = queue


    def generateStateSpace(self,l):
        """
        Generate the full local state space at level l.

        Combines the service state space with the arrival phase states.

        Parameters
        ----------
        l : int
            Level in the QBD process.

        Returns
        -------
        None
            The resulting state space is stored in self.stateSpace.
        """

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
        """
        Calculate the size of the state space accounting for the servers only.

        Parameters
        ----------
        l : int
            Level in the QBD process.

        Returns
        -------
        int
            Number of possible service configurations.
        """

        if l>=self.queue.servers:
            z = comb(self.queue.servers + self.queue.nPhasesService()-1,
             self.queue.nPhasesService()-1, exact=True)
        else:
            z = comb(l + self.queue.nPhasesService()-1,
             self.queue.nPhasesService()-1, exact=True)

        return(z)

    def serviceStateSpace(self,l):
        """
        Generate the state space accounting for servers only.

        Parameters
        ----------
        l : int
            Level in the QBD process.

        Returns
        -------
        numpy.ndarray
            Matrix where each row corresponds to a service configuration.
        """
        
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
        """
        Returns start and end phases of the jumping server.
        Returns [-1,-1] if jump is infeasible.
        s1 and s2 indicates the states (as lists) that are compared.

        Parameters
        ----------
        s1 : list
            Initial state.
        s2 : list
            Target state.

        Returns
        -------
        list
            [start_phase, end_phase] if a valid jump occurs,
            [-1, -1] otherwise.
        """

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
        """
        Returns phase of the newly occupied server.
        Returns -1 if jump is infeasible.
        s1 and s2 indicates the states (as lists) that are compared.

        Parameters
        ----------
        s1 : list
            Initial state.
        s2 : list
            Target state.

        Returns
        -------
        int
            Index of the phase where a server was added,
            or -1 if infeasible.
        """

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
        """
        Returns phase of the newly idle server.
        Returns -1 if jump is infeasible.
        s1 and s2 indicates the states (as lists) that are compared.

        Parameters
        ----------
        s1 : list
            Initial state.
        s2 : list
            Target state.

        Returns
        -------
        int
            Index of the phase where a server was removed,
            or -1 if infeasible.
        """

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
        """
        Returns the start and end phase of the jumping arrival.
        Returns [-1,-1] if jump is infeasible.
        s1 and s2 indicates the states that are compared.

        Parameters
        ----------
        s1 : list
            Initial state.
        s2 : list
            Target state.

        Returns
        -------
        list
            [start_phase, end_phase] if valid,
            [-1, -1] otherwise.
        """

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
        """
        Returns true if neither the servers nor the arrival change phase.
        
        Parameters
        ----------
        s1 : list
            Initial state.
        s2 : list
            Target state.

        Returns
        -------
        bool
            True if neither service nor arrival phases change,
            False otherwise.
        """
        
        diff = s2[0]-s1[0]
        nn = 0
        for i in range(len(diff)):
            if diff[i]==0:
                nn+=1
        if nn==len(diff) and (s1[1]-s2[1])==0:
            return(True)
        else:
            return(False)