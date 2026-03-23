import math
import numpy as np


class BlockUniformization:
    """
    Implements the uniformization method for continuous-time Markov chains,
    exploiting the block matrix structure.
    """

    def __init__(self, bMat, lMat, eps=1e-9):
        """
        Initialize the BlockUniformization object.

        Parameters
        ----------
        bMat : numpy.ndarray
            Block transition matrix representing transitions between blocks.
        lMat : numpy.ndarray
            Local transition matrix (typically generator submatrix).
        eps : float, optional
            Desired numerical precision for truncation (default is 1e-9).
        """
        self.eps = eps
        self.bMat = bMat
        self.lMat = lMat

        # set the uniformization rate
        self.uniRate = self.__uniformRate()

        # create the stochastic blocks
        self.p_bmat, self.p_lmat = self.__stochMat()

    def __uniformRate(self):
        """
        Compute the uniformization rate.

        Returns
        -------
        float
            Maximum absolute value of the diagonal elements of lMat.
        """
        return np.max(np.abs(np.diag(self.lMat)))

    def __stochMat(self):
        """
        Construct the stochastic matrices used in uniformization.

        Returns
        -------
        tuple of numpy.ndarray
            (p_bmat, p_lmat) where both matrices are scaled to form
            stochastic transition matrices.
        """
        return self.bMat * (1.0 / self.uniRate), np.add(
            self.lMat * (1.0 / self.uniRate), np.identity(self.lMat.shape[0])
        )

    def __numbIter(self, t):
        """
        Compute the required number of iterations for the uniformization sum.

        Parameters
        ----------
        t : float
            Time horizon.

        Returns
        -------
        int
            Number of iterations needed to satisfy the error tolerance.
        """

        sigma = 1
        si = 1
        K = 0
        unit = self.uniRate * t
        tol = (1 - self.eps) * math.exp(unit)
        while sigma < tol:
            si = si * ((unit) / (K + 1))
            sigma = sigma + si
            K += 1
        return K

    def run(self, initDist, t):
        """
        Evaluate the cumulative probability using uniformization.

        Handles potential numerical underflow by splitting the computation
        into smaller time steps if necessary.

        Parameters
        ----------
        initDist : numpy.ndarray
            Initial probability distribution (row vector).
        t : float
            Time horizon.

        Returns
        -------
        float
            Cumulative probability of being in non-absorbing states at time t.
        """

        # evaluate risk that self.uniRate*t will cause underflow
        tUnderflow = 70.0 / self.uniRate
        steps = 1
        tvec = np.array([t])
        if t > tUnderflow:
            steps = math.ceil(t / tUnderflow)
            tvec = np.zeros(steps)
            if steps > 1:
                for i in range(steps - 1):
                    tvec[i] = tUnderflow
            tvec[steps - 1] = t - tUnderflow * (steps - 1)

        if steps > 1:
            return self.__evalInParts(initDist, steps, tvec)
        else:
            return self.__evalDirect(initDist, t)

    def __evalInParts(self, initDist, steps, tvec):
        """
        Applies the uniformization algorithm *in parts* to avoid underflow.
        Exploits the block structure returning the cumulative probability over
        the states where the process is *not* absorbed after t units of time.

        Parameters
        ----------
        initDist : numpy.ndarray
            Initial probability distribution.
        steps : int
            Number of sub-intervals.
        tvec : numpy.ndarray
            Array of time intervals for each step.

        Returns
        -------
        float
            Cumulative probability over non-absorbing states.
        """

        # initialize
        newDist = np.copy(initDist)
        y = np.copy(initDist)
        l = self.bMat.shape[0]
        nb = int(initDist.shape[1] / l)

        for stp in range(steps):
            # get number of iterations
            unit = self.uniRate * tvec[stp]
            K = self.__numbIter(tvec[stp])

            # iterate
            for k in range(1, K + 1):

                s_p_mat = np.block(
                    [[self.p_lmat * (unit / k)], [self.p_bmat * (unit / k)]]
                )

                for b in range(nb):
                    if b == (nb - 1):
                        y[0, -l:] = np.matmul(y[0, -l:], s_p_mat[:l, :])
                    else:
                        y[0, (l * b) : (l * (b + 1))] = np.matmul(
                            y[0, (l * b) : (l * (b + 2))], s_p_mat
                        )

                newDist = newDist + y
            # finalize
            newDist *= math.exp(-unit)

            if stp < (steps - 1):
                y = np.copy(newDist)

        return np.sum(newDist)

    def __evalDirect(self, initDist, t):
        """
        Applies the uniformization algorithm.
        Exploits the block structure returning the cumulative probability over
        the states where the process is *not* absorbed after t units of time

        Parameters
        ----------
        initDist : numpy.ndarray
            Initial probability distribution.
        t : float
            Time horizon.

        Returns
        -------
        float
            Cumulative probability over non-absorbing states.
        """

        # initialize
        cmp = np.sum(initDist)  # cumulated probability
        l = self.bMat.shape[0]
        nb = int(initDist.shape[1] / l)

        # get number of iterations
        unit = self.uniRate * t
        K = self.__numbIter(t)
        # iterate
        for k in range(1, K + 1):

            s_p_mat = np.block([[self.p_lmat * (unit / k)], [self.p_bmat * (unit / k)]])

            for b in range(nb):
                if b == (nb - 1):
                    initDist[0, -l:] = np.matmul(initDist[0, -l:], s_p_mat[:l, :])
                else:
                    initDist[0, (l * b) : (l * (b + 1))] = np.matmul(
                        initDist[0, (l * b) : (l * (b + 2))], s_p_mat
                    )
            cmp += np.sum(initDist)

        # finalize
        cmp *= math.exp(-unit)
        return cmp
