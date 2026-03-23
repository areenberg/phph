import math
import numpy as np


class Uniformization:
    """
    Conducts uniformization using a transition rate matrix and
    an initial probability distribution.
    """

    def __init__(self, initDist, tranMat, uniRate=-1, eps=1e-9):
        """
        Initialize the Uniformization object.

        Parameters
        ----------
        initDist : list
            Initial probability distribution.
        tranMat : list
            Transition rate matrix.
        uniRate : list
            Uniformization rate (if negative, it is computed automatically).
        eps : list
            Tolerance for convergence.

        Returns
        -------
        None
            Initializes the Uniformization object.
        """
        self.eps = eps
        self.tranMat = tranMat
        self.initDist = initDist

        # set the uniformization rate
        if uniRate < 0:
            self.uniRate = self.uniformRate()
        else:
            self.uniRate = uniRate
        # create the stochastic matrix
        self.PMat = self.stochMat()

    def uniformRate(self):
        """
        Returns the uniformization rate.

        Parameters
        ----------
        None

        Returns
        -------
        float
            Uniformization rate.
        """
        return np.max(np.abs(np.diag(self.tranMat)))

    def stochMat(self):
        """
        Returns the stochastic matrix associated with the transition
        rate matrix and uniformization rate.

        Parameters
        ----------
        None

        Returns
        -------
        list
            Stochastic matrix.
        """
        return np.add(
            self.tranMat * (1.0 / self.uniRate), np.identity(self.tranMat.shape[0])
        )

    def numbIter(self, t):
        """
        Returns the required number of iterations for the uniformization algorithm.

        Parameters
        ----------
        t : list
            Time horizon.

        Returns
        -------
        int
            Number of iterations.
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

    def run(self, t):
        """
        Applies the uniformization algorithm and returns the state
        distribution after t units of time.

        Parameters
        ----------
        t : list
            Time horizon.

        Returns
        -------
        list
            State distribution at time t.
        """

        # evaluate risk that self.uniRate*t will lead to underflow
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

        # initialize
        self.newDist = np.copy(self.initDist)
        y = np.copy(self.initDist)

        for stp in range(steps):
            # get number of iterations
            unit = self.uniRate * tvec[stp]
            K = self.numbIter(tvec[stp])
            # iterate
            for k in range(1, K + 1):
                y = np.matmul(y, self.PMat * (unit / k))
                self.newDist = self.newDist + y
            # finalize
            self.newDist = self.newDist * math.exp(-unit)

            if stp < (steps - 1):
                y = np.copy(self.newDist)

        return self.newDist
