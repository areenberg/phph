import math
import numpy as np

class Queue:
    """
    The fundamental characteristics of the queue.
    """

    def __init__(self,arrivalInitDistribution,arrivalGenerator,
    serviceInitDistribution,serviceGenerator,
    servers):
        """
        Initialize the queue with arrival and service process parameters.

        Parameters
        ----------
        arrivalInitDistribution : list
            Initial distribution of the arrival process.
        arrivalGenerator : list
            Generator matrix of the arrival process.
        serviceInitDistribution : list
            Initial distribution of the service process.
        serviceGenerator : list
            Generator matrix of the service process.
        servers : list
            Number of servers.

        Returns
        -------
        None
            Initializes the Queue object.
        """

        #Parameters for the arrival process
        self.arrivalInitDistribution = arrivalInitDistribution
        self.arrivalGenerator = arrivalGenerator
        self.arrivalExitRates = -np.sum(arrivalGenerator,axis=1)
        #Parameters for the service process
        self.serviceInitDistribution = serviceInitDistribution
        self.serviceGenerator = serviceGenerator
        self.serviceExitRates = -np.sum(serviceGenerator,axis=1)
        #number of servers
        self.servers = servers
        #check parameters
        self.feasible=True
        self.checkParameters()
        

    def checkParameters(self):
        """
        Checks the feasibility of the input parameters.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Updates the feasibility flag and prints an error if infeasible.
        """
        arrivalRate = 1/self.meanInterArrivalTime()
        serviceRate = 1/self.meanInterServiceTime()
        if arrivalRate>=serviceRate*self.servers:
            self.feasible=False
            print("Error: The model is infeasible since arrivalRate > serviceRate x servers")
            print("arrivalRate / (serviceRate x servers) =",arrivalRate/(serviceRate*self.servers))        

    def nPhasesArrival(self):
        """
        Returns the number of phases in the arrival process.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of arrival phases.
        """
        return(self.arrivalGenerator.shape[0])

    def nPhasesService(self):
        """
        Returns the number of phases in the service process.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of service phases.
        """
        return(self.serviceGenerator.shape[0])

    def meanInterArrivalTime(self):
        """
        Calculates and returns the mean inter-arrival time.

        Parameters
        ----------
        None

        Returns
        -------
        float
            Mean inter-arrival time.
        """
        y = -np.matmul(self.arrivalInitDistribution,np.linalg.inv(self.arrivalGenerator)).sum()
        return(y)

    def varianceInterArrivalTime(self):
        """
        Calculates and returns the variance of the inter-arrival time.

        Parameters
        ----------
        None

        Returns
        -------
        float
            Variance of inter-arrival time.
        """
        y = 2*(np.matmul(self.arrivalInitDistribution,np.linalg.matrix_power(self.arrivalGenerator,-2)).sum()) - math.pow(np.matmul(self.arrivalInitDistribution,np.linalg.inv(self.arrivalGenerator)).sum(),2)
        return(y)

    def meanInterServiceTime(self):
        """
        Calculates and returns the mean inter-service time.

        Parameters
        ----------
        None

        Returns
        -------
        float
            Mean inter-service time.
        """
        y = -np.matmul(self.serviceInitDistribution,np.linalg.inv(self.serviceGenerator)).sum()
        return(y)

    def varianceInterServiceTime(self):
        """
        Calculates and returns the variance of the inter-service time.

        Parameters
        ----------
        None

        Returns
        -------
        float
            Variance of inter-service time.
        """
        y = 2*(np.matmul(self.serviceInitDistribution,np.linalg.matrix_power(self.serviceGenerator,-2)).sum()) - math.pow(np.matmul(self.serviceInitDistribution,np.linalg.inv(self.serviceGenerator)).sum(),2)
        return(y)