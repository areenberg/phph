import math
import numpy as np

class Queue:
    #The fundamental characteristics of the queue

    def __init__(self,arrivalInitDistribution,arrivalGenerator,
    serviceInitDistribution,serviceGenerator,
    servers):

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
        #check the input parameters
        arrivalRate = 1/self.meanInterArrivalTime()
        serviceRate = 1/self.meanInterServiceTime()
        if arrivalRate>=serviceRate*self.servers:
            self.feasible=False
            print("Error: The model is infeasible since arrivalRate > serviceRate x servers")
            print("arrivalRate / (serviceRate x servers) =",arrivalRate/(serviceRate*self.servers))        

    def nPhasesArrival(self):
        return(self.arrivalGenerator.shape[0])

    def nPhasesService(self):
        return(self.serviceGenerator.shape[0])

    def meanInterArrivalTime(self):
        #calculate and return the mean
        #inter-arrival time
        y = -np.matmul(self.arrivalInitDistribution,np.linalg.inv(self.arrivalGenerator)).sum()
        return(y)

    def varianceInterArrivalTime(self):
        #calculate and return the variance
        #inter-arrival time
        y = 2*(np.matmul(self.arrivalInitDistribution,np.linalg.matrix_power(self.arrivalGenerator,-2)).sum()) - math.pow(np.matmul(self.arrivalInitDistribution,np.linalg.inv(self.arrivalGenerator)).sum(),2)
        return(y)

    def meanInterServiceTime(self):
        #calculate and return the mean
        #inter-service time
        y = -np.matmul(self.serviceInitDistribution,np.linalg.inv(self.serviceGenerator)).sum()
        return(y)

    def varianceInterServiceTime(self):
        #calculate and return the variance
        #inter-service time
        y = 2*(np.matmul(self.serviceInitDistribution,np.linalg.matrix_power(self.serviceGenerator,-2)).sum()) - math.pow(np.matmul(self.serviceInitDistribution,np.linalg.inv(self.serviceGenerator)).sum(),2)
        return(y)
