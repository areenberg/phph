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

    def checkParameters(self):
        #check the input parameters
        print()

    def nPhasesArrival(self):
        return(self.arrivalGenerator.shape[0])

    def nPhasesService(self):
        return(self.serviceGenerator.shape[0])

    def meanInterArrivalTime(self):
        #calculate and return the mean
        #inter-arrival time
        y = -np.matmul(self.arrivalInitDistribution,np.linalg.inv(self.arrivalGenerator)).sum()
        return(y)

    def meanInterServiceTime(self):
        #calculate and return the mean
        #inter-service time
        y = -np.matmul(self.serviceInitDistribution,np.linalg.inv(self.serviceGenerator)).sum()
        return(y)
