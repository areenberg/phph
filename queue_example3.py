# -*- coding: utf-8 -*-

#EXAMPLE 3

#In this example we evaluate a PH/PH/c queue, where the
#inter-arrival and service time distributions reduce to
#exponential distributions. We validate the output by 
#comparing it to the analytical results of the corresponding
#M/M/c queue. 

#--------------------------------
#   PREAMPLE
#--------------------------------

import math
import numpy as np
import phph

#--------------------------------
#   FUNCTIONS 
#--------------------------------

#INSERT ANALYTICAL M/M/C FUNCTIONS ...


#--------------------------------
#   PH/PH/C QUEUE 
#--------------------------------

#Server capacity
servers = 5

#Arrival parameters

#Initial distribution
arrivalInitDistribution = np.matrix([[(1/6),(1/3),(1/2)]])

#Phase-type generator
arrivalGenerator = np.matrix([[-13,4,6],
                              [13,-19,3],
                              [3,1,-7]])

#Service parameters

#Initial distribution
serviceInitDistribution = np.matrix([[(5/9),(1/3),(1/9)]])

#Phase-type generator
serviceGenerator = np.matrix([[-3,1,1],
                              [9,-15,5],
                              [11,4,-16]])


#Create the model
mdl = phph.model(arrivalInitDistribution,arrivalGenerator,
                 serviceInitDistribution,serviceGenerator,
                 servers)


#Expected waiting time, E[W] 
print(mdl.meanWaitingTime())
#M/M/c: 0.118

#Expected number of customers in system, E[C]
print(mdl.meanOccupancy())
#M/M/c: 3.35

#Probability that the system is empty, P(C=0)
print(mdl.probEmpty())
#M/M/c: 0.0466

#Probability of waiting longer than t=1 time units, P(W>t)
print(mdl.waitDist(1))
