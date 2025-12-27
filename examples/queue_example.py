# -*- coding: utf-8 -*-

"""
In this example we demonstrate how to use the 'phph' package to evaluate the performance
of an E_2/H_3/C queue.   
"""

#--------------------------------
#   PREAMPLE
#--------------------------------

import phph
import numpy as np #for vectors and matrices

#--------------------------------
#   E_2/H_3/C QUEUE 
#--------------------------------

#Server capacity
servers = 5

#ARRIVAL PARAMETERS

#Initial distribution
arrivalInitDistribution = np.matrix([[1,0]])

#Phase-type generator
arrivalGenerator = np.matrix([[-12,12],
                              [0,-12]])

#SERVICE PARAMETERS

#Initial distribution
serviceInitDistribution = np.matrix([[(1/3),(1/2),(1/6)]])

#Phase-type generator
serviceGenerator = np.matrix([[-2,0,0],
                              [0,-1,0],
                              [0,0,-6]])

#CREATE THE MODEL

mdl = phph.model(arrivalInitDistribution,arrivalGenerator,
                 serviceInitDistribution,serviceGenerator,
                 servers)

#--------------------------------
#   RESULTS 
#--------------------------------

#Expected waiting time, E[W] 
print(mdl.meanWaitingTime())
#0.455024

#Expected number of customers in system, E[C]
print(mdl.meanOccupancy())
#6.896811

#Probability of waiting, P(W>0)
print(mdl.probWait())
#0.562802

#Probability of waiting longer than t=1 time units, P(W>t)
print(mdl.waitDist(1))
#0.162965