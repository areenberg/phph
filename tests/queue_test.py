# -*- coding: utf-8 -*-

"""
In this program, we evaluate a PH/PH/C queue, where the inter-arrival and service time distributions
reduce to exponential distributions. We validate the output by computing the difference from the results of the
corresponding M/M/C queue.
"""

#--------------------------------
#   PREAMPLE
#--------------------------------

import phph
import math
import sys
import numpy as np

#--------------------------------
#   M/M/C FUNCTIONS 
#--------------------------------

def probEmptyMMC(arrivalRate,serviceRate,servers):
    #probability that the M/M/C queue is empty
    rho = arrivalRate/(serviceRate*servers)
    pr=0
    for n in range(servers):
        pr+=math.pow(servers*rho,n)/math.factorial(n)
    pr+=(math.pow(servers*rho,servers)/math.factorial(servers))*(1/(1-rho))    
    return 1/pr

def probNOccupied(arrivalRate,serviceRate,servers,n):
    #probability of n customers in the M/M/c queue
    rho = arrivalRate/(serviceRate*servers)
    if n<servers:
        pr = (math.pow(rho*servers,n)/math.factorial(n))*probEmptyMMC(arrivalRate,serviceRate,servers)
    else:
        pr = ((math.pow(rho,n)*math.pow(servers,servers))/math.factorial(servers))*probEmptyMMC(arrivalRate,serviceRate,servers)
    return pr

def meanWaitingTimeMMC(arrivalRate,serviceRate,servers):
    #expected waiting time in the M/M/c queue
    wait = (math.pow(arrivalRate/serviceRate,servers)*serviceRate)/( math.factorial(servers-1)*math.pow(servers*serviceRate-arrivalRate,2))
    wait *= probEmptyMMC(arrivalRate,serviceRate,servers)
    return wait

def meanOccupancyMMC(arrivalRate,serviceRate,servers):
    #expected occupancy of the M/M/c queue
    return arrivalRate*(meanWaitingTimeMMC(arrivalRate,serviceRate,servers)+(1/serviceRate))

def waitDistMMC(arrivalRate,serviceRate,servers,t):
    #probability that an arriving customer has to wait
    #more than t units of time
    rho = arrivalRate/(serviceRate*servers)
    pr=0
    for n in range(servers):
        pr+=probNOccupied(arrivalRate,serviceRate,servers,n)
    pr += probEmptyMMC(arrivalRate,serviceRate,servers)*((math.pow(servers,servers)*math.pow(rho,servers))/math.factorial(servers))*((math.exp(-serviceRate*servers*(1-rho)*t)-1)/(rho-1))    
    return 1-pr

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

#--------------------------------
#   VALIDATION 
#--------------------------------

#Expected waiting time, E[W], difference
diffExpected = abs(mdl.meanWaitingTime()-meanWaitingTimeMMC(3,1,servers))
print(diffExpected)
#1.068590e-15

#Expected number of customers in system, E[C], difference
diffOccupancy = abs(mdl.meanOccupancy()-meanOccupancyMMC(3,1,servers))
print(diffOccupancy)
#8.881784e-16

#Probability that the system is empty, P(C=0), difference
diffProbEmpty = abs(mdl.probEmpty()-probEmptyMMC(3,1,servers))
print(diffProbEmpty)
#4.254318e-11

#Probability of waiting longer than t=1 time units, P(W>t), difference
diffWait = abs(mdl.waitDist(1)-waitDistMMC(3,1,servers,1))
print(diffWait)
#1.118148e-10

if np.round(diffExpected,9)!=0.0 or np.round(diffOccupancy,9)!=0.0 or np.round(diffProbEmpty,9)!=0.0 or np.round(diffWait,9)!=0.0:
    sys.exit("Validation test failed. The metrics could not be replicated.")
else:
    print("Validation test successful!")