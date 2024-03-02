# -*- coding: utf-8 -*-

#In this example we replicate results from de Smit (1983)
#and Ramaswami & Lucantoni (1985).

#de Smit, J. H. A. (1983). Numerical solution for the multi-server queue with
#hyper-exponential service times. Operations Research Letters, 2(5), 217–224.

#Ramaswami, V., & Lucantoni, D. M. (1985). Algorithms for the multi-server
#queue with phase type service. Communications in Statistics. Stochastic
#Models, 1(3), 393–417. https://doi.org/10.1080/15326348508807020

#--------------------------------
#   PREAMPLE
#--------------------------------

import math
import numpy as np
import phph

#--------------------------------
#   E_5 / H_2 / 15 
#--------------------------------

#Server capacity
servers = 15

#Arrival parameters

#Initial distribution
arrivalInitDistribution = np.matrix([[1.0,0.0,0.0,0.0,0.0]])

#Phase-type generator
arrivalGenerator = np.matrix([[-0.9*servers*5,0.9*servers*5,0.0,0.0,0.0],
                              [0.0,-0.9*servers*5,0.9*servers*5,0.0,0.0],
                              [0.0,0.0,-0.9*servers*5,0.9*servers*5,0.0],
                              [0.0,0.0,0.0,-0.9*servers*5,0.9*servers*5],
                              [0.0,0.0,0.0,0.0,-0.9*servers*5]])

#Service parameters

#Initial distribution
serviceInitDistribution = np.matrix([[(24+math.sqrt(468))/54,
                                      1-((24+math.sqrt(468))/54)]])

#Phase-type generator
serviceGenerator = np.matrix([[-3*serviceInitDistribution[0, 0],0.0],
                              [0.0,-1.5*serviceInitDistribution[0, 1]]])

#Create the model
mdl = phph.model(arrivalInitDistribution,arrivalGenerator,
                 serviceInitDistribution,serviceGenerator,
                 servers)

print("E_5 / H_2 / 15")

#Expected waiting time, E[W] (actual)
print(mdl.meanWaitingTime())
#phph: 0.895427
#de Smit: 0.8943
#Ramaswami & Lucantoni: 0.89540

#Probability of not waiting, P(W=0) (actual)
print(1-mdl.probWait())
#phph: 0.434633
#de Smit: 0.4346
#Ramaswami & Lucantoni: 0.434622

#P(W>7.0550) P(W>4.1200) P(W>2.8563) (actual)
print(mdl.waitDist(7.0550),mdl.waitDist(4.1200),mdl.waitDist(2.8563))
#phph: 0.010000 0.049999 0.100001
#de Smit: 0.01 0.05 0.10
#Ramaswami & Lucantoni: NA

#Expected number of customers in system, E[C] (continuous time)
print(mdl.meanOccupancy())
#phph: 25.58827
#de Smit: 25.588
#Ramaswami & Lucantoni: NA

#--------------------------------
#   E_2 / H_2 / 15 
#--------------------------------

#Server capacity
servers = 15

#Arrival parameters

#Initial distribution
arrivalInitDistribution = np.matrix([[1.0,0.0]])

#Phase-type generator
arrivalGenerator = np.matrix([[-0.9*servers*2,0.9*servers*2],
                              [0.0,-0.9*servers*2]])

#Service parameters

#Initial distribution
serviceInitDistribution = np.matrix([[(24+math.sqrt(468))/54,
                                      1-((24+math.sqrt(468))/54)]])

#Phase-type generator
serviceGenerator = np.matrix([[-3*serviceInitDistribution[0, 0],0.0],
                              [0.0,-1.5*serviceInitDistribution[0, 1]]])

#Create the model
mdl = phph.model(arrivalInitDistribution,arrivalGenerator,
                 serviceInitDistribution,serviceGenerator,
                 servers)

print("E_2 / H_2 / 15")

#Expected waiting time, E[W] (actual)
print(mdl.meanWaitingTime())
#phph: 0.977745
#de Smit: 0.9777
#Ramaswami & Lucantoni: 0.97773  

#Probability of not waiting, P(W=0) (actual)
print(1-mdl.probWait())
#phph: 0.409896
#de Smit: 0.4099
#Ramaswami & Lucantoni: 0.409876

#P(W>7.4924) P(W>4.4048) P(W>3.0753) (actual)
print(mdl.waitDist(7.4924),mdl.waitDist(4.4048),mdl.waitDist(3.0753))
#phph: 0.010000 0.049999 0.100001
#de Smit: 0.01 0.05 0.10
#Ramaswami & Lucantoni: NA

#Expected number of customers in system, E[C] (continuous time)
print(mdl.meanOccupancy())
#phph: 26.69956
#de Smit: 26.700
#Ramaswami & Lucantoni: NA

#--------------------------------
#   M / H_2 / 15 
#--------------------------------

#Server capacity
servers = 15

#Arrival parameters

#Initial distribution
arrivalInitDistribution = np.matrix([[1.0]])

#Phase-type generator
arrivalGenerator = np.matrix([[-0.9*servers]])

#Service parameters

#Initial distribution
serviceInitDistribution = np.matrix([[(24+math.sqrt(468))/54,
                                      1-((24+math.sqrt(468))/54)]])

#Phase-type generator
serviceGenerator = np.matrix([[-3*serviceInitDistribution[0, 0],0.0],
                              [0.0,-1.5*serviceInitDistribution[0, 1]]])

#Create the model
mdl = phph.model(arrivalInitDistribution,arrivalGenerator,
                 serviceInitDistribution,serviceGenerator,
                 servers)

print("M / H_2 / 15")

#Expected waiting time, E[W] (actual)
print(mdl.meanWaitingTime())
#phph: 1.11609 
#de Smit: 1.1161
#Ramaswami & Lucantoni: 1.11606

#Probability of not waiting, P(W=0) (actual)
print(1-mdl.probWait())
#phph: 0.375117
#de Smit: 0.3751
#Ramaswami & Lucantoni: 0.375084

#P(W>8.2209) P(W>4.8783) P(W>3.4390) (actual)
print(mdl.waitDist(8.2209),mdl.waitDist(4.8783),mdl.waitDist(3.4390))
#phph: 0.010000 0.050001 0.100002
#de Smit: 0.01 0.05 0.10
#Ramaswami & Lucantoni: NA

#Expected number of customers in system, E[C] (continuous time)
print(mdl.meanOccupancy())
#phph: 28.56718
#de Smit: 28.567
#Ramaswami & Lucantoni: NA

#--------------------------------
#   H_2 / H_2 / 15 
#--------------------------------

#Server capacity
servers = 15

#Arrival parameters

#Initial distribution
arrivalInitDistribution = np.matrix([[0.90824829,1-0.90824829]])

#Phase-type generator
arrivalGenerator = np.matrix([[-24.52270384,0.0],
                              [0.0,-2.477296157]])


#Service parameters

#Initial distribution
serviceInitDistribution = np.matrix([[(24+math.sqrt(468))/54,
                                      1-((24+math.sqrt(468))/54)]])

#Phase-type generator
serviceGenerator = np.matrix([[-3*serviceInitDistribution[0, 0],0.0],
                              [0.0,-1.5*serviceInitDistribution[0, 1]]])

#Create the model
mdl = phph.model(arrivalInitDistribution,arrivalGenerator,
                 serviceInitDistribution,serviceGenerator,
                 servers)

print("H_2 / H_2 / 15")

#Expected waiting time, E[W] (actual)
print(mdl.meanWaitingTime())
#phph: 2.195077
#de Smit: 2.1951
#Ramaswami & Lucantoni: 2.19318  

#Probability of not waiting, P(W=0) (actual)
print(1-mdl.probWait())
#phph: 0.242127
#de Smit: 0.2421
#Ramaswami & Lucantoni: 0.242164

#P(W>13.7846) P(W>8.4791) P(W>6.1941) (actual)
print(mdl.waitDist(13.7846),mdl.waitDist(8.4791),mdl.waitDist(6.1941))
#phph: 0.010000 0.050000 0.100000 
#de Smit: 0.01 0.05 0.10
#Ramaswami & Lucantoni: NA

#Expected number of customers in system, E[C] (continuous time)
print(mdl.meanOccupancy())
#phph: 43.13353
#de Smit: 43.133
#Ramaswami & Lucantoni: NA
