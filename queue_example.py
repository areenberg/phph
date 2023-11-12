# -*- coding: utf-8 -*-

#--------------------------------
#   PREAMPLE
#--------------------------------

import numpy as np
import math

#import matplotlib.pyplot as plt
from PHPHCSolver.Queue import Queue
from PHPHCSolver.LocalStateSpace import LocalStateSpace
from PHPHCSolver.SubMatrices import SubMatrices





#--------------------------------
#   QUEUE PARAMETERS
#--------------------------------

#server capacity
servers = 4

#Service parameters

#Initial distribution
alpha = np.matrix([[0.5,0.5]])

#Exit rates
s = np.matrix([[0.5*servers],
               [0.3*servers]])

#Phase-type generator
S = np.matrix([[0.0,0.0],
               [10.0,0.0]])
S[0,0] = -(s[0]+S[0,1])
S[1,1] = -(s[1]+S[1,0])


#Arrival parameters

#Initial distribution
gamma = np.matrix([[0.8,0.2]])

#Exit rates
t = np.matrix([[0.9*servers],
               [0.5*servers]])


#Phase-type generator
T = np.matrix([[0.0,0.1],
               [0.2,0.0]])
T[0,0] = -(t[0]+T[0,1])
T[1,1] = -(t[1]+T[1,0])


queue = Queue(gamma,T,t,alpha,S,s,servers)

#--------------------------------
#   EVALUATE QUEUE
#--------------------------------

subMats = SubMatrices(queue)

fMat = subMats.createForwardInhomMatrix(3,4)
bMat = subMats.createBackwardInhomMatrix(3,2)
lMat = subMats.createLocalInhomMatrix(3,fMat,bMat)

print(fMat)
print(bMat)
print(lMat)


#ls = LocalStateSpace(queue)
#ls.generateStateSpace(servers)
#subMats.createForwardMatrix(ls)
#subMats.createBackwardMatrix(ls)
#subMats.createLocalMatrix(ls)

#print(subMats.forwardMat)
#print(subMats.backwardMat)
#print(subMats.localMat)


