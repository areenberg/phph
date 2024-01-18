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
from PHPHCSolver.Solver import Solver





#--------------------------------
#   QUEUE PARAMETERS
#--------------------------------

#server capacity
servers = 15

#Service parameters

#Initial distribution
alpha = np.matrix([[(24+math.sqrt(468))/54,
                    1-((24+math.sqrt(468))/54)]])

#Exit rates
s = np.transpose(np.matrix([[3*alpha[0, 0],
                             1.5*alpha[0, 1]]]))

#Phase-type generator
S = np.matrix([[-s[0,0],0.0],
               [0.0,-s[1,0]]])


#Arrival parameters

#Initial distribution
gamma = np.matrix([[1.0,0.0]])
#gamma = np.matrix([[1.0]])

#Exit rates
t = np.transpose(np.matrix([[0.0,
                             0.9*servers*2]]))
#t = np.matrix([[0.9*servers]])

#Phase-type generator
T = np.matrix([[-0.9*servers*2,0.9*servers*2],
               [0.0,-0.9*servers*2]])
#T = np.matrix([[-0.9*servers]])

queue = Queue(gamma,T,t,alpha,S,s,servers)

#--------------------------------
#   EVALUATE QUEUE
#--------------------------------

sol = Solver(queue)

print(sol.meanWaitingTime())
print(sol.meanOccupancy())
print(sol.meanQueueLength())
