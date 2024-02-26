# -*- coding: utf-8 -*-

#--------------------------------
#   PREAMPLE
#--------------------------------

import numpy as np
import math
import time #for runtime measurements
from PHPHCSolver.Queue import Queue
from PHPHCSolver.Solver import Solver
from PHPHCSolver.SubMatrices import SubMatrices
from PHPHCSolver.LocalStateSpace import LocalStateSpace
from PHPHCSolver.Uniformization import Uniformization
from PHPHCSolver.BlockUniformization import BlockUniformization


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

#start_time = time.perf_counter()
sol = Solver(queue)
#print(time.perf_counter()-start_time)

#start_time = time.perf_counter()
#print(sol.probWait())
#print(time.perf_counter()-start_time)


print(sol.waitDist(3.0753,type="actual"))
print(sol.waitDist(3.0937,type="virtual"))


#print(sol.meanWaitingTime())
