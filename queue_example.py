# -*- coding: utf-8 -*-

#--------------------------------
#   PREAMPLE
#--------------------------------

import numpy as np
import math
from scipy import stats

#import statistics
#import matplotlib.pyplot as plt
from PHPHCSolver.Queue import Queue





#--------------------------------
#   QUEUE PARAMETERS
#--------------------------------

#server capacity
servers = 2

#Service parameters

#Initial distribution
alpha = np.matrix([[ (24+math.sqrt(468))/54 ,
                                1-((24+math.sqrt(468))/54)]])

#Exit rates
s = np.transpose(np.matrix([[ 3*alpha[0,0] ,
                              1.5*alpha[0,1] ]]))

#Phase-type generator
S = np.matrix([[0.0,0.0],
               [0.0,0.0]])
S[0,0] = -s[0]
S[1,1] = -s[1]


#Arrival parameters

#Initial distribution
gamma = np.matrix([[1.0,0.0]])

#Exit rates
t = np.matrix([[0.9*servers],
               [0.9*servers]])

#Phase-type generator
T = np.matrix([[0.0,0.0],
               [0.0,0.0]])
T[0,0] = -t[0]
T[1,1] = -t[1]


queue = Queue(gamma,T,t,alpha,S,s,servers)

#--------------------------------
#   EVALUATE QUEUE
#--------------------------------

print(queue.meanInterArrivalTime())
print(1/(0.9*servers))
