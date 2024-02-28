# -*- coding: utf-8 -*-

#--------------------------------
#   PREAMPLE
#--------------------------------

import math
import numpy as np
import phph

#--------------------------------
#   QUEUE PARAMETERS
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


#--------------------------------
#   EVALUATE QUEUE
#--------------------------------

mdl = phph.model(arrivalInitDistribution,arrivalGenerator,
                 serviceInitDistribution,serviceGenerator,
                 servers)

print(1-mdl.probWait())
print(mdl.meanWaitingTime())
print(mdl.waitDist(3.0753,type="actual"))
print(mdl.waitDist(3.0753,type="virtual"))

