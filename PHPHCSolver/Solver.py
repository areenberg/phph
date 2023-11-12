import math
import numpy as np
from PHPHCSolver.Queue import Queue
from PHPHCSolver.LocalStateSpace import LocalStateSpace
from PHPHCSolver.SubMatrices import SubMatrices


class Solver:
    #evaluate the state distribution of the QBD and various
    #performance metrics of the queue 

    def __init__(self,queue):
        self.queue = queue

