import numpy as np

class LocalStateSpace:
    #Class for generating a local state space at a specific level in the
    #QBD.

    def __init__(self,queue):
        self.queue = queue


    def generateStateSpace(selv,l):
        #generate state space at level l
        print()
