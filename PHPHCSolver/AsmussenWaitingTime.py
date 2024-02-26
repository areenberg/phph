import math
import numpy as np
from PHPHCSolver.Queue import Queue
from PHPHCSolver.SubMatrices import SubMatrices


class AsmussenWaitingTime:
    #DRAFT
    
    #evaluate the waiting time distribution
    #using S. Asmussen's method

    #References:
    #Asmussen, Soeren, and Colm Art O'cinneide. 1998. “Representations for Matrix-Geometric and
    #Matrix-Exponential Steady-State Distributions with Applications to Many-Server Queues.”
    #Communications in Statistics. Stochastic Models 14 (1-2): 369–87.
    #https://doi.org/10.1080/15326349808807477.

    #Asmussen, Soeren, and Jakob R. Moeller. 2001. “Calculation of the Steady State Waiting Time
    #Distribution in GI/PH/c and MAP/PH/c Queues.” Queueing Systems 37 (1-3): 9–29.
    #https://doi.org/10.1023/A:1011083915877.

        def __init__(self,queue,subMats,eps):

            self.queue = queue
            self.subMats=subMats
            self.eps=eps

            #calculate fundamental parameters
            self.__calculateMatrixT()
            #self.__calculateRho()
            #self.__calculateMatrixG()



        def __calculateMatrixT(self):
            #derive matrix T numerically

            #create the matrix S (state changes not accompanied by a service completion)
            rs = np.sum(self.subMats.forwardMat,axis=1)
            S = np.copy(self.subMats.localMat)
            np.fill_diagonal(S,S.diagonal()+rs)
            
            #create the matrix Ajump
            self.Ajump = self.subMats.backwardMat #state changes accompanied by a service completion

            #initialize
            self.T = np.copy(self.S)

            
            #Note 1: Consider the density function H(x) on its scalar form. The scalar form should correspond to the term exp(-st) in the laplace transform.
            #Exploit the formular for the laplace transform of a matrix exponential function to solve the integral.   
            
            #Note 2: Consider the Laplace transform for multiplication of two functions, L(f(x)*g(x)) = L(f(x))*L(g(x))
            #See page 4 of https://www.math.purdue.edu/~neptamin/266Sp21/Laplace_techniques.pdf
            
            
            #lmbIden = self.queue.arrivalRate*np.identity(self.T.shape[0])
            #lmbPi = self.queue.arrivalRate*self.Pi_Inf
            #tol = 1
            #while tol>self.eps:
            #    Mchange = np.matmul(np.linalg.inv(np.subtract(lmbIden,self.T)),lmbPi)
            #    Tnew = np.add(self.Q,Mchange)
            #    tol = abs(np.subtract(Tnew,self.T)).max()
            #    self.T = np.copy(Tnew)
