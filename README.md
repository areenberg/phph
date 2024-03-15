# `phph` -- A Python package for PH/PH/c queueing systems
 
`phph` is the Python package for evaluating the performance of PH/PH/c queueing systems.

## Features

* Available on PyPI.
* Allows for evaluation of multiserver models containing any Phase-type (PH) inter-arrival and service-time distribution.
* Returns fundamental performance metrics such as the expected waiting time, expected occupancy, and the probability of waiting longer than *t* units of time.  
* Users can choose to view the metrics observed by *actual* arriving customers or *virtual* Poisson arrivals.

# Quick start guide

The following shows how to get quickly started with `phph`.

## Installation

Download and install `phph` directly from PyPI. 

```
pip install phph
```

## Usage

Start by specifying the inter-arrival and service-time distributions.

```python
#Import packages
import phph
import numpy as np

#Set the server capacity
servers = 5

#ARRIVAL PARAMETERS - Example of Erlang distribution

#Initial distribution
arrivalInitDistribution = np.matrix([[1,0]])

#Phase-type generator
arrivalGenerator = np.matrix([[-12,12],
                              [0,-12]])

#SERVICE PARAMETERS - Example of hyper-exponential distribution

#Initial distribution
serviceInitDistribution = np.matrix([[(1/3),(1/2),(1/6)]])

#Phase-type generator
serviceGenerator = np.matrix([[-2,0,0],
                              [0,-1,0],
                              [0,0,-6]])

```

Now, create the model object.

```python
mdl = phph.model(arrivalInitDistribution,arrivalGenerator,
                 serviceInitDistribution,serviceGenerator,
                 servers)
```

We can now use the object `mdl` to return various performance metrics. In the following, we calculate the expected waiting time, the expected occupancy, and the probability of waiting.

```python
#Expected waiting time 
print(mdl.meanWaitingTime())
#0.455024

#Expected occupancy
print(mdl.meanOccupancy())
#6.896811

#Probability of waiting
print(mdl.probWait())
#0.562802
```

# User manual

The following shows how to create the model and provides a list of all available performance metrics.

## Object creation

Create the model object with:

```python
mdl = phph.model(arrivalInitDistribution,arrivalGenerator,
                 serviceInitDistribution,serviceGenerator,
                 servers)
```

* `arrivalInitDistribution` is a NumPy row vector defining the initial distribution of the PH distribution associated with **arrivals**.
* `arrivalGenerator` is a NumPy matrix defining the PH generator of the distribution associated with **arrivals**.
* `serviceInitDistribution` is a NumPy row vector defining the initial distribution of the PH distribution associated with **services**.
* `serviceGenerator` is a NumPy matrix defining the PH generator of the distribution associated with **services**.
* `servers` is a non-zero positive integer and defines the number of servers in the queueing system.

## Performance metrics

* `mdl.meanWaitingTime()`. Returns the *actual* (i.e. observed by arriving customers) expected waiting time. 
* `mdl.meanResponse()`. Returns the *actual* expected total time in the system.
* `mdl.probWait(type="actual")`. Returns the probability of waiting. Choose between `"actual"` (default) and `"virtual"` using the argument `type`. 
* `mdl.probEmpty(type="actual")`. Returns the probability that the system is empty. Choose between `"actual"` (default) and `"virtual"` using the argument `type`.
* `mdl.probK(k,type="actual")`. Returns the probability of observing `k` customers in the system on arrival. Choose between `"actual"` (default) and `"virtual"` using the argument `type`.
* `mdl.waitDist(t,type="actual")`. Returns the probability of waiting more than `t` time units. Choose between `"actual"` (default) and `"virtual"` using the argument `type`.
* `mdl.meanQueueLength()`. Returns the expected length of the queue. 
* `mdl.meanOccupancy()`. Returns the expected occupancy.

## Other output

* `mdl.localStateDist(k)`. Returns the local state distribution of level `k`.
* `mdl.localState(k)`. Returns the definition of the local state space of level `k`.