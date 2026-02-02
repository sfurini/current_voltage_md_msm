# Current-voltage characteristics of potassium channels estimated by Molecular Dynamics simulations and Markov State Models

* data: discretized MD trajectories. Each file contains:
    * dt: dumping time of the MD trajectories
    * states: an np.array with shape (number of states) x (number of binding sites)
    * dtrajs: a list of discretized MD trajectories. Each element of the list is an np.array reporting the indexes of the states along that trajectory
    * ftrajs: same as dtrajs but reporting the number of conduction events along the trajectory

* flowmsm: the code used to estimate the current-voltage characteristic
    * step1.py: definition of the transition matrix of the MSMs
    * step1:py: definition of the MSMs
    * step2.py: estimate of the rate matrix
    * step4.py: estimate of the current-voltage characteristics

