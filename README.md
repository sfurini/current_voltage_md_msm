# Current-voltage characteristics of potassium channels estimated by Molecular Dynamics simulations and Markov State Models

* data: discretized MD trajectories. Each file contains:
    * dt: dumping time of the MD trajectories in ns
    * states: an np.array with shape (number of states) x (number of binding sites).

        The elements of the array have this format: [s0\_ion, s1\_ion, s2\_ion, s3\_ion, s4\_ion, sc\_ion, s0\_water, s1\_water, s2\_water, s3\_water, s4\_water]

        As an example a state with ion in S0, S2, S3, and cavity, water in S1, and S4 empty is coded as: [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0]

    * dtrajs: a list of discretized MD trajectories. Each element of the list is an np.array reporting the indexes of the states along that trajectory
    * ftrajs: same as dtrajs but reporting the number of conduction events along the trajectory

* flowmsm: the code used to estimate the current-voltage characteristic
    * step1.py: definition of the transition matrix of the MSMs
    * step1:py: definition of the MSMs
    * step2.py: estimate of the rate matrix
    * step4.py: estimate of the current-voltage characteristics

