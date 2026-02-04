from multiprocessing import Process

import flowmsm

n_boots = 0 # Number of datasets to generate (by bootstrapping)
lags = [1,2,4,6,8,10,20,30,40,50,60,70,80,90,100] # multiple of elementary timestep

# Use this for MthK channel
#prefix = 'data/mthk'
#dVs = [-250, -200, -150, -100, -50, 50, 100, 150, 200, 250]

# Use this for KcsA channel
prefix = 'data/5vk6'
dVs = [400, 350, 300, 250, 200, 100, -100, -200]

def run(dV):
    model = flowmsm.msm.TransitionMatrix('{}.{}mV.dtrajs.npz'.format(prefix, dV))
    model.fit('{}.{}mV.msm.pk'.format(prefix, dV), lags, n_boots)

processes = []
for dV in dVs:
    p = Process(target=run, args=(dV,))
    p.start()
    processes.append(p)
for p in processes:
    p.join()
