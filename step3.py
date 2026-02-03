import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Process

import flowmsm

prefix = 'data/mthk'
channel_name = 'mthk'
dVs = [250, 200, 150, 100, 50, -50, -100, -150, -200, -250]
dVs_fitting = [-250, 250]

prefix = 'data/5vk6'
channel_name = '5vk6'
dVs = [400, 350, 300, 250, 200, 100, -100, -200]
dVs_fitting = [-200, 400]

prefix_fitting_output = '_'.join([str(dV) for dV in np.sort(dVs_fitting)])
lag = 50
prob_min = 0.01

def run(dV):
    pdf = PdfPages('figures.{}.{}.step3.{}mV.pdf'.format(channel_name, prefix_fitting_output, dV))
    with open('{}.step2.{}mV.pk'.format(prefix, dV), 'rb') as fin:
        m = pickle.load(fin)
    m.lags = np.array([x for x in m.lags if x <= lag])
    m.i_boots = m.check_target_presence(target_states)
    if -2 in m.i_boots:
        m.i_boots.remove(-2)
    m.make_target_states_model(target_states, pdf)
    m.make_rate_model(pdf, i_boots = m.i_boots)
    with open('{}.{}.step3.{}mV.pk'.format(prefix, prefix_fitting_output, dV), 'wb') as fout:
        pickle.dump(m, fout)
    pdf.close()


# Fit all models to the same target states and estimate Q
channel = flowmsm.channel.Channel({dV:'{}.step2.{}mV.pk'.format(prefix, dV) for dV in dVs}, channel_name, dVs_fitting = dVs_fitting, use_md_data = False)
target_states = channel.find_high_probability(i_boot = -1, lag = lag, prob_min = prob_min)
print('Number of common states = ',len(target_states))
for i_state, state in enumerate(target_states):
    output = '\t{}'.format(state)
    for dV in dVs_fitting:
        prob = channel.msm_input[dV].probability(-1, lag)
        for i_state_dV, state_dV in enumerate(channel.msm_input[dV].states[-1][lag]):
            if channel.msm_input[dV].repr_state(state_dV) == state:
                output += '\t[dV = {}] {}'.format(dV, prob[i_state_dV])
    print(output)

processes = []
for dV in dVs:
    p = Process(target=run, args=(dV,))
    p.start()
    processes.append(p)
for p in processes:
    p.join()
