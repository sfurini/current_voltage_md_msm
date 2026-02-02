import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Process

import flowmsm

prefix = 'data/mthk'
channel_name = 'mthk'
dVs = [-250, -200, -150, -100, -50, 50, 100, 150, 200, 250]

def run(dV):
    pdf = PdfPages('figures.{}.step1.{}mV.pdf'.format(channel_name, dV))
    m = flowmsm.msm.MSM_from_MD('{}.{}mV'.format(prefix, dV), flowmsm.utils_channel.repr_state, flowmsm.utils_channel.value_state)
    m.make_model(pdf, title = 'original microstates')
    m.repr_state = flowmsm.utils_channel.repr_state_only_ions
    m.value_state = flowmsm.utils_channel.value_state_only_ions
    m.make_model(pdf, title = 'only ions')
    with open('{}.step1.{}mV.pk'.format(prefix, dV), 'wb') as fout:
        pickle.dump(m, fout)
    pdf.close()

# Create initial models for all the membrane voltages
processes = []
for dV in dVs:
    p = Process(target=run, args=(dV,))
    p.start()
    processes.append(p)
for p in processes:
    p.join()

