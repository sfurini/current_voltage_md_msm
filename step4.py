import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Process

import flowmsm

# Use this for MthK channel
#prefix = 'data/mthk'
#channel_name = 'mthk'
#dVs = [250, 200, 150, 100, 50, -50, -100, -150, -200, -250]
#dVs_fitting = [-250, 250]
#dVs_output = np.arange(-250,251,10)
#plot_current_range = [[None, None], [-60, 30]]
#exp_data = {
#   -250:[-52.5456,-55.56651429,-53.87297143,-56.52771429,-55.84114286,-53.91874286],
#   -200:[-52.04211429,-52.95754286,-53.87297143,-51.99634286,-54.05605714,-42.84205714],
#   -150:[-32.90965714,-29.24794286,-32.3604,-42.52165714,-26.45588571,-30.30068571],
#   -100:[-18.81205714,-20.5056,-17.89662857,-21.23794286,-22.7484,-23.0688],
#   -50:[-4.210971429,-5.675657143,-4.93416,-3.295542857,-6.362228571,-2.517428571],
#   50:[6.362228571,8.284628571,6.636857143,6.362228571,7.552285714,7.506514286],
#   100:[11.5344,10.8936,13.73142857,14.96725714,11.76325714,11.5344],
#   150:[16.02,13.09062857,22.38222857,20.59714286,17.89662857,22.06182857],
#   200:[29.56834286,24.07577143,24.62502857,20.78022857,26.18125714,24.30462857],
#   250:[29.56834286,40.64502857,30.84994286,27.8748,29.7972,39.82114286],
#}

# Use this for KcsA channel
prefix = 'data/5vk6'
channel_name = '5vk6'
dVs = [400, 350, 300, 250, 200, 100, -100, -200]
dVs_fitting = [-200, 400]
dVs_output = np.arange(-200,401,10)
plot_current_range = [[None, None], [-2, 12]]
exp_data = {
        -200:[-0.236008929,-0.7209,-0.274628571,-0.370749313,-0.228857143],
        -100:[-0.128160256,-0.283783425,-0.122057143,-0.291272727],
        100:[0.481186813,0.867299722,0.118887136,0.570656772,0.28167033,0.404215213,0.285328386],
        200:[1.919011976,1.607971207,1.637326343,1.607991824,2.124694123,1.748918548,1.684451865,1.525992095],
        250:[2.721084287,3.1465099,2.997985927,2.503831677,2.841061603,4.208391564,2.917928571,3.975221974],
        300:[5.240828571,4.462714286,5.2866,4.783114286,4.6458,3.754210666,4.014781468,4.344341328],
        350:[7.449254974,6.391344907,6.73033374,6.233027026,7.226961273,7.125793053,6.454574001,7.044799162],
        400:[7.450571429,9.916812297,8.213519833,7.832,9.760296801,10.26151314,8.454562744,7.896791496],
}

prefix_fitting_output = '_'.join([str(dV) for dV in np.sort(dVs_fitting)])
lag = 50
  
# Fit the current voltage characteristc
pdf = PdfPages('figures.{}.{}.pdf'.format(channel_name, prefix_fitting_output))
channel = flowmsm.channel.Channel({dV:'{}.{}.step3.{}mV.pk'.format(prefix, prefix_fitting_output, dV) for dV in dVs}, channel_name, dVs_fitting = dVs_fitting, use_md_data = False)
channel.current_voltage(pdf, dVs_output = dVs_output, lag4plot = lag, i_boots = [-1]+channel.find_boots()[-1])
channel.plot_states(pdf, lag)
channel.plot_current_voltage(pdf, lag, exp_data = exp_data, plot_current_range = plot_current_range, i_boots = [-1])
#channel.plot_current_voltage(pdf, lag, exp_data = exp_data, plot_current_range = plot_current_range, i_boots = channel.find_boots()[-1])
pdf.close()

