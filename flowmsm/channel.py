"""
channel.py includes the definition of the class Channel
"""

import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import constants
from scipy.optimize import minimize

from . import msm

class Channel(object):
    """
    Parameters
    ----------
    inputs: dict
        keys: float
            Membrane Potential
        values: inputfile.pk
            Containing the MSM at the corresponding potential

    Attributes
    ----------

    prefix: str
        Used for naming outputs

    msm_input: dict
        keys: float
            Membrane Potential
        values: MSM_from_MD
            Corresponding MSM 

    dVs_input: np.array
        Membrane potentials
        Sorted keys of msm_input

    msm_output: dict
        keys: float
            Membrane potential
        values: MSM_from_matrixes
            Corresponding MSM

    dVs_output: list
        Membrane potentials
        Sorted keys of msm_output

    dt: float
        Timestep

    lags: list
        Mulitple of timesteps
    """
    def __init__(self, inputs, prefix, dVs_fitting = [], use_md_data = True):
        """
        inputs: dict
            keys:   dV
            values: pk inputfiles from class MSM

        prefix: str
            Used for naming outputs
        """
        self.prefix = prefix
        self.dVs_fitting = np.array(dVs_fitting)
        self.msm_output = {}
        self.dVs_output = []
        #--- Read input MSM
        self.msm_input = {}
        for dV, inputfile in inputs.items():
            print('Reading dV = {} from inputfile = {}'.format(dV, inputfile))
            with open(inputfile, 'rb') as fin:
                self.msm_input[dV] = pickle.load(fin)
                for i_boot in self.msm_input[dV].i_boots:
                    if use_md_data:
                        for lag in self.msm_input[dV].lags:
                            self.msm_input[dV].T[i_boot][lag] = self.msm_input[dV].Tmd[i_boot][lag]
                            self.msm_input[dV].F[i_boot][lag] = self.msm_input[dV].Fmd[i_boot][lag]
        self.dVs_input = list(self.msm_input.keys())
        self.dVs_input.sort()
        self.dVs_input = np.array(self.dVs_input)
        self.dt = self.msm_input[self.dVs_input[0]].dt
        self.lags = self.msm_input[self.dVs_input[0]].lags
        #--- Check that the input models are consistent in term of dt and lags
        for idV in range(1, len(self.dVs_input)):
            if self.msm_input[self.dVs_input[idV]].dt != self.dt:
                raise ValueError('ERROR: inconsistent dt in model at dV {}'.format(self.dVs_input[idV]))
            if any(self.msm_input[self.dVs_input[idV]].lags != self.lags):
                raise ValueError('ERROR: inconsistent lags in model at dV {}'.format(self.dVs_input[idV]))

    def find_boots(self):
        """
        Return
        ------
        i_boots_sets    list of list
            internal lists provides all the single i_boots that are common to all the input voltages
            e.g. [[-1,], [0,], [1,], [2,], [3,], [4,], [0, 1, 2, 3, 4]]
        i_boots_set     list
            the list of the i_boots >= 0, i.e. the real bootstraps
        """
        i_boots = set()
        for dV in self.dVs_input:
            if i_boots:
                i_boots = i_boots & set(self.msm_input[dV].i_boots)
            else:
                i_boots = set(self.msm_input[dV].i_boots)
        i_boots = list(i_boots)
        i_boots.sort()
        i_boots_sets = [[i_boots,] for i_boots in i_boots]
        if -2 in i_boots: i_boots.remove(-2)
        if -1 in i_boots: i_boots.remove(-1)
        i_boots_sets.append(i_boots)
        return i_boots_sets, i_boots

    def reset_lags_all_models(self, lags):
        if not isinstance(lags, np.ndarray):
            lags = np.array(lags)
        self.lags = lags
        for dV in self.dVs_input:
            self.msm_input[dV].lags = lags

    def reset_F(self, i_boot, lag):
        for dV in self.dVs_input:
            self.msm_input[dV].set_F_from_states(i_boot, lag)

    def merge_pcca(self, n_clusters_min, n_clusters_max, i_boot, lag_pcca):
        """
        """
        #--- make a copy before the pcca clustering, this is later used to merge the states
        models_original = copy.deepcopy(self)
        #--- find a number of clusters that fits for all input models
        n_clusters_optimal = -1
        for dV in self.dVs_input:
            pcca = GenPCCA(self.msm_input[dV].T[i_boot][lag_pcca])
            n_clusters = pcca.optimal(n_clusters_min, n_clusters_max)
            print('dV = {} i_boot = {} lag = {} n_clusters = {}'.format(dV, i_boot, lag_pcca, n_clusters))
            n_clusters_optimal = max(n_clusters_optimal, n_clusters)
        print('Number of clusters = {}'.format(n_clusters_optimal))
        #--- find the partition functions
        chi = {}
        for dV in self.dVs_input:
            pcca = GenPCCA(self.msm_input[dV].T[i_boot][lag_pcca])
            T_red, chi[dV] = pcca.coarsegrain(n_clusters_optimal) # chi = n_states x n_clusters, chi[i,j] = prob. state_i belongs to cluster_j
            inds_clusters = np.argmax(chi[dV], axis = 1)
            if len(np.unique(inds_clusters)) != n_clusters_optimal:
                raise ValueError('ERROR: clusters not used')
            self.msm_input[dV].merge_clusters(i_boot, lag_pcca, inds_clusters)
        #--- Check same number of states
        n_states = self.check_number_states(i_boot, lags = lag_pcca)
        #--- Reorder the models at all dVs to the same order
        states2order = []
        for dV in self.dVs_input:
            states2order.append(self.msm_input[dV].states[i_boot][lag_pcca])
        inds_same_order = common_order(*states2order)
        #--- Print the states
        print('Merged state with a case-by-case  PCCA')
        for i_state in range(n_states):
            print('State {}'.format(i_state))
            for i_dV, dV in enumerate(self.dVs_input):
                probs = self.msm_input[dV].probability(i_boot, lag_pcca)
                print('\t{}, prob = {}'.format(repr_state_full(self.msm_input[dV].states[i_boot][lag_pcca][inds_same_order[i_dV][i_state],:]), probs[inds_same_order[i_dV][i_state]]))
        #--- Define a dict for refering all macrostates index to the order in the first model
        #       self.dVs
        #           [100, 200, ...]
        #       inds_same_order: list of list
        #           [ [0, 1, 2], [2, 0, 1], ... ] means that state 0 of the first model corresponds to state 2 of the second model
        #       ind_macro_common: dict of dict
        #           100:
        #               0:0
        #               1:1
        #               2:2
        #           200:
        #               2:0
        #               0:1
        #               1:2
        inds_macro_common = {}
        for i_dV, dV in enumerate(self.dVs_input):
            inds_macro_common[dV] = {}
            for ind_common, ind  in enumerate(inds_same_order[i_dV]):
                inds_macro_common[dV][ind] = ind_common
        #--- Find shared association rules
        #       micro2macro: dict
        #           key: repr_state_full of the microstate
        #           value: dict
        #               key: index of the corresponding macrostate in the first model
        #               value: list with elements the probabilities of the microstates associated to the macrostate
        micro2macro = {}
        states_original = {} # here I memorize the states at all dVs
        for i_dV, dV in enumerate(self.dVs_input):
            prob = models_original.msm_input[dV].probability(i_boot, lag_pcca)
            if len(prob) !=  models_original.msm_input[dV].states[i_boot][lag_pcca].shape[0]: # I changed this because when using MD data it is possible that states are deleted 
                raise ValueError('ERROR: use full model')
            states_original[dV] = np.copy(models_original.msm_input[dV].states[i_boot][lag_pcca])
            for i_state, state in enumerate(states_original[dV]):
                key_state = repr_state_full(state)
                if key_state not in micro2macro:
                    micro2macro[key_state] = {}
                for ind_macro, inds_micro in enumerate(self.msm_input[dV].indexes_macro[i_boot][lag_pcca]):
                    if i_state in inds_micro:
                        break
                else:
                    raise ValueError('ERROR: state {} not found in indexes_macro')
                ind_macro_common = inds_macro_common[dV][ind_macro] # this is the index of the macro state in the first(reference) model
                if ind_macro_common not in micro2macro[key_state]:
                    micro2macro[key_state][ind_macro_common] = []
                micro2macro[key_state][ind_macro_common].append(prob[i_state])
        #--- Apply the shared association rules at all lags
        for i_dV, dV in enumerate(self.dVs_input):
            for i_lag, lag in enumerate(self.lags):
                inds_clusters = []
                for i_state, state in enumerate(states_original[dV]):
                    key_state = repr_state_full(state)
                    #--- Find macro state according to shared associatio rules
                    n_best_association = 0
                    prob_best_association = 0
                    for ind_macro in micro2macro[key_state].keys():
                        if len(micro2macro[key_state][ind_macro]) > n_best_association:
                            n_best_association = len(micro2macro[key_state][ind_macro])
                            prob_best_association = sum(micro2macro[key_state][ind_macro])
                            ind_best_macro = ind_macro
                        elif len(micro2macro[key_state][ind_macro]) == n_best_association:
                            if sum(micro2macro[key_state][ind_macro]) > prob_best_association:
                                prob_best_association = sum(micro2macro[key_state][ind_macro])
                                ind_best_macro = ind_macro
                    #--- Find macro state according to this single case
                    for ind_macro, inds_micro in enumerate(self.msm_input[dV].indexes_macro[i_boot][lag_pcca]):
                        if i_state in inds_micro:
                            break
                    else:
                        raise ValueError('ERROR: state {} not found in indexes_macro lag = {}'.format(key_state, lag))
                    ind_macro_common = inds_macro_common[dV][ind_macro] # this is the index of the macro state in the first(reference) model
                    #if ind_best_macro != ind_macro_common:
                    #    print('WARNING: change in clustering at dV {} for state {}'.format(dV, key_state))
                    #inds_clusters.append(ind_best_macro)
                    #--- Always use the one at lag_pcca
                    inds_clusters.append(ind_macro_common)
                models_original.msm_input[dV].merge_clusters(i_boot, lag, inds_clusters)
                if (models_original.msm_input[dV].T[i_boot][lag].shape[0] != n_clusters_optimal):
                    raise ValueError('ERROR: wrong number of final clusters')
            models_original.msm_input[dV].plot_timescales(None, i_boot, title = 'after PCCA')
            models_original.msm_input[dV].plot_current(i_boot, title = 'after PCCA')
        self.msm_input = models_original.msm_input
        #--- Print the states of the final models
        print('Merged state with common clustering ')
        for i_state in range(n_states):
            print('State {}'.format(i_state))
            for i_dV, dV in enumerate(self.dVs_input):
                probs = self.msm_input[dV].probability(i_boot, lag_pcca)
                print('\t{}, prob = {}'.format(repr_state_full(self.msm_input[dV].states[i_boot][lag_pcca][i_state,:]), probs[i_state]))
        self.plot_states(i_boot)

    def find_high_probability(self, i_boot, lag, prob_min):
        """
        Return the shared set of states with prob > prob_min
        """
        #--- find all states with prob > prob_min
        states_high_prob = set()
        for i_dV, dV in enumerate(self.dVs_fitting):
            print('Selecting states with prob > {} at dV = {}'.format(prob_min, dV))
            i_states, states = self.msm_input[dV].states_above_probability_threshold(i_boot, lag, prob_min)
            for state in states:
                states_high_prob.add(state)
        #--- check that the high prob states are present in all models
        states2remove = set()
        for i_dV, dV in enumerate(self.dVs_fitting):
            for state in states_high_prob:
                if not self.msm_input[dV].has_state(i_boot, lag, state):
                    print('WARNING: state at high probability in one model is absent in another one')
                    states2remove.add(state)
        states_high_prob = states_high_prob - states2remove
        return list(states_high_prob)

    def find_n_high_average_probability_states(self, i_boot, lags, n_states = 10):
        """
        Return the shared set of n states with highest average probability
        """
        #--- find probs for all states
        states_prob = {}
        for i_dV, dV in enumerate(self.dVs_fitting):
            for i_lag, lag in enumerate(lags):
                probs = self.msm_input[dV].probability(i_boot, lag)
                for i_state, state in enumerate(self.msm_input[dV].states[i_boot][lag]):
                    state_repr = repr_state(state)
                    if state_repr not in states_prob:
                        states_prob[state_repr] = []
                    states_prob[state_repr].append(probs[i_state])
        states_selected = []
        probs_selected = []
        for state, prob in states_prob.items():
            if len(prob) == len(self.dVs_fitting): # the state appears at all voltages
                states_selected.append(state)
                probs_selected.append(np.mean(prob))
        states_selected = np.array(states_selected)
        probs_selected = np.array(probs_selected)
        states_selected = states_selected[np.argsort(probs_selected)[-n_states:]]
        return list(states_selected), np.sort(probs_selected)[-n_states:]

    def find_n_high_probability_states(self, i_boot, lag, n_states = 10):
        """
        Return the shared set of n states of highest probability
        """
        #--- find n highest probability states for all dVs
        states_high_prob = set()
        for i_dV, dV in enumerate(self.dVs_fitting):
            states_prob = {}
            probs = self.msm_input[dV].probability(i_boot, lag)
            states = np.array([repr_state(state) for state in self.msm_input[dV].states[i_boot][lag]])
            states = states[np.argsort(probs)[-n_states:]]
            states_high_prob = states_high_prob | set(states)
        #--- check that the high prob states are present in all models
        states2remove = set()
        for i_dV, dV in enumerate(self.dVs_fitting):
            for state in states_high_prob:
                if not self.msm_input[dV].has_state(i_boot, lag, state):
                    print('WARNING: state at high probability in one model is absent in another one')
                    states2remove.add(state)
        states_high_prob = states_high_prob - states2remove
        return list(states_high_prob)

    def select_high_probability(self, i_boot, lags, prob_min):
        states_high_prob = self.find_high_probability(i_boot, lags, prob_min)
        #--- merge all the models to the high probability states
        states_high_prob = list(states_high_prob)
        for i_dV, dV in enumerate(self.dVs_input):
            for i_lag, lag in enumerate(lags):
                self.msm_input[dV].merge_target(i_boot = i_boot, lag = lag, target_states = states_high_prob, keep_old_states = True)

    def align_models(self, i_boot, lags = None):
        """
        Force the models at i_boot:lags to have the same states in the same order at all dVs
        """
        #--- Inputs
        if isinstance(lags, int):
            lags = [lags,]
        elif lags is None:
            lags = self.lags
        #--- Find common states
        sets_states = [] # list of states at the various dVs
        for lag in lags:
            for dV, msm in self.msm_input.items():
                sets_states.append(set([repr_state(state) for state in msm.states[i_boot][lag]]))
        common_states = set.intersection(*sets_states) # states common in all conditions
        print('Number of common states: {}'.format(len(common_states)))
        #--- Initialize the models at all dVs
        states2order = []
        for lag in lags:
            for dV in self.dVs_input:
                inds_not_common = [i for i in range(self.msm_input[dV].states[i_boot][lag].shape[0]) if repr_state(self.msm_input[dV].states[i_boot][lag][i,:]) not in common_states]
                print('Removing {} over {} states from dV {} i_boot {} lag {}'.format(len(inds_not_common), self.msm_input[dV].T[i_boot][lag].shape[0], dV, i_boot, lag))
                self.msm_input[dV].T[i_boot][lag], self.msm_input[dV].F[i_boot][lag], self.msm_input[dV].states[i_boot][lag], inds2merge = self.msm_input[dV].merge_remove_indexes(lag, i_boot, inds_not_common)
                self.msm_input[dV].update_indexes_macro(lag, i_boot, inds2merge)
                states2order.append(self.msm_input[dV].states[i_boot][lag])
        #--- Reorder the models at all dVs to the same order
        inds_same_order = common_order(*states2order)
        iorder = 0
        for ilag, lag in enumerate(lags):
            for idV, dV in enumerate(self.dVs_input):
                self.msm_input[dV].T[i_boot][lag] = self.msm_input[dV].T[i_boot][lag][inds_same_order[iorder],:][:,inds_same_order[iorder]]
                if lag in self.msm_input[dV].T[i_boot]:
                    self.msm_input[dV].F[i_boot][lag] = self.msm_input[dV].F[i_boot][lag][inds_same_order[iorder],:][:,inds_same_order[iorder]]
                self.msm_input[dV].states[i_boot][lag] = self.msm_input[dV].states[i_boot][lag][inds_same_order[iorder],:]
                self.msm_input[dV].indexes_macro[i_boot][lag] = [self.msm_input[dV].indexes_macro[i_boot][lag][ind] for ind in inds_same_order[iorder]]
                iorder += 1

    def plot_states(self, pdf, i_boot):
        n_states = self.check_number_states(i_boot)
        n_sites = 6
        for i_dV, dV in enumerate(self.dVs_fitting):
            f = plt.figure()
            for i_site in range(n_sites):
                for i_state in range(n_states):
                    ax = f.add_subplot(n_sites, n_states, i_site*n_states+i_state+1)
                    y = [self.msm_input[dV].states[i_boot][lag][i_state, i_site] for lag in self.lags]
                    ax.plot(self.lags, y, 'o-')
                    ax.set_ylim([0, 1])
            pdf.savefig()

    def reset_originals(self, i_boot, lags = None):
        #--- Inputs
        if isinstance(lags, int):
            lags = [lags,]
        elif lags is None:
            lags = self.lags
        for ilag, lag in enumerate(lags):
            for idV, dV in enumerate(self.dVs_input):
                self.msm_input[dV].reset_originals(i_boot, lag)

    def check_number_states(self, i_boot, lags = None):
        """
        Check that all the models at i_boot have the same number of states at all lags and dVs
        """
        #--- Inputs
        if isinstance(lags, int) or isinstance(lags, np.number):
            lags = [lags,]
        elif lags is None:
            lags = self.lags
        #--- Define reference values
        n_states = self.msm_input[self.dVs_input[0]].states[i_boot][lags[0]].shape[0]
        #--- Test if states are consistent
        for ilag, lag in enumerate(lags):
            for idV, dV in enumerate(self.dVs_input):
                n_states_idV = self.msm_input[self.dVs_input[idV]].states[i_boot][lag].shape[0]
                if n_states_idV != n_states:
                    raise ValueError('ERROR: Number of states at dV {} = {} is not compatible with {}'.format(self.dVs_input[idV], n_states_idV, n_states))
        return n_states

    def check_states(self, i_boots, lags = None, permissive = False, dVs = None):
        """
        Check that all the models at all i_boots have the same states at all lags and dVs
        """
        #--- Inputs
        if isinstance(lags, int) or isinstance(lags, np.number):
            lags = [lags,]
        elif lags is None:
            lags = self.lags
        if dVs is None:
            dVs = self.dVs_input
        if isinstance(i_boots, int) or isinstance(i_boots, np.number):
            i_boots = [i_boots,]
        #--- Define reference values
        n_states = self.msm_input[dVs[0]].states[i_boots[0]][lags[0]].shape[0]
        states = self.msm_input[dVs[0]].states[i_boots[0]][lags[0]]
        #--- Test if states are consistent
        for i_boot in i_boots:
            for ilag, lag in enumerate(lags):
                for idV, dV in enumerate(dVs):
                    n_states_idV = self.msm_input[dVs[idV]].states[i_boot][lag].shape[0]
                    if n_states_idV != n_states:
                        raise ValueError('ERROR: Number of states at dV {} lag {} = {} is not compatible with {}'.format(dVs[idV], lag, n_states_idV, n_states))
                    for i_state in range(n_states):
                        state_i = states[i_state,:]
                        state_i_idV = self.msm_input[dVs[idV]].states[i_boot][lag][i_state,:]
                        if self.msm_input[dVs[idV]].repr_state(state_i_idV) != self.msm_input[dVs[0]].repr_state(state_i):
                            if permissive:
                                print('WARNING: States {} at dV {} is not compatible with state {} at dV {}'.format(state_i_idV, dVs[idV], state_i, dVs[0]))
                            else:
                                raise ValueError('ERROR: States {} at dV {} is not compatible with {}'.format(state_i_idV, dVs[idV], state_i))
        return n_states, states

    def constant_fitting(self, i_boot, lag, dVs_fitting, what = 'T'):
        """
        I use the same interface of linear_fitting but here beta == 0
        """
        if what not in ['T', 'F']:
            raise ValueError('ERROR: wrong parameter in linear_fitting')
        n_states = self.msm_input[dVs_fitting[0]].states[i_boot][lag].shape[0]
        alpha, beta = {}, {}
        for i_state in range(n_states):
            for j_state in range(n_states):
                if what == 'T':
                    fitting = np.array([self.msm_input[dV].T[i_boot][lag][i_state,j_state] for dV in dVs_fitting])
                elif what == 'F':
                    fitting = np.array([self.msm_input[dV].F[i_boot][lag][i_state,j_state] for dV in dVs_fitting])
                alpha[i_state, j_state] = np.mean(fitting)
                beta[i_state, j_state] = 0.0
        return alpha, beta

    def linear_fitting(self, i_boot, i_lag, dVs_fitting, what = 'T'):
        if what not in ['T', 'F', 'Q']:
            raise ValueError('ERROR: wrong parameter in linear_fitting')
        n_states = self.msm_input[dVs_fitting[0]].states[i_boot][self.lags[i_lag]].shape[0]
        alpha, beta = {}, {}
        for i_state in range(n_states):
            for j_state in range(n_states):
                if what == 'T':
                    fitting = np.array([self.msm_input[dV].T[i_boot][self.lags[i_lag]][i_state,j_state] for dV in dVs_fitting])
                elif what == 'F':
                    fitting = np.array([self.msm_input[dV].F[i_boot][self.lags[i_lag]][i_state,j_state] for dV in dVs_fitting])
                elif what == 'Q':
                    fitting = np.array([self.msm_input[dV].Q[i_boot][self.lags[i_lag]][i_state,j_state] for dV in dVs_fitting])
                X = np.concatenate((np.ones((len(dVs_fitting),1)), dVs_fitting.reshape((-1,1))), axis = 1)
                XTX = np.dot(np.transpose(X), X)
                iXTX = np.linalg.inv(XTX)
                XTy = np.dot(np.transpose(X), fitting.reshape((-1,1)))
                pars = np.dot(iXTX, XTy)
                alpha[i_state, j_state] = pars[0,0]
                beta[i_state, j_state] = pars[1,0]
        return alpha, beta

    def exponential_fitting_numerical(self, i_boot, i_lag, dVs_fitting, q_thr, what = 'Q'):

        def residual(x, fitting, dVs_fitting):
            return np.sum(np.power(fitting - x[0]*np.exp(x[1]*dVs_fitting), 2.0))

        def residual_diagonal(x, is_unfitted, dVs_fitting, fitting_out_diagonal, fitting_diagonal, others_Q):
            dist = 0.0
            sum_qs = np.zeros(len(dVs_fitting))
            for i, i_state in enumerate(is_unfitted):
                alpha = x[i*2]
                beta = x[i*2+1]
                for i_dV, dV in enumerate(dVs_fitting):
                    q = alpha*np.exp(beta*dV)
                    sum_qs[i_dV] += q
                    dist += np.power(fitting_out_diagonal[i*len(dVs_fitting)+i_dV] - q, 2.0)
            for i_dV, dV in enumerate(dVs_fitting):
                estimated_sum_q = -others_Q[i_dV] - sum_qs[i_dV]
                dist += np.power(fitting_diagonal[i_dV] - estimated_sum_q, 2.0)
            return dist

        if what not in ['Q']:
            raise ValueError('ERROR: wrong parameter in linear_fitting')
        alpha_anal, beta_anal = self.exponential_fitting(i_boot, self.lags[i_lag], dVs_fitting)
        n_states = self.msm_input[dVs_fitting[0]].states[i_boot][self.lags[i_lag]].shape[0]
        alpha, beta = {}, {}
        for j_state in range(n_states):
            is_unfitted, fitting_out_diagonal = [], []
            for i_state in range(n_states):
                if i_state != j_state:
                    if what == 'T':
                        fitting = np.array([self.msm_input[dV].T[i_boot][self.lags[i_lag]][i_state,j_state] for dV in dVs_fitting])
                    elif what == 'F':
                        fitting = np.array([self.msm_input[dV].F[i_boot][self.lags[i_lag]][i_state,j_state] for dV in dVs_fitting])
                    elif what == 'Q':
                        fitting = np.array([self.msm_input[dV].Q[i_boot][self.lags[i_lag]][i_state,j_state] for dV in dVs_fitting]) # these are all the fitting values for the i,j pair
                    if np.all(fitting < q_thr): # all fitting values are below the threshold, this elements are set to ZERO 
                        print('DEBUG> j_state = {} i_state = {} both zero, fitting = {}'.format(j_state, i_state, fitting))
                        alpha[i_state, j_state] = 0.0
                        beta[i_state, j_state] = 0.0
                    elif np.sum(fitting > q_thr) == 1: # just one element is above the threshold, this will be fitted later on...
                        print('DEBUG> j_state = {} i_state = {} one zero, fitting = {}'.format(j_state, i_state, fitting))
                        res = minimize(residual, [np.exp(alpha_anal[i_state, j_state]), beta_anal[i_state, j_state]], bounds = [(0, 1e10)    , (-0.01, 0.01)], args = (fitting, dVs_fitting), options = {'maxiter':int(1e3)}, tol = 1e-10) # fit with threshold for later refinement
                        alpha[i_state, j_state] = res.x[0]
                        beta[i_state, j_state] = res.x[1]
                        is_unfitted.append(i_state)
                        fitting_out_diagonal.extend([x for x in fitting])
                    else: # here more than one element is above the threshold, we can fit alpha and beta
                        print('DEBUG> j_state = {} i_state = {} both non zero, fitting = {}'.format(j_state, i_state, fitting))
                        res = minimize(residual, [np.exp(alpha_anal[i_state, j_state]), beta_anal[i_state, j_state]], bounds = [(0, None)    , (None, None)], args = (fitting, dVs_fitting), options = {'maxiter':int(1e3)}, tol = 1e-10)
                        alpha[i_state, j_state] = res.x[0]
                        beta[i_state, j_state] = res.x[1]
                    print('DEBUG> j_state = {} i_state = {} alpha = {} beta = {}'.format(j_state, i_state, alpha[i_state, j_state], beta[i_state, j_state]))
            if is_unfitted:
                fitting_diagonal = np.array([self.msm_input[dV].Q[i_boot][self.lags[i_lag]][j_state,j_state] for dV in dVs_fitting]) # the diagonal elements at all the fitting voltages
                x0 = []
                for i_state in is_unfitted:
                    x0.append(alpha[i_state, j_state])
                    x0.append(beta[i_state, j_state])
                    print('DEBUG> BEFORE: i_state = {} alpha = {} beta = {}'.format(i_state, alpha[i_state, j_state], beta[i_state, j_state]))
                others_Q = []
                for dV in dVs_fitting:
                    other_Q = 0.0
                    for i_state in range(n_states):
                        if (i_state != j_state) and i_state not in is_unfitted:
                            other_Q += self.msm_input[dV].Q[i_boot][self.lags[i_lag]][i_state, j_state]
                    others_Q.append(other_Q)
                print('DEBUG> BEFORE: cost = {}'.format(residual_diagonal(x0, is_unfitted, dVs_fitting, fitting_out_diagonal, fitting_diagonal, others_Q)))
                res = minimize(residual_diagonal, x0, bounds = [(0, None), (None, None)]*len(is_unfitted), args = (is_unfitted, dVs_fitting, fitting_out_diagonal, fitting_diagonal, others_Q), options = {'maxiter':int(1e3)}, tol = 1e-10)
                for i, i_state in enumerate(is_unfitted):
                    alpha[i_state, j_state] = res.x[i*2]
                    beta[i_state, j_state] = res.x[i*2+1]
                    print('AFTER: i_state = {} alpha = {} beta = {}'.format(i_state, alpha[i_state, j_state], beta[i_state, j_state]))
                print('DEBUG> AFTER: cost = {}'.format(residual_diagonal(res.x, is_unfitted, dVs_fitting, fitting_out_diagonal, fitting_diagonal, others_Q)))
        return alpha, beta

    def linear_fitting_numerical(self, i_boot, lag, dVs_fitting, what = 'T'):
        def residual(x, fitting, dVs_fitting):
            return np.sum(np.power(fitting - (x[0] + x[1]*dVs_fitting), 2.0))
        if what not in ['T', 'F', 'Q']:
            raise ValueError('ERROR: wrong parameter in linear_fitting')
        n_states = self.msm_input[dVs_fitting[0]].states[i_boot][lag].shape[0]
        alpha, beta = {}, {}
        for i_state in range(n_states):
            for j_state in range(n_states):
                if what == 'T':
                    fitting = np.array([self.msm_input[dV].T[i_boot][lag][i_state,j_state] for dV in dVs_fitting])
                elif what == 'F':
                    fitting = np.array([self.msm_input[dV].F[i_boot][lag][i_state,j_state] for dV in dVs_fitting])
                elif what == 'Q':
                    fitting = np.array([self.msm_input[dV].Q[i_boot][lag][i_state,j_state] for dV in dVs_fitting])
                res = minimize(residual, [0.0, np.mean(fitting)], args = (fitting, dVs_fitting), options = {'maxiter':int(1e6)}, tol = 1e-10)
                alpha[i_state, j_state] = res.x[0]
                beta[i_state, j_state] = res.x[1]
        return alpha, beta

    def exponential_fitting(self, i_boot, lag, dVs_fitting):
        n_states = self.msm_input[dVs_fitting[0]].states[i_boot][lag].shape[0]
        alpha, beta = {}, {}
        min_Q = np.inf
        for dV in dVs_fitting:
            min_Q = min(min_Q, np.min(self.msm_input[dV].Q[i_boot][lag][self.msm_input[dV].Q[i_boot][lag] > 0]))
        for i_state in range(n_states):
            for j_state in range(n_states):
                if i_state != j_state:
                    fitting = np.array([self.msm_input[dV].Q[i_boot][lag][i_state,j_state] for dV in dVs_fitting])
                    fitting[fitting == 0] = 0.1*min_Q # to avoid 0 and so -inf logarithm
                    X = np.concatenate((np.ones((len(dVs_fitting),1)), dVs_fitting.reshape((-1,1))), axis = 1)
                    XTX = np.dot(np.transpose(X), X)
                    iXTX = np.linalg.inv(XTX)
                    XTy = np.dot(np.transpose(X), np.log(fitting.reshape((-1,1))))
                    pars = np.dot(iXTX, XTy)
                    alpha[i_state, j_state] = pars[0,0]
                    beta[i_state, j_state] = pars[1,0]
        return alpha, beta

    def define_linear(self, i_boot, lag, alpha, beta, dV):
        n_states = self.msm_input[self.dVs_fitting[0]].states[i_boot][lag].shape[0]
        F = np.zeros((n_states, n_states))
        for i_state in range(n_states):
            for j_state in range(n_states):
                F[i_state, j_state] = alpha[i_state, j_state] + beta[i_state, j_state]*dV
        return F

    def define_Q(self, i_boot, lag, alpha, beta, dV, linear = False):
        n_states = self.msm_input[self.dVs_fitting[0]].states[i_boot][lag].shape[0]
        Q = np.zeros((n_states, n_states))
        for i_state in range(n_states):
            for j_state in range(n_states):
                if i_state != j_state:
                    if linear:
                        Q[i_state, j_state] = alpha[i_state, j_state] + beta[i_state, j_state]*dV
                    else:
                        Q[i_state, j_state] = alpha[i_state, j_state]*np.exp(beta[i_state, j_state]*dV)
        for j in range(n_states):
            Q[j,j] = -np.sum(Q[:,j])
        return Q

    def check_T(self, i_boot, lag, alpha, beta):
        eps = 0
        n_states = self.msm_input[self.dVs_input[0]].states[i_boot][lag].shape[0]
        T = {}
        for idV, dV in enumerate(self.dVs_output):
            T[dV] = np.zeros( (n_states, n_states) )
            for i_state in range(n_states):
                for j_state in range(n_states):
                    T[dV][i_state, j_state] = alpha[i_state, j_state] + beta[i_state, j_state]*dV
                    if T[dV][i_state, j_state] < 0:
                        T[dV][i_state, j_state] = eps 
                    if T[dV][i_state, j_state] > 1:
                        T[dV][i_state, j_state] = 1 - eps
            T[dV] = T[dV] / np.sum(T[dV], axis = 0).reshape((1,-1))
        return T

    def check_Q(self, i_boot, lag, alpha, beta):
        n_states = self.msm_input[self.dVs_fitting[0]].states[i_boot][lag].shape[0]
        Q = {}
        for idV, dV in enumerate(self.dVs_output):
            Q[dV] = np.zeros( (n_states, n_states) )
            for i_state in range(n_states):
                for j_state in range(n_states):
                    Q[dV][i_state, j_state] = alpha[i_state, j_state] + beta[i_state, j_state]*dV
            for j_state in range(n_states):
                Q[dV][j_state,j_state] = 0.0
                Q[dV][j_state,j_state] = -np.sum(Q[dV][:,j_state])
        return Q

    def current_voltage(self, pdf, dVs_output, i_boots = [-1,], lag4plot = None, plot_dV_range = None):
        """
        Add models at other voltages with the following method:
            * Check that the inputs models are aligned at all boots and lags
            * Estimate Q as exponential function of dVs
            * Estimate F at lag = 1 as linear function of dVs

        Parameters
        ----------
        dVs_fitting: list/np.array
            Values of membrane potential used for fitting
        dVs_output: list/np.array
            Values of membrane potential at which current will be estimated
        """
        def residual_qs_db(x, alpha_start, beta_start, F):
            i_alpha, i_beta = 0, 1
            alpha, beta = {}, {}
            for i_state in range(n_states):
                for j_state in range(n_states):
                    if j_state != i_state:
                        alpha[i_state, j_state] = x[i_alpha]
                        beta[i_state, j_state] = x[i_beta]
                        i_alpha += 2
                        i_beta += 2
            dist = 0.0
            Q_dV0 = self.define_Q(i_boot, lag, alpha, beta, 0.0)
            prob_dV0 = msm.utils_model.probability_Q(Q_dV0)
            for i_state in range(n_states):
                for j_state in range(i_state+1, n_states-1):
                    if j_state != i_state:
                        dist += (prob_dV0[j_state]*Q_dV0[i_state, j_state] - prob_dV0[i_state]*Q_dV0[j_state, i_state])**2.0
            return dist
        q_thr = 1e-5
        #--- Input definition & control
        if isinstance(i_boots, int):
            i_boots = [i_boots,]
        if not isinstance(i_boots, list):
            raise TypeError('ERROR: wrong type for i_boots')
        self.dVs_output = dVs_output
        self.msm_output = {}
        #--- Check states
        n_states, states = self.check_states(i_boots, dVs = self.dVs_fitting)
        #--- Estimate models for all i_boots
        Q = {dV:{} for dV in self.dVs_output}
        alpha = {i_boot:{} for i_boot in i_boots}
        beta = {i_boot:{} for i_boot in i_boots}
        for i_boot in i_boots:
            #--- Fitting F at 1st lag
            print('Estimating F at lag: {}'.format(self.lags[0]))
            alphaF, betaF = self.constant_fitting(i_boot, self.lags[0], self.dVs_fitting, what = 'F')
            F = self.define_linear(i_boot, 1, alphaF, betaF, self.dVs_fitting[0])
            #--- Exponential Fitting Q
            #alpha, beta = {}, {}
            for i_lag, lag in enumerate(self.lags):
                print('Estimating alphas and betas for Q at lag:',lag)
                alpha[i_boot][lag], beta[i_boot][lag] = self.exponential_fitting_numerical(i_boot, i_lag, self.dVs_fitting, q_thr = q_thr)
            #--- Define Q for output models
            for dV in self.dVs_output:
                Q[dV][i_boot] = {}
                for lag in self.lags:
                    Q[dV][i_boot][lag] = self.define_Q(i_boot, lag, alpha[i_boot][lag], beta[i_boot][lag], dV) #, linear = True)
        #--- Create the output models
        for dV in self.dVs_output:
            self.msm_output[dV] = msm.MSM_from_matrixes(self.dt, self.lags, None, F = F, states = states, Q = Q[dV])
        #--- Plot Q
        if lag4plot is not None:
            #self.plot_matrix(pdf, -1, lag4plot, what = 'Q', alpha = alpha, beta = beta)
            self.plot_matrix(pdf, self.find_boots()[-1], lag4plot, what = 'Q', alpha = alpha, beta = beta, plot_dV_range = plot_dV_range)
        #--- propagate F at successive lags
        #self.msm_output[dV].propagate_F(-1)
        #--- Plot T fitting from Q
        #self.plot_matrix(i_boots[0], lag4plot, what = 'I')
        #if lag4plot is not None:
        #    self.plot_matrix(i_boots[0], lag4plot, what = 'T')

    def plot_current_voltage(self, pdf, lag, i_boots = [-1,], exp_data = {}, i_boots_input = None, plot_current_range = None):
        if i_boots_input is None:
            for i_dV, dV in enumerate(self.dVs_input):
                if i_boots_input is None:
                    i_boots_input = set(self.msm_input[dV].i_boots)
                else:
                    i_boots_input = i_boots_input & set(self.msm_input[dV].i_boots)
            i_boots_input = [i_boot for i_boot in i_boots_input if i_boot >= 0]
            i_boots_input.sort()
        if len(i_boots_input) == 0:
            i_boots_input = i_boots
        f = plt.figure()
        ax = f.add_subplot(1,1,1)
        dV_exp = exp_data.keys()
        I_exp = exp_data.values()
        #--- Current in MD trajectories
        for dV_exp, Is_exp in exp_data.items():
            for I_exp in Is_exp:
                ax.plot(dV_exp, I_exp, 'xk', label = None)
        #--- Current in input models
        I = np.empty(len(self.dVs_input))
        I_std = np.zeros(len(self.dVs_input))
        for idV, dV in enumerate(self.dVs_input):
            print('DIO PORCO',dV)
            current = self.msm_input[dV].current_from_md(lag, i_boots_input)
            I[idV] = np.mean(current)
            if len(current) > 1:
                I_std[idV] = np.std(current)
        #if np.all(I_std == 0):
        #    ax.plot(self.dVs_input, I, 'or', label = 'input [Q]')
        #else:
        #    ax.errorbar(self.dVs_input, I, yerr = I_std, label = 'input [Q]', linestyle='', marker = 'o')
        #--- Current in fitting models
        I = np.empty(len(self.dVs_fitting))
        I_std = np.zeros(len(self.dVs_fitting))
        for idV, dV in enumerate(self.dVs_fitting):
            current = self.msm_input[dV].current_from_md(lag, i_boots_input)
            I[idV] = np.mean(current)
            if len(current) > 1:
                I_std[idV] = np.std(current)
        #if np.all(I_std == 0):
        #    ax.plot(self.dVs_fitting, I, 'ob', label = 'fitting [Q]')
        #else:
        #    ax.errorbar(self.dVs_fitting, I, yerr = I_std, label = 'fitting [Q]', linestyle='', marker = 'o')
        #--- Current in output models
        if len(self.dVs_output):
            I = np.empty(len(self.dVs_output))
            I_std = np.zeros(len(self.dVs_output))
            currents = []
            for idV, dV in enumerate(self.dVs_output):
                current = self.msm_output[dV].current_from_Q(lag, i_boots)
                I[idV] = np.mean(current)
                if len(current) > 1:
                    I_std[idV] = np.std(current)
                currents.append(current)
                print('MSM(dV) I = {} pA V = {} mV gamma = {} ± {} pS'.format(I[idV], dV, 1e3*I[idV]/dV, 1e3*I_std[idV]/dV))
            currents = np.array(currents)
            if np.all(I_std == 0):
                ax.plot(self.dVs_output, I, ':xg', label = 'estimated')
            else:
                #ax.plot(self.dVs_output, I, '-xg', label = 'estimated')
                #ax.plot(self.dVs_output, I+I_std, ':g', label = None)
                #ax.plot(self.dVs_output, I-I_std, ':g', label = None)
                plt.fill_between(self.dVs_output, I-I_std, I+I_std, alpha=0.3, label="estimated")
                #for i_boot in range(currents.shape[1]):
                #    ax.plot(self.dVs_output, currents[:,i_boot], ':k', label = None)

        plt.xlabel('Membrane Potential [dV]')
        plt.ylabel('Current [pA]')
        plt.title('{} [{}:{}] ({} ns)'.format(self.prefix, i_boots, lag, lag*self.dt))
        plt.grid()
        plt.legend()
        pdf.savefig()
        if plot_current_range is not None:
            plt.xlim(plot_current_range[0])
            plt.ylim(plot_current_range[1])
        pdf.savefig()
        plt.close()

    def only_MD(self):
        for dV, msm in self.msm_input.items():
            msm.only_MD()

    def TF_from_trajs(self, i_boot):
        for dV, msm in self.msm_input.items():
            msm.TF_from_trajs(i_boot)

    def find_lag(self, i_boot, thrs):
        stable_lags = []
        for dV, msm in self.msm_input.items():
            stable_lags.append(msm.find_lag(i_boot, thrs))
        print('Taus stable at {} for lags >= {}'.format(thrs, max(stable_lags)))

    def plot_timescales(self, pdf, i_boot):
        for dV, msm in self.msm_input.items():
            msm.plot_timescales(pdf, n_timescales = None, i_boots = i_boot)

    def plot_current(self, pdf, i_boot):
        for dV, msm in self.msm_input.items():
            msm.plot_current(pdf, i_boot = i_boot)

    def plot_current_all_lags(self, i_boot):
        f = plt.figure()
        ax = f.add_subplot(1,1,1)
        for dV, msm in self.msm_input.items():
            I = np.empty(len(msm.lags))
            for i_lag, lag in enumerate(msm.lags):
                I[i_lag] = msm.current(lag, i_boot)
            ax.plot(msm.dt*msm.lags, I, 'o-', label = 'input')
        for dV in self.dVs_output:
            I = np.empty(len(self.lags))
            for i_lag, lag in enumerate(self.lags):
                I[i_lag] = self.msm_output[dV].current(lag, i_boot)
            ax.plot(self.dt*np.array(self.lags), I, 'o-', label = 'output')
        plt.xlabel('Sampling period [ns]')
        plt.ylabel('Current [pA]')
        plt.title(self.prefix)
        plt.grid()
        plt.legend()
        pdf.savefig()
        plt.close()

    def exist_rate_matrix(self, dVs, i_boots):
        if isinstance(i_boots, int):
            i_boots = [i_boots,]
        for dV in dVs:
            for i_boot in i_boots:
                if i_boot not in self.msm_input[dV].Q:
                    return False
        return True

    def plot_matrix(self, pdf, i_boots, lag, i_boots_input = None, what = 'T', plot_dV_range = None, **kwargs):
        from scipy import stats
        if len(i_boots) < 2:
            print('WARNING: plot_matrix only works when bootstraps are available')
            return
        i_boots_input = self.find_boots()[-1]
        flag_q_input = self.exist_rate_matrix(self.dVs_input, i_boots_input)
        scale = (constants.e*1e12)/(1e-9)
        if isinstance(i_boots, int):
            i_boots = [i_boots,]
        if what not in ['T', 'F', 'Q', 'I']:
            raise ValueError('ERROR: wrong parameter in linear_fitting')
        n_states, dummy = self.check_states(i_boots, lag)
        probs_input = {}
        probs_fitting = {}
        probs_output = {}
        for i_dV, dV in enumerate(self.dVs_input):
            probs_input[dV] = self.msm_input[dV].probability(i_boots_input, lag, from_Q = flag_q_input)
        for i_dV, dV in enumerate(self.dVs_fitting):
            probs_fitting[dV] = self.msm_input[dV].probability(i_boots, lag, from_Q = True)
        for i_dV, dV in enumerate(self.dVs_output):
            if dV in self.msm_output:
                if lag in self.msm_output[dV].Q[-1]:
                    probs_output[dV] = self.msm_output[dV].probability(i_boots, lag, from_Q = True)
        dist_output_fitting = 0.0
        total_fitting = 0.0
        dVs_output_fitting = np.sort(np.array([dV for dV in set(self.dVs_fitting) & set(self.dVs_output)]))
        for j_state in range(n_states):
            for i_state in range(n_states):
                ts_output = None
                if what == 'T':
                    ts_input = np.array([self.msm_input[dV].T[i_boot][lag][i_state,j_state] for dV in self.dVs_input])
                    if len(self.dVs_output):
                        if self.dVs_output[0] in self.msm_output:
                            if lag in self.msm_output[self.dVs_output[0]].T[-1]:
                                ts_output = np.array([self.msm_output[dV].T[-1][lag][i_state,j_state] for dV in self.dVs_output])
                    ts_fitting = np.array([self.msm_input[dV].T[i_boot][lag][i_state,j_state] for dV in self.dVs_fitting])
                elif what == 'F':
                    if flag_q_input:
                        ts_input = np.array([self.msm_input[dV].F[i_boot][lag][i_state,j_state] for dV in self.dVs_input])
                    if len(self.dVs_output):
                        if self.dVs_output[0] in self.msm_output:
                            if lag in self.msm_output[self.dVs_output[0]].F[-1]:
                                ts_output = np.array([self.msm_output[dV].F[-1][lag][i_state,j_state] for dV in self.dVs_output])
                    ts_fitting = np.array([self.msm_input[dV].F[i_boot][lag][i_state,j_state] for dV in self.dVs_fitting])
                elif what == 'Q':
                    ts_input = []
                    for i_boot_dummy in i_boots_input:
                        ts_input.append(np.array([self.msm_input[dV].Q[i_boot_dummy][lag][i_state,j_state] for dV in self.dVs_input]).reshape((1,-1)))
                    ts_input = np.concatenate(ts_input)
                    if len(self.dVs_output):
                        if self.dVs_output[0] in self.msm_output:
                            if lag in self.msm_output[self.dVs_output[0]].Q[-1]:
                                #ts_output = np.array([self.msm_output[dV].Q[-1][lag][i_state,j_state] for dV in self.dVs_output])
                                ts_output = []
                                for i_boot_dummy in i_boots:
                                    ts_output.append(np.array([self.msm_output[dV].Q[i_boot_dummy][lag][i_state,j_state] for dV in self.dVs_output]).reshape((1,-1)))
                                ts_output = np.concatenate(ts_output)
                    #ts_fitting = np.array([self.msm_input[dV].Q[i_boot][lag][i_state,j_state] for dV in self.dVs_fitting])
                    ts_fitting = []
                    for i_boot_dummy in i_boots:
                        ts_fitting.append(np.array([self.msm_input[dV].Q[i_boot_dummy][lag][i_state,j_state] for dV in self.dVs_fitting]).reshape((1,-1)))
                    ts_fitting = np.concatenate(ts_fitting)
                elif what == 'I':
                    ts_input = np.array([scale*self.msm_input[dV].Q[i_boot][lag][i_state,j_state]*self.msm_input[dV].F[i_boot][self.lags[0]][i_state,j_state]*probs_input[dV][j_state] for dV in self.dVs_input])
                    if len(self.dVs_output):
                        if self.dVs_output[0] in self.msm_output:
                            if lag in self.msm_output[self.dVs_output[0]].Q[-1]:
                                ts_output = np.array([scale*self.msm_output[dV].Q[i_boot][lag][i_state,j_state]*self.msm_output[dV].F[i_boot][self.lags[0]][i_state,j_state]*probs_output[dV][j_state] for dV in self.dVs_output])
                                dist_output_fitting += np.sum(np.abs(np.array([(  scale*self.msm_input[dV].Q[i_boot][lag][i_state,j_state]*self.msm_input[dV].F[i_boot][self.lags[0]][i_state,j_state]*probs_fitting[dV][j_state] 
                                                                                - scale*self.msm_output[dV].Q[i_boot][lag][i_state,j_state]*self.msm_output[dV].F[i_boot][self.lags[0]][i_state,j_state]*probs_output[dV][j_state] )
                                                                               for dV in dVs_output_fitting])))
                                total_fitting += np.sum(np.abs(np.array([(  scale*self.msm_input[dV].Q[i_boot][lag][i_state,j_state]*self.msm_input[dV].F[i_boot][self.lags[0]][i_state,j_state]*probs_fitting[dV][j_state]  )
                                                                               for dV in dVs_output_fitting])))
                    ts_fitting = np.array([scale*self.msm_input[dV].Q[i_boot][lag][i_state,j_state]*self.msm_input[dV].F[i_boot][self.lags[0]][i_state,j_state]*probs_fitting[dV][j_state] for dV in self.dVs_fitting])
                f = plt.figure()
                ax1 = f.add_subplot(1,1,1)
                #ax1.plot(self.dVs_input, np.mean(ts_input, axis = 0), 'xk')
                if len(i_boots_input) > 1:
                    #for ind_seq_boot, i_boot_dummy in enumerate(i_boots_input):
                    #    ax1.plot(self.dVs_input, ts_input[ind_seq_boot,:], label = '{}'.format(i_boot_dummy)) # ok
                    ax1.errorbar(self.dVs_input, np.mean(ts_input, axis = 0), yerr = np.std(ts_input, axis = 0), linestyle ='', label = None, ecolor = 'k', marker = None, capsize = 5)
                ax1.errorbar(self.dVs_fitting, np.mean(ts_fitting, axis = 0), yerr = np.std(ts_fitting, axis = 0), linestyle ='', label = None, ecolor = 'b', marker = None, capsize = 5)
                #ax1.plot(self.dVs_fitting, np.mean(ts_fitting, axis = 0), 'ob')
                if 'alpha' in kwargs and 'beta' in kwargs:
                    if kwargs.get('linear_fitting', False):
                        ax1.plot(self.dVs_output, kwargs['alpha'][i_boot][lag][i_state, j_state] + kwargs['beta'][i_boot][lag][i_state, j_state]*self.dVs_output, ':r')
                    elif what == 'T' or what == 'F':
                        ax1.plot(self.dVs_output, kwargs['alpha'][i_boot][lag][i_state, j_state] + kwargs['beta'][i_boot][lag][i_state, j_state]*self.dVs_output, ':r')
                    elif what == 'Q':
                        if i_state != j_state:
                            qs = np.zeros((len(i_boots), len(self.dVs_output)))
                            betas = [kwargs['beta'][i_boot][lag][i_state, j_state] for i,i_boot in enumerate(i_boots)]
                            t_stat, p_val = stats.ttest_1samp(betas, 0)
                            print('i_state {} j_states {} <beta> {} sigma(beta) {} p-val {}'.format(i_state, j_state, np.mean(betas), np.std(betas), p_val))
                            for i_dV, dV in enumerate(self.dVs_output):
                                for i, i_boot in enumerate(i_boots):
                                    qs[i, i_dV] = kwargs['alpha'][i_boot][lag][i_state, j_state]*np.exp(kwargs['beta'][i_boot][lag][i_state, j_state]*dV)
                            #for i_boot in i_boots:
                            #    #ax1.plot(self.dVs_output, kwargs['alpha'][i_boot][lag][i_state, j_state]*np.exp(kwargs['beta'][i_boot][lag][i_state, j_state]*self.dVs_output), ':r')
                            #    ax1.plot(self.dVs_output, qs[i_boot,:], ':r')
                            #ax1.errorbar(self.dVs_output, np.mean(qs, axis = 0), yerr = np.std(qs, axis = 0), linestyle ='', label = None, ecolor = 'r')
                            plt.fill_between(self.dVs_output, np.mean(qs, axis = 0) - np.std(qs, axis = 0), np.mean(qs, axis = 0) + np.std(qs, axis = 0), alpha=0.3, label="None")
                            plt.ylim(bottom=0)
                        else:
                            #qs = np.zeros(len(self.dVs_output))
                            qs = np.zeros((len(i_boots), len(self.dVs_output)))
                            for i_dV, dV in enumerate(self.dVs_output):
                                for k_state in range(n_states):
                                    if k_state != j_state:
                                        for i, i_boot in enumerate(i_boots):
                                            qs[i, i_dV] -= kwargs['alpha'][i_boot][lag][k_state, j_state]*np.exp(kwargs['beta'][i_boot][lag][k_state, j_state]*dV)
                            #for i_boot in i_boots:
                            #    ax1.plot(self.dVs_output, qs[i_boot,:], ':r')
                            #ax1.errorbar(self.dVs_output, np.mean(qs, axis = 0), yerr = np.std(qs, axis = 0), linestyle ='', label = None, ecolor = 'r')
                            plt.fill_between(self.dVs_output, np.mean(qs, axis = 0) - np.std(qs, axis = 0), np.mean(qs, axis = 0) + np.std(qs, axis = 0), alpha=0.3, label="None")
                            plt.ylim(top=0)
                if ts_output is not None:
                    #ax1.plot(self.dVs_output, ts_output, ':g')
                    if what == 'I':
                        plt.title('{}[{}:{}] i_state = {} j_state = {}, dist = {} / {} = {}'.format(what, i_boot, lag, i_state, j_state, np.round(dist_output_fitting, 2), np.round(total_fitting, 2), np.round(100.0*dist_output_fitting/total_fitting, 1)))
                    else:
                        plt.title('{}[{}:{}] i_state = {} j_state = {}'.format(what, i_boot, lag, i_state, j_state))
                else:
                    plt.title('{}[{}:{}] i_state = {} j_state = {}'.format(what, i_boot, lag, i_state, j_state))
                ax1.set_xlabel('Membrane Potential [mV]')
                ax1.set_ylabel(what + '[ns^-1]')
                if plot_dV_range is not None:
                    plt.xlim(plot_dV_range)
                plt.grid()
                #plt.legend()
                pdf.savefig()
                plt.close()

    def plot_states(self, pdf, lag, i_boots = [-1,], i_boots_input = None, plot_dV_range = None):
        #--- Input definition & control
        if isinstance(i_boots, int):
            i_boots = [i_boots,]
        if not isinstance(i_boots, list):
            raise TypeError('ERROR: wrong type for i_boots')
        if i_boots_input is None:
            for i_dV, dV in enumerate(self.dVs_input):
                if i_boots_input is None:
                    i_boots_input = set(self.msm_input[dV].i_boots)
                else:
                    i_boots_input = i_boots_input & set(self.msm_input[dV].i_boots)
            if len(i_boots) == 1:
                i_boots_input = i_boots
            else:
                i_boots_input = [i_boot for i_boot in i_boots_input if i_boot >= 0]
            i_boots_input.sort()
        flag_q_input = self.exist_rate_matrix(self.dVs_input, i_boots_input)
        #--- check states
        n_states, states = self.check_states(i_boots, lag, dVs = self.dVs_fitting)
        #--- computing probs
        probs_fitting = {'Q':{},'T':{}}
        for i_dV, dV in enumerate(self.dVs_fitting):
            probs_fitting['Q'][dV] = []
            probs_fitting['T'][dV] = []
            for i_boot in i_boots:
                probs_fitting['Q'][dV].append(self.msm_input[dV].probability(i_boot, lag, from_Q = True))
                probs_fitting['T'][dV].append(self.msm_input[dV].probability(i_boot, lag))
            probs_fitting['Q'][dV] = np.array(probs_fitting['Q'][dV])
            probs_fitting['T'][dV] = np.array(probs_fitting['T'][dV])
        probs_input = {}
        for i_dV, dV in enumerate(self.dVs_input):
            probs_input[dV] = self.msm_input[dV].probability(i_boots_input, lag, from_Q = flag_q_input)
            #probs_input[dV] = []
            #for i_boot in i_boots_input:
            #    probs_input[dV].append(self.msm_input[dV].probability(i_boot, lag))
            #probs_input[dV] = np.array(probs_input[dV])
        probs_output = {}
        for i_dV, dV in enumerate(self.dVs_output):
            probs_output[dV] = []
            for i_boot in i_boots:
                probs_output[dV].append(self.msm_output[dV].probability(i_boot, lag, from_Q = True))
            probs_output[dV] = np.array(probs_output[dV])
        #--- Plotting
        for i_state in range(n_states):
            prob_fitting_Q, prob_fitting_T, prob_input, prob_output = np.zeros((len(i_boots), len(self.dVs_fitting))), np.zeros((len(i_boots), len(self.dVs_fitting))), np.zeros((len(i_boots_input), len(self.dVs_input))), np.zeros((len(i_boots), len(self.dVs_output)))
            for i_dV, dV in enumerate(self.dVs_fitting):
                prob_fitting_Q[:,i_dV] = probs_fitting['Q'][dV][:,i_state]
                prob_fitting_T[:,i_dV] = probs_fitting['T'][dV][:,i_state]
            for i_dV, dV in enumerate(self.dVs_input):
                prob_input[:,i_dV] = probs_input[dV][:,i_state]
            for i_dV, dV in enumerate(self.dVs_output):
                prob_output[:,i_dV] = probs_output[dV][:,i_state]
            print(prob_input)
            print(prob_fitting_Q)
            print(prob_fitting_T)
            print(prob_output)
            f = plt.figure()
            ax = f.add_subplot(1,1,1)
            #ax.plot(self.dVs_input, np.mean(prob_input, axis = 0), 'xm', label = 'input')
            if prob_input.shape[0] > 1:
                ax.errorbar(self.dVs_input, np.mean(prob_input, axis = 0), yerr = np.std(prob_input, axis = 0), linestyle ='', label = None, ecolor = 'k', capsize = 5)
            else:
                ax.plot(self.dVs_input, prob_input.flatten(), 'xk')
            #if prob_fitting_Q.shape[0] > 1:
            #    ax.errorbar(self.dVs_fitting, np.mean(prob_fitting_Q, axis = 0), yerr = np.std(prob_fitting_Q, axis = 0), linestyle ='', label = None, ecolor = 'r')
            if prob_fitting_T.shape[0] > 1:
                ax.errorbar(self.dVs_fitting, np.mean(prob_fitting_T, axis = 0), yerr = np.std(prob_fitting_T, axis = 0), linestyle ='', label = None, ecolor = 'b', capsize = 5)
            else:
                ax.plot(self.dVs_fitting, prob_fitting_T.flatten(), 'xb')
            if prob_output.shape[0] > 1:
                #ax.plot(self.dVs_output, np.mean(prob_output, axis = 0) + np.std(prob_output, axis = 0), ':b', label = None)
                #ax.plot(self.dVs_output, np.mean(prob_output, axis = 0) - np.std(prob_output, axis = 0), ':b', label = None)
                #ax.errorbar(self.dVs_output, np.mean(prob_output, axis = 0), yerr = np.std(prob_output, axis = 0), linestyle ='', label = None, ecolor = 'b')
                plt.fill_between(self.dVs_output, np.mean(prob_output, axis = 0) - np.std(prob_output, axis = 0), np.mean(prob_output, axis = 0) + np.std(prob_output, axis = 0), alpha=0.3, label="None")
            else:
                ax.plot(self.dVs_output, np.mean(prob_output, axis = 0), ':b', label = 'output - Q')
            plt.xlabel('Membrane potential [dV]')
            #plt.legend()
            plt.title('[{}:{}] {}'.format(i_boot, lag, self.msm_input[self.dVs_input[0]].repr_state(states[i_state,:])))
            plt.ylim(bottom=0)
            if plot_dV_range is not None:
                plt.xlim(plot_dV_range)
            pdf.savefig()
            plt.close()

    def print_states(self, i_boot, lag):
        n_states, states = self.check_states(i_boot, lag, dVs = self.dVs_fitting)
        for i_state in range(n_states):
            print('State {}'.format(i_state))
            for i_dV, dV in enumerate(self.dVs_input):
                print('\t{}'.format(repr_state(self.msm_input[dV].states[i_boot][lag][i_state,:])))
                print('\t\t{}'.format(repr_state_full(self.msm_input[dV].states[i_boot][lag][i_state,:])))
        print(80*'-')
        for i_state in range(n_states):
            print('\t{}'.format(repr_state(self.msm_input[self.dVs_fitting[0]].states[i_boot][lag][i_state,:])))

    def __str__(self):
        output = ''
        for i_dV, dV in enumerate(self.dVs_input):
            output += 20*'-'+'\n'
            output +='dV = {}\n'.format(dV)
            output += 20*'-'+'\n'
            output += self.msm_input[dV].__str__()+'\n'
        return output
