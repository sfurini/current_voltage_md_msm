"""
Definition of the class MSM
"""

import pickle
import copy

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy import linalg

from . import utils_channel
from . import utils_model

class TransitionMatrix(object):
    def __init__(self, file_dtrajs):
        data = np.load(file_dtrajs)
        self.dt = data['dt']
        self.states = data['states']
        self.dtrajs = data['dtrajs']
        self.ftrajs = data['ftrajs']
        print('Read {} discrete trajectories with timestep {} ns'.format(len(self.dtrajs), self.dt))
    def estimate_T(self, lag, index_trajs, min_block_size = 10000):
        n_states = len(self.states)
        C_full = np.zeros((n_states, n_states))
        for i_traj in index_trajs:
            disc_traj = self.dtrajs[i_traj]
            print('Traj {}, lenght: {}'.format(i_traj, len(disc_traj)))
            for i_state in range(n_states):
                for j_state in range(n_states):
                    C_full[i_state, j_state] += np.sum( (disc_traj[:-lag] == i_state) & (disc_traj[lag:] == j_state) ) 
                    #C_full[i_state, j_state] += np.sum( (disc_traj[:150000-lag] == i_state) & (disc_traj[lag:150000] == j_state) ) 
        print('T matrix computed using {} samples'.format(np.sum(C_full)))
        while True: #--- remove states with zero probability
            prob_states = np.sum(C_full, axis = 1)
            states_local = []
            C = []
            cols_delete = []
            for i, prob in enumerate(prob_states):
                skip = True
                if prob > 0:
                    skip = False
                    from_j_to_i = np.copy(C_full[:,i])
                    from_j_to_i[i] = 0.0
                    if np.sum(from_j_to_i) == 0:
                        print('State {} reached only from itself'.format(self.states[i,:]))
                        skip = True
                    from_i_to_j = np.copy(C_full[i,:])
                    from_i_to_j[i] = 0.0
                    if np.sum(from_j_to_i) == 0: # ERRORE ? dovrebbe essere i_to_j ?
                        print('State {} only goes to itself'.format(self.states[i,:]))
                        skip = True
                if skip:
                    print('Deleting column {}'.format(i))
                    cols_delete.append(i) # si ricorda di cancellare la colonna
                else:
                    C.append(list(C_full[i,:])) # aggiunge la riga i
                    states_local.append(list(self.states[i,:]))
            states_local = np.array(states_local)
            C = np.array(C)
            prob_states = np.sum(C, axis = 1)
            if cols_delete:
                C = np.delete(C, cols_delete, axis = 1)
            prob_states = np.sum(C, axis = 1)
            n_samples = np.sum(C)
            print('Number of samples {} min. samples for microstate {}'.format(n_samples, np.min(prob_states)))
            print('Probability from counts (pre-normalization) [shape {}]:'.format(C.shape))
            for i_state in np.argsort(prob_states)[::-1]:
                print(states_local[i_state,:], prob_states[i_state])
            T = C / prob_states.reshape((-1,1))
            T = T / np.sum(T, axis = 1)
            prob_states /= np.sum(prob_states)
            C_full = C.copy()
            print('Probability from counts (post-normalization) [shape {}]:'.format(T.shape))
            for i_state in np.argsort(prob_states)[::-1]:
                print(states_local[i_state,:], prob_states[i_state])
            if np.min(prob_states) > 0:
                break
        #--- find correspondance between new states and old states
        old2new = {}
        for i_old_state in range(n_states):
            for i_new_state in range(len(states_local)):
                if np.all(self.states[i_old_state,:] == states_local[i_new_state,:]):
                    old2new[i_old_state] = i_new_state
                    break
            else:
                print('Missing conversion for state {}'.format(i_old_state))
                old2new[i_old_state] = n_states # in this way it's out of the range of indexes used in states and to build T
        #--- create local trajectories (the ones used to build T) with the same state numbering of states local
        disc_trajs_local, f_trajs_local = [], []
        for i_traj in index_trajs:
            disc_traj = self.dtrajs[i_traj]
            f_traj = self.ftrajs[i_traj]
            disc_traj_local, f_traj_local = [], []
            for i_time, i_state in enumerate(disc_traj):
                if i_state in old2new:
                    disc_traj_local.append(old2new[i_state])
                    if i_time < len(f_traj):
                        f_traj_local.append(f_traj[i_time])
                    else:
                        f_traj_local.append(0) # for the last state the F is undefined
                else:
                    raise ValueError('ERROR: missing conversion for {} in {}'.format(i_state, old2new))
            if len(disc_traj_local) > min_block_size:
                disc_traj_local = np.array(disc_traj_local, dtype = int)
                f_traj_local = np.array(f_traj_local, dtype = int)
                disc_trajs_local.append(disc_traj_local)
                f_trajs_local.append(f_traj_local)
            else:
                print('WARNING: skipping trajectory with {} samples'.format(len(disc_traj_local)))
        T = np.transpose(T) # to get in standard format Tij = prob j-->i
        print('T computed using {} samples'.format(n_samples))
        print('Range sum_j(T_ij) {}-{}'.format(np.min(np.sum(T, axis = 0)), np.max(np.sum(T, axis = 0))), flush=True)
        prob_states = utils_model.probability(T)
        print('Probability of states from T [shape {}]:'.format(T.shape))
        for i_state in np.argsort(prob_states)[::-1]:
            print(states_local[i_state,:], 'prob.:', prob_states[i_state], '#states to i:',np.sum(T[i_state,:] > 0), '#states from i:',np.sum(T[:,i_state] > 0))
        return T, states_local, disc_trajs_local, f_trajs_local, n_samples
    def fit(self, file_matrix, lags, n_boots):
        fout_pk = open(file_matrix,'wb')
        index_trajs = np.arange(len(self.dtrajs), dtype = int)
        indexes_boot = [np.random.choice(index_trajs, size = int(np.floor(0.5*len(index_trajs))), replace = True) for i_boot in range(n_boots)]
        pickle.dump(self.dt, fout_pk)
        pickle.dump(lags, fout_pk)
        pickle.dump([-1,] + [i_boot for i_boot in range(n_boots)], fout_pk)
        for i_lag, lag in enumerate(lags):
            #--- using all blocks
            print('Using all block-data, lag: {}'.format(lag), flush = True)
            T_tmp, states_tmp, disc_trajs_tmp, f_trajs_tmp, n_samples = self.estimate_T(lag, index_trajs = np.arange(len(self.dtrajs), dtype = int))
            print('T.shape: {}, states.shape: {}'.format(T_tmp.shape, states_tmp.shape), flush = True)
            pickle.dump(T_tmp, fout_pk)
            pickle.dump(n_samples, fout_pk)
            pickle.dump(states_tmp, fout_pk)
            if i_lag == 0:
                pickle.dump(disc_trajs_tmp, fout_pk)
                pickle.dump(f_trajs_tmp, fout_pk)
            for i_boot, index_boot in enumerate(indexes_boot):
                print('lag: {} i_boot: {}'.format(lag, i_boot), flush = True)
                print('index_trajs: {}'.format(index_boot), flush = True)
                T_tmp, states_tmp, disc_trajs_tmp, f_trajs_tmp, n_samples = self.estimate_T(lag, index_trajs = index_boot)
                print('T.shape: {}, states.shape: {}'.format(T_tmp.shape, states_tmp.shape), flush = True)
                pickle.dump(T_tmp, fout_pk)
                pickle.dump(n_samples, fout_pk)
                pickle.dump(states_tmp, fout_pk)
                if i_lag == 0:
                    pickle.dump(disc_trajs_tmp, fout_pk)
                    pickle.dump(f_trajs_tmp, fout_pk)
        fout_pk.close()

class MSM(object):
    """
    Attributes
    ----------
    prefix: str
        Prefix used for output

    dt: float
        Elementary time step of the MD trajectories in ns

    lags: list of ints
        Multiples of dt used to estimate the MSM

    i_boots: list of ints
        Indexes of alternative MSMs
        Negative indexes are used for special cases
            -2 = MSM estimated from entire trajetories
            -1 = MSM estimate from trajectories divided in bits of same length
        Positive values are used for bootstraps iterations

    states: dict
        key: int
            i_boot index
        value: dict
            key: int
                lag as multiple of self.dt
            value: a matrix with shape number_of_states x number_of_features that describe the states at i_boot and lag

    states_original: dict
        key: int
            i_boot index
        value: dict
            key: int
                lag as multiple of self.dt
            value: a matrix with shape number_of_states x number_of_features that describe the states at i_boot and lag
    At first states and states_original are the same
    They differ after merging operation
    States_original never changes, so it can be used to check which initial states contribute to each state after a merging operation by using indexes_macro

    indexes_macro: dict
        key: int
            i_boot index
        value: dict
            key: int
                lag as multiple of self.dt
            value: list of list
    The number of element of the external list is the number of states.
    The internal list gives the indexes in states_original of the states merged into each state of states
    Example: [ [0, 3], [1, 2], ...]
        It means that state 0 in states is the merging of states 0 and 3 in states_original

    T: dict
        key: int
            i_boot index
        value: dict
            key: int
                lag as multiple of self.dt
            value: the transition matrix at that i_boot and lag calculated directly from MD trajectories

    F: dict
        key: int
            i_boot index
        value: dict
            key: int
                lag as multiple of self.dt
            value: the transition matrix at that i_boot and lag calculated directly from MD trajectories

    Q: dict
        key: int
            i_boot index
        value: dict
            key: int
                lag as multiple of self.dt
            value: the Q matrix 

    repr_state: function
        A function that takes as input an array representing a state and that returns a string

    value_state: function
        A function that takes as input an array representing a state and that returns a numeric array
    """

    def has_state(self, i_boot, lag, target_state):
        """
        Check if the model at i_boot:lag has the state target_state

        Return
        ------
        bool
        """
        for state in self.states[i_boot][lag]:
            state = self.repr_state(state)
            if state == target_state:
                return True
        return False

    def set_F_from_states(self, i_boot, lag):
        """
        Calculate the flux matrix at i_boot:lag
        """
        print('Computing F from state for lag {} i_boot {}'.format(lag, i_boot))
        n_states = self.states[i_boot][lag].shape[0]
        self.F[i_boot][lag] = np.zeros((n_states, n_states))
        for i, state_i in enumerate(self.states[i_boot][lag]):
            for j, state_j in enumerate(self.states[i_boot][lag]):
                self.F[i_boot][lag][i,j] = utils_channel.F_from_states(state_i, state_j)

    def reset_originals(self, i_boot, lag):
        """
        Use this if you want indexes_macro and states_original to be aligned with states
        """
        self.states_original[i_boot][lag] = np.copy(self.states[i_boot][lag])
        self.indexes_macro[i_boot][lag] = [[i_state,] for i_state in range(self.T[i_boot][lag].shape[0])]

    def check_lags(self, lags):
        if isinstance(lags, int):
            lags = [lags,]
        elif lags is None:
            lags = self.lags
        return lags

    def align_models(self, i_boot, lags = None, check_probability_md = False):
        """
        Force the models at i_boot:lags to have the same states in the same order
        """
        lags = self.check_lags(lags)
        print('Aligning the models at i_boot {} lags {} to the same states'.format(i_boot, lags))
        #--- Find states that are not common
        inds_not_commons = self.indexes_different_states(len(lags)*[i_boot,], lags, check_probability_md = check_probability_md)
        #--- Remove the not common states
        states2order = []
        for i_lag, lag in enumerate(lags):
            print('Mapping {} states to {} states for lag {}'.format(self.T[i_boot][lag].shape[0], self.T[i_boot][lag].shape[0]-len(inds_not_commons[i_lag]), lag))
            dummy = self.merge_remove_indexes(lag, i_boot, inds_not_commons[i_lag], return_md = check_probability_md)
            if check_probability_md:
                if len(dummy) == 4:
                    self.T[i_boot][lag] = dummy[0]
                    self.Tmd[i_boot][lag] = dummy[1]
                    self.states[i_boot][lag] = dummy[2]
                    inds2merge = dummy[3]
                else:
                    self.T[i_boot][lag] = dummy[0]
                    self.Tmd[i_boot][lag] = dummy[1]
                    self.F[i_boot][lag] = dummy[2]
                    self.Fmd[i_boot][lag] = dummy[3]
                    self.states[i_boot][lag] = dummy[4]
                    inds2merge = dummy[5]
            else:
                if len(dummy) == 3:
                    self.T[i_boot][lag] = dummy[0]
                    self.states[i_boot][lag] = dummy[1]
                    inds2merge = dummy[2]
                else:
                    self.T[i_boot][lag] = dummy[0]
                    self.F[i_boot][lag] = dummy[1]
                    self.states[i_boot][lag] = dummy[2]
                    inds2merge = dummy[3]
            self.update_indexes_macro(lag, i_boot, inds2merge)
            states2order.append(self.states[i_boot][lag])
        #--- Reorder the models at all lags to the same order
        inds_same_order = utils_model.common_order(self.value_state, *states2order)
        for i_lag, lag in enumerate(lags):
                self.T[i_boot][lag] =  self.T[i_boot][lag][inds_same_order[i_lag],:][:,inds_same_order[i_lag]]
                if lag in self.F[i_boot]:
                    self.F[i_boot][lag] =  self.F[i_boot][lag][inds_same_order[i_lag],:][:,inds_same_order[i_lag]]
                if check_probability_md:
                    self.Tmd[i_boot][lag] =  self.Tmd[i_boot][lag][inds_same_order[i_lag],:][:,inds_same_order[i_lag]]
                    if lag in self.Fmd[i_boot]:
                        self.Fmd[i_boot][lag] =  self.Fmd[i_boot][lag][inds_same_order[i_lag],:][:,inds_same_order[i_lag]]
                self.states[i_boot][lag] =  self.states[i_boot][lag][inds_same_order[i_lag],:]
                self.indexes_macro[i_boot][lag] = [self.indexes_macro[i_boot][lag][ind] for ind in inds_same_order[i_lag]]

    def propagate_F(self, i_boot):
        """
        Calculate F at higher lags with the iterative formula starting from lag == 1
        """
        for i_lag in range(1, len(self.lags)):
            #--- Clean previous F values if present 
            if self.lags[i_lag] in self.F[i_boot]:
                del self.F[i_boot][self.lags[i_lag]]
            #--- Find the lag used to propagsate T and F
            lag = self.lags[i_lag]
            delta_lag = int(lag - self.lags[i_lag-1])
            if delta_lag not in self.lags[:i_lag]:
                raise ValueError('ERROR: Missing lag for computing {}'.format(lag))
            print('Computing F at lag {} from {} and {}'.format(lag, self.lags[i_lag-1], delta_lag))
            #--- Force same states in the 3 lags used for the calculation
            inds_not_commons = self.indexes_different_states(3*[i_boot,], [lag, self.lags[i_lag-1], delta_lag])
            #--- Get T, F for calculations
            if len(inds_not_commons[0]) > 0:
                print('Mapping to {} states for lag {}'.format(self.T[i_boot][lag].shape[0]-len(inds_not_commons[0]), lag))
            self.T[i_boot][lag], self.states[i_boot][lag], inds2merge = self.merge_remove_indexes(lag, i_boot, inds_not_commons[0])
            self.update_indexes_macro(lag, i_boot, inds2merge)
            T_prev, F_prev, states_prev, dummy = self.merge_remove_indexes(self.lags[i_lag-1], i_boot, inds_not_commons[1])
            T_step, F_step, states_step, dummy = self.merge_remove_indexes(delta_lag, i_boot, inds_not_commons[2])
            #--- Reorder them to same order
            inds_same_order = utils_model.common_order(self.value_state, self.states[i_boot][lag], states_prev, states_step)
            self.T[i_boot][lag] = self.T[i_boot][lag][inds_same_order[0],:][:,inds_same_order[0]]
            self.states[i_boot][lag] = self.states[i_boot][lag][inds_same_order[0],:]
            self.indexes_macro[i_boot][lag] = [self.indexes_macro[i_boot][lag][ind] for ind in inds_same_order[0]]
            T_prev = T_prev[inds_same_order[1],:][:,inds_same_order[1]]
            F_prev = F_prev[inds_same_order[1],:][:,inds_same_order[1]]
            T_step = T_step[inds_same_order[2],:][:,inds_same_order[2]]
            F_step = F_step[inds_same_order[2],:][:,inds_same_order[2]]
            n_states = self.states[i_boot][lag].shape[0]
            #--- Finally compute F
            self.F[i_boot][lag] = np.zeros( (n_states, n_states) )
            for i_state in range(n_states):
                for j_state in range(n_states):
                    probs_k = T_prev[i_state,:].flatten()*T_step[:,j_state].flatten()
                    num = np.sum( (F_prev[i_state,:].flatten()+F_step[:,j_state].flatten())*probs_k )
                    den = np.sum(probs_k)
                    den = self.T[i_boot][lag][i_state, j_state]
                    if den > 0:
                        self.F[i_boot][lag][i_state, j_state] =  num/den

    def propagate_TF_lag1(self, i_boot):
        """
        Calculate F and T at higher lags with the iterative formulas starting from lag == 1
        """
        for i_lag in range(1, len(self.lags)):
            lag = self.lags[i_lag]
            #--- Clean previous T/F values if present 
            if lag in self.T[i_boot]:
                del self.F[i_boot][lag]
            if lag in self.F[i_boot]:
                del self.F[i_boot][lag]
            #--- Find the lag used to propagsate T and F
            delta_lag = int(lag - self.lags[i_lag-1])
            print('Computing lag {} from {} and {} times lag 1'.format(lag, self.lags[i_lag-1], delta_lag))
            #--- Force same states of lag 1 at the new lag
            self.states[i_boot][lag] = np.copy(self.states[i_boot][self.lags[0]])
            self.indexes_macro[i_boot][lag] =  np.copy(self.indexes_macro[i_boot][self.lags[0]])
            #--- Get initial values for T and F
            T_prev = self.T[i_boot][self.lags[i_lag-1]]
            F_prev = self.F[i_boot][self.lags[i_lag-1]]
            T_step = self.T[i_boot][self.lags[0]]
            F_step = self.F[i_boot][self.lags[0]]
            self.T[i_boot][lag] = np.copy(T_prev)
            n_states = self.states[i_boot][lag].shape[0]
            self.F[i_boot][lag] = np.zeros( (n_states, n_states) )
            #--- Finally compute T/F
            for n in range(delta_lag):
                self.T[i_boot][lag] = np.dot(T_step, self.T[i_boot][lag])
                for i_state in range(n_states):
                    for j_state in range(n_states):
                        # 1st = prev --> 2nd = step
                        probs_k = T_step[i_state,:].flatten()*T_prev[:,j_state].flatten()
                        num = np.sum( (F_step[i_state,:].flatten()+F_prev[:,j_state].flatten())*probs_k )
                        den = np.sum(probs_k)
                        if den > 0:
                            self.F[i_boot][lag][i_state, j_state] =  num/den
                        else:
                            self.F[i_boot][lag][i_state, j_state] =  0
                # 1st = prev --> 2nd = step
                T_prev = np.dot(T_step, T_prev)
                F_prev = np.copy(self.F[i_boot][lag])

    def states_above_probability_threshold(self, i_boot, lag, prob_min):
        """
        Return
        ------
        list    the indexes of the states with probability above prob_min
        list    the corresponding states representation
        """
        print('Selecting states with probability > {} at lag {} i_boot {}'.format(prob_min, lag, i_boot))
        probs = self.probability(i_boot, lag)
        i_states_selected, states_selected = [], []
        prob_total = 0
        for i_state, state in enumerate(self.states[i_boot][lag]):
            if probs[i_state] > prob_min:
                i_states_selected.append(i_state)
                states_selected.append(self.repr_state(self.states[i_boot][lag][i_state,:]))
                print('Selecting state[{}] = {} with prob = {}'.format(i_states_selected[-1], states_selected[-1], probs[i_state]))
                prob_total += probs[i_state]
        print('The selected states have a cumulative probability = {}'.format(prob_total))
        return i_states_selected, states_selected

    def indexes_different_states(self, i_boots, lags, check_probability_md = False):
        """
        lags: list
            List of lagtimes
        i_boots: list
            List of bootstrap indexes

        Return
        ------
        list
            For each combination of lag and i_boot, the indexes of the states that are not common to the other combinations of lags and i_boots
        """
        sets_states = []
        for lag, i_boot in zip(lags, i_boots):
            if check_probability_md:
                states_here = set()
                probs = probability(self.Tmd[i_boot][lag])
                for i_state, prob in enumerate(probs):
                    if prob > 0:
                        states_here.add(self.repr_state(self.states[i_boot][lag][i_state,:]))
                sets_states.append(states_here)
            else:
                sets_states.append(set([self.repr_state(state) for state in self.states[i_boot][lag]]))
        common_states = set.intersection(*sets_states)
        inds_states_not_common = []
        for lag, i_boot in zip(lags, i_boots):
            inds_states_not_common.append([i for i in range(self.states[i_boot][lag].shape[0]) if self.repr_state(self.states[i_boot][lag][i,:]) not in common_states])
        return inds_states_not_common

    def similarity_states_for_merge(self, i_boot, lag, i_state, j_state, method = 'T'):
        """
        flaot   a value that meaure how 'similar' two states are
        float   a second metric that is used of the the first one is the same
        """
        # i_state --> j_state
        if method == 'sumT':
            return -(self.T[i_boot][lag][j_state, i_state] + self.T[i_boot][lag][i_state, j_state]),-(self.T[i_boot][lag][j_state, i_state] + self.T[i_boot][lag][i_state, j_state]),
        elif method == 'T':
            return -self.T[i_boot][lag][j_state, i_state], -self.T[i_boot][lag][j_state, i_state]
        elif method == 'F':
            return np.abs(F_from_states(self.states[i_boot][lag][i_state,:], self.states[i_boot][lag][j_state,:])), -self.T[i_boot][lag][j_state, i_state]
        else:
            raise ValueError('ERROR: wrong method to calculate similarity between states')

    def check_target_presence(self, target_states):
        """
        Check at which i_boots the target_states are present

        Return
        ------
        list    the index of the i_boots where all target_states are present
        """
        i_boots = []
        for i_boot in self.i_boots:
            for lag in self.lags:
                flag_all_target_at_lag = False
                for i_target, target_state in enumerate(target_states):
                    for i_state, state in enumerate(self.states[i_boot][lag]):
                        state = self.repr_state(state)
                        if state == target_state:
                            break
                    else:
                        print('Target {} missing at i_boot {} lag {}'.format(target_state, i_boot, lag))
                        break
                else:
                    flag_all_target_at_lag = True
                if not flag_all_target_at_lag:
                    break
            else:
                i_boots.append(i_boot)
        print('After checking target states i_boots = {}'.format(i_boots))
        return i_boots

    def merge_target(self, i_boot, lag, target_states, keep_old_states = False):
        """
        lag: int
            lag used to calculate proximity
        target_states: list
            self.repr_state of the states used for merging

        Merge the model to a predifined set of target states
        """
        n_states, states = self.check_states(i_boot, permissive = False)
        inds_target_order = []
        inds_target_states = []
        for i_state, state in enumerate(states):
            state = self.repr_state(state)
            for i_target, target_state in enumerate(target_states):
                if state == target_state:
                    inds_target_states.append(i_state) # questi sono gli indici degli stati che diventeranno i target
                    inds_target_order.append(i_target) # questo e' l'indice di target dello stato nel vettore precedente
                    break
        #print('DEBUG> inds_target_states:',inds_target_states)
        #print('DEBUG> inds_target_order:',inds_target_order)
        inds_target_order = np.argsort(inds_target_order) # questo vettore contiene l'ordine con cui sistemare gli stati mappati affinche' siano disposti come in target
        #print('DEBUG> inds_target_order:',inds_target_order)
        #--- check that all target states are present
        for i_target, target_state in enumerate(target_states):
            for i_state, state in enumerate(states):
                state = self.repr_state(state)
                if state == target_state:
                    #print('DEBUG> target_state {} = {} (i_state = {})'.format(i_target, state, i_state))
                    break
            else:
                raise ValueError('ERROR: i_boot {} target state {} is missing'.format(i_boot, target_state))
        if len(inds_target_states) != len(target_states):
            raise ValueError('ERROR: wrong number of target states')
        #--- create the merging lists
        inds2merge = [[] for i in range(len(target_states))]
        for i_state, state in enumerate(states):
            if i_state in inds_target_states:
                i_target = inds_target_states.index(i_state)
                inds2merge[i_target].append(i_state)
            else:
                best_proximity_par0, best_proximity_par1, i_neigh = np.inf, np.inf, np.inf
                for j_state in inds_target_states:
                    proximity = self.similarity_states_for_merge(i_boot, lag, i_state, j_state)
                    if proximity[0] <= best_proximity_par0:
                        if proximity[0] == best_proximity_par0:
                            if proximity[1] < best_proximity_par1:
                                best_proximity_par0 = proximity[0]
                                best_proximity_par1 = proximity[1]
                                i_neigh = j_state
                        else:
                            best_proximity_par0 = proximity[0]
                            best_proximity_par1 = proximity[1]
                            i_neigh = j_state
                i_target = inds_target_states.index(i_neigh)
                inds2merge[i_target].append(i_state)
        for i, inds in enumerate(inds2merge):
            if len(inds) == 0:
                raise ValueError('ERROR: no state mapping to {}',format(target_states[i]))
        print('Merging states to target states for i_boot {}'.format(i_boot))
        for inds in inds2merge:
            print('\tMerging states with indexes:',inds)
            for i in inds:
                print('\t\tstate = {}'.format(self.repr_state(states[i,:])))
        if keep_old_states:
            self.merge_manual(i_boot, inds2merge, inds_for_states = inds_target_states)
        else:
            self.merge_manual(i_boot, inds2merge)
        #--- Reorder the final states as in target_states
        for lag in self.lags:
            self.T[i_boot][lag] = self.T[i_boot][lag][inds_target_order,:][:,inds_target_order]
            self.F[i_boot][lag] = self.F[i_boot][lag][inds_target_order,:][:,inds_target_order]
            self.states[i_boot][lag] = self.states[i_boot][lag][inds_target_order,:]
            if i_boot in self.Tmd:
                if lag in self.Tmd[i_boot]:
                    self.Tmd[i_boot][lag] = self.Tmd[i_boot][lag][inds_target_order,:][:,inds_target_order]
            if i_boot in self.Fmd:
                if lag in self.Fmd[i_boot]:
                    self.Fmd[i_boot][lag] = self.Fmd[i_boot][lag][inds_target_order,:][:,inds_target_order]
            dummy_indexes_macro = self.indexes_macro[i_boot][lag].copy()
            self.indexes_macro[i_boot][lag] = []
            for ind in inds_target_order:
                self.indexes_macro[i_boot][lag].append(dummy_indexes_macro[ind])

    def merge_manual(self, i_boot, inds2merge, inds_for_states = []):
        """
        """
        for lag in self.lags:
            merge_results = self.merge(lag, i_boot, inds2merge.copy())
            self.update_indexes_macro(lag, i_boot, inds2merge)
            if len(merge_results) == 2:
                self.T[i_boot][lag] = merge_results[0]
                if len(inds_for_states):
                    self.states[i_boot][lag] = self.states[i_boot][lag][inds_for_states,:]
                else:
                    self.states[i_boot][lag] = merge_results[1]
            else:
                self.T[i_boot][lag] = merge_results[0]
                self.F[i_boot][lag] = merge_results[1]
                if len(inds_for_states):
                    self.states[i_boot][lag] = self.states[i_boot][lag][inds_for_states,:]
                else:
                    self.states[i_boot][lag] = merge_results[2]
            print('After merging to target for lag {} i_boot {}, n_states = {}'.format(lag, i_boot, self.states[i_boot][lag].shape[0]))

    def merge_remove_indexes(self, lag, i_boot, inds2remove, return_md = False):
        prob = self.probability(i_boot, lag)
        #--- first add the states that are not going to be removed
        inds2merge = [[i_state,] for i_state in range(self.states[i_boot][lag].shape[0]) if i_state not in inds2remove]
        #--- then add the states to remove to one of the states not to remove
        for i_state_2remove in inds2remove:
            best_proximity_par0, best_proximity_par1, i_neigh = np.inf, np.inf, np.inf
            for i_state in range(self.states[i_boot][lag].shape[0]):
                if i_state not in inds2remove:
                    #proximity = self.T[i_boot][lag][i_state, i_state_2remove]
                    #proximity = (self.T[i_boot][lag][i_state, i_state_2remove] + self.T[i_boot][lag][i_state_2remove, i_state])
                    proximity = self.similarity_states_for_merge(i_boot, lag, i_state_2remove, i_state)
                    if proximity[0] <= best_proximity_par0:
                        if proximity[0] == best_proximity_par0:
                            if proximity[1] < best_proximity_par1:
                                best_proximity_par0 = proximity[0]
                                best_proximity_par1 = proximity[1]
                                i_neigh = i_state
                        else:
                            best_proximity_par0 = proximity[0]
                            best_proximity_par1 = proximity[1]
                            i_neigh = i_state
            for i, inds in enumerate(inds2merge):
                if i_neigh in inds:
                    inds2merge[i].append(i_state_2remove)
                    #print('DEBUG> Merging state {} with prob. {} into states {} with prob. {}'.format(self.repr_state(self.states[i_boot][lag][i_state_2remove]), prob[i_state_2remove], self.repr_state(self.states[i_boot][lag][i_neigh]), prob[i_neigh]))
                    break
        #--- sort the indexes inside each sublist
        for i in range(len(inds2merge)):
            inds2merge[i].sort()
        return self.merge(lag, i_boot, inds2merge.copy(), return_md = return_md) + (inds2merge,)

    def merge(self, lag, i_boot, inds2merge, return_md = False):
        """
        return_md
            If True Tmd and Fmd are also returned
            In this case the states are updated considering the probabilities computed from Tmd
        """
        if len(inds2merge) == self.states[i_boot][lag].shape[0]:
            if return_md:
                if lag in self.F[i_boot]:
                    return self.T[i_boot][lag], self.Tmd[i_boot][lag], self.F[i_boot][lag], self.Fmd[i_boot][lag], self.states[i_boot][lag]
                else:
                    return self.T[i_boot][lag], self.Tmd[i_boot][lag], self.states[i_boot][lag]
            else:
                if lag in self.F[i_boot]:
                    return self.T[i_boot][lag], self.F[i_boot][lag], self.states[i_boot][lag]
                else:
                    return self.T[i_boot][lag], self.states[i_boot][lag]
        states_old = np.copy(self.states[i_boot][lag])
        T_old = np.copy(self.T[i_boot][lag])
        if lag in self.F[i_boot]:
            F_old = np.copy(self.F[i_boot][lag])
        if return_md:
            if self.T[i_boot][lag].shape != self.Tmd[i_boot][lag].shape:
                raise ValueError('ERROR: inconsistent T and Tmd shapes')
            Tmd_old = np.copy(self.Tmd[i_boot][lag])
            if lag in self.Fmd[i_boot]:
                if self.F[i_boot][lag].shape != self.Fmd[i_boot][lag].shape:
                    raise ValueError('ERROR: inconsistent F and Fmd shapes')
                Fmd_old = np.copy(self.Fmd[i_boot][lag])
        for i in range(len(inds2merge)):
            inds = inds2merge[i] # these are the ones to be merged
            inds.sort()
            #print('DEBUG> Merging states:', inds)
            n_states2remove = len(inds) - 1
            if n_states2remove > 0:
                #--- compute probabilities of old states
                if return_md:
                    ls, es = np.linalg.eig(Tmd_old)
                else:
                    ls, es = np.linalg.eig(T_old)
                inds_sort = np.argsort(np.abs(ls))[::-1]
                es = es[:,inds_sort]
                probs = np.abs(es[:,0])
                if np.any(probs == 0):
                    print('WARNING: some states have null probability')
                    probs[probs == 0] = np.min(probs[probs > 0])*1e-20
                probs /= np.sum(probs)
                #--- initialize new matrixes
                n_states_old = T_old.shape[0]
                n_states_new =  n_states_old - n_states2remove
                states_new = np.zeros( (n_states_new, states_old.shape[1]) )
                T_new = np.zeros( (n_states_new, n_states_new) )
                if lag in self.F[i_boot]:
                    F_new = np.zeros( (n_states_new, n_states_new) )
                if return_md:
                    Tmd_new = np.zeros( (n_states_new, n_states_new) )
                    if lag in self.Fmd[i_boot]:
                        Fmd_new = np.zeros( (n_states_new, n_states_new) )
                #--- define dictionary for indexes conversion between new and old
                old2new = {}
                i_new = 0
                for i_state in range(n_states_old):
                    if i_state not in inds:
                        old2new[i_state] = i_new
                        i_new += 1
                    else:
                        if inds.index(i_state) == 0:
                            old2new[i_state] = i_new
                            i_merge = i_new
                            i_new += 1
                        else:
                            old2new[i_state] = i_merge
                new2old = {}
                for i_old, i_new in old2new.items():
                    if i_new not in new2old:
                        new2old[i_new] = []
                    new2old[i_new].append(i_old)
                #--- update matrixes
                for i_state in range(n_states_new):
                    i_states_old = new2old[i_state]
                    if len(i_states_old) == 1:
                        i_state_old = i_states_old[0]
                        states_new[i_state, :] = states_old[i_state_old, :]
                    else:
                        if np.sum(probs[i_states_old]) == 0:
                            raise ValueError('ERROR: merging states with cumulative null probability')
                        states_new[i_state, :] = np.sum(states_old[i_states_old, :]*probs[i_states_old].reshape((-1,1)), axis = 0) / np.sum(probs[i_states_old])
                #print('DEBUG> np.sum(probs): ', np.sum(probs))
                #print('DEBUG> probs: ', probs)
                #print('DEBUG> {}-{}'.format(np.min(np.sum(T_old, axis = 0)), np.max(np.sum(T_old, axis = 0))))
                for i_state in range(n_states_new):
                    i_states_old = new2old[i_state]
                    for j_state in range(n_states_new):
                        j_states_old = new2old[j_state]
                        if (len(i_states_old) == 1) and (len(j_states_old) == 1): # here nothing changes
                            i_state_old = i_states_old[0]
                            j_state_old = j_states_old[0]
                            T_new[i_state, j_state] = np.copy(T_old[i_state_old, j_state_old])
                            if lag in self.F[i_boot]:
                                F_new[i_state, j_state] = np.copy(F_old[i_state_old, j_state_old])
                            if return_md:
                                Tmd_new[i_state, j_state] = Tmd_old[i_state_old, j_state_old]
                                if lag in self.Fmd[i_boot]:
                                    Fmd_new[i_state, j_state] = Fmd_old[i_state_old, j_state_old]
                        elif (len(i_states_old) == 1) and (len(j_states_old) > 1): # the new start state merges several old states
                            i_state_old = i_states_old[0]
                            T_new[i_state, j_state] = np.sum(T_old[i_state_old,j_states_old]*probs[j_states_old]) / np.sum(probs[j_states_old])
                            if lag in self.F[i_boot]:
                                den = np.sum(T_old[i_state_old,j_states_old]*probs[j_states_old])
                                if den > 0:
                                    num =  np.sum(F_old[i_state_old,j_states_old]*T_old[i_state_old,j_states_old]*probs[j_states_old])
                                    F_new[i_state, j_state] = num / den 
                            if return_md:
                                Tmd_new[i_state, j_state] = np.sum(Tmd_old[i_state_old,j_states_old]*probs[j_states_old]) / np.sum(probs[j_states_old])
                                if lag in self.Fmd[i_boot]:
                                    den = np.sum(Tmd_old[i_state_old,j_states_old]*probs[j_states_old])
                                    if den > 0:
                                        num =  np.sum(Fmd_old[i_state_old,j_states_old]*Tmd_old[i_state_old,j_states_old]*probs[j_states_old])
                                        Fmd_new[i_state, j_state] = num / den 
                        elif (len(i_states_old) > 1) and (len(j_states_old) == 1): # the new end state merges several old states
                            j_state_old = j_states_old[0]
                            T_new[i_state, j_state] = np.sum(T_old[i_states_old, j_state_old])
                            if lag in self.F[i_boot]:
                                den = np.sum(T_old[i_states_old, j_state_old])
                                if den > 0:
                                    num = np.sum(F_old[i_states_old, j_state_old]*T_old[i_states_old, j_state_old])
                                    F_new[i_state, j_state] =  num / den 
                            if return_md:
                                Tmd_new[i_state, j_state] = np.sum(Tmd_old[i_states_old, j_state_old])
                                if lag in self.Fmd[i_boot]:
                                    den = np.sum(Tmd_old[i_states_old, j_state_old])
                                    if den > 0:
                                        num = np.sum(Fmd_old[i_states_old, j_state_old]*Tmd_old[i_states_old, j_state_old])
                                        Fmd_new[i_state, j_state] =  num / den 
                        elif (len(i_states_old) > 1) and (len(j_states_old) > 1): # both the new start/end merges several old states
                            if i_state != j_state:
                                raise ValueError('ERROR: something wrong in merge, only one macrostate at a time is merged')
                            num, den = 0, 0
                            for i_state_old in i_states_old:
                                den += probs[i_state_old]
                                num += np.sum(T_old[i_states_old,i_state_old]*probs[i_state_old])
                            if den > 0:
                                T_new[i_state,j_state] = num/den
                            if lag in self.F[i_boot]:
                                num, den = 0, 0
                                for i_state_old in i_states_old:
                                    num += np.sum(F_old[i_state_old,j_states_old]*T_old[i_state_old,j_states_old]*probs[j_states_old]) 
                                    den += np.sum(T_old[i_state_old,j_states_old]*probs[j_states_old])
                                if den > 0:
                                    F_new[i_state, j_state] = num / den
                            if return_md:
                                num, den = 0, 0
                                for i_state_old in i_states_old:
                                    den += probs[i_state_old]
                                    num += np.sum(Tmd_old[i_states_old,i_state_old]*probs[i_state_old])
                                if den > 0:
                                    Tmd_new[i_state,j_state] = num/den
                                if lag in self.Fmd[i_boot]:
                                    num, den = 0, 0
                                    for i_state_old in i_states_old:
                                        num += np.sum(Fmd_old[i_state_old,j_states_old]*Tmd_old[i_state_old,j_states_old]*probs[j_states_old]) 
                                        den += np.sum(Tmd_old[i_state_old,j_states_old]*probs[j_states_old])
                                    if den > 0:
                                        Fmd_new[i_state, j_state] = num / den
                #print('DEBUG> {}-{}'.format(np.min(np.sum(T_new, axis = 0)), np.max(np.sum(T_new, axis = 0))))
                #print('DEBUG> {}'.format(np.sum(T_new, axis = 0)))
                #--- updating
                states_old = states_new
                T_old = T_new
                if lag in self.F[i_boot]:
                    F_old = F_new
                if return_md:
                    Tmd_old = Tmd_new
                    if lag in self.Fmd[i_boot]:
                        Fmd_old = Fmd_new
                #--- sorting
                inds_sort = [ind for ind in range(n_states_new)]
                inds_sort[i] = old2new[inds[0]]
                inds_sort[old2new[inds[0]]] = i
                for ind in inds:
                    if old2new[ind] != inds_sort[i]:
                        raise ValueError('ERROR: wrong sorting for merged states')
                inds_sort[i+1:] = list(np.sort(inds_sort[i+1:]))
                states_old = states_old[inds_sort,:]
                T_old = T_old[inds_sort,:][:,inds_sort]
                if lag in self.F[i_boot]:
                    F_old = F_old[inds_sort,:][:,inds_sort]
                if return_md:
                    Tmd_old = Tmd_old[inds_sort,:][:,inds_sort]
                    if lag in self.Fmd[i_boot]:
                        Fmd_old = Fmd_old[inds_sort,:][:,inds_sort]
                inds_sort = np.array(inds_sort)
                inds_new_positions = np.zeros(len(inds_sort), dtype = int)
                for ind, ind_sort in enumerate(inds_sort):
                    inds_new_positions[ind_sort] = ind
                #--- update merging indexes according to new matrixes
                for j in range(i+1, len(inds2merge)):
                    inds2merge[j] = [inds_new_positions[old2new[k]] for k in inds2merge[j]]
            else:
                #--- define dictionary for indexes conversion between new and old
                n_states_old = T_old.shape[0]
                n_states_new =  n_states_old - n_states2remove
                old2new = {}
                i_new = 0
                for i_state in range(n_states_old):
                    if i_state not in inds:
                        old2new[i_state] = i_new
                        i_new += 1
                    else:
                        if inds.index(i_state) == 0:
                            old2new[i_state] = i_new
                            i_merge = i_new
                            i_new += 1
                        else:
                            old2new[i_state] = i_merge
                new2old = {}
                for i_old, i_new in old2new.items():
                    if i_new not in new2old:
                        new2old[i_new] = []
                    new2old[i_new].append(i_old)
                #--- sorting
                inds_sort = [ind for ind in range(n_states_new)]
                inds_sort[i] = old2new[inds[0]]
                inds_sort[old2new[inds[0]]] = i
                for ind in inds:
                    if old2new[ind] != inds_sort[i]:
                        raise ValueError('ERROR: wrong sorting for merged states')
                inds_sort[i+1:] = list(np.sort(inds_sort[i+1:]))
                states_old = states_old[inds_sort,:]
                T_old = T_old[inds_sort,:][:,inds_sort]
                if lag in self.F[i_boot]:
                    F_old = F_old[inds_sort,:][:,inds_sort]
                if return_md:
                    Tmd_old = Tmd_old[inds_sort,:][:,inds_sort]
                    if lag in self.Fmd[i_boot]:
                        Fmd_old = Fmd_old[inds_sort,:][:,inds_sort]
                inds_sort = np.array(inds_sort)
                inds_new_positions = np.zeros(len(inds_sort), dtype = int)
                for ind, ind_sort in enumerate(inds_sort):
                    inds_new_positions[ind_sort] = ind
                #--- update merging indexes according to new matrixes
                for j in range(i+1, len(inds2merge)):
                    inds2merge[j] = [inds_new_positions[old2new[k]] for k in inds2merge[j]]
        if return_md:
            if lag in self.F[i_boot]:
                return T_new, Tmd_new, F_new, Fmd_new, states_new
            else:
                return T_new, Tmd_new, states_new
        else:
            if lag in self.F[i_boot]:
                return T_new, F_new, states_new
            else:
                return T_new, states_new

    def update_indexes_macro(self, lag, i_boot, inds2merge):
        indexes_macro = []
        for i, inds in enumerate(inds2merge):
            indexes_macro.append([])
            for ind in inds:
                indexes_macro[-1].extend(self.indexes_macro[i_boot][lag][ind])
        self.indexes_macro[i_boot][lag] = indexes_macro.copy()

    def merge_clusters(self, i_boot, lag, inds_clusters = None):
        """
        inds_clusters: np.array
            Same length as states
            Element i = index of the cluster state i belongs
        """
        if isinstance(inds_clusters, str):
            if inds_clusters == 'random':
                inds_clusters = np.random.choice(np.arange(2, dtype = int), self.states[i_boot][lag].shape[0], replace = True)
            else:
                raise ValueError('ERROR: inds_clusters needs to be a list or "random"')
        if len(inds_clusters) != self.states[i_boot][lag].shape[0]:
            raise ValueError('ERROR: number of states in inds_clusters {} not consistent with {}'.format(len(inds_clusters), self.states[i_boot][lag].shape[0]))
        inds2merge = {}
        for i_state in range(self.states[i_boot][lag].shape[0]):
            i_cluster = inds_clusters[i_state]
            if i_cluster not in inds2merge:
                inds2merge[i_cluster] = []
            inds2merge[i_cluster].append(i_state)
        inds2merge = list(inds2merge.values())
        merge_results = self.merge(lag, i_boot, inds2merge.copy())
        self.update_indexes_macro(lag, i_boot, inds2merge)
        if len(merge_results) == 2:
            self.T[i_boot][lag] = merge_results[0]
            self.states[i_boot][lag] = merge_results[1]
        else:
            self.T[i_boot][lag] = merge_results[0]
            self.F[i_boot][lag] = merge_results[1]
            self.states[i_boot][lag] = merge_results[2]
        print('After merging to cluster for lag {} i_boot {}, n_states = {}'.format(lag, i_boot, self.states[i_boot][lag].shape[0]))

    def probability_md(self, i_boot, lag):
        return utils_model.probability(self.Tmd[i_boot][int(lag)])

    def probability(self, i_boots, lag, from_Q = False):
        if from_Q:
            if isinstance(i_boots, int):
                if lag not in self.Q[i_boots]:
                    raise ValueError('ERROR: missing lag {} for {} i_boots = {}'.format(lag, self.Q[i_boots].keys(), i_boots))
                return  utils_model.probability_Q(self.Q[i_boots][int(lag)])
            else:
                probs = []
                for i_boot in i_boots:
                    if lag not in self.Q[i_boot]:
                        raise ValueError('ERROR: missing lag {} for {} i_boots = {}'.format(lag, self.Q[i_boot].keys(), i_boot))
                    probs.append(utils_model.probability_Q(self.Q[i_boot][int(lag)]).reshape((1,-1)))
                return np.concatenate(probs)
        else:
            if isinstance(i_boots, int):
                if lag not in self.T[i_boots]:
                    raise ValueError('ERROR: missing lag {} for {} i_boots = {}'.format(lag, self.T[i_boots].keys(), i_boots))
                return  utils_model.probability(self.T[i_boots][int(lag)])
            else:
                probs = []
                for i_boot in i_boots:
                    if lag not in self.T[i_boot]:
                        raise ValueError('ERROR: missing lag {} for {} i_boots = {}'.format(lag, self.T[i_boot].keys(), i_boot))
                    probs.append(utils_model.probability(self.T[i_boot][int(lag)]).reshape((1,-1)))
                return np.concatenate(probs)

    def eigenvalues(self, n_eigs, lag, i_boot):
        ls, es = np.linalg.eig(self.T[i_boot][lag])
        ls = ls[np.argsort(np.abs(ls))[::-1]]
        converged = np.all(np.abs(ls[1:n_eigs+1]) < 1)
        if not converged:
            raise ValueError('ERROR: eigenvalues > 1')
        return np.abs(ls[:n_eigs])

    def eigenvectors(self, n_eigs, lag, i_boot):
        ls, es = np.linalg.eig(self.T[i_boot][lag])
        inds_sort = np.argsort(np.abs(ls))[::-1]
        ls = ls[inds_sort]
        es = es[:,inds_sort]
        converged = np.all(np.abs(ls[1:n_eigs+1]) < 1)
        if not converged:
            raise ValueError('ERROR: eigenvalues > 1')
        return es[:,:n_eigs]

    def timescales(self, n_timescales, lag, i_boot = []):
        if isinstance(i_boot, list) or isinstance(i_boot, np.ndarray):
            if len(i_boot) == 0:
                i_boot = self.i_boots
            return np.concatenate([self.timescales(n_timescales, lag, ind) for ind in i_boot], axis = 0)
        else:
            ls, es = np.linalg.eig(self.T[i_boot][lag])
            ls = ls[np.argsort(np.abs(ls))[::-1]]
            converged = np.all(np.abs(ls[1:n_timescales+1]) < 1)
            if not converged:
                raise ValueError('ERROR: eigenvalues > 1')
            taus = -1.0 / np.log(np.abs(ls))
            return taus[1:n_timescales+1].reshape((1, -1))

    def timescales_from_md(self, n_timescales, lag, i_boot = []):
        if isinstance(i_boot, list) or isinstance(i_boot, np.ndarray):
            if len(i_boot) == 0:
                i_boot = self.i_boots
            return np.concatenate([self.timescales(n_timescales, lag, ind) for ind in i_boot], axis = 0)
        else:
            ls, es = np.linalg.eig(self.Tmd[i_boot][lag])
            ls = ls[np.argsort(np.abs(ls))[::-1]]
            converged = np.all(np.abs(ls[1:n_timescales+1]) < 1)
            if not converged:
                raise ValueError('ERROR: eigenvalues > 1')
            taus = -1.0 / np.log(np.abs(ls))
            return taus[1:n_timescales+1].reshape((1, -1))

    def current(self, lag, i_boot = []):
        scale = (constants.e*1e12)/(self.dt*lag*1e-9)
        if isinstance(i_boot, list) or isinstance(i_boot, np.ndarray):
            if len(i_boot) == 0:
                i_boot = [ind for ind in self.i_boots if ind >= 0]
            return [self.current(lag, ind) for ind in i_boot]
        else:
            return scale*np.sum(self.F[i_boot][lag]*self.T[i_boot][lag]*self.probability(i_boot, lag).reshape((1,-1)))

    def current_from_md(self, lag, i_boot):
        scale = (constants.e*1e12)/(self.dt*lag*1e-9)
        if isinstance(i_boot, list) or isinstance(i_boot, np.ndarray):
            if len(i_boot) == 0:
                i_boot = [ind for ind in self.i_boots if ind >= 0]
            return [self.current_from_md(lag, ind) for ind in i_boot]
        else:
            return scale*np.sum(self.Fmd[i_boot][lag]*self.Tmd[i_boot][lag]*self.probability_md(i_boot, lag).reshape((1,-1)))

    def current_from_Q(self, lag, i_boot):
        scale = (constants.e*1e12)/(1e-9)
        if isinstance(i_boot, list) or isinstance(i_boot, np.ndarray):
            if len(i_boot) == 0:
                i_boot = [ind for ind in self.i_boots if ind >= 0]
            return [self.current_from_Q(lag, ind) for ind in i_boot]
        else:
            ls, es = np.linalg.eig(self.Q[i_boot][lag])
            inds_sort = np.argsort(np.abs(ls))
            es = es[:,inds_sort]
            prob_states = np.abs(es[:,0])
            prob_states /= np.sum(prob_states)
            return scale*np.sum(self.F[i_boot][self.lags[0]]*(self.Q[i_boot][lag] - self.Q[i_boot][lag]*np.eye(len(self.Q[i_boot][lag])))*prob_states.reshape((1,-1)))

    def plot_timescales_at_lag(self, n_timescales, pdf,  i_boot, lag, title = ''):
        if n_timescales is None:
            n_timescales = self.T[i_boot][lag].shape[0]-1
        taus = self.dt*lag*self.timescales(n_timescales, lag, i_boot).flatten()
        f = plt.figure()
        ax = f.add_subplot(1,1,1)
        ax.plot(1+np.arange(n_timescales), taus, 'bo', label = 'from T')
        #if self.Tmd[i_boot]:
        #    dummy_msm = MSM_from_matrixes(self.dt, lag, self.Tmd[i_boot][lag], self.Fmd[i_boot][lag], self.states[i_boot][lag])
        #    taus = self.dt*lag*dummy_msm.timescales(n_timescales, lag, i_boot = -1)
        #    ax.plot(1+np.arange(n_timescales), taus, 'x', label = 'from Tmd')
        plt.xlabel('Index')
        plt.ylabel('Timescale [ns]')
        plt.legend()
        #plt.grid()
        plt.title('{} {} lag = {} ns'.format(self.prefix, title, self.dt*lag))
        plt.xlim([0, n_timescales + 1])
        plt.ylim([0, None])
        plt.xticks(np.arange(1, n_timescales + 1))
        pdf.savefig()
        #plt.yscale('log')
        #pdf.savefig()
        plt.close()

    def plot_eigenvalues_at_lag(self, n_timescales, i_boot, lag):
        taus = self.eigenvalues(n_timescales+1, lag, i_boot)
        f = plt.figure()
        ax = f.add_subplot(1,1,1)
        ax.plot(1+np.arange(len(taus)), taus, 'o')
        plt.xlabel('Index')
        plt.ylabel('|eigenvalue|')
        plt.grid()
        pdf.savefig()
        plt.yscale('log')
        pdf.savefig()
        plt.close()

    def plot_timescales(self, pdf, i_boot, n_timescales = None, title = ''):
        if n_timescales is None:
            n_timescales = self.T[i_boot][lag].shape[0]-1
        f = plt.figure()
        ax = f.add_subplot(1,1,1)
        taus = np.empty((len(self.lags), n_timescales))
        for i_lag, lag in enumerate(self.lags):
            taus[i_lag, :] = self.dt*lag*self.timescales(n_timescales, lag, i_boot)
        for i_tau in range(n_timescales):
            ax.plot(self.dt*self.lags, taus[:, i_tau], 'o-')
        ax.plot(self.dt*self.lags, self.dt*self.lags, ':k')
        plt.xlim(left = 0)
        plt.ylim(bottom = 0)
        plt.xlabel('Sampling period [ns]')
        plt.ylabel('Timescale [ns]')
        plt.title('{} {}'.format(self.prefix, title))
        plt.grid()
        pdf.savefig()
        plt.close()

    def minimal_Q(self, i_boot):
        minQ = np.inf
        for i_lag, lag in enumerate(self.lags[-1:]):
            Q = np.copy(self.Q[i_boot][lag])
            Q[Q == 0] = np.inf
            for i in range(Q.shape[0]):
                Q[i,i] = np.inf
            new_min_Q = np.min(Q)
            if new_min_Q < minQ:
                minQ = np.min(Q)
                i_min, j_min  =  np.unravel_index(np.argmin(Q), Q.shape)
        return minQ, i_min, j_min

    def Q_below_threshold(self, i_boot, thr):
        n_states = self.Q[i_boot][self.lags[0]].shape[0]
        inds = []
        for i in range(n_states):
            for j in range(n_states):
                if i != j:
                    for i_lag, lag in enumerate(self.lags[-1:]):
                        if self.Q[i_boot][lag][i,j] <= thr:
                            inds.append((i,j))
                            break
        return inds

    def estimate_Q_constrained(self, pdf, i_boot, fix2zero = []):
        n_states, states = self.check_states(i_boot)
        print('Estimating Qs for model with {} states using {} free parameters'.format(n_states, n_states**2 - n_states - len(fix2zero)))
        self.Q[i_boot] = {}
        n_unk = int(n_states**2.0 - n_states)
        for i_lag, lag in enumerate(self.lags):
            print('Estimating Q at lag {} for i_boot {}'.format(lag, i_boot))
            x0 = np.zeros(n_unk)
            k = 0
            pairing = []
            for i in range(self.T[i_boot][lag].shape[0]):
                for j in range(self.T[i_boot][lag].shape[0]):
                    if i != j:
                        if (i,j) not in fix2zero:
                            pairing.append((i,j))
                            if i_lag == 0:
                                x0[k] = self.T[i_boot][lag][i,j] / (self.dt*lag) # for the shortest lag take the numerical estimate of the derivative
                            else:
                                x0[k] = self.Q[i_boot][self.lags[i_lag-1]][i,j] # for the following lags, use the value estimated at the previous lag
                            k += 1
            I = self.current_from_md(lag, i_boot) # set as target the current estimated from MD
            dummy = utils_model.estimate_Q(pairing, self.n_samples[i_boot][lag], self.dt, lag, self.T[i_boot][lag], self.F[i_boot][lag], self.T[i_boot][self.lags[0]], self.F[i_boot][self.lags[0]], x0, I, np.inf)
            self.Q[i_boot][lag] = np.copy(dummy[0])
        return copy.deepcopy(self.Q[i_boot])

    def estimate_Q(self, pdf, i_boot, ws_current = [0.0, ], fix2zero = []):
        n_states, states = self.check_states(i_boot)
        print('Estimating Qs for model with {} states using {} free parameters'.format(n_states, n_states**2 - n_states - len(fix2zero)))
        self.Q[i_boot] = {}
        n_unk = int(n_states**2.0 - n_states)
        dummy_ws_current = np.copy(ws_current)
        for i_lag, lag in enumerate(self.lags):
            qs = []
            cost_matrix = np.zeros(len(ws_current))
            cost_current = np.zeros(len(ws_current))
            for i_w_current, w_current in enumerate(ws_current):
                print('Estimating Q at lag {} for i_boot {}  with w_current = {}'.format(lag, i_boot,  w_current))
                x0 = np.zeros(n_unk)
                k = 0
                pairing = []
                for i in range(self.T[i_boot][lag].shape[0]):
                    for j in range(self.T[i_boot][lag].shape[0]):
                        if i != j:
                            if (i,j) not in fix2zero:
                                pairing.append((i,j))
                                if i_lag == 0:
                                    x0[k] = self.T[i_boot][lag][i,j] / (self.dt*lag) # for the shortest lag take the numerical estimate of the derivative
                                else:
                                    x0[k] = self.Q[i_boot][self.lags[i_lag-1]][i,j] # for the following lags, use the value estimated at the previous lag
                                k += 1
                I = self.current_from_md(lag, i_boot) # set as target the current estimated from MD
                dummy = utils_model.estimate_Q(pairing, self.n_samples[i_boot][lag], self.dt, lag, self.T[i_boot][lag], self.F[i_boot][lag], self.T[i_boot][self.lags[0]], self.F[i_boot][self.lags[0]], x0, I, w_current)
                qs.append(dummy[0])
                cost_matrix[i_w_current] = -dummy[1]
                cost_current[i_w_current] = dummy[5]
            cost_matrix = (cost_matrix - np.nanmin(cost_matrix))/(np.nanmax(cost_matrix) - np.nanmin(cost_matrix))
            cost_current = (cost_current - np.nanmin(cost_current))/(np.nanmax(cost_current) - np.nanmin(cost_current))
            cost = cost_matrix + cost_current
            ind_best = np.argmin(cost)
            print('Selecting as best Q the one at w_current = {}'.format(ws_current[ind_best]))
            f = plt.figure()
            ax1 = f.add_subplot(3,1,1)
            plt.title('i_boot {} lag {}\ncost: {}'.format(i_boot, lag, cost))
            ax2 = f.add_subplot(3,1,2)
            ax3 = f.add_subplot(3,1,3)
            ax1.plot(dummy_ws_current, cost_matrix,'o-')
            ax1.plot(dummy_ws_current[ind_best], cost_matrix[ind_best],'or')
            ax1.set_xscale('log')
            plt.ylabel('cost_matrix')
            ax2.plot(dummy_ws_current, cost_current,'o-')
            ax2.plot(dummy_ws_current[ind_best], cost_current[ind_best],'or')
            ax2.set_xscale('log')
            plt.ylabel('cost_current')
            ax3.plot(dummy_ws_current, cost,'o-')
            ax3.plot(dummy_ws_current[ind_best], cost[ind_best],'or')
            ax3.set_xscale('log')
            plt.ylabel('cost')
            plt.xlabel('w')
            pdf.savefig()
            plt.close()
            self.Q[i_boot][lag] = np.copy(qs[ind_best])
        return copy.deepcopy(self.Q[i_boot])

    def estimate_disconnected_states(self, i_boot):
        fix2zero = []
        n_states, states = self.check_states(i_boot)
        for i_state, state_i in enumerate(states):
            for j_state, state_j in enumerate(states):
                if i_state != j_state:
                    n_steps = utils_channel.F_from_states(state_i, state_j, return_steps = True)
                    ##T_min = min(self.T[i_boot][self.lags[0]][i_state, j_state], self.T[i_boot][self.lags[0]][j_state, i_state])
                    #if self.T[i_boot][self.lags[0]][i_state, j_state] == 0 and n_steps > 1:
                    ##if n_steps > 1:
                    #    fix2zero.append((i_state,j_state))
        print('Number of disconnetted states: {}'.format(len(fix2zero)))
        return fix2zero

    def estimate_connectivity(self, i_boot, alpha = 0.05):
        from scipy.stats import chi2
        n_states, states = self.check_states(i_boot)
        #--- estimate Q with full connectivity
        Q_prev, log_like_prev = self.estimate_Q(i_boot)
        self.plot_Q(i_boot)
        self.plot_current(i_boot)
        fix2zero = []
        for i in range(n_states*n_states - 2*n_states):
            #--- define a threshold
            minQ, i_minQ, j_minQ  = self.minimal_Q(i_boot)
            print('Minimal value of Q > 0: {}'.format(minQ))
            print('\tstate_i = {}'.format(self.repr_state(self.states[i_boot][self.lags[0]][i_minQ])))
            print('\tstate_j = {}'.format(self.repr_state(self.states[i_boot][self.lags[0]][j_minQ])))
            #--- fix to zero the elements below the threshold
            fix2zero_new = self.Q_below_threshold(i_boot, minQ)
            n_pars_remove = len(set(fix2zero_new) - set(fix2zero))
            fix2zero = list(set(fix2zero) | set(fix2zero_new))
            print('Setting to zero the following elements ({} new elements)'.format(n_pars_remove))
            for inds in fix2zero:
                print('\tstate_i = {} index: {}'.format(self.repr_state(self.states[i_boot][self.lags[0]][inds[0]]), inds[0]))
                print('\tstate_j = {} index: {}'.format(self.repr_state(self.states[i_boot][self.lags[0]][inds[1]]), inds[1]))
                print('\t----')
            #--- estimate Q with reduced connectivity
            Q_now, log_like_now = self.estimate_Q(i_boot, fix2zero = fix2zero)
            self.plot_Q(i_boot)
            self.plot_current(i_boot)
            log_like_ratio = -2*(log_like_now - log_like_prev)
            print('log_like_prev:',log_like_prev)
            print('log_like_now:',log_like_now)
            print('log likelihood ratio:',log_like_ratio)
            print('Min log likelihood ratio:',np.min(log_like_ratio))
            chi2_stat = np.min(log_like_ratio)
            f = plt.figure()
            ax = f.add_subplot(1,1,1)
            ax.plot(self.dt*self.lags, log_like_ratio, '-o')
            plt.xlabel('Time [ns]')
            plt.ylabel('log likelihood ratio')
            plt.xscale('log')
            plt.title('Comparing models with {} Vs {} null elements'.format(len(fix2zero), len(fix2zero)-n_pars_remove))
            plt.grid()
            pdf.savefig()
            plt.close()
            #--- compare the Likelihood Ratio with the Chi2 distribution
            p_val = 1-chi2.cdf(chi2_stat, n_pars_remove)
            print('p-val = {}'.format(p_val))
            # null hypothesis: the complex model is NOT better
            # reject the null hypothesis --> the complex model is better, so keep the complex model and stop reducing
            if p_val < alpha: # stop when NOT  rejecting the null hypothesis --> the 
                print('Stop connectivity algorithm, resetting the previous estimate of Q')
                for i_lag, lag in enumerate(self.lags):
                    self.Q[i_boot][lag] = Q_prev[lag]
                break
            log_like_prev = log_like_now
            Q_prev = Q_now

    def plot_Q(self, pdf, i_boots):
        """
        """
        if isinstance(i_boots, int):
            i_boots = [i_boots,]
        n_states, states = self.check_states(i_boots)
        if len(i_boots) == 1: # plots if only one boot is requested
            i_boot = i_boots[0]
            Tpower = {}
            expQ = {}
            Tpower_fromQ1 = {}
            for i_lag, lag in enumerate(self.lags):
                Tpower[lag] = np.linalg.matrix_power(self.T[i_boot][self.lags[0]], lag)
                expQ[lag] = linalg.expm(self.Q[i_boot][lag]*self.dt*lag)
                Tpower_fromQ1[lag] = np.linalg.matrix_power(expQ[self.lags[0]], lag)
            #--- Plotting of T, exp(Q*dt)
            total_error = np.zeros(len(self.lags))
            aver_t = np.zeros(len(self.lags))
            for i_state in range(n_states):
                for j_state in range(n_states):
                    f = plt.figure()
                    ax = f.add_subplot(3,1,1)
                    qs = [self.Q[i_boot][lag][i_state, j_state] for lag in self.lags]
                    ax.plot(self.dt*self.lags, qs, 'o-')
                    plt.ylabel('Q_ij')
                    plt.title('i_boot = {} i_state = {} j_state = {}\nstate_i = {}\nstate_j = {}'.format(i_boot, i_state, j_state, self.repr_state(self.states[i_boot][self.lags[0]][i_state]), self.repr_state(self.states[i_boot][self.lags[0]][j_state])))
                    plt.xscale('log')
                    plt.grid()
                    ax = f.add_subplot(3,1,2)
                    ts = np.array([self.T[i_boot][lag][i_state, j_state] for lag in self.lags])
                    tps = [Tpower[lag][i_state, j_state] for lag in self.lags]
                    tps_Q1 = [Tpower_fromQ1[lag][i_state, j_state] for lag in self.lags]
                    expqs = np.array([expQ[lag][i_state, j_state] for lag in self.lags])
                    for i_lag, lag in enumerate(self.lags):
                        total_error[i_lag] += np.abs(expqs[i_lag] - ts[i_lag])
                        aver_t[i_lag] += ts[i_lag]
                    ax.plot(self.dt*self.lags, ts, 'o-', label = 'T')
                    ax.plot(self.dt*self.lags, tps, '^-', label = 'T[0]^lag')
                    ax.plot(self.dt*self.lags, expqs, 'x-', label = 'exp(Q*dt)')
                    ax.plot(self.dt*self.lags, tps_Q1, '+-', label = 'exp(Q*[dt = 1])^lag')
                    plt.ylabel('T_ij')
                    plt.xscale('log')
                    plt.grid()
                    plt.legend()
                    ax = f.add_subplot(3,1,3)
                    ax.plot(self.dt*self.lags, expqs-ts, 'o-', label = 'exp(Q*dt)-T')
                    plt.xlabel('Time [ns]')
                    plt.ylabel('error')
                    plt.xscale('log')
                    plt.grid()
                    plt.legend()
                    pdf.savefig()
                    plt.close()
            #--- Plot cumulative distance T - exp(Q*dt)
            f = plt.figure()
            ax = f.add_subplot(1,1,1)
            ax.plot(self.dt*self.lags, total_error, 'o-')
            plt.title('i_boot = {}'.format(i_boot))
            plt.xlabel('Time [ns]')
            plt.ylabel('Total matrix error')
            plt.grid()
            plt.xscale('log')
            pdf.savefig()
            plt.close()
            #--- Plot cumulative distance T - exp(Q*dt) normalized to average T value
            f = plt.figure()
            ax = f.add_subplot(1,1,1)
            ax.plot(self.dt*self.lags, total_error / aver_t, 'o-')
            plt.title('i_boot = {}'.format(i_boot))
            plt.xlabel('Time [ns]')
            plt.ylabel('Total matrix error / average T')
            plt.grid()
            plt.xscale('log')
            pdf.savefig()
            plt.close()
        else:
            expQ = {}
            for i_boot in i_boots:
                expQ[i_boot] = {}
                for i_lag, lag in enumerate(self.lags):
                    expQ[i_boot][lag] = linalg.expm(self.Q[i_boot][lag]*self.dt*lag)
            for j_state in range(n_states):
                for i_state in range(n_states):
                    f = plt.figure()
                    #ax = f.add_subplot(2,1,1)
                    #qs = np.zeros((len(i_boots), len(self.lags)))
                    #for ind_seq_boot, i_boot in enumerate(i_boots):
                    #    qs[ind_seq_boot,:] = [self.Q[i_boot][lag][i_state, j_state] for lag in self.lags]
                    #    ax.plot(self.dt*self.lags, qs[ind_seq_boot,:], 'b.')
                    #ax.errorbar(self.dt*self.lags, np.mean(qs, axis = 0), yerr =  np.std(qs, axis = 0), color = 'b')
                    #plt.ylabel('Q_ij')
                    #plt.title('i_state = {} j_state = {}\nstate_i = {}\nstate_j = {}'.format(i_state, j_state, self.repr_state(self.states[i_boot][self.lags[0]][i_state]), self.repr_state(self.states[i_boot][self.lags[0]][j_state])))
                    #plt.xscale('log')
                    #plt.grid()
                    ax = f.add_subplot(1,1,1)
                    ts = np.zeros((len(i_boots), len(self.lags)))
                    expqs = np.zeros((len(i_boots), len(self.lags)))
                    for ind_seq_boot, i_boot in enumerate(i_boots):
                        ts[ind_seq_boot,:] = np.array([self.T[i_boot][lag][i_state, j_state] for lag in self.lags])
                        #ax.plot(self.dt*self.lags, ts[ind_seq_boot,:], 'r.', label = None)
                        expqs[ind_seq_boot,:] = np.array([expQ[i_boot][lag][i_state, j_state] for lag in self.lags])
                        #ax.plot(self.dt*self.lags, expqs[ind_seq_boot,:], 'b.', label = None)
                    ax.errorbar(self.dt*self.lags, np.mean(ts, axis = 0), yerr =  np.std(ts, axis = 0), color = 'r', label = 'T')
                    ax.errorbar(self.dt*self.lags, np.mean(expqs, axis = 0), yerr =  np.std(expqs, axis = 0), color = 'b', label = 'exp(Q*dt)')
                    #ax.fill_between(self.dt*self.lags, np.mean(ts, axis = 0) -  np.std(ts, axis = 0), np.mean(ts, axis = 0) +  np.std(ts, axis = 0), alpha = 0.3, label = 'T')
                    #ax.fill_between(self.dt*self.lags, np.mean(expqs, axis = 0) -  np.std(expqs, axis = 0), np.mean(expqs, axis = 0) +  np.std(expqs, axis = 0), alpha = 0.3, label = 'exp(Q*dt)')
                    plt.ylabel('T_ij')
                    plt.ylim([0, 1.0])
                    plt.xlim(left = 0)
                    #plt.xscale('log')
                    #plt.legend()
                    plt.title('i_state = {} j_state = {}\nstate_i = {}\nstate_j = {}'.format(i_state, j_state, self.repr_state(self.states[i_boot][self.lags[0]][i_state]), self.repr_state(self.states[i_boot][self.lags[0]][j_state])))
                    plt.grid()
                    pdf.savefig()
                    plt.close()

    def plot_current(self, pdf, i_boot, title = ''):
        f = plt.figure()
        ax = f.add_subplot(1,1,1)
        I = np.empty(len(self.lags))
        for i_lag, lag in enumerate(self.lags):
            I[i_lag] = self.current(lag, i_boot)
        #print('DEBUG> I(T,F) = ',I)
        ax.plot(self.dt*self.lags, I, 'o-', label = 'from T,F')
        if i_boot in self.Q:
            for i_lag, lag in enumerate(self.lags):
                I[i_lag] = self.current_from_Q(lag, i_boot)
            ax.plot(self.dt*self.lags, I, 'o-', label = 'from Q')
            #print('DEBUG> I(Q) = ',I)
        plt.xlabel('Sampling period [ns]')
        plt.ylabel('Current [pA]')
        plt.title('{} [{}] {}'.format(self.prefix, i_boot, title))
        plt.legend()
        plt.grid()
        pdf.savefig()
        plt.xscale('log')
        pdf.savefig()
        plt.close()

    def check_states(self, i_boots, permissive = False, check_md = False):
        """
        Check that all the models at i_boot have the same states at all lags
        """
        if isinstance(i_boots, int) or isinstance(i_boots, np.number):
            i_boots = [i_boots,]
        #--- Define reference values
        n_states = self.states[i_boots[0]][self.lags[0]].shape[0]
        states = self.states[i_boots[0]][self.lags[0]]
        #--- Test if states are consistent
        for i_boot in i_boots:
            for ilag, lag in enumerate(self.lags):
                n_states_now = self.states[i_boot][lag].shape[0]
                if n_states_now != n_states:
                    if permissive:
                        return False
                    raise ValueError('ERROR: Number of states at lag {} i_boot {} = {} is not compatible with {}'.format(lag ,i_boot, n_states_now, n_states))
                if self.T[i_boot][lag].shape[0] != n_states:
                    if permissive:
                        return False
                    raise ValueError('ERROR: Number of states from T at lag {} i_boot {} = {} is not compatible with {}'.format(lag, i_boot, self.T[i_boot][lag].shape[0], n_states))
                if check_md:
                    if self.Tmd[i_boot][lag].shape[0] != n_states:
                        if permissive:
                            return False
                        raise ValueError('ERROR: Number of states from T at lag {} i_boot {} = {} is not compatible with {}'.format(lag, i_boot, self.Tmd[i_boot][lag].shape[0], n_states))
                for i_state in range(n_states):
                    state_i = states[i_state,:]
                    state_i_now = self.states[i_boot][lag][i_state,:]
                    if self.repr_state(state_i_now) != self.repr_state(state_i):
                        if permissive:
                            return False
                        else:
                            raise ValueError('ERROR: States {} at lag {} i_boot {} is not compatible with {}'.format(state_i_now, lag, i_boot, state_i))
        if permissive:
            return True
        else:
            return n_states, states

    def check_number_states(self, i_boot, permissive = False, check_md = False):
        """
        Check that all the models at i_boot have the same states at all lags
        """
        #--- Define reference values
        n_states = self.states[i_boot][self.lags[0]].shape[0]
        states = self.states[i_boot][self.lags[0]]
        #--- Test if states are consistent
        for ilag, lag in enumerate(self.lags):
            n_states_now = self.states[i_boot][lag].shape[0]
            if n_states_now != n_states:
                if permissive:
                    return False
                else:
                    raise ValueError('ERROR: Number of states at lag {} = {} is not compatible with {}'.format(lag, n_states_now, n_states))
        if permissive:
            return True
        else:
            return n_states
    
    def simulate(self, pdf, i_boot, lag, n_steps):
        #--- Generate initial state
        p = self.probability(i_boot, lag)
        p_cum = np.cumsum(p)
        r = np.random.uniform(low = 0, high = 1)
        dtraj = np.ones(n_steps, dtype = int)
        ftraj = np.ones(n_steps)
        ftraj[0] = 0.0
        dtraj[0] = np.where(p_cum > r)[0][0]
        for i_step in range(1, n_steps):
            p_from_last = self.T[i_boot][lag][:,dtraj[i_step-1]]
            p_cum = np.cumsum(p_from_last)
            r = np.random.uniform(low = 0, high = 1)
            dtraj[i_step] = np.where(p_cum > r)[0][0]
            ftraj[i_step] = self.F[i_boot][lag][dtraj[i_step], dtraj[i_step-1]]
        traj = self.states[i_boot][lag][dtraj,:]
        traj_for_plot = np.transpose(traj[:,:6]) # so S0 is first row, S1 second row, etc...
        time = self.dt*lag*np.arange(n_steps)
        f = plt.figure()
        ax = f.add_subplot(2,1,1)
        ax.matshow(traj_for_plot, aspect='auto')
        ax.set_yticks([0,1,2,3,4,5])
        ax.set_yticklabels(['S0', 'S1', 'S2', 'S3', 'S4', 'C'])
        inds = np.array(ax.get_xticks()[1:-1], dtype = int)
        ax.set_xticks(inds)
        labels = ['{}'.format(t) for t in time[inds]]
        ax.set_xticklabels(labels)
        ax.set_xlabel('Time [ns]')
        ax = f.add_subplot(2,1,2)
        ax.plot(time, np.cumsum(ftraj), '-')
        plt.xlim(left = 0)
        plt.xlim(right = time[-1])
        plt.xlabel('Time [ns]')
        plt.ylabel('Cumulative charge movement')
        pdf.savefig()
        plt.close()

    def print_states(self, i_boot, lag):
        n_states, states = self.check_states(i_boot)
        probs = self.probability(i_boot, lag)
        probs_Q = None
        if i_boot in self.Q:
            if lag in self.Q[i_boot]:
                probs_Q = self.probability(i_boot, lag, from_Q = True)
        for i_state in range(n_states):
            if probs_Q is not None:
                print('state {} prob. = {} prob_Q. = {}\n\t{}'.format(i_state, probs[i_state], probs_Q[i_state],  self.repr_state(self.states[i_boot][lag][i_state,:])))
            else:
                print('state {} prob. = {}\n\t{}'.format(i_state, probs[i_state],  self.repr_state(self.states[i_boot][lag][i_state,:])))
        for i_state in np.argsort(probs)[::-1]:
            print(self.repr_state(states[i_state,:]), '{0:6.3f}'.format(probs[i_state]))

    def repr_msm(self, i_boot, lag, what = ''):
        np.set_printoptions(threshold=np.inf, formatter={'float': '{: 4.3f}'.format}, linewidth = np.inf)
        output  = '# what = {}\n'.format(what)
        output += 'i_boot = {}\n'.format(i_boot)
        output += 'lag = {}\n'.format(lag)
        output += 'T = \n{}\n'.format(str(self.T[i_boot][lag]))
        output += 'F = \n{}\n'.format(str(self.F[i_boot][lag]))
        output += 'states = \n{}\n'.format(str(self.states[i_boot][lag]))
        output += 'states_original = \n{}\n'.format(str(self.states_original[i_boot][lag]))
        output += 'indexes_macro = \n{}\n'.format(str(self.indexes_macro[i_boot][lag]))
        np.set_printoptions(threshold=1000, formatter={'float': '{: 4.3f}'.format}, linewidth = 80)
        return output

    def __str__(self):
        output  = '-------------\n'
        output += 'Markov Models\n'
        output += '-------------\n'
        output += 'dt = {} ns\n'.format(self.dt)
        output += 'Lags: {}\n'.format(','.join([str(lag) for lag in self.lags]))
        output += 'Bootstraps: {}\n'.format(self.i_boots)
        for i_boot in self.i_boots:
            for i_lag, lag in enumerate(self.lags):
                output += 'i_boot {0:<6d} lag {1:<6d}\n'.format(i_boot, lag)
                output += '\tnumber of states = {}\n'.format(self.states[i_boot][lag].shape[0])
                output += '\tshape transition matrix = {}\n'.format(self.T[i_boot][lag].shape)
                if i_boot in self.Tmd:
                    if lag in self.Tmd[i_boot]:
                        output += '\tshape transition matrix md = {}\n'.format(self.Tmd[i_boot][lag].shape)
                output += '\tnumber of samples used to calculate T = {}\n'.format(self.n_samples[i_boot][lag])
                output += '\tnumber of not-null elements in T: {}/{}\n'.format(np.sum(self.T[i_boot][lag] > 0), np.prod(self.T[i_boot][lag].shape))
                for i_state in range(self.states[i_boot][lag].shape[0]):
                    output += 'state:  {}\n'.format(self.repr_state(self.states[i_boot][lag][i_state,:]))
                    for i_state_original in self.indexes_macro[i_boot][lag][i_state]:
                        output += '\t{}\t{}\n'.format(self.repr_state(self.states_original[i_boot][lag][i_state_original,:]), i_state_original)
                #output += 'T = \n'+str(self.T[i_boot][lag])+'\n'
                #output += 'F = \n'+str(self.F[i_boot][lag])+'\n'
        return output[:-1]

class MSM_from_MD(MSM):
    """
    Attributes
    ----------
    The same of the parent class +

    fileinput: str
        File in pk format used to read data about T, states, and discretized trajectories

    n_samples: dict
        key: int
            i_boot index
        value: dict
            key: int
                lag as multiple of self.dt
            value: number of samples used to compute the transition matrix

    Tmd: dict
        key: int
            i_boot index
        value: dict
            key: int
                lag as multiple of self.dt
            value: the transition matrix at that i_boot and lag calculated directly from MD trajectories

    Fmd: dict
        key: int
            i_boot index
        value: dict
            key: int
                lag as multiple of self.dt
            value: the flux matrix at that i_boot and lag calculated directly from MD trajectories

    Fmd_samples: dict
        key: int
            i_boot index
        value: dict
            key: int
                lag as multiple of self.dt
            value: dict
                key: tuple
                    i_state,j_state
                value: np.array
                    all the elements of F observed for the pair of states i_state,j_state at that i_boot and lag
    """

    def __init__(self, prefix, repr_state, value_state):
        """
        prefix: str
            Name of output files starts with this string
            an input file named $prefix.msm.pk is expected
        dt: float
            Timestep in ns
        """
        self.prefix = prefix
        self.fileinput = '{}.msm.pk'.format(self.prefix)
        self.states, self.states_original = {}, {}
        self.indexes_macro = {}
        self.T, self.F, self.Q = {}, {}, {}
        self.n_samples = {}
        self.Tmd, self.Fmd = {}, {}
        self.Fmd_samples = {}
        self.repr_state = repr_state
        self.value_state = value_state
        print('Reading data from: {}'.format(self.fileinput))
        with open(self.fileinput, 'rb') as fin:
            self.dt = pickle.load(fin)
            self.lags = list(pickle.load(fin))
            self.i_boots = pickle.load(fin)
            if isinstance(self.i_boots, int):
                self.i_boots = [-2,-1] + [i_boot for i_boot in range(self.i_boots)]
            for i_boot in self.i_boots:
                self.states[i_boot], self.states_original[i_boot] = {}, {}
                self.indexes_macro[i_boot] = {}
                self.T[i_boot], self.F[i_boot] = {}, {}
                self.n_samples[i_boot] = {}
                self.Tmd[i_boot], self.Fmd[i_boot] = {}, {}
                self.Fmd_samples[i_boot] = {}
            for i_lag, lag in enumerate(self.lags):
                for i_boot in self.i_boots:
                    self.T[i_boot][lag] = pickle.load(fin)
                    self.n_samples[i_boot][lag] = int(pickle.load(fin))
                    self.states[i_boot][lag] = pickle.load(fin)
                    self.indexes_macro[i_boot][lag] = [[i,] for i in np.arange(self.T[i_boot][lag].shape[0], dtype = int)]
                    print('\ti_boot: {} lag: {} shape(T): {} shape(states): {} n_sample: {}'.format(i_boot, lag, i_boot, self.T[i_boot][lag].shape, self.states[i_boot][lag].shape, self.n_samples[i_boot][lag]))
                    if i_lag == 0:
                        disc_trajs = pickle.load(fin)
                        f_trajs = pickle.load(fin)
                        total_time, conduction_events = 0, 0
                        for i_traj in range(len(disc_trajs)):
                            disc_traj = disc_trajs[i_traj]
                            f_traj = f_trajs[i_traj]
                            total_time += self.dt*len(disc_traj)
                            conduction_events += np.sum(f_traj)/7
                            print('\tTrajectory {}/{} nsteps = {} nsteps in F = {} time = {} ns conduction events = {}'.format(i_traj, len(disc_trajs), len(disc_traj), len(f_traj), self.dt*len(disc_traj), np.sum(f_traj)/7))
                        print('\tTotal simulation time = {} ns'.format(total_time))
                        print('\tNumber of conduction events = {}'.format(conduction_events))
                        print('\tAverage current = {} pA'.format(160.2*conduction_events/total_time))
                        while True:
                            run_again = False
                            #--- get rid of isolated states
                            print('Testing for the presence of isolated states')
                            inds = np.ones(self.T[i_boot][lag].shape[0], dtype = bool)
                            for i_row, i_col in zip(*np.where(self.T[i_boot][lag] == 1)):
                                if i_row == i_col:
                                    inds[i_row] = False
                                    run_again = True
                            if np.sum(np.logical_not(inds)):
                                print('Removing {} isolated states'.format(np.sum(np.logical_not(inds))))
                                self.remove_states(lag, i_boot, inds)
                            #--- get rid of states with null prob
                            print('Testing for the presence of states with null probability')
                            p = self.probability(i_boot, lag)
                            inds = (p == 0)
                            if np.sum(inds):
                                run_again = True
                                print('Removing {} states with null probability'.format(np.sum(inds)))
                                self.remove_states(lag, i_boot, np.logical_not(inds))
                            if not run_again:
                                break
                    self.states_original[i_boot][lag] = np.copy(self.states[i_boot][lag])
        self.lags = np.array(self.lags)

    def remove_states(self, lag, i_boot, inds):
        """
        inds: the ones to keep
        """
        print('Lag {} i_boot {} removing {} states'.format(lag, i_boot, np.sum(np.logical_not(inds))))
        for ind, ind_flag in enumerate(inds):
            if not ind_flag:
                print('\tremoving state {}'.format(self.repr_state(self.states[i_boot][lag][ind,:])))
        self.T[i_boot][lag] = self.T[i_boot][lag][:,inds][inds,:]
        self.T[i_boot][lag] = self.T[i_boot][lag] / np.sum(self.T[i_boot][lag], axis = 0)
        if lag in self.F[i_boot]:
            self.F[i_boot][lag] = self.F[i_boot][lag][:,inds][inds,:]
        if lag in self.Tmd[i_boot]:
            self.Tmd[i_boot][lag] = self.Tmd[i_boot][lag][:,inds][inds,:]
            self.Tmd[i_boot][lag] = self.Tmd[i_boot][lag] / np.sum(self.Tmd[i_boot][lag], axis = 0)
            if lag in self.Fmd[i_boot]:
                self.Fmd[i_boot][lag] = self.Fmd[i_boot][lag][:,inds][inds,:]
        self.states[i_boot][lag] = self.states[i_boot][lag][inds,:]
        self.indexes_macro[i_boot][lag] = [[i,] for i in np.arange(self.T[i_boot][lag].shape[0], dtype = int)]

    def merge_same_repr_state(self, i_boot):
        for lag in self.lags:
            print('Merging states with same occupancy for i_boot {} lag {}'.format(i_boot, lag))
            inds2merge = {}
            for i_state in range(self.states[i_boot][lag].shape[0]):
                state = self.repr_state(self.states[i_boot][lag][i_state,:])
                if state not in inds2merge:
                    inds2merge[state] = []
                inds2merge[state].append(i_state)
            inds2merge = [inds for inds in inds2merge.values()]
            for inds in inds2merge:
                print('\tMerging states with indexes:',inds)
                for i in inds:
                    print('\t\tstate = {}'.format(self.repr_state(self.states[i_boot][lag][i,:])))
            merge_results = self.merge(lag, i_boot, inds2merge.copy())
            self.update_indexes_macro(lag, i_boot, inds2merge)
            if len(merge_results) == 2:
                self.T[i_boot][lag] = merge_results[0]
                self.states[i_boot][lag] = merge_results[1]
            else:
                self.T[i_boot][lag] = merge_results[0]
                self.F[i_boot][lag] = merge_results[1]
                self.states[i_boot][lag] = merge_results[2]
            print('After merging same ion states for lag {} i_boot {}, n_states = {}'.format(lag, i_boot, self.states[i_boot][lag].shape[0]))

    def TF_from_trajs(self, i_boot):
        #--- read states, disc_trajs, and f_trajs from fileinput
        states, disc_trajs, f_trajs = {}, {}, {}
        with open(self.fileinput, 'rb') as fin:
            dt = pickle.load(fin)
            lags = list(pickle.load(fin))
            i_boots = pickle.load(fin)
            if isinstance(i_boots, int):
                i_boots = [-2,-1] + [i_boot for i_boot in range(i_boots)]
            for i in i_boots:
                states[i] = {}
            for i_lag, lag in enumerate(lags):
                for i in i_boots:
                    dummy = pickle.load(fin) # T
                    dummy = pickle.load(fin) # n_samples
                    states[i][lag] = pickle.load(fin)
                    if i_lag == 0:
                        disc_trajs[i] = pickle.load(fin)
                        f_trajs[i] = pickle.load(fin)
        for i_lag, lag in enumerate(self.lags):
            #--- do the mapping from microstates to macrostates
            n_macrostates = len(self.indexes_macro[i_boot][lag])
            micro2macro = {}
            for i_macro, indexes in enumerate(self.indexes_macro[i_boot][lag]):
                for index in indexes:
                    micro2macro[index] = i_macro
            print('Building T/F for lag {} i_boot {} over {} states using MD data'.format(lag, i_boot, n_macrostates))
            #--- compute T and F
            F, T = {}, {}
            for i_traj, disc_traj in enumerate(disc_trajs[i_boot]): # cicle over all the discretized trajectories for this i_boot
                f_traj = f_trajs[i_boot][i_traj]
                for t_start in range(len(disc_traj) - lag): # all available starting time points, for having at least one sample after lag
                    t_end = t_start + lag
                    i_start = disc_traj[t_start] # index of the start-state
                    i_end = disc_traj[t_end] # index of the end-state
                    if i_start in micro2macro and i_end in micro2macro:
                        i_macro_start = micro2macro[i_start]
                        i_macro_end = micro2macro[i_end]
                        pair = (i_macro_end, i_macro_start) # pairing for the F matrix
                        if pair not in F:
                            F[pair] = []
                            T[pair] = 0
                        F[pair].append(np.sum(f_traj[t_start:t_end])) # remember that f_traj start with the F of the 0->1 transition
                        T[pair] += 1
                    else:
                        if i_start not in micro2macro:
                            print('WARNING: missing start state {} at lag {}'.format(i_start, lag))
                        if i_end not in micro2macro:
                            print('WARNING: missing end state {} at lag {}'.format(i_end, lag))
            self.Tmd[i_boot][lag] = np.zeros((n_macrostates, n_macrostates)) # np.zeros((n_macrostates, n_macrostates))
            self.Fmd[i_boot][lag] = np.zeros((n_macrostates, n_macrostates))
            self.Fmd_samples[i_boot][lag] = {}
            for i_state in range(n_macrostates):
                for j_state in range(n_macrostates):
                    pair = i_state, j_state
                    if pair in F:
                        self.Tmd[i_boot][lag][i_state, j_state] = T[pair]
                        self.Fmd[i_boot][lag][i_state, j_state] = np.mean(F[pair])/7.0
                        self.Fmd_samples[i_boot][lag][i_state, j_state] = np.array(F[pair])/7.0
                    else:
                        self.Fmd_samples[i_boot][lag][i_state, j_state] = []
            while True:
                run_again = False
                #--- First using MD data
                inds = np.zeros(self.Tmd[i_boot][lag].shape[0], dtype = bool)
                print('Checking states to remove after MD data retrieval - starting test of Tmd:', len(inds))
                #------ get rid of states with null probability
                counts = np.sum(self.Tmd[i_boot][lag], axis = 0)
                for ind in np.where(counts == 0)[0]:
                    print('\tnull probability for state {}'.format(ind))
                    inds[ind] = True
                #------ get rid of isolated states
                for ind in range(self.Tmd[i_boot][lag].shape[0]):
                    dummy = self.Tmd[i_boot][lag][:,ind].copy()
                    if np.isnan(dummy[ind]):
                        inds[ind] = True
                        print('\tnan state {}'.format(ind))
                    else:
                        dummy[ind] = 0
                        if np.sum(dummy) == 0:
                            inds[ind] = True
                            print('\tisolated state {}'.format(ind))
                if np.sum(inds):
                    run_again = True
                    self.remove_states(lag, i_boot, np.logical_not(inds))
                #--- Then using original data
                inds = np.zeros(self.T[i_boot][lag].shape[0], dtype = bool)
                print('Checking states to remove after MD data retrieval - starting test of T:', len(inds))
                #------ get rid of states with null probability
                counts = np.sum(self.T[i_boot][lag], axis = 0)
                for ind in np.where(counts == 0)[0]:
                    print('\tnull probability for state {}'.format(ind))
                    inds[ind] = True
                #------ get rid of isolated or nan states
                for ind in range(self.T[i_boot][lag].shape[0]):
                    dummy = self.T[i_boot][lag][:,ind].copy()
                    if np.isnan(dummy[ind]):
                        inds[ind] = True
                        print('\tnan state {}'.format(ind))
                    else:
                        dummy[ind] = 0
                        if np.sum(dummy) == 0:
                            inds[ind] = True
                            print('\tisolated state {}'.format(ind))
                if np.sum(inds):
                    run_again = True
                    self.remove_states(lag, i_boot, np.logical_not(inds))
                #--- are we done ?
                if not run_again:
                    break
            self.Tmd[i_boot][lag] = self.Tmd[i_boot][lag] / np.sum(self.Tmd[i_boot][lag], axis = 0)
            self.T[i_boot][lag] = self.T[i_boot][lag] / np.sum(self.T[i_boot][lag], axis = 0)

    def find_lag(self, i_boot, thrs = 0.01):
        slow_taus = np.empty(len(self.lags))
        for i_lag, lag in enumerate(self.lags):
            slow_taus[i_lag] = self.dt*lag*self.timescales(1, lag, i_boot).flatten()
        delta_taus = ((slow_taus[1:] - slow_taus[:-1])/slow_taus[:-1]) / ((self.lags[1:] - self.lags[:-1])/self.lags[:-1])
        ind_lag = np.where(delta_taus < thrs)[0][0] # first lag that increase by less than thrs% with respect to the following one
        return self.lags[ind_lag]
    
    def plot_timescales(self, pdf, i_boots = [], n_timescales = None, title = ''):
        if isinstance(i_boots, int):
            i_boots = [i_boots,]
        if len(i_boots) == 0:
            i_boots = [i_boot for i_boot in self.i_boots if i_boot > 0]
        if n_timescales is None:
            n_timescales = np.inf
            for i_boot in i_boots:
                for i_lag, lag in enumerate(self.lags):
                    n_timescales = min(n_timescales, self.T[i_boot][lag].shape[0]-1)
            n_timescales = int(n_timescales)
        f = plt.figure()
        ax = f.add_subplot(1,1,1)
        taus = np.empty((len(self.lags), n_timescales))
        taus_std = np.zeros((len(self.lags), n_timescales))
        for i_lag, lag in enumerate(self.lags):
            time_constant = self.dt*lag*self.timescales(n_timescales, lag, i_boots)
            taus[i_lag, :] = np.mean(time_constant, axis = 0)
            if time_constant.shape[0] > 1:
                taus_std[i_lag, :] = np.std(time_constant, axis = 0)
        for i_tau in range(n_timescales):
            if np.all(taus_std == 0):
                ax.plot(self.dt*self.lags, taus[:, i_tau], 'o-', label = 'from T')
            else:
                ax.errorbar(self.dt*self.lags, taus[:, i_tau], yerr =  taus_std[:, i_tau], label = 'from T')
        if all([i_boot in self.Tmd for i_boot in i_boots]):
            taus = np.empty((len(self.lags), n_timescales))
            taus_std = np.zeros((len(self.lags), n_timescales))
            for i_lag, lag in enumerate(self.lags):
                time_constant = self.dt*lag*self.timescales_from_md(n_timescales, lag, i_boots)
                taus[i_lag, :] = np.mean(time_constant, axis = 0)
                if time_constant.shape[0] > 1:
                    taus_std[i_lag, :] = np.std(time_constant, axis = 0)
            for i_tau in range(n_timescales):
                if np.all(taus_std == 0):
                    ax.plot(self.dt*self.lags, taus[:, i_tau], 'o-', label = 'from Tmd')
                else:
                    ax.errorbar(self.dt*self.lags, taus[:, i_tau], yerr =  taus_std[:, i_tau], label = 'from Tmd')
        ax.plot(self.dt*self.lags, self.dt*self.lags, ':k')
        plt.xlim(left = self.dt)
        plt.ylim(bottom = 0)
        plt.xlabel('Sampling period [ns]')
        plt.ylabel('Timescale [ns]')
        plt.title('{} i_boots = {} {}'.format(self.prefix, i_boots, title))
        plt.grid()
        pdf.savefig()
        plt.xscale('log')
        pdf.savefig()
        plt.close()

    def plot_current(self, pdf, i_boots = [], title = ''):
        if isinstance(i_boots, int):
            i_boots = [i_boots,]
        if len(i_boots) == 0:
            i_boots = [i_boot for i_boot in self.i_boots if i_boot > 0]
        f = plt.figure()
        ax = f.add_subplot(1,1,1)
        I = np.empty(len(self.lags))
        I_std = np.zeros(len(self.lags))
        for i_lag, lag in enumerate(self.lags):
            current = self.current(lag, i_boots)
            I[i_lag] = np.mean(current)
            if len(current) > 1:
                I_std[i_lag] = np.std(current)
        if np.all(I_std == 0):
            ax.plot(self.dt*self.lags, I, 'o-', label = 'from T,F')
        else:
            ax.errorbar(self.dt*self.lags, I, yerr =  I_std, label = 'from T,F')
        plot_md = True
        for i_boot in i_boots:
            if i_boot not in self.Tmd:
                plot_md = False
                break
            for lag in self.lags:
                if lag not in self.Tmd[i_boot]:
                    plot_md = False
                    break
        if plot_md:
            I = np.empty(len(self.lags))
            I_std = np.zeros(len(self.lags))
            for i_lag, lag in enumerate(self.lags):
                current = self.current_from_md(lag, i_boots)
                I[i_lag] = np.mean(current)
                if len(current) > 1:
                    I_std[i_lag] = np.std(current)
            if np.all(I_std == 0):
                ax.plot(self.dt*self.lags, I, 'o-', label = 'from MD')
            else:
                ax.errorbar(self.dt*self.lags, I, yerr =  I_std, label = 'from MD')
        if all([i_boot in self.Q for i_boot in i_boots]):
            I = np.empty(len(self.lags))
            I_std = np.zeros(len(self.lags))
            for i_lag, lag in enumerate(self.lags):
                current = self.current_from_Q(lag, i_boots)
                I[i_lag] = np.mean(current)
                if len(current) > 1:
                    I_std[i_lag] = np.std(current)
            if np.all(I_std == 0):
                ax.plot(self.dt*self.lags, I, 'o-', label = 'from Q')
            else:
                ax.errorbar(self.dt*self.lags, I, yerr =  I_std, label = 'from Q')
        plt.xlabel('Sampling period [ns]')
        plt.ylabel('Current [pA]')
        plt.title('{} i_boots = {} {}'.format(self.prefix, i_boots, title))
        plt.grid()
        plt.legend()
        pdf.savefig()
        plt.xscale('log')
        pdf.savefig()
        plt.close()

    def plot_F(self, i_boot):
        n_states = self.states[i_boot][self.lags[0]].shape[0]
        distance = np.zeros(len(self.lags))
        n_out95conf = np.zeros(len(self.lags))
        for i in range(n_states):
            for j in range(n_states):
                f = np.zeros(len(self.lags))
                fmd = np.zeros(len(self.lags))
                #emd = np.zeros(len(self.lags))
                emd = np.zeros((2,len(self.lags)))
                for i_lag, lag in enumerate(self.lags):
                    if (self.F[i_boot][lag].shape[0] != n_states):
                        raise ValueError('ERROR: inconsistent dimensions')
                    f[i_lag] = self.F[i_boot][lag][i,j]
                    fmd[i_lag] = self.Fmd[i_boot][lag][i,j]
                    #emd[i_lag] = np.std(self.Fmd_samples[i_boot][lag][i,j])
                    distance[i_lag] += (f[i_lag] - fmd[i_lag])**2.0
                    if len(self.Fmd_samples[i_boot][lag][i,j]) > 10:
                        perc_low = np.percentile(self.Fmd_samples[i_boot][lag][i,j], 5)
                        perc_high = np.percentile(self.Fmd_samples[i_boot][lag][i,j], 95)
                        perc_50 = np.median(self.Fmd_samples[i_boot][lag][i,j])
                        emd[0, i_lag] = perc_50 - perc_low
                        emd[1, i_lag] = perc_high - perc_50
                        if (f[i_lag] < perc_low) or (f[i_lag] > perc_high):
                            n_out95conf[i_lag] += 1
                fig = plt.figure()
                ax = fig.add_subplot(1,1,1)
                ax.plot(self.dt*self.lags, f, 'o-', label = 'from T, F')
                ax.plot(self.dt*self.lags, fmd, 'x-', label = 'from MD')
                #ax.errorbar(self.dt*self.lags, fmd, yerr = emd, label = 'from MD')
                plt.legend()
                plt.xlabel('Lagtime [ns]')
                plt.ylabel('Fij')
                plt.xscale('log')
                plt.grid()
                pdf.savefig()
                plt.close()
        distance = np.sqrt(distance)/(n_states*n_states)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(self.dt*self.lags, distance, 'o-')
        plt.xlabel('Lagtime [ns]')
        plt.ylabel('|F - Fmd|')
        plt.xscale('log')
        plt.grid()
        pdf.savefig()
        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(self.dt*self.lags, n_out95conf, 'o-')
        plt.xlabel('Lagtime [ns]')
        plt.ylabel('dist > 95 conf.')
        plt.xscale('log')
        plt.grid()
        pdf.savefig()
        plt.close()

    def make_model(self, pdf, n_timescales = None, title = 'MSM'):
        i_boots_to_remove = []
        for i_boot in self.i_boots:
            self.merge_same_repr_state(i_boot)
            self.align_models(i_boot)
            self.set_F_from_states(i_boot, self.lags[0])
            self.propagate_F(i_boot)
            if not self.check_states(i_boot, permissive = True):
                print('WARNING: i_boot {} needs to be removed'.format(i_boot))
                i_boots_to_remove.append(i_boot)
                continue
            self.plot_current(pdf, i_boot, title = title)
            self.plot_timescales(pdf, i_boots = i_boot, n_timescales = n_timescales, title = title)
            self.print_states(i_boot, self.lags[-1])
        for i_boot in i_boots_to_remove:
            self.i_boots.remove(i_boot)
        i_boots = [i_boot for i_boot in self.i_boots if i_boot >= 0]
        if len(i_boots) > 1:
            self.plot_timescales(pdf, i_boots = i_boots, n_timescales = n_timescales, title = title)
            self.plot_current(pdf, i_boots = i_boots, title = title)

    def make_target_states_model(self, target_states, pdf, n_timescales = None):
        for i_boot in self.i_boots:
            self.align_models(i_boot = i_boot)
            self.merge_target(i_boot = i_boot, lag = self.lags[-1], target_states = target_states, keep_old_states = True)
            self.TF_from_trajs(i_boot)
            n_states, states = self.check_states(i_boot, check_md =  True)
            self.plot_timescales(pdf, i_boots = i_boot, title = 'target states')
            self.plot_current(pdf, i_boots = i_boot, title = 'target states')
            self.set_F_from_states(i_boot = i_boot, lag = self.lags[0])
            self.propagate_F(i_boot)
            self.plot_timescales(pdf, i_boots = i_boot, title = 'target states (after reset)')
            self.plot_current(pdf, i_boots = i_boot, title = 'target states (after reset)')
            self.print_states(i_boot,self.lags[-1])
        if len(self.i_boots) > 1:
            self.plot_timescales(pdf, n_timescales = n_timescales, title = 'target states (after reset)')
            self.plot_current(pdf, title = 'target states (after reset)')

    def make_rate_model(self, pdf, i_boots = [-1,]):
        for i_boot in i_boots:
            #fix2zero = self.estimate_disconnected_states(i_boot)
            #self.estimate_Q(pdf, i_boot, ws_current = np.logspace(0,2,10))
            self.estimate_Q_constrained(pdf, i_boot)
            self.plot_Q(pdf, i_boot)
            self.plot_current(pdf, i_boots = i_boot, title = 'Q')
        if len(i_boots) > 1:
            self.plot_Q(pdf, i_boots)
            self.plot_timescales(pdf, title = 'target states (after reset)')
            self.plot_current(pdf, title = 'target states (after reset)')

class MSM_from_matrixes(MSM):
    def __init__(self, dt, lags, T, F, states, Q = None):
        """
        MSM initialized manually from matrixes

        Alternative mode of initialization are possible (see Cases below)
        """
        self.prefix = 'MSM manually defined using input matrixes'
        self.Tmd = {}
        self.dt = dt
        if ((isinstance(lags, int) or isinstance(lags, np.number)) and isinstance(T, np.ndarray) and isinstance(states, np.ndarray) 
            and (isinstance(F, np.ndarray) or F is None) and (Q is None)):
            # Case 1:
            #   lags is a single number
            #   T is np.array --> the transition matrix at that lag
            #   states is np.array --> the states at that lag
            #   F is np.array or None --> the flux matrix at that lag
            #   Q is undefined
            self.lags = np.array([lags,])
            self.i_boots = [-1,]
            self.T = {-1:{lags:T}}
            self.F = {-1:{lags:F}}
            self.states = {-1:{lags:states}}
            self.states_original = copy.deepcopy(self.states)
            self.Q = {}
            self.indexes_macro = {-1:{lags:[[i,] for i in np.arange(self.T[-1][lags].shape[0], dtype = int)]}}
        elif (isinstance(lags, int) or isinstance(lags, np.number)) and (T is None) and isinstance(states, np.ndarray) and (isinstance(F, np.ndarray) or F is None) and isinstance(Q, np.ndarray):
            # Case 2:
            #   lags is a single number
            #   states is np.array --> the states at that lag
            #   F is np.array or None --> the flux matrix at that lag
            #   Q is np.array
            #   T is None --> the transition matrix is calculated from Q
            self.lags = np.array([lags,])
            self.i_boots = [-1,]
            self.T = {-1:{lags:linalg.expm(Q*self.dt*lags)}}
            self.F = {-1:{lags:F}}
            self.Q = {-1:{lags:Q}}
            self.states = {-1:{lags:states}}
            self.indexes_macro = {-1:{lags:[[i,] for i in np.arange(self.T[-1][lags].shape[0], dtype = int)]}}
        elif (isinstance(lags, list) or isinstance(lags, np.ndarray)) and isinstance(T, dict) and isinstance(states, dict) and (isinstance(F, dict) or F is None) and (Q is None):
            # Case 3:
            #   lags is a sequence of values
            #   T is a dict --> used to define self.T[-1]
            #   states is a dict --> used to define self.states[-1]
            #   F is a dict or None--> used to define self.F[-1]
            #   Q is undefined
            if isinstance(lags, list):
                lags = np.array(lags)
            self.lags = lags
            self.i_boots = [-1,]
            self.T = {-1:T}
            if F is None:
                self.F = {-1:{}}
            else:
                self.F = {-1:F}
            self.states = {-1:states}
            self.original_states = {-1:copy.deepcopy(states)}
            self.Q = {-1:Q}
            self.indexes_macro = {-1:{}}
            for lag in self.lags:
                self.indexes_macro[-1][lag] = [[i,] for i in np.arange(self.T[-1][lag].shape[0], dtype = int)]
        elif (isinstance(lags, list) or isinstance(lags, np.ndarray)) and (T is None) and (isinstance(F, np.ndarray)) and isinstance(states, np.ndarray) and (isinstance(Q, dict)):
            # Case 4:
            #   lags is a sequence of values
            #   Q is a dict of dict --> used to define self.Q [HERE MORE i_boots are possible]
            #   states is an np.array --> used to define states for all i_boots:lag
            #   F is an np.array --> used to define states for all i_boots:lag
            #   T is None --> it is computed from Q
            if isinstance(lags, list):
                lags = np.array(lags)
            self.lags = lags
            self.Q = Q
            self.i_boots = list(self.Q.keys())
            self.i_boots.sort()
            self.F, self.T, self.states, self.indexes_macro = {}, {}, {}, {}
            self.n_samples = {}
            self.repr_state = utils_channel.repr_state
            for i_boot in self.i_boots:
                self.F[i_boot] = {self.lags[0]:F}
                self.T[i_boot] = {lag:linalg.expm(Q[i_boot][lag]*self.dt*lag) for lag in self.lags}
                self.n_samples[i_boot] = {lag:0 for lag in self.lags}
                self.states[i_boot] = {lag:states for lag in self.lags}
                self.indexes_macro[i_boot] = {lag:[[i,] for i in np.arange(self.Q[i_boot][lag].shape[0], dtype = int)] for lag in self.lags}
            self.states_original = copy.deepcopy(self.states)
        else:
            raise TypeError('ERROR: wrong input types for class MSM_from_matrixes')
