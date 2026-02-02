"""
Functions for working on the markov state models
"""

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import linear_sum_assignment
from scipy import constants
from scipy import linalg

def common_order(value_state, *args):
    """
    Compute indexes to rearrange the states of a series of MSMs so that they are in the same order

    Parameters
    ----------
    *args   instances of class MSM

    Return
    ------
    list of list of int
        How to sort the states of each MSM to have them in the same order
    """
    states_ref = args[0]
    n_states = states_ref.shape[0]
    inds_common_orders = [np.arange(n_states),] # for the first one keep the original order
    for states in args[1:]:
        if n_states != states.shape[0]:
            raise ValueError('ERROR: impossible to sort with the same order with {} and {} states'.format(n_states, states.shape[0]))
        C = np.empty((n_states, n_states))
        states_i = []
        for i in range(n_states):
            state_i = value_state(states_ref[i,:])
            for j in range(n_states):
                state_j = value_state(states[j,:])
                C[i,j] = np.linalg.norm(state_i-state_j)
        row_ind, col_ind = linear_sum_assignment(C)
        inds_common_orders.append(col_ind)
    return inds_common_orders

def detect_cumulative_changes(x, threshold=1):
    """
    Return
    ------
    list of int
        Indexes of arr where the cumulative change is above arr
        at each event the cumulative change is resetted
    """
    events_up, events_down = [], []
    cumx = np.cumsum(x)
    for i in range(len(x)):
        if cumx[i] >= threshold:
            events_up.append(i)
            cumx = cumx - cumx[i]
        if cumx[i] <= -threshold:
            events_down.append(i)
            cumx = cumx - cumx[i]
    return events_up, events_down

def split_by_mask(data, mask):
    """
    Split `data` into contiguous segments based on binary `mask` values (0 or 1).

    Parameters
    ----------
    data : np.ndarray
        Array to be split.
    mask : np.ndarray
        Binary array (same length as data) containing 0s and 1s.

    Returns
    -------
    list0, list1 : list of np.ndarray
        Two lists: one for segments where mask == 0, one where mask == 1.
    """
    data = np.asarray(data)
    mask = np.asarray(mask)
    if data.shape != mask.shape:
        raise ValueError("data and mask must have the same shape")
    # Find indices where mask changes (0→1 or 1→0)
    change_points = np.nonzero(np.diff(mask))[0] + 1
    segments = np.split(data, change_points)
    mask_segments = np.split(mask, change_points)
    list0 = [seg for seg, m in zip(segments, mask_segments) if m[0] == 0]
    list1 = [seg for seg, m in zip(segments, mask_segments) if m[0] == 1]
    return list0, list1

def probability(T):
    """
    Compute state probabilites from the transition matrix of the MSM

    Parameters
    ----------
    T   np.array
        Transition matrix of the MSM

    Return
    ------
    np.array    State probabilities computed using the eigenvector with eigenvalue 1
    """
    ls, es = np.linalg.eig(T)
    inds_sort = np.argsort(np.abs(ls))[::-1]
    es = es[:,inds_sort]
    prob_states = np.abs(es[:,0])
    prob_states /= np.sum(prob_states)
    return prob_states

def probability_Q(Q):
    """
    Compute state probabilities from the rate matrix of the MSM

    Parameters
    ----------
    Q   np.array
        Rate matrix of the MSM

    Return
    ------
    np.array    State probabilities computed using the eigenvector with eigenvalue 0
    """
    ls, es = np.linalg.eig(Q)
    inds_sort = np.argsort(np.abs(ls))
    es = es[:,inds_sort]
    prob_states = np.abs(es[:,0])
    prob_states /= np.sum(prob_states)
    return prob_states

def estimate_Q(pairing, n_samples, dt, lag, T, F, T0, F0, x0, I_target, w_current):
    """
    Numerical estimate of the rate matrix Q

    Parameters
    ----------
    pairing list
        list of pairs i,j
        Q is assumed different from zero only for the pairs in this list

    n_samples   int
        Number of samples that was used to estimate T

    dt  float
        The minimal timestep of the trajectories used to estimate the MSM

    lag int
        The period used to sample the trajectories when estimating the MSM
        --> the period of the MSM is dt*lag

    T   np.array
        The transition matrix

    F   np.array
        The flux matrix

    T0  np.array
        The transition matrix at the first lag (i.e. lag == 1)

    F0  np.array
        The flux matrix at the first lag (i.e. lag == 1)

    x0  np.array
        Initial conditions for estimating Q

    I_target    float
        The target current

    w_current   float
        This parameters define how to deal with current distance in the optimization procedure

    Return
    ------
    np.array    The estiamated rate matrix

    float   log_likelihood

    float   dist_matrix_max

    float   dist_matrix_mean

    float   dist_current

    float   np.abs(dist_current/I_target)
    """
    def current(Q):
        """
        Compute current from Q
        """
        scale = (constants.e*1e12)/(1e-9)
        prob_states = probability_Q(Q)
        IQ = scale*np.sum(F0*Q*prob_states.reshape((1,-1)))
        return IQ
    def residual(x, pairing, n_samples, dt, lag, T, w_current, return_what = 'cost'):
        Q = np.zeros((T.shape[0], T.shape[0]))
        for k, pair in enumerate(pairing):
            Q[*pair] = x[k]
        for j in range(T.shape[0]):
            Q[j,j] = -np.sum(Q[:,j])
        IQ = current(Q)
        expQ = np.linalg.matrix_power(linalg.expm(Q*dt), lag)
        log_likelihood = np.sum(T*n_samples*probability(T).reshape((1,-1))*np.log(expQ))
        dist_current = np.abs(IQ - I_target)
        dist_matrix_max = np.max(np.abs(expQ - T))
        dist_matrix_mean = np.mean(np.abs(expQ - T))
        dist_matrix = -log_likelihood
        if return_what == 'cost':
            return dist_matrix + w_current*dist_current
        elif return_what == 'log_likelihood':
            return log_likelihood
        elif return_what == 'metrics':
            return log_likelihood, dist_matrix_max, dist_matrix_mean, dist_current, np.abs(dist_current/I_target)
        else:
            raise ValueError('ERROR: unknown return for residual function {}'.format(return_what))
    def constraint(x):
        Q = np.zeros((T.shape[0], T.shape[0]))
        for k, pair in enumerate(pairing):
            Q[*pair] = x[k]
        for j in range(T.shape[0]):
            Q[j,j] = -np.sum(Q[:,j])
        IQ = current(Q)
        dist_current = (IQ - I_target)
        return dist_current
    def constraint_1(x, perc = 0.05):
        Q = np.zeros((T.shape[0], T.shape[0]))
        for k, pair in enumerate(pairing):
            Q[*pair] = x[k]
        for j in range(T.shape[0]):
            Q[j,j] = -np.sum(Q[:,j])
        IQ = current(Q)
        dist_current = IQ - (I_target - perc*np.abs(I_target))
        return dist_current
    def constraint_2(x, perc = 0.05):
        Q = np.zeros((T.shape[0], T.shape[0]))
        for k, pair in enumerate(pairing):
            Q[*pair] = x[k]
        for j in range(T.shape[0]):
            Q[j,j] = -np.sum(Q[:,j])
        IQ = current(Q)
        dist_current = -IQ + (I_target + perc*np.abs(I_target))
        return dist_current
    print('Estimating Q using {} free parameters'.format(len(pairing)))
    if np.isinf(w_current):
        i_min_tests = 0
        while True:
            res = minimize(residual, x0, method = 'SLSQP' # COBYLA' #trust-constr' #method = 'SLSQP'
                       , bounds = len(x0)*[(0.0, None),]
                       , args = (pairing, n_samples, dt, lag, T, 0.0)
                       , constraints = ({'type':'ineq', 'fun':constraint_1,}, {'type':'ineq', 'fun':constraint_2,}) 
                       , options = {'maxiter':int(1e3)}, tol = 1e-30)
                       #, constraints = ({'type':'eq', 'fun':constraint,}) 
            i_min_tests += 1
            if constraint_1(res.x) > 0 and constraint_2(res.x) > 0:
                if i_min_tests > 1:
                    print('Solution within contraints found after {} attempts'.format(i_min_tests))
                break
            if i_min_tests > 7:
                print('WARNING: Solution within contraints NOT found after {} attempts'.format(i_min_tests))
                break
            if i_min_tests < 2:
                w_current_init_cond = 1e0
            elif i_min_tests < 3:
                w_current_init_cond = 1e1
            elif i_min_tests < 4:
                w_current_init_cond = 1e2
            elif i_min_tests < 5:
                w_current_init_cond = 1e3
            else:
                w_current_init_cond = 1e4
            res = minimize(residual, x0, bounds = len(x0)*[(0.0, None),], args = (pairing, n_samples, dt, lag, T, w_current_init_cond), options = {'maxiter':int(1e3)}, tol = 1e-30)
            x0 = res.x
    else:
        res = minimize(residual, x0, bounds = len(x0)*[(0.0, None),], args = (pairing, n_samples, dt, lag, T, w_current), options = {'maxiter':int(1e3)}, tol = 1e-30)
    if np.any(res.x < 0):
        raise ValueError('ERROR: negative element in Q')
    Q = np.zeros((T.shape[0], T.shape[0]))
    for k, pair in enumerate(pairing):
        Q[*pair] = res.x[k]
    for j in range(T.shape[0]):
        Q[j,j] = -np.sum(Q[:,j])
    lk = residual(res.x, pairing, n_samples, dt, lag, T, 0.0, return_what = 'metrics')
    return Q, *lk
