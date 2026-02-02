"""
utils_channel.py includes functions that depends on the specific channel under analysis
"""

import numpy as np

def F_from_states(state_i, state_j, max_n_steps = 10, return_steps = False):
    """
    Compute the amount of elementary charge that moves across the channel when going from state_i to state_j
        considering only the configuration of ions

    Parameters
    ----------
    state_[ij]: np.array
        State of the ion channel coded as 0's and 1's with this scheme:
        [s0_ion, s1_ion, s2_ion, s3_ion, s4_ion, sc_ion, s0_water, s1_water, s2_water, s3_water, s4_water]

    max_n_steps: int
        fter this number of steps the it gives up and raise an error

    return_steps: bool
        If True returns the number of steps not the charge movement

    Return
    ------
    float:   elementary charge (or number of steps) for the transition state_i --> state_j
    """
    def generate_new_states(state_old, F_prev = 0):
        new_states, F_new_states = [], []
        if state_old[0] == 1: # from S0 to out
            new_state = state_old.copy()
            new_state[0] = 0
            new_states.append(new_state)
            F_new_states.append(1 + F_prev)
            if state_old[1] == 0: # moving inward from S0
                new_state = state_old.copy()
                new_state[0] = 0
                new_state[1] = 1
                new_states.append(new_state)
                F_new_states.append(-1 + F_prev)
        elif state_old[0] == 0: # from out to S0
            new_state = state_old.copy()
            new_state[0] = 1
            new_states.append(new_state)
            F_new_states.append(-1 + F_prev)
        if state_old[5] == 1: # from cavity to in
            new_state = state_old.copy()
            new_state[5] = 0
            new_states.append(new_state)
            F_new_states.append(-1 + F_prev)
            if state_old[4] == 0: # moving outward from cavity
                new_state = state_old.copy()
                new_state[5] = 0
                new_state[4] = 1
                new_states.append(new_state)
                F_new_states.append(+1 + F_prev)
        elif state_old[5] == 0: # from cavity to S4
            new_state = state_old.copy()
            new_state[5] = 1
            new_states.append(new_state)
            F_new_states.append(1 + F_prev)
        for i in range(1,5):
            if state_old[i] == 1:
                if state_old[i-1] == 0: # moving outward
                    new_state = state_old.copy()
                    new_state[i] = 0
                    new_state[i-1] = 1
                    new_states.append(new_state)
                    F_new_states.append(1 + F_prev)
                if state_old[i+1] == 0: # moving inward
                    new_state = state_old.copy()
                    new_state[i] = 0
                    new_state[i+1] = 1
                    new_states.append(new_state)
                    F_new_states.append(-1 + F_prev)
        return new_states, F_new_states
    state_i = np.round(state_i)
    state_j = np.round(state_j)
    if np.all(state_i[:6] == state_j[:6]):
        return 0
    new_states_next, F_new_states_next = generate_new_states(state_j)
    n_steps = 1
    while True:
        for i_state, new_state in enumerate(new_states_next):
            if np.all(new_state[:6] == state_i[:6]):
                if return_steps:
                    return n_steps
                else:
                    return F_new_states_next[i_state]/7.0
        new_states = new_states_next.copy()
        F_new_states = F_new_states_next.copy()
        new_states_next, F_new_states_next = [], []
        for j, new_state in enumerate(new_states):
            new_states_dummy, F_new_states_dummy = generate_new_states(new_state, F_new_states[j])
            new_states_next.extend(new_states_dummy)
            F_new_states_next.extend(F_new_states_dummy)
        n_steps += 1
        if n_steps > max_n_steps:
            raise ValueError('ERROR: could not go from {} to {} in {} steps'.format(state_i, state_j, max_n_steps))

def repr_state(state):
    return ' '.join(['{0:6.3f}'.format(np.round(site, decimals = 0)) for site in state])

def repr_state_only_ions(state):
    inds_ions = [0, 1, 2, 3, 4, 5,]
    return ' '.join(['{0:6.3f}'.format(np.round(site, decimals = 0)) for site in state[inds_ions]])

def value_state(state):
    return state

def value_state_only_ions(state):
    inds_ions = [0, 1, 2, 3, 4, 5,]
    return state[inds_ions]

def repr_state_full(state):
    return ' '.join(['{0:3.1f}'.format(site) for site in state])

def label(state):
    output = ''
    if state[0] > 0.5:
        output = 'K'
    elif state[6] > 0.5:
        output = 'w'
    else:
        output = '-'
    if state[1] > 0.5:
        output += 'K'
    elif state[7] > 0.5:
        output += 'w'
    else:
        output += '-'
    if state[2] > 0.5:
        output += 'K'
    elif state[8] > 0.5:
        output += 'w'
    else:
        output += '-'
    if state[3] > 0.5:
        output += 'K'
    elif state[9] > 0.5:
        output += 'w'
    else:
        output += '-'
    if state[4] > 0.5:
        output += 'K'
    elif state[10] > 0.5:
        output += 'w'
    else:
        output += '-'
    if state[5] > 0.5:
        output += 'K'
    else:
        output += 'w'
    return output

#    def plot_states(self, pdf, i_boot, lag):
#        n_states = self.states[i_boot][lag].shape[0]
#        inds_sort = [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5]
#        #                         0     1     2     3     4    5      6      7      8      9     10
#        label_sites = np.array(['S0', 'S1', 'S2', 'S3', 'S4', 'C', 'wS0', 'wS1', 'wS2', 'wS3', 'wS4'])
#        sites = np.arange(self.states[i_boot][lag].shape[1])
#        prob = self.probability(i_boot, lag)
#        for i_state in range(n_states):
#            state = self.states[i_boot][lag][i_state, :]
#
#            #f = plt.figure()
#            #ax = f.add_subplot(1,1,1)
#            #ax.bar(sites, state[inds_sort])
#            #plt.xticks(sites, label_sites[inds_sort])
#            #plt.title('prob = {}'.format(prob[i_state]))
#            #pdf.savefig()
#            #plt.close()
#
#            f = plt.figure()
#            ax = f.add_subplot(1,1,1)
#            ions = state[[0, 1, 2, 3, 4, 5]]
#            water = np.array(list(state[[6, 7, 8, 9, 10]]) + [1-state[5]])
#            for i in range(5):
#                if water[i] + ions[i] > 1:
#                    water[i] = water[i] - (ions[i] + water[i] - 1)
#            empty = np.ones(6) - ions - water
#            empty[empty<0] = 0
#            ax.bar([0, 1, 2, 3, 4, 5], ions, edgecolor='black', linewidth=1, facecolor='lime')
#            ax.bar([0, 1, 2, 3, 4, 5], water, bottom = ions, edgecolor='black', linewidth=1, facecolor='skyblue')
#            ax.bar([0, 1, 2, 3, 4, 5], empty, bottom = ions+water, edgecolor='black', linewidth=1,facecolor='white')
#            plt.xticks([0, 1, 2, 3, 4, 5], ['S0', 'S1', 'S2', 'S3', 'S4', 'C'])
#            plt.title('prob = {}'.format(prob[i_state]))
#            plt.ylim([0, 1.1])
#            #plt.grid()
#            pdf.savefig()
#            plt.close()
#
#            #i_states_original = self.indexes_macro[i_boot][lag][i_state]
#            #f = plt.figure()
#            #for i_site in range(len(sites)):
#            #    ax = f.add_subplot(len(sites), 1, i_site+1)
#            #    ax.plot(np.arange(len(i_states_original)), self.states_original[i_boot][lag][i_states_original, i_site], 'o-')
#            #    ax.set_ylim([0, 1])
#            #    ax.set_ylabel(label_sites[i_site])
#            #pdf.savefig()
#            #plt.close()
# def csv_for_gephy(self, i_boot, lag, prefix_output, prob_min = 0.0):
#     def n_ions(state):
#         return np.sum(state[0:5])
#     def waters_s2s3(state):
#         return state[8]+state[9]
#     def waters_s2s3_combs(state):
#         return state[8]+2*state[9]
#     def waters_s1s2s3s4_combs(state):
#         return state[7]+2*state[8]+4*state[9]+8*state[10]
#     def waters_s0s1(state):
#         return state[6]+state[7]
#     def ions_s0s1(state):
#         return state[0]+state[1]
#     def ions_s4c(state):
#         return state[4]+state[5]
#     csv_nodes = open('{}.nodes.csv'.format(prefix_output), 'wt')
#     csv_nodes.write('Id,node,Label,n_ions_sf,waters_s0s1,waters_s2s3,waters_s2s3_combs,waters_s1s2s3s4_combs,ions_s0s1,ions_s4c,water_s0,water_s1,Prob\n')
#     probs = self.probability(i_boot, lag)
#     for i_state, state in enumerate(self.states[i_boot][lag]):
#         if probs[i_state] > prob_min:
#             if (state[0] == 1 and state[6] == 1) or  (state[1] == 1 and state[7] == 1) or (state[2] == 1 and state[8] == 1) or (state[3] == 1 and state[9] == 1) or (state[4] == 1 and state[10] == 1):
#                 print(state[0:5]*state[6:],''.join([str(site) for site in state]), utils_channel.label(state), probs[i_state])
#             csv_nodes.write('{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(i_state,''.join([str(site) for site in state]), utils_channel.label(state), n_ions(state), waters_s0s1(state), waters_s2s3(state), waters_s2s3_combs(state), waters_s1s2s3s4_combs(state), ions_s0s1(state), ions_s4c(state), state[6], state[7],  probs[i_state]))
#     csv_nodes.close()
#     csv_edges = open('{}.edges.csv'.format(prefix_output), 'wt')
#     csv_edges.write('Source,Target,Type,Id,Weight\n')
#     i_edge = 0
#     for i_state, state_i in enumerate(self.states[i_boot][lag]):
#         if probs[i_state] > prob_min:
#             for j_state, state_j in enumerate(self.states[i_boot][lag]):
#                 if j_state != i_state and probs[j_state] > prob_min:
#                     if self.T[i_boot][lag][i_state,j_state] > 0:
#                         csv_edges.write('{},{},Directed,{},{}\n'.format(j_state, i_state, i_edge, self.T[i_boot][lag][i_state,j_state]))
#                         i_edge += 1
#     csv_edges.close()
#
# def coming_from(self, source, sink, i_boot, lag, pdf, title):
#     #[s0_ion, s1_ion, s2_ion, s3_ion, s4_ion, sc_ion, s0_water, s1_water, s2_water, s3_water, s4_water]
#     # 0       1       2       3       4       5       6         7         8         9         10
#     probs = self.probability(i_boot, lag)
#     source_states, probs_source_states, sink_states = [], [], []
#     for i_state, state_i in enumerate(self.states[i_boot][lag]):
#         if source(state_i):
#             p_i = 0.0 # probability to reach any state in sink from i_state
#             set_sinks = set()
#             if probs[i_state] > 0.0:
#                 for j_state, state_j in enumerate(self.states[i_boot][lag]):
#                     if sink(state_j):
#                         if self.T[i_boot][lag][j_state, i_state] > 0.0:
#                             p_i += probs[i_state] * self.T[i_boot][lag][j_state, i_state]
#                             set_sinks.add(utils_channel.label(state_j))
#                             #print(state_i, state_j, probs[i_state], self.T[i_boot][lag][j_state, i_state])
#             if p_i > 0.0:
#                 source_states.append(state_i)
#                 probs_source_states.append(p_i)
#                 sink_states.append(set_sinks)
#     source_states = np.array(source_states)
#     probs_source_states = np.array(probs_source_states)
#     sink_states = np.array(sink_states)
#     probs_source_states /= np.sum(probs_source_states)
#     inds_sort = np.argsort(probs_source_states)[::-1]
#     probs_source_states = probs_source_states[inds_sort]
#     source_states = source_states[inds_sort,:]
#     sink_states = sink_states[inds_sort]
#     for i_state, state_i in enumerate(source_states):
#         print(state_i, utils_channel.label(state_i),  probs_source_states[i_state], sink_states[i_state])
#     ave_state = np.sum(source_states * probs_source_states.reshape((-1,1)), axis = 0)
#     n_states = self.states[i_boot][lag].shape[0]
#     inds_sort = [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5]
#     inds_sites_water = [1, 3, 5, 7, 9]
#     inds_water = [6, 7, 8, 9, 10]
#     #                         0     1     2     3     4    5      6      7      8      9     10
#     label_sites = np.array(['S0', 'S1', 'S2', 'S3', 'S4', 'C', 'wS0', 'wS1', 'wS2', 'wS3', 'wS4'])
#     sites = np.arange(len(ave_state))
#     f = plt.figure()
#     ax = f.add_subplot(1,1,1)
#     sites = [0, 1, 2, 3, 4, 5]
#     ions = ave_state[[0, 1, 2, 3, 4, 5]]
#     water = np.array(list(ave_state[[6, 7, 8, 9, 10]]) + [ave_state[5]-1])
#     for i in range(5):
#         if water[i] + ions[i] > 1:
#             water[i] = water[i] - (ions[i] + water[i] - 1)
#     empty = np.ones(6) - ions - water
#     empty[empty<0] = 0
#     ax.bar(sites, ions, edgecolor='black', linewidth=1, facecolor='lime')
#     ax.bar(sites, water, bottom = ions, edgecolor='black', linewidth=1, facecolor='skyblue')
#     ax.bar(sites, empty, bottom = ions+water, edgecolor='black', linewidth=1,facecolor='white')
#     plt.xticks(sites, ['S0', 'S1', 'S2', 'S3', 'S4', 'C'])
#     plt.title(title)
#     plt.ylim([0, 1.1])
#     pdf.savefig()
#     plt.close()
#
# def run(self, i_boot, lag, n_steps, pdf):
#     """
#     Run a simulation of n_steps starting from start_state
#
#     Return
#     ------
#     np.array(dtype = int) with n_steps elements
#         The discrete state in the trajectory
#     """
#     dtraj = np.zeros(n_steps, dtype = int)
#     ftraj = np.zeros(n_steps, dtype = int)
#     p = self.probability(i_boot, lag)
#     p_cum = np.cumsum(p)
#     r = np.random.uniform(low = 0, high = 1)
#     dtraj[0] =  np.where(p_cum > r)[0][0]
#     for i_step in range(1, n_steps):
#         p_from_last = self.T[i_boot][lag][:,dtraj[i_step-1]]
#         p_cum = np.cumsum(p_from_last)
#         r = np.random.uniform(low = 0, high = 1)
#         dtraj[i_step] = np.where(p_cum > r)[0][0]
#         ftraj[i_step] = (self.F[i_boot][lag][dtraj[i_step], dtraj[i_step-1]])
#     events_up, events_down = utils_model.detect_cumulative_changes(ftraj)
#     events = np.zeros(n_steps)
#     events[events_up] = +1
#     events[events_down] = -1
#     eigv = self.eigenvectors(2, i_boot = i_boot, lag = lag)
#     slow_eigv = eigv[:,1]
#     mask = np.array(slow_eigv[dtraj] > 0, dtype = np.int64)
#     #mask = np.array([kwk(self.states[i_boot][lag][i_state,:]) for i_state in dtraj], dtype = int)
#     bits0, bits1 = utils_model.split_by_mask(events, mask)
#     f = plt.figure()
#     ax = f.add_subplot(1,1,1)
#     ax.plot(np.arange(n_steps), np.cumsum(ftraj), 'b-')
#     ax.plot(events_up, np.cumsum(ftraj)[events_up], 'rv')
#     ax.plot(events_down, np.cumsum(ftraj)[events_down], 'g^')
#     pdf.savefig()
#     plt.close()
#     f = plt.figure()
#     ax = f.add_subplot(1,1,1)
#     ax.plot(np.arange(n_steps), mask, '-')
#     pdf.savefig()
#     plt.close()
#     n_events, len_events = [], []
#     print('Probs1 = {}'.format(np.sum(mask == 1) / len(mask)))
#     for bit0 in bits0:
#         #f = plt.figure()
#         #ax = f.add_subplot(1,1,1)
#         #ax.plot(np.arange(len(bit0)), np.cumsum(bit0), '-')
#         #pdf.savefig()
#         #plt.close()
#         n_events.append(np.cumsum(bit0)[-1])
#         len_events.append(len(bit0))
#     print('Number of events: {}±{} / {}±{}'.format(np.mean(n_events), np.std(n_events), np.mean(len_events), np.std(len_events)))
#     n_events, len_events = [], []
#     for bit1 in bits1:
#         #f = plt.figure()
#         #ax = f.add_subplot(1,1,1)
#         #ax.plot(np.arange(len(bit1)), np.cumsum(bit1), '-')
#         #pdf.savefig()
#         #plt.close()
#         if len(bit1) > 100:
#             n_events.append(np.cumsum(bit1)[-1])
#             len_events.append(len(bit1))
#     print('Number of events: {}±{} / {}±{}'.format(np.mean(n_events), np.std(n_events), np.mean(len_events), np.std(len_events)))
#     return dtraj
