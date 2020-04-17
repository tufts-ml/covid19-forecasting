"""
PatientTrajectory.py

Examples
--------

# Patient that goes through all stages but eventually recovers
>>> K = 5
>>> p = PatientTrajectory()
>>> p.state_ids = [0, 1, 2, 3, 4]
>>> p.durations = [5, 4, 3, 2, 2]
>>> p.health_state_ids = [0, 0, 0, 0, 1]
>>> p.is_terminal_0 = 0

>>> T = np.sum(p.durations) + 1
>>> empty_count_TK = np.zeros((T, K), dtype=np.int32)

## Check terminal counts is accurate
>>> term_T1 = p.update_terminal_count_matrix(empty_count_TK.copy()[:,0], 0)
>>> np.sum(term_T1)
0

## Check occupancy counts are accurate
>>> occ_TK = p.update_count_matrix(empty_count_TK.copy(), 0)
>>> np.allclose(np.sum(occ_TK[:-1,:], axis=1), 1.0)
True
>>> occ_TK
array([[1, 0, 0, 0, 0],
       [1, 0, 0, 0, 0],
       [1, 0, 0, 0, 0],
       [1, 0, 0, 0, 0],
       [1, 0, 0, 0, 0],
       [0, 1, 0, 0, 0],
       [0, 1, 0, 0, 0],
       [0, 1, 0, 0, 0],
       [0, 1, 0, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 0, 1, 0],
       [0, 0, 0, 1, 0],
       [0, 0, 0, 0, 1],
       [0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0]], dtype=int32)

## Check admit counts track entry to each stage
>>> admit_TK = p.update_admit_count_matrix(empty_count_TK.copy(), 0)
>>> np.sum(admit_TK)
5
>>> admit_TK[[0] + np.cumsum(p.durations).tolist()]
array([[1, 0, 0, 0, 0],
       [0, 1, 0, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 0, 1, 0],
       [0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0]], dtype=int32)

## Check discharge has only one entry
>>> discharge_TK = p.update_discharge_count_matrix(empty_count_TK.copy(), 0)
>>> np.sum(discharge_TK)
1
>>> discharge_TK[np.sum(p.durations), :]
array([0, 0, 0, 0, 1], dtype=int32)
"""


HEALTH_STATE_ID_TO_NAME = {0: 'Declining', 1: 'Recovering'}

import numpy as np

class PatientTrajectory(object):
    ''' Represents a simulated patient trajectory using semi-markov model

    Attributes
    ----------
    durations: list_of_int
        List of integer giving duration of current (ordinal state, health state) pair
    state_ids : list_of_int
        List of integer indicator (indicating current ordinal state)
    health_state_ids : list_of_int
        List of which binary health state (declining, recovering)
    is_terminal_0 : int
        Indicates if final state is terminal, declining
    '''

    def __init__(self, start_state=None, config_dict=None, prng=None, next_state_map=None, state_name_to_id=None):
        ''' Construct a PatientTrajectory from provided input

        Args
        ----
        start_state : str
            Name of state the patient starts in
        config_dict : dict
            Dict containing parameters of semi-markov process (read from JSON file)
        prng : numpy RandomState

        Returns
        -------
        Newly constructed PatientTrajectory instance
        '''
        self.durations = list()
        self.state_ids = list()
        self.health_state_ids = list()
        self.is_terminal = 0
        choices = range(len(config_dict['age_groups']))
        age_probas=list()
        for age_group in config_dict['age_groups']:
            age_probas.append(config_dict['proba_Presenting_given_%s'%age_group])
        self.age_group_id = prng.choice(choices, p=age_probas)

        if start_state is None:
            # Shortcut for testing. Avoid need to mockup config dict, etc.
            pass
        else:
            self.simulate_trajectory(start_state, config_dict, prng, next_state_map, state_name_to_id)

    def simulate_trajectory(self, start_state, config_dict, prng, next_state_map, state_name_to_id):
        ## Simulate trajectory
        state = start_state
        health_state_id = 0
        while health_state_id < 1 and state != 'TERMINAL':
            health_state_id = prng.rand() < config_dict['proba_Recovering_given_%s' % state]
            choices_and_probas_dict = config_dict['pmf_timesteps_%s_%s' % (HEALTH_STATE_ID_TO_NAME[health_state_id], state)]
            choices = np.fromiter(choices_and_probas_dict.keys(), dtype=np.int32)
            probas = np.fromiter(choices_and_probas_dict.values(), dtype=np.float64)
            duration = prng.choice(choices, p=probas)
            self.state_ids.append(state_name_to_id[state])
            self.health_state_ids.append(health_state_id)
            self.durations.append(duration)
            state = next_state_map[state]
            self.is_terminal_0 = (state == 'TERMINAL' and health_state_id < 1)


    def update_count_matrix(self, count_TK, t_start):
        ''' Update count matrix tracking population of each state at each time

        Returns
        -------
        count_TK : 2D array with shape (T, K)
            One row for each timestep
            One column for each state
        '''
        t = t_start
        for ii in range(len(self.state_ids)):
            count_TK[t:t+self.durations[ii], self.state_ids[ii]] += 1
            t += self.durations[ii]
        return count_TK

    def update_terminal_count_matrix(self, count_T1, t_start):
        ''' Update count matrix tracking population of each state at each time

        Returns
        -------
        count_T1 : 2D array with shape (T, 1)
            One row for each timestep
            One column only, for terminal state
        '''
        if self.is_terminal_0:
            t_terminal = t_start + np.sum(self.durations)
            count_T1[t_terminal, 0] += 1
        return count_T1

    def update_admit_count_matrix(self, count_TK, t_start):
        ''' Update count matrix tracking "newly admitted" population into each state at each time

        Returns
        -------
        count_TK : 2D array with shape (T, K)
            One row for each timestep
            One column for each state
        '''
        t = t_start
        L = len(self.durations)
        for ii in range(L):
            count_TK[t, self.state_ids[ii]] += 1
            t = t + self.durations[ii]
        return count_TK

    def update_discharge_count_matrix(self, count_TK, t_start):
        ''' Update count matrix tracking "recovery" from each state at each time

        Returns
        -------
        count_TK : 2D array with shape (T, K)
            One row for each timestep
            One column for each state
        '''
        if not self.is_terminal_0:
            count_TK[t_start + np.sum(self.durations), self.state_ids[-1]] += 1
        return count_TK
    
    def update_count_by_age_matrix(self, count_TKA, t_start):
        ''' Update count matrix tracking population of each state at each time

        Returns
        -------
        count_TK : 2D array with shape (T, K, A)
            One dimension for each timestep
            One dimension for each state
            One dimension for each age group
        '''
        t = t_start
        for ii in range(len(self.state_ids)):
            count_TKA[t:t+self.durations[ii], self.state_ids[ii], self.age_group_id] += 1
            t += self.durations[ii]
        return count_TKA

    def update_terminal_count_by_age_matrix(self, count_T1A, t_start):
        ''' Update count matrix tracking population of each state at each time

        Returns
        -------
        count_T1 : 2D array with shape (T, 1)
            One row for each timestep
            One column only, for terminal state
            One dimension for each age group
        '''
        if self.is_terminal_0:
            t_terminal = t_start + np.sum(self.durations)
            count_T1A[t_terminal, 0, self.age_group_id] += 1
        return count_T1A

    def update_admit_count_by_age_matrix(self, count_TKA, t_start):
        ''' Update count matrix tracking "newly admitted" population into each state at each time

        Returns
        -------
        count_TK : 2D array with shape (T, K)
            One dimension for each timestep
            One dimension for each state
            One dimension for each age group
        '''
        t = t_start
        L = len(self.durations)
        for ii in range(L):
            count_TKA[t, self.state_ids[ii], self.age_group_id] += 1
            t = t + self.durations[ii]
        return count_TKA

    def update_discharge_count_by_age_matrix(self, count_TKA, t_start):
        ''' Update count matrix tracking "recovery" from each state at each time

        Returns
        -------
        count_TK : 2D array with shape (T, K)
            One dimension for each timestep
            One dimension for each state
            One dimension for each age group
        '''
        if not self.is_terminal_0:
            count_TKA[t_start + np.sum(self.durations), self.state_ids[-1], self.age_group_id] += 1
        return count_TKA