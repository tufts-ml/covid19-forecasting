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

>>> t_st = 0 # start time, 

## Check terminal counts is accurate
>>> term_T1 = p.update_terminal_count_matrix(empty_count_TK.copy()[:,0], t_st)
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

>>> trans_time_ids = np.hstack([[0], np.cumsum(p.durations)])

## Check admit counts has only one entry
>>> admit_TK = p.update_admit_count_matrix(empty_count_TK.copy(), 0)
>>> np.sum(admit_TK)
1
>>> admit_TK[trans_time_ids]
array([[1, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0]], dtype=int32)

## Check discharge counts has only one entry
>>> discharge_TK = p.update_discharge_count_matrix(empty_count_TK.copy(), 0)
>>> np.sum(discharge_TK)
1
>>> discharge_TK[trans_time_ids, :]
array([[0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 1]], dtype=int32)"""


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

    def __init__(self, start_state=None, config_dict=None, prng=None, next_state_map=None, state_name_to_id=None, t=None):
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

        if start_state is None:
            # Shortcut for testing. Avoid need to mockup config dict, etc.
            pass
        else:
            self.simulate_trajectory(start_state, config_dict, prng, next_state_map, state_name_to_id, t)

    def simulate_trajectory(self, start_state, config_dict, prng, next_state_map, state_name_to_id, t):
        ## Simulate trajectory
        state = start_state
        health_state_id = 0
        while (state != 'TERMINAL' and state != 'RELEASE'):

            if health_state_id < 1:
                health_state_id = prng.rand() < config_dict['proba_Recovering_given_%s' % state]

            choices_and_probas_dict = config_dict['pmf_duration_%s_%s' % (HEALTH_STATE_ID_TO_NAME[health_state_id], state)]
            choices = np.fromiter(choices_and_probas_dict.keys(), dtype=np.int32)
            probas = np.fromiter(choices_and_probas_dict.values(), dtype=np.float64)
            assert np.allclose(1.0, np.sum(probas))
            duration = prng.choice(choices, p=probas)
            if len(self.state_ids) == 0 and t <= 0:
                try:
                    choices_and_probas_dict = config_dict['pmf_initial_duration_spent_%s' % (state)]
                    choices = np.fromiter(choices_and_probas_dict.keys(), dtype=np.int32)
                    probas = np.fromiter(choices_and_probas_dict.values(), dtype=np.float64)
                    assert np.allclose(1.0, np.sum(probas))
                    duration_spent = prng.choice(choices, p=probas)
                    duration = np.maximum(duration - duration_spent, 1)
                except KeyError:
                    pass
            self.state_ids.append(state_name_to_id[state])
            self.health_state_ids.append(health_state_id)
            self.durations.append(duration)
            previous_state = state
            state = next_state_map[state+HEALTH_STATE_ID_TO_NAME[health_state_id]]
            
            try:
                if prng.rand()<config_dict['proba_Die_given_%s' % (previous_state)] and health_state_id < 1 :
                    state = 'TERMINAL' 
            except KeyError:  # proba_Die_given_STATE not specified in params.json, so premature death from this STATE does not exist
                pass
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
        ''' Update count matrix tracking "newly admitted" at each state, time

        Returns
        -------
        count_TK : 2D array with shape (T, K)
            One row for each timestep
            One column for each state
            Exactly one 1 entry for each patient in simulation.
        '''
        count_TK[t_start, self.state_ids[0]] += 1
        return count_TK

    def update_discharge_count_matrix(self, count_TK, t_start):
        ''' Update count matrix tracking "recovery" from each state at each time

        Returns
        -------
        count_TK : 2D array with shape (T, K)
            One row for each timestep
            One column for each state
            At most one 1 entry for each patient in simulation,
        '''
        if not self.is_terminal_0:
            count_TK[t_start + np.sum(self.durations), self.state_ids[-1]] += 1
        return count_TK
