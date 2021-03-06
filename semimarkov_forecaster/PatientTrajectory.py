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
>>> p.is_terminal_0 = False

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
       [0, 0, 0, 0, 1]], dtype=int32)

>>> summary_dict = p.get_length_of_stay_summary_dict()
>>> summary_dict['duration_State00']
5
>>> summary_dict['duration_State00+Recovering']
0
>>> summary_dict['duration_State00+Declining']
5
>>> summary_dict['duration_State01']
4
>>> summary_dict['duration_State02']
3
>>> summary_dict['duration_State03']
2
>>> summary_dict['duration_State04']
2
"""

import numpy as np
import pandas as pd
from collections import defaultdict

HEALTH_STATE_ID_TO_NAME = {0: 'Declining', 1: 'Recovering'}

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
    is_terminal_0 : bool
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
        self.state_name_to_id = state_name_to_id
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
            try:
                assert(np.allclose(1.0, np.sum(probas)))
            except AssertionError as e:
                L = len(probas)
                diagnostic_df = pd.DataFrame(
                            np.hstack([probas, np.cumsum(probas)]).reshape((2,L)).T,
                            columns=['probas', 'cumsum'])
                raise ValueError("Probabilities do not sum to one for state %s,%s\n%s" % (
                    state,
                    HEALTH_STATE_ID_TO_NAME[health_state_id],
                    str(diagnostic_df)))

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
            
            # Progress to the next state
            next_state = next_state_map[state+HEALTH_STATE_ID_TO_NAME[health_state_id]]

            # Allow option for premature terminal state
            if health_state_id < 1:
                try:
                    if prng.rand() < config_dict['proba_Die_after_Declining_%s' % (state)]:
                        next_state = 'TERMINAL' 
                except KeyError:  # proba_Die not specified, so premature death from this STATE does not exist
                    pass
            
            # Advance to next state
            state = next_state

            # End while loop block. Continue if not terminal or recovered.


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

    def get_length_of_stay_summary_dict(self):
        ''' Compute summary statistics about this patient's entire stay

        Returns
        -------
        summary_dict : dict 
            Dictionary with string keys and count values
        '''
        if self.state_name_to_id is None:
            state_id_to_name = dict([
                (a, 'State%02d' % a) for a in range(1+np.max(self.state_ids))])
        else:
            state_id_to_name = dict(
                zip(self.state_name_to_id.values(),
                    self.state_name_to_id.keys()))

        L = len(self.durations)
        summary_dict = defaultdict(int)
        summary_dict['is_Terminal'] = int(self.is_terminal_0)
        summary_dict['is_InICU'] = 0
        summary_dict['is_OnVent'] = 0
        summary_dict['duration_All'] = np.sum(self.durations)
        for ll in range(L):
            health_state = HEALTH_STATE_ID_TO_NAME[self.health_state_ids[ll]]
            state_name = state_id_to_name[self.state_ids[ll]]
            duration = self.durations[ll]
            summary_dict['duration_' + state_name] += duration
            summary_dict['duration_' + state_name + "+" + health_state] += duration

            if state_name.count("ICU"):
                summary_dict['is_InICU'] = 1
            if state_name.count("OnVent"):
                summary_dict['is_OnVent'] = 1

        summary_dict['duration_All'] = np.sum(self.durations)
        return summary_dict
