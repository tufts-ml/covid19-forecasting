HEALTH_STATE_ID_TO_NAME = {0: 'Declining', 1: 'Recovering'}

import numpy as np

class PatientTrajectory(object):
	''' Represents a simulated patient trajectory

	Attributes
	----------
	durations: list_of_int
	state_ids : list_of_str
	health_state_ids : list_of_int
	'''

	def __init__(self, start_state, config_dict, prng, next_state_map, state_name_to_id):
		self.durations = list()
		self.state_ids = list()
		self.health_state_ids = list()
		self.is_terminal = 0

		## Simulate trajectory
		state = start_state
		health_state_id = 0
		while health_state_id < 1 and not self.is_terminal:
			health_state_id = prng.rand() < config_dict['proba_Recovering_given_%s' % state]
			choices_and_probas_dict = config_dict['pmf_timesteps_%s_%s' % (HEALTH_STATE_ID_TO_NAME[health_state_id], state)]
			choices = np.fromiter(choices_and_probas_dict.keys(), dtype=np.int32)
			probas = np.fromiter(choices_and_probas_dict.values(), dtype=np.float64)
			duration = prng.choice(choices, p=probas)
			self.state_ids.append(state_name_to_id[state])
			self.health_state_ids.append(health_state_id)
			self.durations.append(duration)
			state = next_state_map[state]
			self.is_terminal = (state == 'TERMINAL')

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
		if self.is_terminal:
			count_TK[t, -1] += self.is_terminal
		return count_TK


	def update_discharge_count_matrix(self, count_TK, t_start):
		''' Update count matrix tracking "recovery" from each state at each time

		Returns
		-------
		count_TK : 2D array with shape (T, K)
			One row for each timestep
			One column for each state
		'''
		if not self.is_terminal:
			count_TK[t_start + np.sum(self.durations), self.state_ids[-1]] += 1
		return count_TK