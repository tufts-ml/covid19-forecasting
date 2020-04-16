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


HEALTH_STATE_ID_TO_NAME = {0: 'Declining', 1: 'MildRecovering', 2:'FullRecovering'}

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

		if start_state is None:
			# Shortcut for testing. Avoid need to mockup config dict, etc.
			pass
		else:
			self.simulate_trajectory(start_state, config_dict, prng, next_state_map, state_name_to_id)

	def simulate_trajectory(self, start_state, config_dict, prng, next_state_map, state_name_to_id):
        #每次initialize 一个person, 在init中会call simulate_trajectory, 并完成simulate 此person的trajectory.

		print('Simulating trajectory for this person: \n')       
		print('Inside simulate_trajectory, passed in start_state is {}\n'.format(start_state))
		print('Inside simulate_trajectory, avaliable progressions are: {}'.format(next_state_map.keys()))        
		## Simulate trajectory
		state = start_state
		health_state_id = 0 
		while health_state_id < 2 and state != 'TERMINAL': #meaning this person did not die, did not full recover
			#health_state_id = prng.rand() < config_dict['proba_Recovering_given_%s' % state]
			#when arrive at each state, toss the coin to decide what health_state this person will get
			conditional_health_state_pmf = [config_dict["proba_Declining_given_%s" % state], config_dict["proba_MildRecovering_given_%s" % state], config_dict["proba_FullRecovering_given_%s" % state]]
			print('conditional_health_state_pmf at current state is {}'.format(conditional_health_state_pmf))
			assert np.isclose(np.sum(conditional_health_state_pmf),1), 'not valid conditional health state pmf at current state, sum is %s'%np.sum(conditional_health_state_pmf)            
			health_state_id = prng.choice([0,1,2], 1, p=conditional_health_state_pmf)[0]
            
			print('at state {}, this person is {} \n'.format(state, HEALTH_STATE_ID_TO_NAME[int(health_state_id)]))
			print('Using {}:'.format('pmf_timesteps_%s_%s' % (HEALTH_STATE_ID_TO_NAME[health_state_id], state))) 
            
			choices_and_probas_dict = config_dict['pmf_timesteps_%s_%s' % (HEALTH_STATE_ID_TO_NAME[health_state_id], state)]
			print('Inside simulate_trajectory, choices_and_probas_dict for start_state {} is {}'.format(start_state, choices_and_probas_dict))
            #print
			#print('Inside simulate_trajectory, choices_and_probas_dict is {}'.format(choices_and_probas_dict))
            
			choices = np.fromiter(choices_and_probas_dict.keys(), dtype=np.int32)
            #print
			print('Inside simulate_trajectory, choices is {}'.format(choices))
            
			probas = np.fromiter(choices_and_probas_dict.values(), dtype=np.float64)
            #print
			print('Inside simulate_trajectory, probas is {}'.format(probas))
            
			duration = prng.choice(choices, p=probas)
            #print
			print('Inside simulate_trajectory, duration is {}'.format(duration))
            
			self.state_ids.append(state_name_to_id[state]) #not know state_name_to_id
			print('Inside simulate_trajectory, self.state_ids is {}'.format(self.state_ids))            
			self.health_state_ids.append(health_state_id)
			print('Inside simulate_trajectory, self.health_state_ids is {}'.format(self.health_state_ids))
			self.durations.append(duration)
			print('Inside simulate_trajectory, self.durations is {}'.format(self.durations)) 
			#hz: here need to change
			if health_state_id < 2:
				state = next_state_map['%s_%s' % (state, HEALTH_STATE_ID_TO_NAME[health_state_id])] 
				print('This person advancing to {} \n'.format(state))
			else:
				print('This person recovered from current state')
			self.is_terminal_0 = (state == 'TERMINAL')                
# 			state = next_state_map[state]
# 			if  health_state_id < 1:            
# 				print('This person advancing to {} \n'.format(state))
# 			else:
# 				print('This person recovered from current state')         
# 			self.is_terminal_0 = (state == 'TERMINAL' and health_state_id < 1)
		print('##########End of this person####### \n')          


	def update_count_matrix(self, count_TK, t_start):
		''' Update count matrix tracking population of each state at each time

		Returns
		-------
		count_TK : 2D array with shape (T, K)
			One row for each timestep
			One column for each state
		'''
		t = t_start
		for ii in range(len(self.state_ids)): #what does state_ids mean
			#print('self.durations[ii] is {}, self.state_ids[ii] is {}'.format(self.durations[ii], self.state_ids[ii]))
            
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