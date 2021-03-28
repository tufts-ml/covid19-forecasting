import os
import json
import argparse
import numpy as np
import scipy.stats
from scipy import stats
import pandas as pd
import tqdm
import json
import warnings
from copy import deepcopy
import pickle
import itertools
from semimarkov_forecaster import *

def run_simulation(random_seed, config_dict, states, func_name, approximate=None):
    prng = np.random.RandomState(random_seed)
    
    T = config_dict['num_timesteps']
    K = len(states) # Num states (not including terminal)

    ## Number of *previous* timesteps to simulate
    # Defaults to 0 if not provided
    Tpast = config_dict.get('num_past_timesteps', 0)

    ## Preallocate admit, discharge, and occupancy
    Tmax = 10 * T + Tpast
    occupancy_count_TK = np.zeros((Tmax, K), dtype=np.float64)
    discharge_count_TK = np.zeros((Tmax, K), dtype=np.float64)
    terminal_count_T1 = np.zeros((Tmax, 1), dtype=np.float64)

    health_ids = [0, 1]

    HEALTH_STATE_ID_TO_NAME = {0: 'Declining', 1: 'Recovering'}

    H = len(health_ids)

    duration_cdf_HKT = np.zeros((H, K, Tmax))
    for health, stage in itertools.product(health_ids, np.arange(K)):
        choices_and_probas_dict = config_dict[
            'pmf_duration_%s_%s' % (
                HEALTH_STATE_ID_TO_NAME[health], states[stage])]
        choices = np.fromiter(choices_and_probas_dict.keys(), dtype=np.int32)
        probas = np.fromiter(choices_and_probas_dict.values(), dtype=np.float64)
        for c, p in zip(choices, probas):
            assert c >= 1
            duration_cdf_HKT[health, stage, c - 1] = p
        duration_cdf_HKT[health, stage, :] = np.cumsum(duration_cdf_HKT[health, stage, :])

    sim_kwargs = {}
    sim_kwargs['duration_cdf_HKT'] = duration_cdf_HKT
    sim_kwargs['pRecover_K'] = np.asarray([
        float(config_dict['proba_Recovering_given_%s' % stage])
        for stage in states])
    sim_kwargs['pDieAfterDeclining_K'] = np.asarray([
        float(config_dict.get('proba_Die_after_Declining_%s' % stage, 0.0))
        for stage in states])

    ## Read functions to sample the incoming admissions to each state
    admissions_per_state_Tplus1K = np.zeros((T+1, K), dtype=np.int32)
    for ss, state in enumerate(states):
        try:
            csvfile = config_dict['pmf_num_per_timestep_%s' % state]
        except KeyError:
            try:
                csvfile = config_dict['pmf_num_per_timestep_{state}']
            except KeyError:
                continue
        
        csv_df = pd.read_csv(csvfile)
        state_key = 'num_%s' % state
        if state_key not in csv_df.columns:
            continue
        admissions_per_state_Tplus1K[:, ss] = np.array(csv_df[state_key][:T+1])

    init_num_per_state_Tpastplus1K = np.zeros((Tpast+1, K), dtype=np.int32)
    for ss, state in enumerate(states):
        N_new_dict = config_dict['init_num_%s' % state]
        if isinstance(N_new_dict, dict):
            indices = np.array(list(N_new_dict.keys()), dtype=np.int32)
            init_num_per_state_Tpastplus1K[indices, ss] = np.array(list(N_new_dict.values()), dtype=np.int32)
        else:
            init_num_per_state_Tpastplus1K[0, ss] = np.int32(N_new_dict)

    L = K * 2
    M = (np.sum(admissions_per_state_Tplus1K) + np.sum(init_num_per_state_Tpastplus1K)) * L * 2 + 1
    prng = np.random.RandomState(random_seed)
    rand_vals_M = prng.rand(M)

    sim_kwargs['durations_L'] = np.zeros(L, dtype=np.int32)
    sim_kwargs['stage_ids_L'] = -99 * np.ones(L, dtype=np.int32)
    sim_kwargs['health_ids_L'] = -99 * np.ones(L, dtype=np.int32)
    sim_kwargs['occupancy_count_TK'] = occupancy_count_TK
    sim_kwargs['discharge_count_TK'] = discharge_count_TK
    sim_kwargs['terminal_count_T1'] = terminal_count_T1

    if approximate is not None:
        init_num_per_state_Tpastplus1K = np.rint(init_num_per_state_Tpastplus1K.astype(np.float64) / float(approximate)).astype(np.int32)
        admissions_per_state_Tplus1K = np.rint(admissions_per_state_Tplus1K.astype(np.float64) / float(approximate)).astype(np.int32)

    sim_kwargs['init_num_per_state_Tpastplus1K'] = init_num_per_state_Tpastplus1K
    sim_kwargs['admissions_per_state_Tplus1K'] = admissions_per_state_Tplus1K
    states_by_id = np.array([0, 1, 2], dtype=np.int32)
    
    if func_name.count('cython'):
        occupancy_count_TK, discharge_count_TK, terminal_count_T1 = run_forecast__cython(Tpast=Tpast, T=T, Tmax=Tmax, states=states_by_id, rand_vals_M=rand_vals_M, **sim_kwargs)
    else:
        occupancy_count_TK, discharge_count_TK, terminal_count_T1 = run_forecast__python(Tpast=Tpast, T=T, Tmax=Tmax, states=states_by_id, rand_vals_M=rand_vals_M, **sim_kwargs)

    # Save only the first T + 1 tsteps (with index 0, 1, 2, ... T)
    t0 = 0
    tf = T + Tpast + 1
    occupancy_count_TK = np.asarray(occupancy_count_TK)[t0:tf]
    discharge_count_TK = np.asarray(discharge_count_TK)[t0:tf]
    terminal_count_T1 = np.asarray(terminal_count_T1)[t0:tf]

    if approximate is not None:
        occupancy_count_TK = np.rint(occupancy_count_TK * float(approximate)).astype(np.int32)
        discharge_count_TK = np.rint(discharge_count_TK * float(approximate)).astype(np.int32)
        terminal_count_T1 = np.rint(terminal_count_T1 * float(approximate)).astype(np.int32)

    ## Write results to spreadsheet
    col_names = ['n_%s' % s for s in states]
    results_df = pd.DataFrame(occupancy_count_TK, columns=col_names)
    results_df["timestep"] = np.arange(-Tpast, -Tpast + tf)
    results_df["n_TERMINAL"] = terminal_count_T1[:,0]

    discharge_col_names = ['n_discharged_%s' % s for s in states]
    for k, col_name in enumerate(discharge_col_names):
        results_df[col_name] = discharge_count_TK[:, k]

    return results_df


HEALTH_STATE_ID_TO_NAME = {0: 'Declining', 1: 'Recovering', 'Declining': 0, 'Recovering': 1}

class ABCSampler(object):

    def __init__(self, seed, start_epsilon, annealing_constant, T_y, train_test_split, config_dict, func_name, num_timesteps, num_simulations, approximate=None):
        self.T_y = T_y # vector of true summary statistics
        self.epsilon = start_epsilon
        self.annealing_constant = annealing_constant
        self.seed = seed
        self.config_dict = config_dict
        self.func_name = func_name
        self.num_timesteps = num_timesteps
        self.num_simulations = num_simulations
        self.train_test_split = train_test_split # integer timestep for now, then will be a date
        self.approximate = approximate

        self.state_name_id_map = {}
        for s, state in enumerate(config_dict['states']):
            self.state_name_id_map[state] = s
            self.state_name_id_map[s] = state

    def initialize_theta(self, algorithm, abc_prior_dict, params_init=None):
        self.params_init = params_init
        self.abc_prior_dict = abc_prior_dict

        if algorithm == 'abc':
            self.abc_prior = self.initialize_abc_prior(abc_prior_dict)
        else:
            self.abc_prior = None
        
        if params_init is None:
            self.theta_init = self.select_theta_init_from_prior(self.abc_prior)
        else:
            self.theta_init = self.select_theta_init_from_name(params_init, self.abc_prior)


    def initialize_abc_prior(self, abc_prior_dict):
        '''
        Initializes internal representation of the prior distributions for all parameters.
        '''
        states = self.config_dict['states']
        
        health = [np.array(abc_prior_dict['prior_Health_given_%s' % (state)]) for state in states]
        die_after_declining = [np.array(abc_prior_dict['prior_Die_after_Declining_%s' % (state)]) for state in states if state != "OnVentInICU"]
        
        durations = {'Declining': [], 'Recovering': []}
        for state in states:
            for health_state in ['Declining', 'Recovering']:
                lam_dict = abc_prior_dict['prior_duration_%s_%s' % (health_state, state)]['lam']
                tau_dict = abc_prior_dict['prior_duration_%s_%s' % (health_state, state)]['tau']

                a = lam_dict['lower']
                b = lam_dict['upper']
                lam_mean = lam_dict['mean']
                lam_stddev = lam_dict['stddev']
                alpha = (a - lam_mean) / lam_stddev
                beta = (b - lam_mean) / lam_stddev

                tau_mean = tau_dict['mean']
                tau_stddev = tau_dict['stddev']

                durations[health_state].append({'lam': stats.truncnorm(alpha, beta, loc=lam_mean, scale=lam_stddev), 'tau': stats.norm(loc=tau_mean, scale=tau_stddev), 'lower': lam_dict['lower'], 'upper': lam_dict['upper']})

        abc_prior = {'health': health, 'die_after_declining': die_after_declining, 'durations': durations}
        
        return abc_prior

    def select_theta_init_from_prior(self, prior):
        '''
        Samples a set of parameters from the prior.

        Arguments:
            - prior: prior over the parameters specified in the internal representation (e.g. output of function self.initialize_abc_prior())

        Returns:
            - theta_init: parameters (theta) specified in the internal representation
        '''
        states = self.config_dict['states']

        health = [np.random.dirichlet(params) for params in prior['health']]
        die_after_declining = [np.random.dirichlet(params) for params in prior['die_after_declining']]

        durations = {'Declining': [], 'Recovering': []}
        for s, state in enumerate(states):
            for health_state in ['Declining', 'Recovering']:
                durations[health_state].append({'lam': prior['durations'][health_state][s]['lam'].rvs(size=1)[0], 'tau': prior['durations'][health_state][s]['tau'].rvs(size=1)[0], 'lower': prior['durations'][health_state][s]['lower'], 'upper': prior['durations'][health_state][s]['upper']})

        theta_init = {'num timesteps': self.num_timesteps, 'health': health, 'die_after_declining': die_after_declining, 'durations': durations}

        return theta_init

    def select_theta_init_from_name(self, params_init, prior=None):
        '''
        Sets theta_init to a specific set of parameters, specified by a predetermined name. Used mostly for experimental purposes.
        The prior, in the internal representation form, may be included.
        '''
        states = self.config_dict['states']

        if params_init == 'posterior_south_tees':
            
            with open('NHS_data/new_data/formatted_data/configs/good_params_minus150_south_tees.json', 'r') as f:
                params = json.load(f)

            health = [np.array([1 - params['proba_Recovering_given_%s' % state], params['proba_Recovering_given_%s' % state]]) for state in states]
            die_after_declining = [np.array([1 - params['proba_Die_after_Declining_%s' % state], params['proba_Die_after_Declining_%s' % state]]) for state in states[:-1]]

            durations = {'Declining': [], 'Recovering': []}
            for s, state in enumerate(states):
                for health_state in ['Declining', 'Recovering']:
                    durations[health_state].append({'lam': params['pmf_duration_%s_%s' % (health_state, state)]['lam'], 'tau': params['pmf_duration_%s_%s' % (health_state, state)]['tau'], 'lower': prior['durations'][health_state][s]['lower'], 'upper': prior['durations'][health_state][s]['upper']})

        theta_init = {'health': health, 'die_after_declining': die_after_declining, 'durations': durations}

        return theta_init

    def update_config_dict_given_theta(self, theta):
        '''
        self.config_dict is a dictionary containing the information necessary for our hospital model to run a simulation.
        This function modifies the model parameters specified in self.config_dict with the parameters specified by theta.
        '''

        states = self.config_dict['states']

        for s in range(len(states)):
            # updating health state probabilities
            self.config_dict['proba_Recovering_given_%s' % states[s]] = theta['health'][s][1]

            if states[s] != "OnVentInICU":
                self.config_dict['proba_Die_after_Declining_%s' % states[s]] = theta['die_after_declining'][s][1]

            # updating durations probabilities
            for health_state in ["Declining", "Recovering"]:
                choices = list(self.config_dict['pmf_duration_%s_%s' % (health_state, states[s])].keys())

                lam = theta['durations'][health_state][s]['lam']
                tau = theta['durations'][health_state][s]['tau']

                probas = scipy.special.softmax(scipy.stats.poisson.logpmf(np.arange(len(choices)), lam) / np.power(10, tau))

                for c, choice in enumerate(choices):
                    # update each individual choice with the value in theta
                    self.config_dict['pmf_duration_%s_%s' % (health_state, states[s])][choice] = probas[c]

    def simulate_dataset(self, theta):
        '''
        Simulate the dataset given the parameters in theta
        '''

        # first update config dict given the parameters in theta
        self.update_config_dict_given_theta(theta)

        # print(self.config_dict['proba_Die_after_Declining_InGeneralWard']) # have to be 0.0 when not using DieAfterDeclining
        # print(self.config_dict['proba_Die_after_Declining_OffVentInICU']) # have to be 0.0 when not using DieAfterDeclining

        states = self.config_dict['states']

        T = None
        for i in range(self.num_simulations):
            # running the simulation. notice that the random seed is not used. if used, every simulation with the same
            # parameters would be the same
            
            results_df = run_simulation(None, self.config_dict, states, self.func_name, approximate=self.approximate)

            train_df = results_df[results_df['timestep'] >= 0][results_df['timestep'] <= self.train_test_split]

            # condensing the results in a summary vector
            T_x = []
            weights = []
            for col_name in self.config_dict['summary_statistics_names']:
                if col_name == "n_InICU":
                    T_x.append(train_df["n_OffVentInICU"] + train_df["n_OnVentInICU"])
                elif col_name == "n_occupied_beds":
                    T_x.append(train_df["n_InGeneralWard"] + train_df["n_OffVentInICU"] + train_df["n_OnVentInICU"])
                elif col_name == "n_discharges":
                    T_x.append(train_df["n_discharged_InGeneralWard"] + train_df["n_discharged_OffVentInICU"] + train_df["n_discharged_OnVentInICU"])
                elif col_name == "n_TERMINAL_5daysSmoothed":
                    T_x.append(train_df["n_TERMINAL"])
                else:
                    T_x.append(train_df[col_name])

                weights.append(np.linspace(0.5, 1.5, train_df["n_InGeneralWard"].shape[0])*float(self.config_dict['summary_statistics_weights'][col_name]))

            T_x = np.asarray(T_x).flatten()
            weights = np.asarray(weights).flatten()

            # accumulating summary statistics from multiple runs
            if T is None:
                T = T_x
            else:
                T += T_x

        # averaging summary statistics from multiple runs
        T_x = T / float(self.num_simulations)

        return T_x, weights, results_df # return just the last simulation

    # @profile
    def abc_mcmc(self, theta_init, num_iterations, dir_scale_tuple, lam_stddev_tuple, tau_stddev_tuple):
        accepted_thetas = []
        accepted_distances = []
        all_distances = []
        accepted_alphas = []
        all_alphas = []
        accepted_test_forecasts = []
        num_accepted = 0
        theta = theta_init
        self.best_distance = self.epsilon
        self.epsilon_trace = []
        self.num_iterations = num_iterations

        original_num_timesteps = self.config_dict['num_timesteps']
        self.config_dict['num_timesteps'] = self.train_test_split

        log_prior_prev = self.calc_log_proba_prior(theta)

        # create range of proposal parameters (dirichlet, lambda, tau)
        min_dir_scale, max_dir_scale = dir_scale_tuple
        min_lam_stddev, max_lam_stddev = lam_stddev_tuple
        min_tau_stddev, max_tau_stddev = tau_stddev_tuple
        dir_scale_list = np.linspace(max_dir_scale, min_dir_scale, num_iterations)
        lam_stddev_list = np.linspace(max_lam_stddev, min_lam_stddev, num_iterations)
        tau_stddev_list = np.linspace(max_tau_stddev, min_tau_stddev, num_iterations)

        for n in range(num_iterations):
            print("Iteration #%d" % (n + 1))
            # import time
            # start = time.time()

            if n == int(num_iterations * (3/4) * (1/4)) or n == int(num_iterations * (3/4) * (2/4)) or n == int(num_iterations * (3/4) * (3/4)): # distance-threshold decay
                self.epsilon += 0.05 # distance-threshold decay

            # if n == int(num_iterations * (3/4)):
            #     self.config_dict['num_timesteps'] = original_num_timesteps

            if n == int(num_iterations * (3/4)): # distance-threshold decay
                self.epsilon += 0.015 # distance-threshold decay

            dir_scale = dir_scale_list[n]
            lam_stddev = lam_stddev_list[n]
            tau_stddev = tau_stddev_list[n]

            for state_index in range(len(self.config_dict['states'])):
                # draw from proposal distribution
                theta_health_prime = deepcopy(theta['health'])
                p_prime_D = self.draw_proposal_distribution_categorical(theta_health_prime[state_index], dir_scale)
                theta_health_prime[state_index] = p_prime_D

                # create a theta_prime and simulate dataset
                theta_prime = deepcopy(theta)
                theta_prime['health'] = theta_health_prime
                T_x, weights, test_df = self.simulate_dataset(theta_prime)
                distance = self.calc_distance(T_x, weights)
                all_distances.append(distance)

                if self.accept(distance):
                    accepted_distances.append(distance)

                    # calculate alpha(theta, theta_prime)
                    log_prior_prime = self.calc_log_proba_prior(theta_prime)
                    log_prop_prime = 0.0
                    log_prop_prev = 0.0
                    for s, state in enumerate(self.config_dict['states']):
                        log_prop_prime += self.calc_log_proba_proposal_distribution_categorical(theta['health'][s], theta_prime['health'][s], dir_scale)
                        log_prop_prev  += self.calc_log_proba_proposal_distribution_categorical(theta_prime['health'][s], theta['health'][s], dir_scale)

                        if state != 'OnVentInICU':
                            log_prop_prime += self.calc_log_proba_proposal_distribution_categorical(theta['die_after_declining'][s], theta_prime['die_after_declining'][s], dir_scale)
                            log_prop_prev  += self.calc_log_proba_proposal_distribution_categorical(theta_prime['die_after_declining'][s], theta['die_after_declining'][s], dir_scale)
                        
                        log_prop_prime += self.calc_log_proba_proposal_distribution_truncatednormal(theta['durations']['Declining'][s]['lam'], theta_prime['durations']['Declining'][s]['lam'], lam_stddev, theta['durations']['Declining'][s]['lower'], theta['durations']['Declining'][s]['upper'])
                        log_prop_prev  += self.calc_log_proba_proposal_distribution_truncatednormal(theta_prime['durations']['Declining'][s]['lam'], theta['durations']['Declining'][s]['lam'], lam_stddev, theta['durations']['Declining'][s]['lower'], theta['durations']['Declining'][s]['upper'])
                        log_prop_prime += self.calc_log_proba_proposal_distribution_truncatednormal(theta['durations']['Recovering'][s]['lam'], theta_prime['durations']['Recovering'][s]['lam'], lam_stddev, theta['durations']['Declining'][s]['lower'], theta['durations']['Declining'][s]['upper'])
                        log_prop_prev  += self.calc_log_proba_proposal_distribution_truncatednormal(theta_prime['durations']['Recovering'][s]['lam'], theta['durations']['Recovering'][s]['lam'], lam_stddev, theta['durations']['Declining'][s]['lower'], theta['durations']['Declining'][s]['upper'])

                    # eq. 1.3.2
                    alpha = (log_prior_prime + log_prop_prime) - (log_prior_prev + log_prop_prev)
                    # print("Alpha: %.3f" % np.exp(alpha))
                    all_alphas.append(np.exp(alpha))

                    if np.random.random() < np.exp(alpha):
                        self.best_distance = distance
                        theta = deepcopy(theta_prime)
                        accepted_thetas.append(theta)
                        accepted_alphas.append(np.exp(alpha))
                        accepted_test_forecasts.append(test_df)
                        num_accepted += 1
                        log_prior_prev = log_prior_prime
                    else:
                        accepted_alphas.append(np.NaN)

                else:
                    accepted_distances.append(np.NaN)
                    accepted_alphas.append(np.NaN)
                    all_alphas.append(np.NaN)

                self.update_epsilon(n)

                if self.state_name_id_map[state_index] != "OnVentInICU": # ProbaDieAfterDecliningOnVentInICU is always 1.0
                    # draw from proposal distribution
                    theta_health_prime = deepcopy(theta['die_after_declining'])
                    p_prime_D = self.draw_proposal_distribution_categorical(theta_health_prime[state_index], dir_scale*2) # make this proposal a bit narrower
                    theta_health_prime[state_index] = p_prime_D

                    # create a theta_prime and simulate dataset
                    theta_prime = deepcopy(theta)
                    theta_prime['die_after_declining'] = theta_health_prime
                    T_x, weights, test_df = self.simulate_dataset(theta_prime)
                    distance = self.calc_distance(T_x, weights)
                    all_distances.append(distance)

                    if self.accept(distance):
                        accepted_distances.append(distance)

                        # calculate alpha(theta, theta_prime)
                        log_prior_prime = self.calc_log_proba_prior(theta_prime)
                        log_prop_prime = 0.0
                        log_prop_prev = 0.0
                        for s, state in enumerate(self.config_dict['states']):
                            log_prop_prime += self.calc_log_proba_proposal_distribution_categorical(theta['health'][s], theta_prime['health'][s], dir_scale)
                            log_prop_prev  += self.calc_log_proba_proposal_distribution_categorical(theta_prime['health'][s], theta['health'][s], dir_scale)

                            if state != 'OnVentInICU':
                                log_prop_prime += self.calc_log_proba_proposal_distribution_categorical(theta['die_after_declining'][s], theta_prime['die_after_declining'][s], dir_scale)
                                log_prop_prev  += self.calc_log_proba_proposal_distribution_categorical(theta_prime['die_after_declining'][s], theta['die_after_declining'][s], dir_scale)
                            
                            log_prop_prime += self.calc_log_proba_proposal_distribution_truncatednormal(theta['durations']['Declining'][s]['lam'], theta_prime['durations']['Declining'][s]['lam'], lam_stddev, theta['durations']['Declining'][s]['lower'], theta['durations']['Declining'][s]['upper'])
                            log_prop_prev  += self.calc_log_proba_proposal_distribution_truncatednormal(theta_prime['durations']['Declining'][s]['lam'], theta['durations']['Declining'][s]['lam'], lam_stddev, theta['durations']['Declining'][s]['lower'], theta['durations']['Declining'][s]['upper'])
                            log_prop_prime += self.calc_log_proba_proposal_distribution_truncatednormal(theta['durations']['Recovering'][s]['lam'], theta_prime['durations']['Recovering'][s]['lam'], lam_stddev, theta['durations']['Declining'][s]['lower'], theta['durations']['Declining'][s]['upper'])
                            log_prop_prev  += self.calc_log_proba_proposal_distribution_truncatednormal(theta_prime['durations']['Recovering'][s]['lam'], theta['durations']['Recovering'][s]['lam'], lam_stddev, theta['durations']['Declining'][s]['lower'], theta['durations']['Declining'][s]['upper'])

                        # eq. 1.3.2
                        alpha = (log_prior_prime + log_prop_prime) - (log_prior_prev + log_prop_prev)
                        # print("Alpha: %.3f" % np.exp(alpha))
                        all_alphas.append(np.exp(alpha))

                        if np.random.random() < np.exp(alpha):
                            self.best_distance = distance
                            theta = deepcopy(theta_prime)
                            accepted_thetas.append(theta)
                            accepted_alphas.append(np.exp(alpha))
                            accepted_test_forecasts.append(test_df)
                            num_accepted += 1
                            log_prior_prev = log_prior_prime
                        else:
                            accepted_alphas.append(np.NaN)

                    else:
                        accepted_distances.append(np.NaN)
                        accepted_alphas.append(np.NaN)
                        all_alphas.append(np.NaN)

                    self.update_epsilon(n)

                for health_state in ["Declining", "Recovering"]:

                    ######################### lam ###############################

                    # draw from proposal distribution
                    theta_durations_prime = deepcopy(theta['durations'][health_state])
                    lam_prime = self.draw_proposal_distribution_truncatednormal(theta_durations_prime[state_index]['lam'], lam_stddev, theta_durations_prime[state_index]['lower'], theta_durations_prime[state_index]['upper'])
                    theta_durations_prime[state_index]['lam'] = lam_prime

                    # create a theta_prime and simulate dataset
                    theta_prime = deepcopy(theta)
                    theta_prime['durations'][health_state] = theta_durations_prime
                    T_x, weights, test_df = self.simulate_dataset(theta_prime)
                    distance = self.calc_distance(T_x, weights)
                    all_distances.append(distance)

                    if self.accept(distance):
                        accepted_distances.append(distance)

                        # calculate alpha(theta, theta_prime)
                        log_prior_prime = self.calc_log_proba_prior(theta_prime)
                        log_prop_prime = 0.0
                        log_prop_prev = 0.0
                        for s, state in enumerate(self.config_dict['states']):
                            log_prop_prime += self.calc_log_proba_proposal_distribution_categorical(theta['health'][s], theta_prime['health'][s], dir_scale)
                            log_prop_prev  += self.calc_log_proba_proposal_distribution_categorical(theta_prime['health'][s], theta['health'][s], dir_scale)

                            if state != 'OnVentInICU':
                                log_prop_prime += self.calc_log_proba_proposal_distribution_categorical(theta['die_after_declining'][s], theta_prime['die_after_declining'][s], dir_scale)
                                log_prop_prev  += self.calc_log_proba_proposal_distribution_categorical(theta_prime['die_after_declining'][s], theta['die_after_declining'][s], dir_scale)
                            
                            log_prop_prime += self.calc_log_proba_proposal_distribution_truncatednormal(theta['durations']['Declining'][s]['lam'], theta_prime['durations']['Declining'][s]['lam'], lam_stddev, theta['durations']['Declining'][s]['lower'], theta['durations']['Declining'][s]['upper'])
                            log_prop_prev  += self.calc_log_proba_proposal_distribution_truncatednormal(theta_prime['durations']['Declining'][s]['lam'], theta['durations']['Declining'][s]['lam'], lam_stddev, theta['durations']['Declining'][s]['lower'], theta['durations']['Declining'][s]['upper'])
                            log_prop_prime += self.calc_log_proba_proposal_distribution_truncatednormal(theta['durations']['Recovering'][s]['lam'], theta_prime['durations']['Recovering'][s]['lam'], lam_stddev, theta['durations']['Declining'][s]['lower'], theta['durations']['Declining'][s]['upper'])
                            log_prop_prev  += self.calc_log_proba_proposal_distribution_truncatednormal(theta_prime['durations']['Recovering'][s]['lam'], theta['durations']['Recovering'][s]['lam'], lam_stddev, theta['durations']['Declining'][s]['lower'], theta['durations']['Declining'][s]['upper'])

                        # eq. 1.3.2
                        alpha = (log_prior_prime + log_prop_prime) - (log_prior_prev + log_prop_prev)
                        # print("Alpha: %.3f" % np.exp(alpha))
                        all_alphas.append(np.exp(alpha))

                        if np.random.random() < np.exp(alpha):
                            self.best_distance = distance
                            theta = deepcopy(theta_prime)
                            accepted_thetas.append(theta)
                            accepted_alphas.append(np.exp(alpha))
                            accepted_test_forecasts.append(test_df)
                            num_accepted += 1
                            log_prior_prev = log_prior_prime
                        else:
                            accepted_alphas.append(np.NaN)

                    else:
                        accepted_distances.append(np.NaN)
                        accepted_alphas.append(np.NaN)
                        all_alphas.append(np.NaN)

                    self.update_epsilon(n)

                    ######################### tau ###############################

                    # draw from proposal distribution
                    theta_durations_prime = deepcopy(theta['durations'][health_state])
                    tau_prime = self.draw_proposal_distribution_normal(theta_durations_prime[state_index]['tau'], tau_stddev)
                    theta_durations_prime[state_index]['tau'] = tau_prime

                    # create a theta_prime and simulate dataset
                    theta_prime = deepcopy(theta)
                    theta_prime['durations'][health_state] = theta_durations_prime
                    T_x, weights, test_df = self.simulate_dataset(theta_prime)
                    distance = self.calc_distance(T_x, weights)
                    all_distances.append(distance)

                    if self.accept(distance):
                        accepted_distances.append(distance)

                        # calculate alpha(theta, theta_prime)
                        log_prior_prime = self.calc_log_proba_prior(theta_prime)
                        log_prop_prime = 0.0
                        log_prop_prev = 0.0
                        for s, state in enumerate(self.config_dict['states']):
                            log_prop_prime += self.calc_log_proba_proposal_distribution_categorical(theta['health'][s], theta_prime['health'][s], dir_scale)
                            log_prop_prev  += self.calc_log_proba_proposal_distribution_categorical(theta_prime['health'][s], theta['health'][s], dir_scale)

                            if state != 'OnVentInICU':
                                log_prop_prime += self.calc_log_proba_proposal_distribution_categorical(theta['die_after_declining'][s], theta_prime['die_after_declining'][s], dir_scale)
                                log_prop_prev  += self.calc_log_proba_proposal_distribution_categorical(theta_prime['die_after_declining'][s], theta['die_after_declining'][s], dir_scale)
                            
                            log_prop_prime += self.calc_log_proba_proposal_distribution_truncatednormal(theta['durations']['Declining'][s]['lam'], theta_prime['durations']['Declining'][s]['lam'], lam_stddev, theta['durations']['Declining'][s]['lower'], theta['durations']['Declining'][s]['upper'])
                            log_prop_prev  += self.calc_log_proba_proposal_distribution_truncatednormal(theta_prime['durations']['Declining'][s]['lam'], theta['durations']['Declining'][s]['lam'], lam_stddev, theta['durations']['Declining'][s]['lower'], theta['durations']['Declining'][s]['upper'])
                            log_prop_prime += self.calc_log_proba_proposal_distribution_truncatednormal(theta['durations']['Recovering'][s]['lam'], theta_prime['durations']['Recovering'][s]['lam'], lam_stddev, theta['durations']['Declining'][s]['lower'], theta['durations']['Declining'][s]['upper'])
                            log_prop_prev  += self.calc_log_proba_proposal_distribution_truncatednormal(theta_prime['durations']['Recovering'][s]['lam'], theta['durations']['Recovering'][s]['lam'], lam_stddev, theta['durations']['Declining'][s]['lower'], theta['durations']['Declining'][s]['upper'])

                        # eq. 1.3.2
                        alpha = (log_prior_prime + log_prop_prime) - (log_prior_prev + log_prop_prev)
                        # print("Alpha: %.3f" % np.exp(alpha))
                        all_alphas.append(np.exp(alpha))

                        if np.random.random() < np.exp(alpha):
                            self.best_distance = distance
                            theta = deepcopy(theta_prime)
                            accepted_thetas.append(theta)
                            accepted_alphas.append(np.exp(alpha))
                            accepted_test_forecasts.append(test_df)
                            num_accepted += 1
                            log_prior_prev = log_prior_prime
                        else:
                            accepted_alphas.append(np.NaN)

                    else:
                        accepted_distances.append(np.NaN)
                        accepted_alphas.append(np.NaN)
                        all_alphas.append(np.NaN)

                    self.update_epsilon(n)
            # end = time.time()
            # print('Elapsed time: %.2f' % (end - start))

        return accepted_thetas, accepted_distances, num_accepted, all_distances, accepted_alphas, all_alphas, accepted_test_forecasts

    def draw_samples(self, algorithm='abc', num_iterations=100, dir_scale=100, lam_stddev=0.5, tau_stddev=0.1):

        if algorithm == 'abc':
            return self.abc_mcmc(self.theta_init, num_iterations, dir_scale, lam_stddev, tau_stddev)
        else:
            print("Invalid algorithm specification '%s'" % algorithm)
            print("Valid options are:")
            print("\t- 'abc' to run ABC MCMC sampling")
            exit(1)

    def update_epsilon(self, n):

        if n < int(self.num_iterations * (3/4)): # distance-threshold decay
            self.epsilon = max(self.epsilon * self.annealing_constant, self.best_distance) # distance-threshold decay
        
        # self.epsilon = self.epsilon * self.annealing_constant # simulated annealing
        
        self.epsilon_trace.append(self.epsilon)

    def calc_distance(self, T_x, weights=None):
        if weights is None:
            weights = np.ones(T_x.shape)

        # mean absolute error normalized by the maximum individual statistic value at each timestep and for each summary statistic
        a = np.array([T_x, self.T_y])
        denom = np.max(a, axis=0)
        distance = np.nansum((weights * np.abs(np.diff(a, axis=0))) / denom) / T_x.shape[0]

        # print("Distance: %.3f" % distance)
        return distance

    def accept(self, distance):
        return distance < self.epsilon # distance-threshold decay
        
        # return (distance < self.best_distance) or (np.random.random() < np.exp(- distance / self.epsilon)) # simulated annealing

    def calc_log_proba_prior(self, theta):
        log_proba = 0.0
        log_proba += np.sum([scipy.stats.dirichlet.logpdf(theta['health'][s] + 1e-12, self.abc_prior['health'][s]) for s in range(len(theta['health']))])
        try:
            log_proba += np.sum([scipy.stats.dirichlet.logpdf(theta['die_after_declining'][s], self.abc_prior['die_after_declining'][s]) for s in range(len(theta['die_after_declining']))])
        except:
            log_proba += np.sum([scipy.stats.dirichlet.logpdf(theta['die_after_declining'][s] + 1e-12, self.abc_prior['die_after_declining'][s]) for s in range(len(theta['die_after_declining']))])
        
        for health_state in ['Declining', 'Recovering']:
            for sp, state_params in enumerate(theta['durations'][health_state]):
                lam = state_params['lam']
                tau = state_params['tau']
                log_proba += self.abc_prior['durations'][health_state][sp]['lam'].logpdf(lam)
                log_proba += self.abc_prior['durations'][health_state][sp]['tau'].logpdf(tau)

        # TODO
        # if some entry of p_prime_D is zero AND some other entry is one, there will be an error
        return log_proba

    def calc_log_proba_proposal_distribution_categorical(self, p_D, p_prime_D, scale):
        '''
        Computes p(p_prime_D | p_D, scale), where both p_prime_D and p_D are categorical random variables.
        '''
        try:
            return scipy.stats.dirichlet.logpdf(p_prime_D, (p_D * scale) + 1)
        except:
            return scipy.stats.dirichlet.logpdf(p_prime_D + 1e-12, (p_D * scale) + 1) # in case some entry of p_prime_D is zero
        # TODO
        # if some entry of p_prime_D is zero AND some other entry is one, there will be an error

    def calc_log_proba_proposal_distribution_truncatednormal(self, mean, mean_prime, stddev, lower, upper):
        '''
        Computes p(mean_prime | mean, stddev, lower, upper)
        '''
        alpha = (lower - mean) / stddev
        beta = (upper - mean) / stddev
        return stats.truncnorm.logpdf(mean_prime, alpha, beta, loc=mean, scale=stddev)

    def draw_proposal_distribution_categorical(self, p_D, scale):
        return np.random.dirichlet((p_D * scale) + 1)

    def draw_proposal_distribution_normal(self, mean, stddev):
        return np.random.normal(mean, stddev, 1)[0]

    def draw_proposal_distribution_truncatednormal(self, mean, stddev, lower, upper):
        alpha = (lower - mean) / stddev
        beta = (upper - mean) / stddev
        return stats.truncnorm.rvs(alpha, beta, loc=mean, scale=stddev, size=1)[0]

    def save_thetas_to_json(self, thetas, filename):
        states = self.config_dict['states']

        json_to_save = {'last_thetas': []}
        for theta in thetas:
            params_dict = {}
            for s in range(len(states)):
                # updating health state probabilities
                params_dict['proba_Recovering_given_%s' % states[s]] = theta['health'][s][1]
                if states[s] != "OnVentInICU":
                    params_dict['proba_Die_after_Declining_%s' % states[s]] = theta['die_after_declining'][s][1]
                else:
                    params_dict['proba_Die_after_Declining_%s' % states[s]] = 1.0

                # updating durations probabilities
                for health_state in ["Declining", "Recovering"]:
                    choices = list(self.config_dict['pmf_duration_%s_%s' % (health_state, states[s])].keys())

                    lam = theta['durations'][health_state][s]['lam']
                    tau = theta['durations'][health_state][s]['tau']

                    params_dict['pmf_duration_%s_%s' % (health_state, states[s])] = {'lam': lam, 'tau': tau}

                    probas = scipy.special.softmax(scipy.stats.poisson.logpmf(np.arange(len(choices)), lam) / np.power(10, tau))

                    for c, choice in enumerate(choices):
                        # update each individual choice with the value in theta
                        params_dict['pmf_duration_%s_%s' % (health_state, states[s])][choice] = probas[c]
                            
            json_to_save['last_thetas'].append(deepcopy(params_dict))

        with open(filename, 'w+') as f:
            json.dump(json_to_save, f, indent=1)


def save_stats_to_csv(stats, filename):
    df = pd.DataFrame(stats)
    df.to_csv(filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hospital', default='MA_NovToFeb_61days')
    parser.add_argument('--config_template', default='USA/MA_data/config') # template for config_file
    parser.add_argument('--input_template', default='USA/MA_data/')
    parser.add_argument('--output_template', default='USA/MA_data/')
    parser.add_argument('--random_seed', default=101, type=int) # currently not using it 
    parser.add_argument('--algorithm', default='abc')
    parser.add_argument('--func_name', default='python')
    parser.add_argument('--num_iterations', default=10, type=int) # number of sampling iterations 
                                                                   # each iteration has an inner loop through each probabilistic parameter vector
    parser.add_argument('--num_simulations', default=1, type=int)
    parser.add_argument('--start_epsilon', default=0.7, type=float)
    parser.add_argument('--annealing_constant', default=0.999992)
    parser.add_argument('--train_test_split', default=61) # currently an integer timestep, ultimately will be a string date
    parser.add_argument('--dir_scale', default='100-100') # scale parameter for the dirichlet proposal distribution (min-max, linearly interpolated)
    parser.add_argument('--lam_stddev', default='0.5-0.5') # stddev parameter for lambda (truncated normal) (min-max, linearly interpolated)
    parser.add_argument('--tau_stddev', default='0.1-0.1') # stddev parameter for tau (normal) (min-max, linearly interpolated)

    parser.add_argument('--params_init', default='None')
    parser.add_argument('--abc_prior_type', default='OnCDCTableReasonable')
    parser.add_argument('--approximate', default='5')
    parser.add_argument('--abc_prior_config_template', default='priors/abc_prior_config')

    args, unknown_args = parser.parse_known_args()

    unk_keys = map(lambda s: s[2:].strip(), unknown_args[::2])
    unk_vals = unknown_args[1::2]
    unk_dict = dict(zip(unk_keys, unk_vals))

    seed = int(args.random_seed)
    num_iterations = int(args.num_iterations)
    num_simulations = int(args.num_simulations)
    start_epsilon = float(args.start_epsilon)
    annealing_constant = float(args.annealing_constant)
    train_test_split = int(args.train_test_split)
    dir_scale = list(map(int, args.dir_scale.split('-')))
    lam_stddev = list(map(float, args.lam_stddev.split('-')))
    tau_stddev = list(map(float, args.tau_stddev.split('-')))
    algorithm = args.algorithm
    func_name = args.func_name

    if args.approximate == 'None':
        approximate = None
    else:
        approximate = int(args.approximate)

    if args.params_init == 'None':
        params_init = None
    else:
        params_init = args.params_init
    if algorithm == 'abc' or algorithm == 'abc_exp':
        with open(args.abc_prior_config_template + '_%s.json' % args.abc_prior_type, 'r') as f:
            abc_prior = json.load(f)
        thetas_output = args.output_template + '_%s_last_thetas_%s.json' % (args.hospital, args.abc_prior_type)
        stats_output = args.output_template + '_%s_stats_%s.csv' % (args.hospital, args.abc_prior_type)
        test_forecasts_output = args.output_template + '_%s_last_test_forecasts_%s.csv' % (args.hospital, args.abc_prior_type)
    else:
        abc_prior = None
        thetas_output = args.output_template + '_last_thetas_%s.json' % (params_init)
        stats_output = args.output_template + '_stats_%s.csv' % (params_init)

    config_file = args.config_template + '_%s.json' % args.hospital
    inputfile = args.input_template + '%s.csv' % args.hospital

    ## Load JSON
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    num_timesteps = config_dict['num_timesteps']

    true_df = pd.read_csv(inputfile)
    true_df = true_df[true_df['timestep'] <= train_test_split] # only retain training data
    print(true_df)

    # condensing the true results in a summary vector
    T_y = []
    for col_name in config_dict['summary_statistics_names']:
        T_y.append(true_df[col_name])
    T_y = np.asarray(T_y).flatten()

    # initalize sampler
    sampler = ABCSampler(seed, start_epsilon, annealing_constant, T_y, train_test_split, config_dict, func_name, num_timesteps, num_simulations, approximate=approximate)

    sampler.initialize_theta(algorithm, abc_prior, params_init)
    print(sampler.theta_init)

    import time
    start = time.time()
    accepted_thetas, accepted_distances, num_accepted, all_distances, accepted_alphas, all_alphas, accepted_test_forecasts = sampler.draw_samples(algorithm, num_iterations, dir_scale, lam_stddev, tau_stddev)
    end = time.time()
    print('Elapsed time with %s on %d iterations: %9.3f sec' % (func_name, num_iterations, end - start))

    num_to_save = 2000
    last_thetas = [deepcopy(accepted_thetas[0])] + accepted_thetas[-num_to_save:] # also save the first theta
    sampler.save_thetas_to_json(last_thetas, thetas_output)


    stats = {'all_distances': all_distances, 'accepted_distances': accepted_distances, 'all_alphas': all_alphas, 'accepted_alphas': accepted_alphas, 'epsilon_trace': sampler.epsilon_trace}
    save_stats_to_csv(stats, stats_output)

    # save forecasts on test set for last 1000 samples
    test_forecasts_to_save = accepted_test_forecasts[-num_to_save:]
    for i, df in enumerate(test_forecasts_to_save):
        df['index'] = i

    df = pd.concat(test_forecasts_to_save)
    df.to_csv(test_forecasts_output)