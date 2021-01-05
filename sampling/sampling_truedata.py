import os
import json
import argparse
import numpy as np
import scipy.stats
import pandas as pd
# import tqdm
import json
import warnings
from copy import deepcopy
from semimarkov_forecaster import PatientTrajectory

# This is copy-pasted from 'run_forecast.py', with the only difference that, instead of saving
# results to a csv file, the function returns the pandas DataFrame
def run_simulation(random_seed, config_dict, states, state_name_to_id, next_state_map):
    print("random_seed=%s <<<" % random_seed)
    prng = np.random.RandomState(random_seed)
    
    T = config_dict['num_timesteps']
    K = len(states) # Num states (not including terminal)

    ## Number of *previous* timesteps to simulate
    # Defaults to 0 if not provided
    Tpast = config_dict.get('num_past_timesteps', 0)

    ## Preallocate admit, discharge, and occupancy
    Tmax = 10 * T + Tpast
    occupancy_count_TK = np.zeros((Tmax, K), dtype=np.float64)
    admit_count_TK = np.zeros((Tmax, K), dtype=np.float64)
    discharge_count_TK = np.zeros((Tmax, K), dtype=np.float64)
    terminal_count_T1 = np.zeros((Tmax, 1), dtype=np.float64)

    ## Read functions to sample the incoming admissions to each state
    sample_func_per_state = dict()
    for state in states:
        try:
            pmfstr_or_csvfile = config_dict['pmf_num_per_timestep_%s' % state]
        except KeyError:
            try:
                pmfstr_or_csvfile = config_dict['pmf_num_per_timestep_{state}']
            except KeyError:
                continue
        
        # First, try to replace wildcards
        if isinstance(pmfstr_or_csvfile, str):
            for key, val in list(args.__dict__.items()) + list(unk_dict.items()):
                wildcard_key = "{%s}" % key
                if pmfstr_or_csvfile.count(wildcard_key):
                    pmfstr_or_csvfile = pmfstr_or_csvfile.replace(wildcard_key, str(val))
                    print("WILDCARD: %s" % pmfstr_or_csvfile)
        
        if isinstance(pmfstr_or_csvfile, dict):
            pmfdict = pmfstr_or_csvfile
            choices = np.fromiter(pmfdict.keys(), dtype=np.int32)
            probas = np.fromiter(pmfdict.values(), dtype=np.float64)           
            def sample_incoming_count(t, prng):
                return prng.choice(choices, p=probas)
            sample_func_per_state[state] = sample_incoming_count
        elif pmfstr_or_csvfile.startswith('scipy.stats'):
            # Avoid evals on too long of strings for safety reasons
            assert len(pmfstr_or_csvfile) < 40
            pmf = eval(pmfstr_or_csvfile)
            def sample_incoming_count(t, prng):
                return pmf.rvs(random_state=prng)
            sample_func_per_state[state] = sample_incoming_count
            # TODO other parsing for other ways of specifying a pmf over pos ints?

        elif os.path.exists(pmfstr_or_csvfile):
            # Read incoming counts from provided file
            csv_df = pd.read_csv(pmfstr_or_csvfile)
            state_key = 'num_%s' % state
            if state_key not in csv_df.columns:
                continue
            # TODO Verify here that all necessary rows are accounted for
            def sample_incoming_count(t, prng):
                row_ids = np.flatnonzero(csv_df['timestep'] == t)
                if len(row_ids) == 0:
                    if t < 10:
                        raise ValueError("Error in file %s: No matching timestep for t=%d" % (
                            pmfstr_or_csvfile, t))
                    else:
                        warnings.warn("No matching timesteps t>%d found in file %s. Assuming 0" % (
                            csv_df['timestep'].max(), pmfstr_or_csvfile)) 
                        return 0
                if len(row_ids) > 1:
                    raise ValueError("Error in file %s: Must have exactly one matching timestep for t=%d" % (
                        pmfstr_or_csvfile, t))
                return np.array(csv_df['num_%s' % state].values[row_ids[0]], dtype=int) # guard against float input. 
            sample_func_per_state[state] = sample_incoming_count
        # Parsing failed!
        else:
            raise ValueError("Bad PMF specification: %s" % pmfstr_or_csvfile)

    ## Simulation what happens to initial population
    for t in range(-Tpast, 1, 1):
        for state in states:
            N_new_dict = config_dict['init_num_%s' % state]
            if isinstance(N_new_dict, dict):
                N_new = N_new_dict["%s" % t]
            else:
                N_new = int(N_new_dict)
            for n in range(N_new):
                p = PatientTrajectory(state, config_dict, prng, next_state_map, state_name_to_id, t)
                occupancy_count_TK = p.update_count_matrix(occupancy_count_TK, Tpast + t)
                admit_count_TK = p.update_admit_count_matrix(admit_count_TK, Tpast + t)
                discharge_count_TK = p.update_discharge_count_matrix(discharge_count_TK, Tpast + t)
                terminal_count_T1 = p.update_terminal_count_matrix(terminal_count_T1, Tpast + t)

    ## Simulation what happens as new patients added at each step
    #for t in tqdm.tqdm(range(1, T+1)):
    for t in range(1, T+1):
        for state in states:
            if state not in sample_func_per_state:
                continue
            sample_incoming_count = sample_func_per_state[state]
            N_t = sample_incoming_count(t, prng)
            for n in range(N_t):
                p = PatientTrajectory(state, config_dict, prng, next_state_map, state_name_to_id, t)
                occupancy_count_TK = p.update_count_matrix(occupancy_count_TK, Tpast + t)        
                admit_count_TK = p.update_admit_count_matrix(admit_count_TK, Tpast + t)
                discharge_count_TK = p.update_discharge_count_matrix(discharge_count_TK, Tpast + t)
                terminal_count_T1 = p.update_terminal_count_matrix(terminal_count_T1, Tpast + t)


    # Save only the first T + 1 tsteps (with index 0, 1, 2, ... T)
    t0 = 0
    tf = T + Tpast + 1
    occupancy_count_TK = occupancy_count_TK[t0:tf]
    admit_count_TK = admit_count_TK[t0:tf]
    discharge_count_TK = discharge_count_TK[t0:tf]
    terminal_count_T1 = terminal_count_T1[t0:tf]

    # Only considering census counts
    col_names = ['n_%s' % s for s in states]
    results_df = pd.DataFrame(occupancy_count_TK, columns=col_names)

    results_df["timestep"] = np.arange(-Tpast, -Tpast + tf)

    results_df["n_TERMINAL"] = terminal_count_T1[:,0]

    admit_col_names = ['n_admitted_%s' % s for s in states]
    for k, col_name in enumerate(admit_col_names):
        results_df[col_name] = admit_count_TK[:, k]

    discharge_col_names = ['n_discharged_%s' % s for s in states]
    for k, col_name in enumerate(discharge_col_names):
        results_df[col_name] = discharge_count_TK[:, k]

    # results_df.to_csv(output_file,
    #     columns=['timestep'] + col_names + ['n_TERMINAL'] + admit_col_names + discharge_col_names,
    #     index=False, float_format="%.0f")

    return results_df

'''
theta: 
    dict with keys: 'health', 'transitions', durations'
    theta['num timesteps'] = N
    theta['health'] = list of length S, each containing array of shape (2,) (treating health probas as 2-dim dirichlet)
    theta['durations'] = list of length S with arrays of shape (H, D) [(D,) in simple example]
'''
SUMMARY_STATISTICS_NAMES = ["n_discharges", "n_occupied_beds"] #, "n_admitted_InGeneralWard", "n_admitted_OffVentInICU", "n_admitted_OnVentInICU", "n_discharged_InGeneralWard", "n_discharged_OffVentInICU", "n_discharged_OnVentInICU"]

HEALTH_STATE_ID_TO_NAME = {0: 'Declining', 1: 'Recovering', 'Declining': 0, 'Recovering': 1}

class ABCSampler(object):

    def __init__(self, seed, start_epsilon, lowest_epsilon, annealing_constant, T_y, config_dict, num_timesteps, num_simulations):
        self.T_y = T_y
        self.epsilon = start_epsilon
        self.lowest_epsilon = lowest_epsilon
        self.annealing_constant = annealing_constant
        self.seed = seed
        self.config_dict = config_dict
        self.num_timesteps = num_timesteps
        self.num_simulations = num_simulations

        self.state_name_id_map = {}
        for s, state in enumerate(config_dict['states']):
            self.state_name_id_map[state] = s
            self.state_name_id_map[s] = state

    def initialize_theta(self, algorithm, abc_prior_dict, params_init=None):
        self.params_init = params_init

        if algorithm == 'abc' or algorithm == 'abc_exp':
            self.abc_prior = self.initialize_abc_prior(abc_prior_dict)
        else:
            self.abc_prior = None
        
        if params_init is None:
            self.theta_init = self.select_theta_init_from_prior(self.abc_prior)
        else:
            self.theta_init = self.select_theta_init_from_name(params_init)


    def initialize_abc_prior(self, abc_prior_dict):
        states = self.config_dict['states']
        
        health = [np.array(abc_prior_dict['prior_Health_given_%s' % (state)]) for state in states]
        declining = [np.array(list(abc_prior_dict['prior_duration_Declining_%s' % (state)].values())) for state in states]
        recovering = [np.array(list(abc_prior_dict['prior_duration_Recovering_%s' % (state)].values())) for state in states]

        abc_prior = {'health': health, 'transitions': None, 'durations': {'Declining': declining, 'Recovering': recovering}}
        
        return abc_prior

    def select_theta_init_from_prior(self, prior):
        health = [np.random.dirichlet(params) for params in prior['health']]
        declining = [np.random.dirichlet(params) for params in prior['durations']['Declining']]
        recovering = [np.random.dirichlet(params) for params in prior['durations']['Recovering']]

        theta_init = {'num timesteps': self.num_timesteps, 'health': health, 'transitions': None, 'durations': {'Declining': declining, 'Recovering': recovering}}

        return theta_init

    def update_config_dict_given_theta(self, theta):
        states = self.config_dict['states']

        for s in range(len(states)):
            # updating health state probabilities
            self.config_dict['proba_Recovering_given_%s' % states[s]] = theta['health'][s][1]

            if self.state_name_id_map[s] != 'Presenting': # skipping presenting state for durations
                # updating durations probabilities
                for health_state in ["Declining", "Recovering"]:
                    choices = list(self.config_dict['pmf_duration_%s_%s' % (health_state, states[s])].keys())
                    for d in range(theta['durations'][health_state][s].shape[0]):
                        # update each individual choice with the value in theta
                        self.config_dict['pmf_duration_%s_%s' % (health_state, states[s])][choices[d]] = theta['durations'][health_state][s][d]
            else:
                # setting presenting duration probability. This is just a placeholder that simplifies code
                for health_state in ["Declining", "Recovering"]:
                    self.config_dict['pmf_duration_%s_%s' % (health_state, states[s])]['1'] = 1.0

    # @profile
    def simulate_dataset(self, theta):
        # simulate the dataset given the parameters in theta

        # update config dict given the parameters in theta
        self.update_config_dict_given_theta(theta)

        # getting params ready for run_simulation (copied from run_forecast)
        states = config_dict['states']
        state_name_to_id = dict()
        next_state_map = dict()
        for ss, state in enumerate(states):
            state_name_to_id[state] = ss
            if ss < len(states) - 1:
                next_state_map[state] = states[ss+1]
            else:
                next_state_map[state] = 'TERMINAL'

        T = None
        for i in range(self.num_simulations):
            # running the simulation. notice that the random seed is not used. if used, every simulation with the same
            # parameters would be the same
            results_df = run_simulation(np.random.randint(100), self.config_dict, states, state_name_to_id, next_state_map)

            # condensing the results in a summary vector
            T_x = []
            for col_name in SUMMARY_STATISTICS_NAMES:
                if col_name == "n_InICU":
                    T_x.append(results_df["n_OffVentInICU"] + results_df["n_OnVentInICU"])
                elif col_name == "n_occupied_beds":
                    T_x.append(results_df["n_InGeneralWard"] + results_df["n_OffVentInICU"] + results_df["n_OnVentInICU"])
                elif col_name == "n_discharges":
                    T_x.append(results_df["n_discharged_InGeneralWard"] + results_df["n_discharged_OffVentInICU"] + results_df["n_discharged_OnVentInICU"])
                else:
                    T_x.append(results_df[col_name])
            T_x = np.asarray(T_x).flatten()

            # accumulating summary statistics from multiple runs
            if T is None:
                T = T_x
            else:
                T += T_x

        # averaging summary statistics from multiple runs
        T_x = T / float(num_simulations)

        return T_x

    def fixed_params(self, theta_init, num_iterations):
        accepted_thetas = []
        accepted_distances = []
        all_distances = []
        num_accepted = 0
        theta = theta_init
        best_distance = np.Inf

        for n in range(num_iterations):
            print("Iteration #%d" % (n + 1))

            T_x = self.simulate_dataset(theta)
            distance = self.calc_distance(T_x)
            all_distances.append(distance)

            if self.accept(distance):
                accepted_thetas.append(theta)
                num_accepted += 1

        return accepted_thetas, accepted_distances, num_accepted, all_distances, None, None

    def simulated_annealing(self, theta_init, num_iterations, scale):
        accepted_thetas = []
        accepted_distances = []
        all_distances = []
        num_accepted = 0
        theta = theta_init
        best_distance = np.Inf

        for n in range(num_iterations):
            print("Iteration #%d" % (n + 1))

            for s in range(len(self.config_dict['states']) - 1):
                # sample health states

                # draw from proposal distribution
                theta_health_prime = deepcopy(theta['health'])
                p_prime_D = self.draw_proposal_distribution_categorical(theta_health_prime[s], scale)
                theta_health_prime[s] = p_prime_D

                # create a theta_prime and simulate dataset
                theta_prime = deepcopy(theta)
                theta_prime['health'] = theta_health_prime
                T_x = self.simulate_dataset(theta_prime)

                distance = self.calc_distance(T_x)
                all_distances.append(distance)
                # uncomment section of line below ('or' statement) for Simulated Annealing (epsilon is used as the temperature)
                print(np.exp(- distance / self.epsilon))
                if distance < best_distance or np.random.random() < np.exp(- distance / self.epsilon):
                    theta = deepcopy(theta_prime)
                    accepted_thetas.append(theta)
                    accepted_distances.append(deepcopy(distance))
                    num_accepted += 1
                    best_distance = distance


                # sample durations
                for health_state in ["Declining", "Recovering"]:
                    # draw from proposal distribution
                    theta_durations_prime = deepcopy(theta['durations'][health_state])
                    p_prime_D = self.draw_proposal_distribution_categorical(theta_durations_prime[s], scale)
                    theta_durations_prime[s] = p_prime_D

                    # create a theta_prime and simulate dataset
                    theta_prime = deepcopy(theta)
                    theta_prime['durations'][health_state] = theta_durations_prime
                    T_x = self.simulate_dataset(theta_prime)

                    distance = self.calc_distance(T_x)
                    all_distances.append(distance)
                    # uncomment section of line below ('or' statement) for Simulated Annealing (epsilon is used as the temperature)
                    print(np.exp(- distance / self.epsilon))
                    if distance < best_distance or np.random.random() < np.exp(- distance / self.epsilon):
                        theta = deepcopy(theta_prime)
                        accepted_thetas.append(theta)
                        accepted_distances.append(deepcopy(distance))
                        num_accepted += 1
                        best_distance = distance

                    self.update_epsilon()
        
        return accepted_thetas, accepted_distances, num_accepted, all_distances, None, None

    def abc_mcmc_experiments(self, theta_init, num_iterations, scale):
        theta = theta_init
        T_x = self.simulate_dataset(theta)
        original_distance = self.calc_distance(T_x)
        best_distance = np.Inf
        log_prior_prev = self.calc_log_proba_dirichlet_prior(theta)

        distances = {'Recovering': [], 'Declining': []}
        diff_distances = {'Recovering': [], 'Declining': []}
        for n in range(num_iterations):
            print("Iteration #%d" % (n + 1))

            s = 0 # InGeneralWard
            theta_durations_prime = deepcopy(theta['durations']['Recovering'])
            print(theta_durations_prime[s])
            p_prime_D = self.draw_proposal_distribution_categorical(theta_durations_prime[s], scale)
            print(p_prime_D)

            for health_state in ['Recovering', 'Declining']:
                theta_durations_prime = deepcopy(theta['durations'][health_state])
                theta_durations_prime[s] = p_prime_D

                # create a theta_prime and simulate dataset
                theta_prime = deepcopy(theta)
                theta_prime['durations'][health_state] = theta_durations_prime
                T_x = self.simulate_dataset(theta_prime)
                distance = self.calc_distance(T_x)
                distances[health_state].append(distance)
                diff_distances[health_state].append(distance - original_distance)

                if self.accept(distance):
                    log_prior_prime = self.calc_log_proba_dirichlet_prior(theta_prime)
                    log_prop_prime = 0.0
                    log_prop_prev = 0.0
                    for s in range(len(self.config_dict['states']) - 1):
                        # health
                        log_prop_prime += self.calc_log_proba_proposal_distribution_categorical(theta['health'][s], theta_prime['health'][s], scale)
                        log_prop_prev  += self.calc_log_proba_proposal_distribution_categorical(theta_prime['health'][s], theta['health'][s], scale)
                        
                        # durations
                        log_prop_prime += self.calc_log_proba_proposal_distribution_categorical(theta['durations']['Declining'][s], theta_prime['durations']['Declining'][s], scale)
                        log_prop_prev  += self.calc_log_proba_proposal_distribution_categorical(theta_prime['durations']['Declining'][s], theta['durations']['Declining'][s], scale)
                        log_prop_prime += self.calc_log_proba_proposal_distribution_categorical(theta['durations']['Recovering'][s], theta_prime['durations']['Recovering'][s], scale)
                        log_prop_prev  += self.calc_log_proba_proposal_distribution_categorical(theta_prime['durations']['Recovering'][s], theta['durations']['Recovering'][s], scale)
                    
                    # eq. 1.3.2
                    alpha = (log_prior_prime + log_prop_prime) - (log_prior_prev + log_prop_prev)
                    print("Alpha: %.5f" % np.exp(alpha))
        
        import matplotlib.pyplot as plt
        plt.scatter(distances['Recovering'], distances['Declining'])
        plt.xlabel('Recovering')
        plt.ylabel('Declining')
        plt.savefig('abc_results/distances_InGeneralWard_scatterplot_%s.png' % self.params_init)

        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
        for i, health_state in enumerate(distances):
            ax[i].hist(distances[health_state])
            ax[i].set_title(health_state)
        plt.savefig('abc_results/distances_InGeneralWard_histograms_%s.png' % self.params_init)

        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
        for i, health_state in enumerate(distances):
            ax[i].hist(diff_distances[health_state])
            ax[i].set_title(health_state)
        plt.savefig('abc_results/diff_distances_InGeneralWard_histograms_%s.png' % self.params_init)

        exit(1)


    # @profile
    def abc_mcmc(self, theta_init, num_iterations, scale):
        accepted_thetas = []
        accepted_distances = []
        all_distances = []
        accepted_alphas = []
        all_alphas = []
        num_accepted = 0
        theta = theta_init
        self.best_distance = self.epsilon
        self.epsilon_trace = []

        T_x = self.simulate_dataset(theta)
        # log_pi_eps_prev = self.calc_log_pi_epsilon(T_x)
        log_prior_prev = self.calc_log_proba_dirichlet_prior(theta)

        for n in range(num_iterations):
            print("Iteration #%d" % (n + 1))

            for s in range(len(self.config_dict['states'])):
                # draw from proposal distribution
                theta_health_prime = deepcopy(theta['health'])
                p_prime_D = self.draw_proposal_distribution_categorical(theta_health_prime[s], scale)
                theta_health_prime[s] = p_prime_D

                # create a theta_prime and simulate dataset
                theta_prime = deepcopy(theta)
                theta_prime['health'] = theta_health_prime
                T_x = self.simulate_dataset(theta_prime)
                distance = self.calc_distance(T_x)
                all_distances.append(distance)

                if self.accept(distance):
                    accepted_distances.append(distance)

                    # calculate alpha(theta, theta_prime)
                    # log_pi_eps_prime = self.calc_log_pi_epsilon(T_x)
                    # print(log_pi_eps_prime)
                    log_prior_prime = self.calc_log_proba_dirichlet_prior(theta_prime)
                    log_prop_prime = 0.0
                    log_prop_prev = 0.0
                    for s in range(len(self.config_dict['states'])):
                        # health
                        log_prop_prime += self.calc_log_proba_proposal_distribution_categorical(theta['health'][s], theta_prime['health'][s], scale)
                        log_prop_prev  += self.calc_log_proba_proposal_distribution_categorical(theta_prime['health'][s], theta['health'][s], scale)
                        
                        if self.state_name_id_map[s] != 'Presenting': # skipping presenting state for durations
                            # durations
                            log_prop_prime += self.calc_log_proba_proposal_distribution_categorical(theta['durations']['Declining'][s], theta_prime['durations']['Declining'][s], scale)
                            log_prop_prev  += self.calc_log_proba_proposal_distribution_categorical(theta_prime['durations']['Declining'][s], theta['durations']['Declining'][s], scale)
                            log_prop_prime += self.calc_log_proba_proposal_distribution_categorical(theta['durations']['Recovering'][s], theta_prime['durations']['Recovering'][s], scale)
                            log_prop_prev  += self.calc_log_proba_proposal_distribution_categorical(theta_prime['durations']['Recovering'][s], theta['durations']['Recovering'][s], scale)

                    # eq. 1.3.2
                    alpha = (log_prior_prime + log_prop_prime) - (log_prior_prev + log_prop_prev)
                    print("Alpha: %.3f" % np.exp(alpha))
                    all_alphas.append(np.exp(alpha))

                    if np.random.random() < np.exp(alpha):
                        self.best_distance = distance
                        theta = deepcopy(theta_prime)
                        accepted_thetas.append(theta)
                        accepted_alphas.append(np.exp(alpha))
                        num_accepted += 1
                        # log_pi_eps_prev = log_pi_eps_prime
                        log_prior_prev = log_prior_prime
                    else:
                        accepted_alphas.append(np.NaN)

                else:
                    accepted_distances.append(np.NaN)
                    accepted_alphas.append(np.NaN)
                    all_alphas.append(np.NaN)

                self.update_epsilon()

                if self.state_name_id_map[s] != 'Presenting': # skipping presenting state for durations
                    for health_state in ["Declining", "Recovering"]:
                        # draw from proposal distribution
                        theta_durations_prime = deepcopy(theta['durations'][health_state])
                        p_prime_D = self.draw_proposal_distribution_categorical(theta_durations_prime[s], scale)
                        theta_durations_prime[s] = p_prime_D

                        # create a theta_prime and simulate dataset
                        theta_prime = deepcopy(theta)
                        theta_prime['durations'][health_state] = theta_durations_prime
                        T_x = self.simulate_dataset(theta_prime)
                        distance = self.calc_distance(T_x)
                        all_distances.append(distance)

                        if self.accept(distance):
                            accepted_distances.append(distance)

                            # calculate alpha(theta, theta_prime)
                            # log_pi_eps_prime = self.calc_log_pi_epsilon(T_x)
                            # print(log_pi_eps_prime)
                            log_prior_prime = self.calc_log_proba_dirichlet_prior(theta_prime)
                            log_prop_prime = 0.0
                            log_prop_prev = 0.0
                            for s in range(len(self.config_dict['states'])):
                                # health
                                log_prop_prime += self.calc_log_proba_proposal_distribution_categorical(theta['health'][s], theta_prime['health'][s], scale)
                                log_prop_prev  += self.calc_log_proba_proposal_distribution_categorical(theta_prime['health'][s], theta['health'][s], scale)
                                
                                if self.state_name_id_map[s] != 'Presenting': # skipping presenting state for durations
                                    # durations
                                    log_prop_prime += self.calc_log_proba_proposal_distribution_categorical(theta['durations']['Declining'][s], theta_prime['durations']['Declining'][s], scale)
                                    log_prop_prev  += self.calc_log_proba_proposal_distribution_categorical(theta_prime['durations']['Declining'][s], theta['durations']['Declining'][s], scale)
                                    log_prop_prime += self.calc_log_proba_proposal_distribution_categorical(theta['durations']['Recovering'][s], theta_prime['durations']['Recovering'][s], scale)
                                    log_prop_prev  += self.calc_log_proba_proposal_distribution_categorical(theta_prime['durations']['Recovering'][s], theta['durations']['Recovering'][s], scale)
                            
                            # eq. 1.3.2
                            alpha = (log_prior_prime + log_prop_prime) - (log_prior_prev + log_prop_prev)
                            print("Alpha: %.3f" % np.exp(alpha))
                            all_alphas.append(np.exp(alpha))

                            if np.random.random() < np.exp(alpha):
                                self.best_distance = distance
                                theta = deepcopy(theta_prime)
                                accepted_thetas.append(theta)
                                accepted_alphas.append(np.exp(alpha))
                                num_accepted += 1
                                # log_pi_eps_prev = log_pi_eps_prime
                                log_prior_prev = log_prior_prime
                            else:
                                accepted_alphas.append(np.NaN)

                        else:
                            accepted_distances.append(np.NaN)
                            accepted_alphas.append(np.NaN)
                            all_alphas.append(np.NaN)

                        self.update_epsilon()

        return accepted_thetas, accepted_distances, num_accepted, all_distances, accepted_alphas, all_alphas

    def draw_samples(self, algorithm='abc', num_iterations=100, scale=10.0):

        if algorithm == 'fixed':
            return self.fixed_params(self.theta_init, num_iterations)
        elif algorithm == 'sa':
            return self.simulated_annealing(self.theta_init, num_iterations, scale)
        elif algorithm == 'abc':
            return self.abc_mcmc(self.theta_init, num_iterations, scale)
        elif algorithm == 'abc_exp':
            return self.abc_mcmc_experiments(self.theta_init, num_iterations, scale)
        else:
            print("Invalid algorithm specification '%s'" % algorithm)
            print("Valid options are:")
            print("\t- 'fixed' to run simulations with fixed parameters specified in theta_init")
            print("\t- 'sa' to run simulated annealing")
            print("\t- 'abc' to run ABC MCMC sampling")
            exit(1)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.annealing_constant, self.best_distance)
        self.epsilon_trace.append(self.epsilon)

    def calc_distance(self, T_x):
        # euclidean distance
        # distance = scipy.spatial.distance.euclidean(T_x, self.T_y)

        # mean squared error
        # distance = (np.nansum(np.power(T_x - self.T_y, 2)) / T_x.shape[0]) / DENOM_NUM_PRESENTING

        # compute distance as average percent deviation from T_y !!!
        a = np.array([T_x, self.T_y])
        denom = np.max(a, axis=0)
        distance = np.nansum((np.abs(np.diff(a, axis=0))) / denom) / T_x.shape[0]

        # mean squared error, but multiplied by 2 for the later stages
        # distance = np.power(T_x - self.T_y, 2)
        # for i, name in enumerate(SUMMARY_STATISTICS_NAMES):
        #     if name == 'n_OnVentInICU' or 'n_TERMINAL':
        #         distance[(i * self.num_timesteps) : ((i + 1) * self.num_timesteps)] = distance[(i * self.num_timesteps) : ((i + 1) * self.num_timesteps)] * 2

        # distance = np.nansum(distance) / T_x.shape[0]

        print("Distance: %.3f" % distance)
        return distance

    def accept(self, distance):
        return distance < self.epsilon

    def uniform_kernel_density(self, x):
        if np.abs(x) < 1:
            return 0.5
        else:
            return 0.0

    def calc_log_pi_epsilon(self, T_x):
        # eq. 1.2.3
        return np.log(self.uniform_kernel_density(self.calc_distance(T_x) / self.epsilon) + 1e-12) - np.log(self.epsilon)

    def calc_log_proba_dirichlet_prior(self, theta):
        log_proba = 0.0
        log_proba += np.sum([scipy.stats.dirichlet.logpdf(theta['health'][s] + 1e-12, self.abc_prior['health'][s]) for s in range(len(theta['health']))])
        log_proba += np.sum([scipy.stats.dirichlet.logpdf(theta['durations']['Declining'][s] + 1e-12, self.abc_prior['durations']['Declining'][s]) for s in range(len(theta['durations']['Declining'])) if self.state_name_id_map[s] != 'Presenting'])
        log_proba += np.sum([scipy.stats.dirichlet.logpdf(theta['durations']['Recovering'][s] + 1e-12, self.abc_prior['durations']['Recovering'][s]) for s in range(len(theta['durations']['Recovering'])) if self.state_name_id_map[s] != 'Presenting'])
        # TODO
        # if some entry of p_prime_D is zero AND some other entry is one, there will be an error
        return log_proba

    def calc_log_proba_proposal_distribution_categorical(self, p_D, p_prime_D, scale):
        try:
            return scipy.stats.dirichlet.logpdf(p_prime_D, (p_D * scale) + 1)
        except:
            return scipy.stats.dirichlet.logpdf(p_prime_D + 1e-12, (p_D * scale) + 1) # in case some entry of p_prime_D is zero
        # TODO
        # if some entry of p_prime_D is zero AND some other entry is one, there will be an error

    def draw_proposal_distribution_categorical(self, p_D, scale):
        return np.random.dirichlet((p_D * scale) + 1)

    def save_thetas_to_json(self, thetas, filename):
        states = self.config_dict['states']

        json_to_save = {'last_thetas': []}
        for theta in thetas:
            params_dict = {}
            for s in range(len(states)):
                # updating health state probabilities
                params_dict['proba_Recovering_given_%s' % states[s]] = theta['health'][s][1]

                # updating durations probabilities
                for health_state in ["Declining", "Recovering"]:
                    choices = list(self.config_dict['pmf_duration_%s_%s' % (health_state, states[s])].keys())
                    params_dict['pmf_duration_%s_%s' % (health_state, states[s])] = {}
                    for d in range(theta['durations'][health_state][s].shape[0]):
                        # update each individual choice with the value in theta
                        params_dict['pmf_duration_%s_%s' % (health_state, states[s])][choices[d]] = theta['durations'][health_state][s][d]
                            
            json_to_save['last_thetas'].append(deepcopy(params_dict))

        with open(filename, 'w+') as f:
            json.dump(json_to_save, f, indent=1)


def save_stats_to_csv(stats, filename):
    df = pd.DataFrame(stats)
    df.to_csv(filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hospital', default='university_hospitals_of_north_midlands_nhs_trust_Training')
    parser.add_argument('--config_template', default='NHS_data/new_data/formatted_data/params_simple') # template for config_file
    parser.add_argument('--input_template', default='NHS_data/new_data/formatted_data/')
    parser.add_argument('--output_template', default='NHS_results/abc_noVent')
    parser.add_argument('--random_seed', default=101, type=int) # currently not using it 
    parser.add_argument('--algorithm', default='abc')
    parser.add_argument('--num_iterations', default=2000, type=int) # number of sampling iterations 
                                                                   # each iteration has an inner loop through each probabilistic parameter vector
    parser.add_argument('--num_simulations', default=5, type=int)
    parser.add_argument('--start_epsilon', default=1.0, type=float)
    parser.add_argument('--lowest_epsilon', default=15.0, type=float) # unused in abc
    parser.add_argument('--annealing_constant', default=0.9999)
    parser.add_argument('--scale', default=100, type=int) # scale parameter for the dirichlet proposal distribution

    parser.add_argument('--params_init', default='None')
    parser.add_argument('--abc_prior_type', default='20MaxEach_uniform')
    parser.add_argument('--abc_prior_config_template', default='NHS_data/abc_prior_config')

    args, unknown_args = parser.parse_known_args()

    unk_keys = map(lambda s: s[2:].strip(), unknown_args[::2])
    unk_vals = unknown_args[1::2]
    unk_dict = dict(zip(unk_keys, unk_vals))

    seed = int(args.random_seed)
    num_iterations = int(args.num_iterations)
    num_simulations = int(args.num_simulations)
    start_epsilon = float(args.start_epsilon)
    lowest_epsilon = float(args.lowest_epsilon)
    annealing_constant = float(args.annealing_constant)
    scale = int(args.scale)
    algorithm = args.algorithm

    if args.params_init == 'None':
        params_init = None
    else:
        params_init = args.params_init
    if algorithm == 'abc':
        with open(args.abc_prior_config_template + '_%s.json' % args.abc_prior_type, 'r') as f:
            abc_prior = json.load(f)
        thetas_output = args.output_template + '_%s_last_thetas_%s.json' % (args.hospital, args.abc_prior_type)
        stats_output = args.output_template + '_%s_stats_%s.csv' % (args.hospital, args.abc_prior_type)
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
    print(true_df)


    # condensing the true results in a summary vector
    T_y = []
    for col_name in SUMMARY_STATISTICS_NAMES:
        T_y.append(true_df[col_name])
    T_y = np.asarray(T_y).flatten()

    # initalize sampler
    sampler = ABCSampler(seed, start_epsilon, lowest_epsilon, annealing_constant, T_y, config_dict, num_timesteps, num_simulations)

    sampler.initialize_theta(algorithm, abc_prior, params_init)
    print(sampler.theta_init)

    accepted_thetas, accepted_distances, num_accepted, all_distances, accepted_alphas, all_alphas = sampler.draw_samples(algorithm, num_iterations, scale)

    num_to_save = int(len(accepted_thetas) * 0.5) # save only the second half of the parameters
    last_thetas = [accepted_thetas[0]] + accepted_thetas[-num_to_save:] # also save the first theta
    sampler.save_thetas_to_json(last_thetas, thetas_output)

    print(len(sampler.epsilon_trace))
    print(len(accepted_distances))

    stats = {'all_distances': all_distances, 'accepted_distances': accepted_distances, 'all_alphas': all_alphas, 'accepted_alphas': accepted_alphas, 'epsilon_trace': sampler.epsilon_trace}
    save_stats_to_csv(stats, stats_output)