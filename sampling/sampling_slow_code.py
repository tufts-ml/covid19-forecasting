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
from semimarkov_forecaster import PatientTrajectory
import multiprocessing

# This is copy-pasted from 'run_forecast.py', with the only difference that, instead of saving
# results to a csv file, the function returns the pandas DataFrame
# @profile
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
    summary_dict_list = list()

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
        
        # # First, try to replace wildcards
        # if isinstance(pmfstr_or_csvfile, str):
        #     for key, val in list(args.__dict__.items()) + list(unk_dict.items()):
        #         wildcard_key = "{%s}" % key
        #         if pmfstr_or_csvfile.count(wildcard_key):
        #             pmfstr_or_csvfile = pmfstr_or_csvfile.replace(wildcard_key, str(val))
        #             print("WILDCARD: %s" % pmfstr_or_csvfile)
        
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
            # @profile
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

    # print("--------------------------------------------")
    # print("Simulating for %3d timesteps with seed %5d" % (T, args.random_seed))
    # print("Initial population size: %5d" % (np.sum([
    #     config_dict['init_num_%s' % s] for s in config_dict['states']])))
    # print("--------------------------------------------")
    ## Simulate what happens to initial population
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
                # summary_dict_list.append(p.get_length_of_stay_summary_dict()) # don't need this

    ## Simulate what happens as new patients added at each step
    for t in range(1, T+1): # tqdm.tqdm(range(1, T+1)):
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
                # summary_dict_list.append(p.get_length_of_stay_summary_dict()) # don't need this


    # Save only the first T + 1 tsteps (with index 0, 1, 2, ... T)
    t0 = 0
    tf = T + Tpast + 1
    occupancy_count_TK = occupancy_count_TK[t0:tf]
    admit_count_TK = admit_count_TK[t0:tf]
    discharge_count_TK = discharge_count_TK[t0:tf]
    terminal_count_T1 = terminal_count_T1[t0:tf]

    ## Write results to spreadsheet
    # print("----------------------------------------")
    # print("Writing results to %s" % (output_file))
    # print("----------------------------------------")
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

    # los_summary_df = pd.DataFrame(summary_dict_list)
    # los_columns = (
    #     ['is_Terminal', 'is_InICU', 'is_OnVent', 'duration_All']
    #     + ['duration_%s' % s for s in config_dict['states']]
    #     + ['duration_%s+Declining' % s for s in config_dict['states']]
    #     + ['duration_%s+Recovering' % s for s in config_dict['states']]
    #     )
    # # los_summary_df.to_csv(output_file.replace('results', 'los_results_per_patient'), columns=los_columns,
    # #     index=False, float_format="%.0f")

    # statistics = []
    # # Obtain percentile summaries of durations across population
    # for query in ["", "is_InICU == 1", "is_InICU == 0", "is_OnVent == 1"]:

    #     collapsed_dict_list = list()
    #     if len(query) > 0:
    #         q_df = los_summary_df.query(query).copy()
    #     else:
    #         q_df = los_summary_df
    #     filled_vals = q_df[los_columns].values.copy().astype(np.float32)
    #     filled_vals[np.isnan(filled_vals)] = 0

    #     def my_count(x, *args, **kwargs):
    #         return np.sum(np.isfinite(x), *args, **kwargs)

    #     for metric_name, func in [
    #             ('count', my_count),
    #             ('mean', np.mean)]:
    #         collapsed_dict = dict(zip(
    #             los_columns,
    #             func(filled_vals, axis=0)))
    #         collapsed_dict['summary_name'] = metric_name
    #         collapsed_dict_list.append(collapsed_dict)
    #     for perc in [25, 50, 75]:
    #         collapsed_dict = dict(zip(
    #             los_columns,
    #             np.percentile(filled_vals, perc, axis=0)))
    #         collapsed_dict['summary_name'] = 'percentile_%06.3f' % perc
    #         collapsed_dict_list.append(collapsed_dict)
    #     # print(collapsed_dict_list)
    #     collapsed_df = pd.DataFrame(collapsed_dict_list, index=[adict['summary_name'] for adict in collapsed_dict_list])

    #     if len(query) > 0:
    #         sane_query_suffix = "_" + query.replace(" == ", '').replace("_", "")
    #     else:
    #         sane_query_suffix = ""

    #     if sane_query_suffix == '_isOnVent1':
    #         for perc in [25, 50, 75]:
    #             statistics.append(collapsed_df.loc['percentile_%06.3f' % perc, 'duration_OnVentInICU'])
    #     elif sane_query_suffix == '_isInICU0' or sane_query_suffix == '_isInICU1':
    #         for perc in [25, 50, 75]:
    #             statistics.append(collapsed_df.loc['percentile_%06.3f' % perc, 'duration_All'])
    #     elif len(query) == 0:
    #         for st in ['is_Terminal', 'is_OnVent', 'is_InICU']:
    #             statistics.append(collapsed_df.loc['mean', st])

    #     # collapsed_df.to_csv(output_file.replace('results', 'los_results_summary%s' % sane_query_suffix),
    #     #     columns=['summary_name'] + los_columns,
    #     #     index=False, float_format='%.2f')

    return results_df # np.array(statistics)

'''
theta: 
    dict with keys: 'health', 'transitions', durations'
    theta['num timesteps'] = N
    theta['health'] = list of length S, each containing array of shape (2,) (treating health probas as 2-dim dirichlet)
    theta['durations'] = list of length S with arrays of shape (H, D) [(D,) in simple example]
'''
SUMMARY_STATISTICS_NAMES = ["n_discharges", "n_occupied_beds", "n_OnVentInICU"] #, "n_admitted_InGeneralWard", "n_admitted_OffVentInICU", "n_admitted_OnVentInICU", "n_discharged_InGeneralWard", "n_discharged_OffVentInICU", "n_discharged_OnVentInICU"]

HEALTH_STATE_ID_TO_NAME = {0: 'Declining', 1: 'Recovering', 'Declining': 0, 'Recovering': 1}

class ABCSampler(object):

    def __init__(self, seed, start_epsilon, lowest_epsilon, annealing_constant, T_y, train_test_split, config_dict, num_timesteps, num_simulations):
        self.T_y = T_y
        self.epsilon = start_epsilon
        self.lowest_epsilon = lowest_epsilon
        self.annealing_constant = annealing_constant
        self.seed = seed
        self.config_dict = config_dict
        self.num_timesteps = num_timesteps
        self.num_simulations = num_simulations
        self.train_test_split = train_test_split # integer timestep for now, then will be a date

        self.state_name_id_map = {}
        for s, state in enumerate(config_dict['states']):
            self.state_name_id_map[state] = s
            self.state_name_id_map[s] = state

    def initialize_theta(self, algorithm, abc_prior_dict, params_init=None):
        self.params_init = params_init
        self.abc_prior_dict = abc_prior_dict

        if algorithm == 'abc' or algorithm == 'abc_exp':
            print(abc_prior_dict)
            print(params_init)
            self.abc_prior = self.initialize_abc_prior(abc_prior_dict)
        else:
            self.abc_prior = None
        
        if params_init is None:
            self.theta_init = self.select_theta_init_from_prior(self.abc_prior)
        else:
            self.theta_init = self.select_theta_init_from_name(params_init, self.abc_prior)


    def initialize_abc_prior(self, abc_prior_dict):
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
        states = self.config_dict['states']

        health = [np.random.dirichlet(params) for params in prior['health']]
        die_after_declining = [np.random.dirichlet(params) for params in prior['die_after_declining']]

        durations = {'Declining': [], 'Recovering': []}
        for s, state in enumerate(states):
            for health_state in ['Declining', 'Recovering']:
                durations[health_state].append({'lam': prior['durations'][health_state][s]['lam'].rvs(size=1)[0], 'tau': prior['durations'][health_state][s]['tau'].rvs(size=1)[0], 'lower': prior['durations'][health_state][s]['lower'], 'upper': prior['durations'][health_state][s]['upper']})

        theta_init = {'num timesteps': self.num_timesteps, 'health': health, 'die_after_declining': die_after_declining, 'durations': durations}

        return theta_init

    def select_theta_init_from_name(self, params_init, prior):
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

    # @profile
    def simulate_dataset(self, theta):
        # simulate the dataset given the parameters in theta

        # update config dict given the parameters in theta
        self.update_config_dict_given_theta(theta)

        # # getting params ready for run_simulation (copied from run_forecast)
        states = self.config_dict['states']
        state_name_to_id = dict()
        next_state_map = dict()
        for ss, state in enumerate(states):
            state_name_to_id[state] = ss
            # if ss < len(states):
            if ss == 0: ## next_state_map for recovery... HEALTH_STATE_ID_TO_NAME = {0: 'Declining', 1: 'Recovering'}
                next_state_map[state+'Recovering'] = 'RELEASE'
                next_state_map[state+'Declining'] = states[ss+1]
            elif ss == len(states)-1:
                next_state_map[state+'Recovering'] = states[ss-1]
                next_state_map[state+'Declining'] = 'TERMINAL'
            else:
                next_state_map[state+'Recovering'] = states[ss-1]
                next_state_map[state+'Declining'] = states[ss+1]

        state_name_to_id['TERMINAL'] = len(states)

        T = None
        for i in range(self.num_simulations):
            # running the simulation. notice that the random seed is not used. if used, every simulation with the same
            # parameters would be the same # np.random.randint(1000000)
            
            import time
            start = time.time()
            results_df = run_simulation(None, self.config_dict, states, state_name_to_id, next_state_map)
            end = time.time()
            print(end-start)

            train_df = results_df[results_df['timestep'] <= self.train_test_split]
            test_df = results_df[results_df['timestep'] > self.train_test_split]

            results_df = train_df
            # print(results_df)
            # condensing the results in a summary vector
            T_x = []
            weights = []
            for col_name in SUMMARY_STATISTICS_NAMES:
                if col_name == "n_InICU":
                    T_x.append(results_df["n_OffVentInICU"] + results_df["n_OnVentInICU"])
                elif col_name == "n_occupied_beds":
                    T_x.append(results_df["n_InGeneralWard"] + results_df["n_OffVentInICU"] + results_df["n_OnVentInICU"])
                elif col_name == "n_discharges":
                    T_x.append(results_df["n_discharged_InGeneralWard"] + results_df["n_discharged_OffVentInICU"] + results_df["n_discharged_OnVentInICU"])
                else:
                    T_x.append(results_df[col_name])

                weights.append(np.linspace(0.5, 1.5, results_df["n_InGeneralWard"].shape[0]))

            T_x = np.asarray(T_x).flatten()
            weights = np.asarray(weights).flatten()

            # accumulating summary statistics from multiple runs
            if T is None:
                T = T_x
            else:
                T += T_x

        # averaging summary statistics from multiple runs
        T_x = T / float(self.num_simulations)

        return T_x, weights, test_df # return just the last simulation on testing

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

            for state_index in range(len(self.config_dict['states']) - 1):
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

    def abc_mcmc_experiments(self, theta_init, num_iterations, dir_scale_tuple, lam_stddev_tuple, tau_stddev_tuple):
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

        temp_distances = []
        for _ in range(20):
            T_x, weights, test_df = self.simulate_dataset(theta)
            temp_distances.append(self.calc_distance(T_x, weights))

        print(np.mean(temp_distances))
        print(np.std(temp_distances))
        print(np.min(temp_distances))

        all_distances.append(np.min(temp_distances))

        accepted_thetas.append(theta)
        accepted_distances.append(np.NaN)
        accepted_alphas.append(np.NaN)
        all_alphas.append(np.NaN)
        self.epsilon_trace.append(self.epsilon)

        log_prior_prev = self.calc_log_proba_prior(theta)

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
                accepted_thetas.append(theta_prime)

                if self.accept(distance):
                    accepted_distances.append(distance)

                    # calculate alpha(theta, theta_prime)
                    # log_pi_eps_prime = self.calc_log_pi_epsilon(T_x)
                    # print(log_pi_eps_prime)
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
                    print("Alpha: %.3f" % np.exp(alpha))
                    all_alphas.append(np.exp(alpha))

                    if np.random.random() < np.exp(alpha):
                        self.best_distance = distance
                        # theta = deepcopy(theta_prime)
                        # accepted_thetas.append(theta)
                        accepted_alphas.append(np.exp(alpha))
                        accepted_test_forecasts.append(test_df)
                        num_accepted += 1
                        # log_pi_eps_prev = log_pi_eps_prime
                        log_prior_prev = log_prior_prime
                    else:
                        accepted_alphas.append(np.NaN)

                else:
                    accepted_distances.append(np.NaN)
                    accepted_alphas.append(np.NaN)
                    all_alphas.append(np.NaN)

                self.update_epsilon(n)

                if self.state_name_id_map[state_index] != "OnVentInICU":
                    # draw from proposal distribution
                    theta_health_prime = deepcopy(theta['die_after_declining'])
                    p_prime_D = self.draw_proposal_distribution_categorical(theta_health_prime[state_index], dir_scale)
                    theta_health_prime[state_index] = p_prime_D

                    # create a theta_prime and simulate dataset
                    theta_prime = deepcopy(theta)
                    theta_prime['die_after_declining'] = theta_health_prime
                    T_x, weights, test_df = self.simulate_dataset(theta_prime)
                    distance = self.calc_distance(T_x, weights)
                    all_distances.append(distance)
                    accepted_thetas.append(theta_prime)

                    if self.accept(distance):
                        accepted_distances.append(distance)

                        # calculate alpha(theta, theta_prime)
                        # log_pi_eps_prime = self.calc_log_pi_epsilon(T_x)
                        # print(log_pi_eps_prime)
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
                        print("Alpha: %.3f" % np.exp(alpha))
                        all_alphas.append(np.exp(alpha))

                        if np.random.random() < np.exp(alpha):
                            self.best_distance = distance
                            # theta = deepcopy(theta_prime)
                            # accepted_thetas.append(theta)
                            accepted_alphas.append(np.exp(alpha))
                            accepted_test_forecasts.append(test_df)
                            num_accepted += 1
                            # log_pi_eps_prev = log_pi_eps_prime
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
                    accepted_thetas.append(theta_prime)

                    if self.accept(distance):
                        accepted_distances.append(distance)

                        # calculate alpha(theta, theta_prime)
                        # log_pi_eps_prime = self.calc_log_pi_epsilon(T_x)
                        # print(log_pi_eps_prime)
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
                        print("Alpha: %.3f" % np.exp(alpha))
                        all_alphas.append(np.exp(alpha))

                        if np.random.random() < np.exp(alpha):
                            self.best_distance = distance
                            # theta = deepcopy(theta_prime)
                            # accepted_thetas.append(theta)
                            accepted_alphas.append(np.exp(alpha))
                            accepted_test_forecasts.append(test_df)
                            num_accepted += 1
                            # log_pi_eps_prev = log_pi_eps_prime
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
                    accepted_thetas.append(theta_prime)

                    if self.accept(distance):
                        accepted_distances.append(distance)

                        # calculate alpha(theta, theta_prime)
                        # log_pi_eps_prime = self.calc_log_pi_epsilon(T_x)
                        # print(log_pi_eps_prime)
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
                        print("Alpha: %.3f" % np.exp(alpha))
                        all_alphas.append(np.exp(alpha))

                        if np.random.random() < np.exp(alpha):
                            self.best_distance = distance
                            # theta = deepcopy(theta_prime)
                            # accepted_thetas.append(theta)
                            accepted_alphas.append(np.exp(alpha))
                            accepted_test_forecasts.append(test_df)
                            num_accepted += 1
                            # log_pi_eps_prev = log_pi_eps_prime
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

        T_x, weights, test_df = self.simulate_dataset(theta)
        # log_pi_eps_prev = self.calc_log_pi_epsilon(T_x)
        log_prior_prev = self.calc_log_proba_prior(theta)

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
                    # log_pi_eps_prime = self.calc_log_pi_epsilon(T_x)
                    # print(log_pi_eps_prime)
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
                    print("Alpha: %.3f" % np.exp(alpha))
                    all_alphas.append(np.exp(alpha))

                    if np.random.random() < np.exp(alpha):
                        self.best_distance = distance
                        theta = deepcopy(theta_prime)
                        accepted_thetas.append(theta)
                        accepted_alphas.append(np.exp(alpha))
                        accepted_test_forecasts.append(test_df)
                        num_accepted += 1
                        # log_pi_eps_prev = log_pi_eps_prime
                        log_prior_prev = log_prior_prime
                    else:
                        accepted_alphas.append(np.NaN)

                else:
                    accepted_distances.append(np.NaN)
                    accepted_alphas.append(np.NaN)
                    all_alphas.append(np.NaN)

                self.update_epsilon(n)

                if self.state_name_id_map[state_index] != "OnVentInICU":
                    # draw from proposal distribution
                    theta_health_prime = deepcopy(theta['die_after_declining'])
                    p_prime_D = self.draw_proposal_distribution_categorical(theta_health_prime[state_index], dir_scale)
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
                        # log_pi_eps_prime = self.calc_log_pi_epsilon(T_x)
                        # print(log_pi_eps_prime)
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
                        print("Alpha: %.3f" % np.exp(alpha))
                        all_alphas.append(np.exp(alpha))

                        if np.random.random() < np.exp(alpha):
                            self.best_distance = distance
                            theta = deepcopy(theta_prime)
                            accepted_thetas.append(theta)
                            accepted_alphas.append(np.exp(alpha))
                            accepted_test_forecasts.append(test_df)
                            num_accepted += 1
                            # log_pi_eps_prev = log_pi_eps_prime
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
                        # log_pi_eps_prime = self.calc_log_pi_epsilon(T_x)
                        # print(log_pi_eps_prime)
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
                        print("Alpha: %.3f" % np.exp(alpha))
                        all_alphas.append(np.exp(alpha))

                        if np.random.random() < np.exp(alpha):
                            self.best_distance = distance
                            theta = deepcopy(theta_prime)
                            accepted_thetas.append(theta)
                            accepted_alphas.append(np.exp(alpha))
                            accepted_test_forecasts.append(test_df)
                            num_accepted += 1
                            # log_pi_eps_prev = log_pi_eps_prime
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
                        # log_pi_eps_prime = self.calc_log_pi_epsilon(T_x)
                        # print(log_pi_eps_prime)
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
                        print("Alpha: %.3f" % np.exp(alpha))
                        all_alphas.append(np.exp(alpha))

                        if np.random.random() < np.exp(alpha):
                            self.best_distance = distance
                            theta = deepcopy(theta_prime)
                            accepted_thetas.append(theta)
                            accepted_alphas.append(np.exp(alpha))
                            accepted_test_forecasts.append(test_df)
                            num_accepted += 1
                            # log_pi_eps_prev = log_pi_eps_prime
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

    def get_params(self, theta):
        states = self.config_dict['states']
        params_dict = {}
        for s in range(len(states)):
            # updating health state probabilities
            params_dict['proba_Recovering_given_%s' % states[s]] = theta['health'][s][1]
            # print('proba_Recovering_given_%s: %.3f' % (states[s], theta['health'][s][1]))

            if states[s] != "OnVentInICU":
                params_dict['proba_Die_after_Declining_%s' % states[s]] = theta['die_after_declining'][s][1]
                # print('proba_Die_after_Declining_%s: %.3f' % (states[s], theta['die_after_declining'][s][1]))
            else:
                params_dict['proba_Die_after_Declining_%s' % states[s]] = 1.0
                # print('proba_Die_after_Declining_%s: %.3f' % (states[s], 1.0))

            # updating durations probabilities
            for health_state in ["Declining", "Recovering"]:
                choices = list(self.config_dict['pmf_duration_%s_%s' % (health_state, states[s])].keys())

                lam = theta['durations'][health_state][s]['lam']
                tau = theta['durations'][health_state][s]['tau']

                params_dict['pmf_duration_%s_%s' % (health_state, states[s])] = {'lam': lam, 'tau': tau}
                # print('pmf_duration_%s_%s: lam - %.3f, tau %.3f' % (health_state, states[s], lam, tau))

                probas = scipy.special.softmax(scipy.stats.poisson.logpmf(np.arange(len(choices)), lam) / np.power(10, tau))

                for c, choice in enumerate(choices):
                    # update each individual choice with the value in theta
                    params_dict['pmf_duration_%s_%s' % (health_state, states[s])][choice] = probas[c]
        # print('######################################################')
        return params_dict

    def print_params(self, theta):
        states = self.config_dict['states']
        # params_dict = {}
        print('######################################################')
        for s in range(len(states)):
            # updating health state probabilities
            # params_dict['proba_Recovering_given_%s' % states[s]] = theta['health'][s][1]
            print('proba_Recovering_given_%s: %.3f' % (states[s], theta['health'][s][1]))

            if states[s] != "OnVentInICU":
                # params_dict['proba_Die_after_Declining_%s' % states[s]] = theta['die_after_declining'][s][1]
                print('proba_Die_after_Declining_%s: %.3f' % (states[s], theta['die_after_declining'][s][1]))
            else:
                # params_dict['proba_Die_after_Declining_%s' % states[s]] = 1.0
                print('proba_Die_after_Declining_%s: %.3f' % (states[s], 1.0))

            # updating durations probabilities
            for health_state in ["Declining", "Recovering"]:
                choices = list(self.config_dict['pmf_duration_%s_%s' % (health_state, states[s])].keys())

                lam = theta['durations'][health_state][s]['lam']
                tau = theta['durations'][health_state][s]['tau']

                # params_dict['pmf_duration_%s_%s' % (health_state, states[s])] = {'lam': lam, 'tau': tau}
                print('pmf_duration_%s_%s: lam - %.3f, tau %.3f' % (health_state, states[s], lam, tau))

                # probas = scipy.special.softmax(scipy.stats.poisson.logpmf(np.arange(len(choices)), lam) / np.power(10, tau))

                # for c, choice in enumerate(choices):
                #     # update each individual choice with the value in theta
                #     params_dict['pmf_duration_%s_%s' % (health_state, states[s])][choice] = probas[c]
        print('######################################################')


    def draw_samples(self, algorithm='abc', num_iterations=100, dir_scale=100, lam_stddev=0.5, tau_stddev=0.1):

        if algorithm == 'fixed':
            return self.fixed_params(self.theta_init, num_iterations)
        elif algorithm == 'sa':
            return self.simulated_annealing(self.theta_init, num_iterations, scale)
        elif algorithm == 'abc':
            return self.abc_mcmc(self.theta_init, num_iterations, dir_scale, lam_stddev, tau_stddev)
        elif algorithm == 'abc_exp':
            return self.abc_mcmc_experiments(self.theta_init, num_iterations, dir_scale, lam_stddev, tau_stddev)
        else:
            print("Invalid algorithm specification '%s'" % algorithm)
            print("Valid options are:")
            print("\t- 'fixed' to run simulations with fixed parameters specified in theta_init")
            print("\t- 'sa' to run simulated annealing")
            print("\t- 'abc' to run ABC MCMC sampling")
            exit(1)

    def update_epsilon(self, n):
        if n < self.num_iterations * (3/4):
            self.epsilon = max(self.epsilon * self.annealing_constant, self.best_distance)
            # self.epsilon = self.epsilon * self.annealing_constant
        self.epsilon_trace.append(self.epsilon)

    def calc_distance(self, T_x, weights=None):
        if weights is None:
            weights = np.ones(T_x.shape)

        # euclidean distance
        # distance = scipy.spatial.distance.euclidean(T_x, self.T_y)

        # mean squared error
        # distance = (np.nansum(np.power(T_x - self.T_y, 2)) / T_x.shape[0]) / DENOM_NUM_PRESENTING

        # mean absolute error normalized by the maximum individual statistic value at each timestep and for each summary statistic
        a = np.array([T_x, self.T_y])
        denom = np.max(a, axis=0)
        distance = np.nansum((weights * np.abs(np.diff(a, axis=0))) / denom) / T_x.shape[0]

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
        # return (distance < self.best_distance) or (np.random.random() < np.exp(- distance / self.epsilon))

    def uniform_kernel_density(self, x):
        if np.abs(x) < 1:
            return 0.5
        else:
            return 0.0

    def calc_log_pi_epsilon(self, T_x):
        # eq. 1.2.3
        return np.log(self.uniform_kernel_density(self.calc_distance(T_x) / self.epsilon) + 1e-12) - np.log(self.epsilon)

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
        try:
            return scipy.stats.dirichlet.logpdf(p_prime_D, (p_D * scale) + 1)
        except:
            return scipy.stats.dirichlet.logpdf(p_prime_D + 1e-12, (p_D * scale) + 1) # in case some entry of p_prime_D is zero
        # TODO
        # if some entry of p_prime_D is zero AND some other entry is one, there will be an error

    def calc_log_proba_proposal_distribution_truncatednormal(self, mean, mean_prime, stddev, lower, upper):
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
    # try:
    df = pd.DataFrame(stats)
    df.to_csv(filename)
    # except:
    #     import pickle
    #     with open(filename.replace('.csv', '.pkl'), 'wb') as f:
    #         pickle.dump(stats, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hospital', default='south_tees_hospitals_nhs_foundation_trust_TrainingAndTesting')
    parser.add_argument('--config_template', default='NHS_data/new_data/formatted_data/configs/config') # template for config_file
    parser.add_argument('--input_template', default='NHS_data/new_data/formatted_data/')
    parser.add_argument('--output_template', default='NHS_results/TEST')
    parser.add_argument('--random_seed', default=101, type=int) # currently not using it 
    parser.add_argument('--algorithm', default='abc')
    parser.add_argument('--num_iterations', default=1, type=int) # number of sampling iterations 
                                                                   # each iteration has an inner loop through each probabilistic parameter vector
    parser.add_argument('--num_simulations', default=1, type=int)
    parser.add_argument('--start_epsilon', default=1.0, type=float)
    parser.add_argument('--lowest_epsilon', default=15.0, type=float) # unused in abc
    parser.add_argument('--annealing_constant', default=0.99999)
    parser.add_argument('--train_test_split', default=60) # currently an integer timestep, ultimately will be a string date
    parser.add_argument('--dir_scale', default='100-100') # scale parameter for the dirichlet proposal distribution (min-max, linearly interpolated)
    parser.add_argument('--lam_stddev', default='0.5-0.5') # stddev parameter for lambda (truncated normal) (min-max, linearly interpolated)
    parser.add_argument('--tau_stddev', default='0.1-0.1') # stddev parameter for tau (normal) (min-max, linearly interpolated)

    parser.add_argument('--params_init', default='None')
    parser.add_argument('--abc_prior_type', default='for_cdc_tableSpikedDurs')
    parser.add_argument('--abc_prior_config_template', default='toy_data_experiment/abc_prior_config')

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
    train_test_split = int(args.train_test_split)
    dir_scale = list(map(int, args.dir_scale.split('-')))
    lam_stddev = list(map(float, args.lam_stddev.split('-')))
    tau_stddev = list(map(float, args.tau_stddev.split('-')))
    algorithm = args.algorithm

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
    print(true_df)

    true_df = true_df[true_df['timestep'] <= train_test_split]

    # condensing the true results in a summary vector
    T_y = []
    for col_name in SUMMARY_STATISTICS_NAMES:
        T_y.append(true_df[col_name])
    T_y = np.asarray(T_y).flatten()

    # initalize sampler
    sampler = ABCSampler(seed, start_epsilon, lowest_epsilon, annealing_constant, T_y, train_test_split, config_dict, num_timesteps, num_simulations)

    sampler.initialize_theta(algorithm, abc_prior, params_init)
    print(sampler.theta_init)

    accepted_thetas, accepted_distances, num_accepted, all_distances, accepted_alphas, all_alphas, accepted_test_forecasts = sampler.draw_samples(algorithm, num_iterations, dir_scale, lam_stddev, tau_stddev)

    # num_to_save = int(len(accepted_thetas) * 0.5) # save only the second half of the parameters
    # last_thetas = [deepcopy(accepted_thetas[0])] + accepted_thetas[-num_to_save:] # also save the first theta
    last_thetas = accepted_thetas # also save the first theta
    sampler.save_thetas_to_json(last_thetas, thetas_output)

    print(len(sampler.epsilon_trace))
    print(len(accepted_distances))

    stats = {'all_distances': all_distances, 'accepted_distances': accepted_distances, 'all_alphas': all_alphas, 'accepted_alphas': accepted_alphas, 'epsilon_trace': sampler.epsilon_trace}
    save_stats_to_csv(stats, stats_output)

    test_forecasts_to_save = accepted_test_forecasts[-1000:]
    for i, df in enumerate(test_forecasts_to_save):
        df['index'] = i

    df = pd.concat(test_forecasts_to_save)
    df.to_csv(test_forecasts_output)