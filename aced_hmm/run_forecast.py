import numpy as np
import pandas as pd
import itertools
import argparse
import json
import os
import sys
import time
from tqdm import tqdm
import scipy
from aced_hmm.simulator import run_forecast__python
from aced_hmm.print_parameters import pprint_params

def run_simulation(random_seed, output_file, config_dict, states, func_name, approximate=None):
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
        state_key = 'n_admitted_%s' % state
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
        from aced_hmm.simulator import run_forecast__cython
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

    results_df = results_df[results_df['timestep'] >= 0]

    results_df.to_csv(output_file,
        columns=['timestep'] + col_names + ['n_TERMINAL'] + discharge_col_names,
        index=False, float_format="%.0f")

def update_config_given_sample(config_dict, parameters, i):
    num_thetas = len(parameters['proba_Recovering_given_InGeneralWard'])
    index = num_thetas - 1 - i
    for param_name in parameters:
        if 'duration' in param_name:
            config_dict[param_name] = {}
            for choice in parameters[param_name]:
                if choice not in ['lam', 'tau']:
                    config_dict[param_name][choice] = parameters[param_name][choice][index]
        else:
            config_dict[param_name] = parameters[param_name][index]
    return config_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--func_name', default='python', type=str)
    parser.add_argument('--config_path', default='results/US/MA-20201111-20210111-20210211/config_after_abc.json')
    parser.add_argument('--output_dir', default='results/US/MA-20201111-20210111-20210211/individual_forecasts')
    parser.add_argument('--output_file', default='results_after_abc-{{random_seed}}.csv')
    parser.add_argument('--approximate', default='5')
    parser.add_argument('--random_seed', default=1001, type=int)
    parser.add_argument('--num_seeds', default=None) # None value here defaults to 1 when running with fixed parameters, 
                                                     # and to total number of samples in the samples file when running from multiple samples
    args = parser.parse_args()

    if args.approximate == 'None' or args.approximate is None:
        approximate = None
    else:
        approximate = int(args.approximate)

    if args.num_seeds == 'None' or args.num_seeds is None:
        num_seeds = None
    else:
        num_seeds = int(args.num_seeds)

    with open(args.config_path, 'r') as f:
        config_dict = json.load(f)

    with open(config_dict['parameters'], 'r') as f:
        parameters = json.load(f)

    ###############

    if isinstance(parameters['proba_Recovering_given_InGeneralWard'], list):
        print('Forecasting from samples ...')
        run_from_samples = True

        if num_seeds is None:
            num_seeds = len(parameters['proba_Recovering_given_InGeneralWard'])

        if len(parameters['proba_Recovering_given_InGeneralWard']) < num_seeds:
            print('Too many samples requested: there are %d available samples, you have requested %d.\nExiting.' % (len(parameters['proba_Recovering_given_InGeneralWard']), num_seeds))
            exit(1)

        # initialize parameters in config dictionary
        for param_name in parameters:
            config_dict[param_name] = None # can just be None here, it will change dynamically

        print('Using %d parameter samples, each with a distinct random seed.' % (num_seeds))
    else:
        print('Forecasting with fixed parameters ...')
        run_from_samples = False

        if num_seeds is None:
            num_seeds = 1

        # initialize parameters in config dictionary
        for param_name in parameters:
            config_dict[param_name] = parameters[param_name]

        print('Using %d random seeds to generate simulations using the following parameters:' % (num_seeds))
        pprint_params(config_dict)

    if not os.path.exists(args.output_dir): # creates directory if it does not exist
        from pathlib import Path
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    output_file_base = os.path.join(args.output_dir, args.output_file)
    states = config_dict['states']
    print('--------------------------------------------')
    print('           Running %d simulations' % (num_seeds))
    print('--------------------------------------------')
    for i in tqdm(range(num_seeds)):
        seed = int(args.random_seed) + i
        output_file = output_file_base.replace("{{random_seed}}", "random_seed=%s" % str(seed))

        if run_from_samples:
            config_dict = update_config_given_sample(config_dict, parameters, i)

        run_simulation(seed, output_file, config_dict, states, args.func_name, approximate=approximate)
