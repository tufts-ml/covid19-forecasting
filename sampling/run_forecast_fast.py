import numpy as np
import pandas as pd
import itertools
import argparse
import json
import os
import sys
import time
from semimarkov_forecaster import run_forecast__python, run_forecast__cython

def run_simulation(random_seed, output_file, config_dict, states, func_name):
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
    admissions_per_state_Tplus1K = np.zeros((T+1, K), dtype=int)
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
        admissions_per_state_Tplus1K[:, ss] = np.array(csv_df[state_key])

    init_num_per_state_K = np.zeros(K, dtype=int)
    for ss, state in enumerate(states):
        init_num_per_state_K[ss] = config_dict['init_num_%s' % state]

    L = K * 2
    M = (np.sum(admissions_per_state_Tplus1K) + np.sum(init_num_per_state_K)) * L * 2 + 1
    prng = np.random.RandomState(random_seed)
    rand_vals_M = prng.rand(M)

    sim_kwargs['durations_L'] = np.zeros(L, dtype=np.int32)
    sim_kwargs['stage_ids_L'] = -99 * np.ones(L, dtype=np.int32)
    sim_kwargs['health_ids_L'] = -99 * np.ones(L, dtype=np.int32)
    sim_kwargs['occupancy_count_TK'] = occupancy_count_TK
    sim_kwargs['discharge_count_TK'] = discharge_count_TK
    sim_kwargs['terminal_count_T1'] = terminal_count_T1
    sim_kwargs['init_num_per_state_K'] = init_num_per_state_K
    sim_kwargs['admissions_per_state_Tplus1K'] = admissions_per_state_Tplus1K
    states_by_id = np.array([0, 1, 2])
    
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

    ## Write results to spreadsheet
    col_names = ['n_%s' % s for s in states]
    results_df = pd.DataFrame(occupancy_count_TK, columns=col_names)
    results_df["timestep"] = np.arange(-Tpast, -Tpast + tf)
    results_df["n_TERMINAL"] = terminal_count_T1[:,0]

    discharge_col_names = ['n_discharged_%s' % s for s in states]
    for k, col_name in enumerate(discharge_col_names):
        results_df[col_name] = discharge_count_TK[:, k]

    results_df.to_csv(output_file,
        columns=['timestep'] + col_names + ['n_TERMINAL'] + discharge_col_names,
        index=False, float_format="%.0f")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=102, type=int)
    parser.add_argument('--func_name', default='python', type=str)
    parser.add_argument('--params_json_file', default='MA_data/config_MA_test_dataset.json')
    parser.add_argument('--output_file', default='results_test_fast_python.csv')
    args = parser.parse_args()

    with open(args.params_json_file, 'r') as f:
        config_dict = json.load(f)

    states = config_dict['states']

    start_time_sec = time.time()
    run_simulation(args.seed, args.output_file, config_dict, states, args.func_name)
    elapsed_time_sec = time.time() - start_time_sec
    print("Finished after %9.3f seconds." % (elapsed_time_sec))