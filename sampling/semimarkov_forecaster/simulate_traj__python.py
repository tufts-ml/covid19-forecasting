import argparse
import json
import os
import sys
import numpy as np
import itertools
import pandas as pd

def simulate_traj__python(
        start_stage,
        pRecover_K,
        pDieAfterDeclining_K,
        duration_cdf_HKT,
        stage_ids_L,
        health_ids_L,
        durations_L,
        rand_vals_M,
        ):
    ''' Simulate trajectory of a patient, using pure python numpy code
    Returns
    -------
    n_rand_used : int
        Number of random values used
    n_steps : int
        Number of distinct stage / health-state combinations
    is_terminal : int
        Indicates if terminal health state reached.
    stage_ids_L : 1D array, shape (L,)
        Integer identifier for each distinct stage traversed
    health_ids_L : 1D array, shape (L,)
        Integer identifier for each distinct health state traversed
    durations_L : 1D array, shape (L,)
        Integer identifier for duration spent in each stage, health combo
    '''
    MAX_STAGE = pRecover_K.size
    Dmax = duration_cdf_HKT.shape[2]

    stage = start_stage
    health = 0
    mm = 0
    ll = 0
    is_terminal = 0
    while (0 <= stage) and (stage < MAX_STAGE):
        if health == 0:
            health = int(rand_vals_M[mm] < pRecover_K[stage])
            mm += 1

        for d in range(1, Dmax):
            if rand_vals_M[mm] < duration_cdf_HKT[health, stage, d - 1]:
                break
        mm += 1

        durations_L[ll] = d
        stage_ids_L[ll] = stage
        health_ids_L[ll] = health
        ll += 1

        if health > 0:
            next_stage = stage - 1
        else:
            next_stage = stage + 1
        if next_stage >= MAX_STAGE:
            is_terminal = 1
            break

        if health == 0:
            if rand_vals_M[mm] < pDieAfterDeclining_K[stage]:
                is_terminal = 1
                mm += 1
                break
            mm += 1
        stage = next_stage

            
    return (mm, ll, is_terminal, stage_ids_L, health_ids_L, durations_L)

def run_forecast__python(
        Tpast,
        T,
        Tmax,
        states,
        occupancy_count_TK,
        discharge_count_TK,
        terminal_count_T1,
        rand_vals_M,
        init_num_per_state_K,
        admissions_per_state_Tplus1K,
        pRecover_K,
        pDieAfterDeclining_K,
        duration_cdf_HKT,
        stage_ids_L,
        health_ids_L,
        durations_L):

    mm = 0
    ## Simulate what happens to initial population
    for t in range(-Tpast, 1, 1):
        for state in states:
            N_new = init_num_per_state_K[state]

            for n in range(N_new):
                rand_vals_M = rand_vals_M[mm:]

                durations_L[:] = 0
                stage_ids_L[:] = -99
                health_ids_L[:] = -99

                mm, curL, is_terminal, _, _, _ = simulate_traj__python(state, pRecover_K, pDieAfterDeclining_K, duration_cdf_HKT, stage_ids_L, health_ids_L, durations_L, rand_vals_M)

                tp = Tpast + t
                for ii in range(curL):
                    occupancy_count_TK[tp:tp+durations_L[ii], stage_ids_L[ii]] += 1
                    tp += durations_L[ii]

                t_start = Tpast + t
                if not is_terminal:
                    discharge_count_TK[t_start + np.sum(durations_L[:curL]), stage_ids_L[:curL][-1]] += 1

                t_start = Tpast + t
                if is_terminal:
                    t_terminal = t_start + np.sum(durations_L[:curL])
                    terminal_count_T1[t_terminal, 0] += 1

    ## Simulate what happens as new patients added at each step
    for t in range(1, T+1):
        for state in states:
            N_t = admissions_per_state_Tplus1K[t, state]
            for n in range(N_t):
                rand_vals_M = rand_vals_M[mm:]

                durations_L[:] = 0
                stage_ids_L[:] = -99
                health_ids_L[:] = -99

                mm, curL, is_terminal, _, _, _ = simulate_traj__python(state, pRecover_K, pDieAfterDeclining_K, duration_cdf_HKT, stage_ids_L, health_ids_L, durations_L, rand_vals_M)

                tp = Tpast + t
                for ii in range(curL):
                    occupancy_count_TK[tp:tp+durations_L[ii], stage_ids_L[ii]] += 1
                    tp += durations_L[ii]

                t_start = Tpast + t
                if not is_terminal:
                    discharge_count_TK[t_start + np.sum(durations_L[:curL]), stage_ids_L[:curL][-1]] += 1

                t_start = Tpast + t
                if is_terminal:
                    t_terminal = t_start + np.sum(durations_L[:curL])
                    terminal_count_T1[t_terminal, 0] += 1

    return occupancy_count_TK, discharge_count_TK, terminal_count_T1

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
