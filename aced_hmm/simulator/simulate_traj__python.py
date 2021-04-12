import argparse
import json
import os
import sys
import numpy as np
import itertools
import pandas as pd
from scipy import stats

def simulate_traj__python(
        start_stage,
        pRecover_K,
        pDieAfterDeclining_K,
        duration_cdf_HKT,
        stage_ids_L,
        health_ids_L,
        durations_L,
        rand_vals_M,
        tweak_durs=False):
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

        if tweak_durs and health > 0 and d > 1:
            p_one = duration_cdf_HKT[health, stage, 0]
            r = 1/(1-p_one) - 1
            value = d * (1.0 - 0.25*(1+r))
            integer = np.floor(value).astype(np.int32)
            decimals = value - integer
            d = integer + stats.bernoulli.rvs(decimals)

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
        init_num_per_state_Tpastplus1K,
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
            N_new = init_num_per_state_Tpastplus1K[t, state]

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

    tweak_durs = False
    ## Simulate what happens as new patients added at each step
    for t in range(1, T+1):
        for state in states:
            N_t = admissions_per_state_Tplus1K[t, state]
            for n in range(N_t):
                rand_vals_M = rand_vals_M[mm:]

                durations_L[:] = 0
                stage_ids_L[:] = -99
                health_ids_L[:] = -99

                # if t > 61:
                #     # tweak_durs = True

                mm, curL, is_terminal, _, _, _ = simulate_traj__python(state, pRecover_K, pDieAfterDeclining_K, duration_cdf_HKT, stage_ids_L, health_ids_L, durations_L, rand_vals_M, tweak_durs)

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
