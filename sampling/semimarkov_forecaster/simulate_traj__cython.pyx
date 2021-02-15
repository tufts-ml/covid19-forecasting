import argparse
import json
import os
import sys
import numpy as np
import itertools

def simulate_traj__cython(
        int start_stage,
        double[:] pRecover_K,
        double[:] pDieAfterDeclining_K,
        double[:,:,:] duration_cdf_HKT,
        int[:] stage_ids_L,
        int[:] health_ids_L,
        int[:] durations_L,
        double[:] rand_vals_M,
        ):
    """ Faster cython version to simulate trajectory
    Returns
    -------
    """
    cdef int MAX_STAGE = pRecover_K.shape[0]
    cdef int Dmax = duration_cdf_HKT.shape[2]
    cdef int stage = start_stage
    cdef int next_stage = 0
    cdef int health = 0

    cdef int mm = 0
    cdef int ll = 0
    cdef int d = 0
    cdef int is_terminal = 0
    while (0 <= stage) and (stage < MAX_STAGE):
        if health == 0:
            health = rand_vals_M[mm] < pRecover_K[stage]
            mm += 1

        for d in range(1, Dmax):
            if rand_vals_M[mm] < duration_cdf_HKT[health, stage, d-1]:
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
            mm += 1
            if rand_vals_M[mm] < pDieAfterDeclining_K[stage]:
                is_terminal = 1
                break

        # Advance to next step of simulation
        stage = next_stage

    return (mm, ll, is_terminal)

def run_forecast__cython(
        int Tpast,
        int T,
        int Tmax,
        int[:] states,
        double[:,:] occupancy_count_TK,
        double[:,:] discharge_count_TK,
        double[:,:] terminal_count_T1,
        double[:] rand_vals_M,
        int[:] init_num_per_state_K,
        int[:,:] admissions_per_state_Tplus1K,
        double[:] pRecover_K,
        double[:] pDieAfterDeclining_K,
        double[:,:,:] duration_cdf_HKT,
        int[:] stage_ids_L,
        int[:] health_ids_L,
        int[:] durations_L,):

    cdef int mm = 0
    cdef int N_new = 0
    cdef int n = 0
    cdef int curL = 0
    cdef int is_terminal = 0
    cdef int t = 0
    cdef int tp = 0
    cdef int ii = 0
    cdef int t_start = 0
    cdef int t_terminal = 0
    cdef int state = 0

    cdef int jj = 0
    ## Simulate what happens to initial population
    for t in range(-Tpast, 1, 1):
        for state in states:
            N_new = init_num_per_state_K[state]

            for n in range(N_new):
                rand_vals_M = rand_vals_M[mm:]

                durations_L[:] = 0
                stage_ids_L[:] = -99
                health_ids_L[:] = -99

                mm, curL, is_terminal = simulate_traj__cython(state, pRecover_K, pDieAfterDeclining_K, duration_cdf_HKT, stage_ids_L, health_ids_L, durations_L, rand_vals_M)

                tp = Tpast + t
                for ii in range(curL):
                    for jj in range(durations_L[ii]):
                        occupancy_count_TK[tp+jj, stage_ids_L[ii]] += 1
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

                mm, curL, is_terminal = simulate_traj__cython(state, pRecover_K, pDieAfterDeclining_K, duration_cdf_HKT, stage_ids_L, health_ids_L, durations_L, rand_vals_M)

                tp = Tpast + t
                for ii in range(curL):
                    for jj in range(durations_L[ii]):
                        occupancy_count_TK[tp+jj, stage_ids_L[ii]] += 1
                    tp += durations_L[ii]

                t_start = Tpast + t
                if not is_terminal:
                    discharge_count_TK[t_start + np.sum(durations_L[:curL]), stage_ids_L[:curL][-1]] += 1

                t_start = Tpast + t
                if is_terminal:
                    t_terminal = t_start + np.sum(durations_L[:curL])
                    terminal_count_T1[t_terminal, 0] += 1

    return occupancy_count_TK, discharge_count_TK, terminal_count_T1