import argparse
import json
import os
import sys
import numpy as np
import itertools
import time
from simulate_traj__cython import simulate_traj__cython

def simulate_many_traj__python(
        P=1,
        F=1,
        init_day_neg_ids_P,
        init_counts_PK,
        admit_counts_F,
        **sim_kwargs):
    ''' Simulate trajectory of many patients

    Args
    ----
    P : int, >= 1
        Number of previous days to track including t=0 ("first day").
    F : int, >= 1
        Number of days to produce forecasts for, including t=1, 2, ... F
    admit_counts_F : 1D array, size F
        Number of patients to add at stage 0 for each day
    init_counts_PK : 2D array, size (P, K)

    Returns
    -------
    '''
    F = admit_counts_F.size
    P2, K = init_counts_PK.shape
    assert P == P2

    cushion = 40
    T = P + F + cushion
    occupancy_count_TK = np.zeros((T, K))
    terminal_count_T1 = np.zeros((T))

    L = K * 2
    M = L * 2 + 1

    # Draw many possible random values needed in advance
    rand_vals_M = prng.rand(1000 * M)

    ## Simulate what happens to initial population
    init_day_ids_P = P + init_day_neg_ids_P
    assert init_day_ids_P.min() == 0
    assert init_day_ids_P.max() < P
    for pp, tstart in enumerate(init_day_ids_P):
        for kk, Ncur in enumerate(range(init_counts_PK[pp])):
            for _ in range(Ncur):

                if 'durations_L' not in sim_kwargs:
                    sim_kwargs['durations_L'] = np.zeros(L, dtype=np.int32)
                    sim_kwargs['stage_ids_L'] = -99 * np.ones(L, dtype=np.int32)
                    sim_kwargs['health_ids_L'] = -99 * np.ones(L, dtype=np.int32)
                else:
                    # Reuse existing memory!
                    sim_kwargs['durations_L'][:] = 0
                    sim_kwargs['stage_ids_L'][:] = -99
                    sim_kwargs['health_ids_L'][:] = -99

            mm, ll, is_terminal = simulate_traj(
                start_stage=kk, rand_vals_M=rand_vals_M, **sim_kwargs)
            rand_vals_M = rand_vals_M[mm:] # jump ahead to next rand value
            if rand_vals_M.size < M:
                rand_vals_M = prng.rand(1000 * M)

            D_L = np.cumsum(sim_kwargs['durations_L'][:ll])
            for l, k in enumerate(sim_kwargs['stage_ids_L']):
                occupancy_count_TK[(tstart + D_L[l]):(tstart + D_L[l+1]), k] += 1
            if is_terminal:
                terminal_count_T1[tstart + D] = 1
            else:
                discharge_count_T1[tstart + D] = 1

    occupancy_count_TK = occupancy_count_TK[P:(P+F)]
    terminal_count_T1 = terminal_count_T1[P:(P+F)]
    discharge_count_T1 = discharge_count_T1[P:(P+F)]
    return (occupancy_count_TK, terminal_count_T1, discharge_count_T1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1001, type=int)
    parser.add_argument('--n_trials', default=3, type=int)
    parser.add_argument('--func_name', default='python', type=str)
    parser.add_argument('--params_json_file', default=None)
    args = parser.parse_args()

    with open(args.params_json_file, 'r') as f:
        config_dict = json.load(f)

    stage_names = ['InGeneralWard', 'OffVentInICU', 'OnVentInICU']
    health_ids = [0, 1]

    HEALTH_STATE_ID_TO_NAME = {0: 'Declining', 1: 'Recovering'}

    H = len(health_ids)
    K = len(stage_names)

    Tmax = 25
    duration_cdf_HKT = np.zeros((H, K, Tmax))
    for health, stage in itertools.product(health_ids, np.arange(K)):
        choices_and_probas_dict = config_dict[
            'pmf_duration_%s_%s' % (
                HEALTH_STATE_ID_TO_NAME[health], stage_names[stage])]
        choices = np.fromiter(choices_and_probas_dict.keys(), dtype=np.int32)
        probas = np.fromiter(choices_and_probas_dict.values(), dtype=np.float64)
        for c, p in zip(choices, probas):
            assert c >= 1
            duration_cdf_HKT[health, stage, c - 1] = p
        duration_cdf_HKT[health, stage, :] = np.cumsum(duration_cdf_HKT[health, stage, :])

    sim_kwargs = dict(
        start_stage=0)
    sim_kwargs['duration_cdf_HKT'] = duration_cdf_HKT
    sim_kwargs['pRecover_K'] = np.asarray([
        float(config_dict['proba_Recovering_given_%s' % stage])
        for stage in stage_names])
    sim_kwargs['pDieAfterDeclining_K'] = np.asarray([
        float(config_dict.get('proba_Die_after_Declining_%s' % stage, 0.0))
        for stage in stage_names])

    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    for k, v in sim_kwargs.items():
        if isinstance(v, np.ndarray):
            print(k)
            print('-' * len(k))
            print(v)
            print()

    L = K * 2
    M = L * 2 + 1
    prng = np.random.RandomState(args.seed)
    start_time_sec = time.time()
    for trial in range(args.n_trials):
        # Draw all possible random values needed in advance
        rand_vals_M = prng.rand(M)

        if trial == 0:
            sim_kwargs['durations_L'] = np.zeros(L, dtype=np.int32)
            sim_kwargs['stage_ids_L'] = -99 * np.ones(L, dtype=np.int32)
            sim_kwargs['health_ids_L'] = -99 * np.ones(L, dtype=np.int32)
        else:
            sim_kwargs['durations_L'][:] = 0
            sim_kwargs['stage_ids_L'][:] = -99
            sim_kwargs['health_ids_L'][:] = -99

        if args.func_name.count("cython"):
            mm, curL, is_terminal = simulate_traj__cython(rand_vals_M=rand_vals_M, **sim_kwargs)
        else:
            mm, curL, is_terminal, _, _, _ = simulate_traj__python(rand_vals_M=rand_vals_M, **sim_kwargs)

        #print(is_terminal, sim_kwargs['durations_L'][:curL], sim_kwargs['stage_ids_L'][:curL])
    elapsed_time_sec = time.time() - start_time_sec
    print("Finished %d trials after %9.3f sec" % (args.n_trials, elapsed_time_sec))
    print("Last sim durations: " + str(sim_kwargs['durations_L']))
        