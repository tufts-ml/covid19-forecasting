import argparse
import json
import os
import sys
import numpy as np
import itertools
import time
from simulate_traj__cython import simulate_traj__cython

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
            mm += 1
            if rand_vals_M[mm] < pDieAfterDeclining_K[stage]:
                is_terminal = 1
                break
        stage = next_stage

            
    return (mm, ll, int(health < 0), stage_ids_L, health_ids_L, durations_L)


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
        