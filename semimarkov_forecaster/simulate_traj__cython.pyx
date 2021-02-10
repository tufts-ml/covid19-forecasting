import argparse
import json
import os
import sys
import numpy as np
import itertools

def simulate_traj__cython(
        int start_stage,
        double[:] pRecover_K,
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
            stage -= 1
        else:
            stage += 1
        if stage >= MAX_STAGE:
            is_terminal = 1
            break

    return (mm, ll, is_terminal)
