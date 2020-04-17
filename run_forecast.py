import os
import json
import argparse
import numpy as np
import scipy.stats
import pandas as pd
import tqdm
import warnings
from semimarkov_forecaster import PatientTrajectory

def run_simulation(random_seed, output_file, config_dict, states, state_name_to_id, next_state_map, age_groups, age_group_to_id):
    print("random_seed=%s <<<" % random_seed)
    prng = np.random.RandomState(random_seed)
    T = config_dict['num_timesteps']
    K = len(states) # Num states (not including terminal)
    A = len(age_groups)

    ## Preallocate admit, discharge, and occupancy
    Tmax = 10 * T
    occupancy_count_TK = np.zeros((Tmax, K), dtype=np.float64)
    admit_count_TK = np.zeros((Tmax, K), dtype=np.float64)
    discharge_count_TK = np.zeros((Tmax, K), dtype=np.float64)
    terminal_count_T1 = np.zeros((Tmax, 1), dtype=np.float64)
    
    ## age dependent pre-allocation
    occupancy_count_TKA = np.zeros((Tmax, K, A), dtype=np.float64)
    admit_count_TKA = np.zeros((Tmax, K, A), dtype=np.float64)
    discharge_count_TKA = np.zeros((Tmax, K, A), dtype=np.float64)
    terminal_count_T1A = np.zeros((Tmax, 1, A), dtype=np.float64)

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
                        warnings.warn("No matching timestep found in file %s. Assuming 0" % (pmfstr_or_csvfile)) 
                        return 0
                if len(row_ids) > 1:
                    raise ValueError("Error in file %s: Must have exactly one matching timestep for t=%d" % (
                        pmfstr_or_csvfile, t))
                return np.array(csv_df['num_%s' % state].values[row_ids[0]], dtype=int) # guard against float input. 
            sample_func_per_state[state] = sample_incoming_count
        # Parsing failed!
        else:
            raise ValueError("Bad PMF specification: %s" % pmfstr_or_csvfile)

    print("----------------------------------------")
    print("Simulating for %d timesteps with seed %d" % (T, args.random_seed))
    print("----------------------------------------")
    ## Simulation what happens to initial population
    for initial_state in states:
        for n in range(config_dict['num_%s' % initial_state]):
            p = PatientTrajectory(initial_state, config_dict, prng, next_state_map, state_name_to_id)
            occupancy_count_TK = p.update_count_matrix(occupancy_count_TK, 0)
            admit_count_TK = p.update_admit_count_matrix(admit_count_TK, 0)
            discharge_count_TK = p.update_discharge_count_matrix(discharge_count_TK, 0)
            terminal_count_T1 = p.update_terminal_count_matrix(terminal_count_T1, 0)
            
            # Uncomment if you want to track counts by age
            occupancy_count_TKA = p.update_count_by_age_matrix(occupancy_count_TKA, 0)
            admit_count_TKA = p.update_admit_count_by_age_matrix(admit_count_TKA, 0)
            discharge_count_TKA = p.update_discharge_count_by_age_matrix(discharge_count_TKA, 0)
            terminal_count_T1A = p.update_terminal_count_by_age_matrix(terminal_count_T1A, 0)
            
    ## Simulation what happens as new patients added at each step
    for t in tqdm.tqdm(range(1, T+1)):
        for state in states:
            if state not in sample_func_per_state:
                continue
            sample_incoming_count = sample_func_per_state[state]
            N_t = sample_incoming_count(t, prng)
            for n in range(N_t):
                p = PatientTrajectory(state, config_dict, prng, next_state_map, state_name_to_id)
                occupancy_count_TK = p.update_count_matrix(occupancy_count_TK, t)        
                admit_count_TK = p.update_admit_count_matrix(admit_count_TK, t)
                discharge_count_TK = p.update_discharge_count_matrix(discharge_count_TK, t)
                terminal_count_T1 = p.update_terminal_count_matrix(terminal_count_T1, t)
                
                # Uncomment if you want to track counts by age
                occupancy_count_TKA = p.update_count_by_age_matrix(occupancy_count_TKA, t)
                admit_count_TKA = p.update_admit_count_by_age_matrix(admit_count_TKA, t)
                discharge_count_TKA = p.update_discharge_count_by_age_matrix(discharge_count_TKA, t)
                terminal_count_T1A = p.update_terminal_count_by_age_matrix(terminal_count_T1A, t)

    # Save only the first T + 1 tsteps (with index 0, 1, 2, ... T)
    occupancy_count_TK = occupancy_count_TK[:(T+1)]
    admit_count_TK = admit_count_TK[:(T+1)]
    discharge_count_TK = discharge_count_TK[:(T+1)]
    terminal_count_T1 = terminal_count_T1[:(T+1)]
    
    # sanity check, all counts summed over age groups must be equal to total counts
#     occupancy_count_TKA = occupancy_count_TKA[:(T+1)]
#     admit_count_TKA = admit_count_TKA[:(T+1)]
#     discharge_count_TKA = discharge_count_TKA[:(T+1)]
#     terminal_count_T1A = terminal_count_T1A[:(T+1)]
    
#     occupancy_count_TK_age_sum = np.sum(occupancy_count_TKA, axis=2)
#     admit_count_TK_age_sum = np.sum(admit_count_TKA, axis=2)
#     discharge_count_TK_age_sum = np.sum(discharge_count_TKA, axis=2)
#     terminal_count_TK_age_sum = np.sum(terminal_count_T1A, axis=2)
#     from IPython import embed; embed()
    
    print("----------------------------------------------")
    print("Printing Age Group Counts for 5 time steps")
    print("----------------------------------------------")
    for t in range(5):
        print("----------------------------------------------")
        print('Time Step : %d'%t)
        print("----------------------------------------------")
        age_counts_dict = dict()
        for age_idx,age_group in enumerate(age_groups):
            age_counts_dict['occupancy_%s'%age_group]=occupancy_count_TKA[t,:,age_idx]
        age_counts_dict['occupancy_total']=occupancy_count_TK[t,:]
#             age_counts_dict['admitted_%s'%age_group]=admit_count_TKA[t,:,age_idx]
#             age_counts_dict['discharged_%s'%age_group]=discharge_count_TKA[t,:,age_idx]
#             age_counts_dict['terminal_%s'%age_group]=terminal_count_T1A[t,0,age_idx]
        print(age_counts_dict)
    #         for state_idx,state in enumerate(states): 

            
            
    ## Write results to spreadsheet
    print("----------------------------------------")
    print("Writing results to %s" % (output_file))
    print("----------------------------------------")
    col_names = ['n_%s' % s for s in states]
    results_df = pd.DataFrame(occupancy_count_TK, columns=col_names)
    results_df["timestep"] = np.arange(0, T+1)

    results_df["n_TERMINAL"] = terminal_count_T1[:,0]

    admit_col_names = ['n_admitted_%s' % s for s in states]
    for k, col_name in enumerate(admit_col_names):
        results_df[col_name] = admit_count_TK[:, k]

    discharge_col_names = ['n_discharged_%s' % s for s in states]
    for k, col_name in enumerate(discharge_col_names):
        results_df[col_name] = discharge_count_TK[:, k]

    results_df.to_csv(output_file,
        columns=['timestep'] + col_names + ['n_TERMINAL'] + admit_col_names + discharge_col_names,
        index=False, float_format="%.0f")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default=None)
    parser.add_argument('--output_file', default='results.csv')
    parser.add_argument('--random_seed', default=101, type=int)
    parser.add_argument('--num_seeds', default=1, type=int)

    args, unknown_args = parser.parse_known_args()

    unk_keys = map(lambda s: s[2:].strip(), unknown_args[::2])
    unk_vals = unknown_args[1::2]
    unk_dict = dict(zip(unk_keys, unk_vals))

    config_file = args.config_file
    output_file = args.output_file

    ## Load JSON
    with open(config_file, 'r') as f:
        config_dict = json.load(f)

    for key in config_dict:
        print(key, config_dict[key])

    ## Summarize the probabilistic model
    print("----------------------------------------")
    print("Loaded SemiMarkovModel from config_file:")
    print("----------------------------------------")
    states = config_dict['states']
    age_groups = config_dict['age_groups']
    state_name_to_id = dict()
    next_state_map = dict()
    age_group_to_id=dict()
    for ss, state in enumerate(states):
        state_name_to_id[state] = ss
        if ss < len(states) - 1:
            next_state_map[state] = states[ss+1]
        else:
            next_state_map[state] = 'TERMINAL'
        p_recover = config_dict["proba_Recovering_given_%s" % state]
        p_decline = 1.0 - p_recover
        
    for ss, age_group in enumerate(age_groups):
        age_group_to_id[age_group] = ss

        print("State #%d %s" % (ss, state))
        print("    prob. %.3f recover" % (p_recover))
        print("    prob. %.3f advance to state %s" % (p_decline, next_state_map[state]))
    state_name_to_id['TERMINAL'] = len(states)

    output_file_base = output_file
    for seed in range(args.random_seed, args.random_seed + args.num_seeds):
        output_file = output_file_base.replace("random_seed=%s" % args.random_seed, "random_seed=%s" % str(seed))
        run_simulation(seed, output_file, config_dict, states, state_name_to_id, next_state_map, age_groups, age_group_to_id)


