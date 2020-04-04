import json
import argparse
import numpy as np
import scipy.stats
import pandas as pd
import tqdm
from semimarkov_forecaster import PatientTrajectory

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default=None)
    parser.add_argument('--output_file', default='results.csv')
    parser.add_argument('--random_seed', default=101, type=int)
    parser.add_argument('--presenting_list', default = None)
    
    

    args = parser.parse_args()

    use_presenting_list = False
    if not args.presenting_list is None:
        
        print("Reading Presenting List...")
        presenting = pd.read_csv(args.presenting_list)
        try:
            presenting = presenting["hospitalized"].values
            use_presenting_list = True
        except KeyError:
            print("No column named 'hospitalized'. Exiting....")
            use_presenting_list = False
            exit()

        print("Done")
    else:
        print("No presenting List provided, using config settings...")

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
    state_name_to_id = dict()
    next_state_map = dict()
    for ss, state in enumerate(states):
        state_name_to_id[state] = ss
        if ss < len(states) - 1:
            next_state_map[state] = states[ss+1]
        else:
            next_state_map[state] = 'TERMINAL'
        p_recover = config_dict["proba_Recovering_given_%s" % state]
        p_decline = 1.0 - p_recover

        print("State #%d %s" % (ss, state))
        print("    prob. %.3f recover" % (p_recover))
        print("    prob. %.3f advance to state %s" % (p_decline, next_state_map[state]))
    state_name_to_id['TERMINAL'] = len(states)

    prng = np.random.RandomState(args.random_seed)
    T = config_dict['num_timesteps']
    K = len(states) + 1 # Num states including terminal
    occupancy_count_TK = np.zeros((10 * T, K), dtype=np.float64)
    discharge_count_TK = np.zeros((10 * T, K), dtype=np.float64)


    print("----------------------------------------")
    print("Simulating for %d timesteps with seed %d" % (T, args.random_seed))
    print("----------------------------------------")
    ## Simulation what happens to initial population
    for initial_state in states:
        for n in range(config_dict['num_%s' % initial_state]):
            p = PatientTrajectory(initial_state, config_dict, prng, next_state_map, state_name_to_id)
            occupancy_count_TK = p.update_count_matrix(occupancy_count_TK, 0)
            discharge_count_TK = p.update_discharge_count_matrix(discharge_count_TK, 0)

    ## Simulation what happens as new patients added at each step
    for t in tqdm.tqdm(range(1, T+1)):
        for state in states:
            try:
                pmf = config_dict['pmf_num_per_timestep_%s' % states[0]]
                if pmf.startswith('scipy.stats') and len(pmf) < 40:
                    pmf = eval(pmf)
            except KeyError:
                continue
            except ValueError:
                raise ValueError("Bad PMF")
            
            N=0
            if not use_presenting_list:
                # Sample the number entering at current state at current timestep
                N = pmf.rvs(random_state=prng)
            else:
                
                N=int(presenting[t-1])
            
            for n in range(N):
                p = PatientTrajectory(states[0], config_dict, prng, next_state_map, state_name_to_id)
                occupancy_count_TK = p.update_count_matrix(occupancy_count_TK, t)        
                discharge_count_TK = p.update_discharge_count_matrix(discharge_count_TK, t)

    occupancy_count_TK = occupancy_count_TK[:(T+1)] # Save only the first T + 1 tsteps
    discharge_count_TK = discharge_count_TK[:(T+1)] # Save only the first T + 1 tsteps


    ## Write results to spreadsheet
    print("----------------------------------------")
    print("Writing results to %s" % (args.output_file))
    print("----------------------------------------")
    col_names = ['n_%s' % s for s in states + ['TERMINAL']]
    results_df = pd.DataFrame(occupancy_count_TK, columns=col_names)
    results_df["timestep"] = np.arange(0, T+1)

    discharge_col_names = ['n_discharged_%s' % s for s in states]
    for k, col_name in enumerate(discharge_col_names):
        results_df[col_name] = discharge_count_TK[:, k]

    results_df.to_csv(args.output_file,
        columns=['timestep'] + col_names + discharge_col_names,
        index=False, float_format="%.0f")
