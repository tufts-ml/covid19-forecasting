import argparse
import scipy.stats
import numpy as np
import pandas as pd
import json

def summarize_pmf(pmf_dict):
    keys = np.fromiter(pmf_dict.keys(), dtype=np.float64)
    probas = np.fromiter(pmf_dict.values(), dtype=np.float64)
    dist = scipy.stats.rv_discrete(name='custom', values=(keys, probas))
    samps = dist.rvs(size=10000, random_state=0)
    return "mean %.1f | 1st%% %.1f   10th%% %.1f   90th%% %.1f   99th%% %.1f" % (
        dist.mean(),
        np.percentile(samps, 1),
        np.percentile(samps, 10),
        np.percentile(samps, 90),
        np.percentile(samps, 99))


def pprint_params(params_dict):
    states = params_dict['states']
    next_state_map = dict()
    for ss, state in enumerate(states):
        if ss < len(states) - 1:
            next_state_map[state] = states[ss+1]
        else:
            next_state_map[state] = 'TERMINAL'

        p_recover = params_dict["proba_Recovering_given_%s" % state]
        p_decline = 1.0 - p_recover

        print("State #%d %s" % (ss, state))
        print("    prob. %.3f recover" % (p_recover))
        print("    prob. %.3f advance to state %s" % (p_decline, next_state_map[state]))
        print("    Recovering Duration: %s" % (
            summarize_pmf(params_dict['pmf_duration_Recovering_%s' % state]))) 
        print("    Declining Duration:  %s" % (
            summarize_pmf(params_dict['pmf_duration_Declining_%s' % state]))) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--params_json_file',
        help="Path to JSON file with simulation parameters",
        default=None)
    args = parser.parse_args()

    with open(args.params_json_file, 'r') as f:
        params = json.load(f)
    print("----------------------------------------")
    pprint_params(params)
    print("----------------------------------------")




