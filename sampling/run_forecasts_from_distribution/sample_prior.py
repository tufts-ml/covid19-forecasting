import numpy as np
import scipy
from scipy import stats
import json
import argparse

def sample_params_from_prior(prior_dict, random_seed, num_samples):
    samples = {'last_thetas': []}
    for i in range(num_samples):
        np.random.seed(random_seed + i)
        sample = {}
        for state in prior_dict['states']:
            sample['proba_Recovering_given_%s' % state] = np.random.dirichlet(prior_dict['prior_Health_given_%s' % state])[1]

            if state != 'OnVentInICU':
                sample['proba_Die_after_Declining_%s' % state] = np.random.dirichlet(prior_dict['prior_Die_after_Declining_%s' % state])[1]

            for health in ['Recovering', 'Declining']:
                lam_dict = prior_dict['prior_duration_%s_%s' % (health, state)]['lam']
                tau_dict = prior_dict['prior_duration_%s_%s' % (health, state)]['tau']

                a = lam_dict['lower']
                b = lam_dict['upper']
                lam_mean = lam_dict['mean']
                lam_stddev = lam_dict['stddev']
                alpha = (a - lam_mean) / lam_stddev
                beta = (b - lam_mean) / lam_stddev

                tau_mean = tau_dict['mean']
                tau_stddev = tau_dict['stddev']

                lam = stats.truncnorm(alpha, beta, loc=lam_mean, scale=lam_stddev)
                tau = stats.norm(loc=tau_mean, scale=tau_stddev)
                
                lam_sample = lam.rvs(size=1)[0]
                tau_sample = tau.rvs(size=1)[0]
                choices = scipy.special.softmax(scipy.stats.poisson.logpmf(np.arange(int(b)), lam_sample) / np.power(10, tau_sample))
                sample['pmf_duration_%s_%s' % (health, state)] = {}
                for c, choice in enumerate([str(x) for x in range(1, int(b)+1)]):
                    sample['pmf_duration_%s_%s' % (health, state)][choice] = choices[c]
                    
        samples['last_thetas'].append(sample)
        
    return samples

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior_file', default='configs/example_prior.json')
    parser.add_argument('--samples_file', default='configs/samples_from_prior.json')
    parser.add_argument('--random_seed', default=101, type=int)
    parser.add_argument('--num_samples', default=100, type=int)

    args, unknown_args = parser.parse_known_args()

    unk_keys = map(lambda s: s[2:].strip(), unknown_args[::2])
    unk_vals = unknown_args[1::2]
    unk_dict = dict(zip(unk_keys, unk_vals))

    with open(args.prior_file, 'r') as f:
        prior_dict = json.load(f)

    with open(args.samples_file, 'w+') as f:
        json.dump(sample_params_from_prior(prior_dict, args.random_seed, args.num_samples), f, indent=1)