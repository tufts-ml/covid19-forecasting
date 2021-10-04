import numpy as np
import scipy.stats

def calc_cost(alpha, beta, size=100000, random_state=0, ideal_props=None):
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    
    my_dist = scipy.stats.beta(alpha, beta)
    samps_S = my_dist.rvs(size=size, random_state=random_state)

    cost = 0.0
    for pp in ['02', '20', '50', '80', '98']:
        desired_perc = ideal_props['p' + pp]
        actual_perc = np.percentile(samps_S, float(pp))
        weight = ideal_props['w' + pp]
        cost += weight * np.abs(desired_perc - actual_perc)
    return cost

def sample(alpha, beta, size=100000, random_state=0, ideal_props=None):
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    
    my_dist = scipy.stats.beta(alpha, beta)
    return my_dist.rvs(size=size, random_state=random_state)