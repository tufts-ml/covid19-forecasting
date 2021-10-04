import numpy as np
import scipy.stats

def calc_cost(t, w, random_state=0, size=100000, ideal_props={}):
    samps_S = sample(t, w, random_state, size, ideal_props)

    cost = 0.0
    for pp in ['02', '20', '50', '80', '98']:
        desired_perc = ideal_props['p' + pp]
        actual_perc = np.percentile(samps_S, float(pp))
        weight = ideal_props['w' + pp]
        cost += weight * np.abs(desired_perc - actual_perc)
    return cost

def sample(t, w, random_state=0, size=100000, ideal_props={}):
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    mean_val = float(ideal_props['p50'])
    bg_alpha_beta = ideal_props.get('bg_alpha_beta', 1.1)

    fg_dist = scipy.stats.beta(mean_val * t, (1-mean_val) * t)
    bg_dist = scipy.stats.beta(bg_alpha_beta, bg_alpha_beta)
    S = int(size)
    samps_S = fg_dist.rvs(random_state=random_state, size=S)
    keepmask_fg_S = random_state.rand(S) < w
    Sbg = S - np.sum(keepmask_fg_S)
    samps_S[np.logical_not(keepmask_fg_S)] = bg_dist.rvs(random_state=random_state, size=Sbg)
    return samps_S