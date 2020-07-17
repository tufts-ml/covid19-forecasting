'''
ggp_grid_search.py
------------------
Perform grid search over a set of hyperparameters for a GGP fit to the given count data.
Score model on last 20% of examples (i.e., most recent counts).
Write best model parameters to JSON file with the given filename.
Plot best heldout log likelihood for each timescale prior on the given axis.

Model: Gaussian Process with Negative Binomial likelihood

--- Parameters ---                                --- Priors ---
                                        TruncatedNormal (lower bounded at 0)
c:     value of Constant mean fn for GP         mu=TUNED, sigma=10
a:     amplitude of SqExp cov fn for GP         mu=TUNED, sigma=10
l:     time-scale of SqExp cov fn for GP        mu=TUNED, sigma=5
alpha: Negative Binomial dispersion parameter   mu=TUNED, sigma=500
'''

import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import itertools
from NegativeBinomialGP import NegativeBinomialGP

def ggp_grid_search(counts, output_model_file, ax):
    T = len(counts)
    t = np.arange(T)[:,None]

    # Holding out last 20% of counts
    n_valid = int(.2 * len(counts))
    t_tr = t[:-n_valid]
    t_va = t[-n_valid:]
    y_tr = counts[:-n_valid]
    y_va = counts[-n_valid:]

    ### Initialize hyperparameter spaces ###
    l_mus = [1, 4, 7, 10, 13, 16]
    c_mus = [1, 4, 7, 10]
    a_mus = [1, 4, 7, 10]
    alpha_mus = [500, 1000, 1500, 2000, 2500, 3000]
    params_to_search = list(itertools.product(c_mus, a_mus, alpha_mus))

    score_per_time_scale = list()
    params_per_time_scale = list()

    for l in l_mus:

        score_list = list()

        for c_mu, a_mu, alpha_mu in params_to_search:

            ### Fit and score a model with the current parameters ###
            model_dict = {
                "c": [c_mu, 10],
                "a": [a_mu, 10],
                "l": [l, 5],
                "alpha": [alpha_mu, 500]
            }

            model = NegativeBinomialGP(model_dict)
            model.fit(t_tr, y_tr)
            score = model.score(t_va, y_va)
            printf(f'\nScore on heldout set = {score}')
            score_list.append(score)

        ### Choose the best model for the current time-scale ###
        best_id = np.argmax(score_list)
        best_score = score_list[best_id]
        best_params = params_to_search[best_id]

        print(f'Best prior params for l = {l}: ', end='')
        print(f'mean = {best_params[0]} | ', end='')
        print(f'amplitude = {best_params[1]} | ', end='')
        print(f'alpha = {best_params[2]}')
        print(f'score = {best_score}\n')

        score_per_time_scale.append(best_score)
        params_per_time_scale.append(best_params)

    ### Choose the best model overall ###
    best_id = np.argmax(score_per_time_scale)
    best_score = score_per_time_scale[best_id]
    best_params = params_per_time_scale[best_id]

    print('Best prior params overall: ', end='')
    print(f'l = {l_mus[best_id]} | ', end='')
    print(f'mean = {best_params[0]} | ', end='')
    print(f'amplitude = {best_params[1]} | ', end='')
    print(f'alpha = {best_params[2]} | ', end='')
    print(f'score = {best_score}\n')

    ### Plot best score for each timescale prior assumption ###
    ax.set_title('GGP Performance vs Time-Scale Prior')
    ax.set_xlabel('Time-scale prior mean')
    ax.set_ylabel('Heldout log lik')
    ax.plot(l_mus, score_per_time_scale, 's-')

    ### Write best model parameters to json file ###
    model = dict()
    model["c"] = [best_params[0], 10]
    model["a"] = [best_params[1], 10]
    model["l"] = [l_mus[best_id], 5]
    model["alpha"] = [best_params[2], 500]

    with open(output_model_file, 'w') as f:
        json.dump(model, f, indent=4)

