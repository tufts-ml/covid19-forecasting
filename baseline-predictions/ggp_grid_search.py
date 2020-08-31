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
from datetime import date
from datetime import timedelta

from GPGP import GPGP
from plot_forecasts import plot_forecasts

def ggp_grid_search(counts, output_model_file, perf_ax, forecast_ax, end):
    T = int(.8 * len(counts))
    y_tr = counts[:T]
    y_va = counts[T:]
    F = len(y_va)

    ### Initialize hyperparameter spaces ###
    l_mus = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
    c_mus = [4]

    score_per_time_scale = list()
    params_per_time_scale = list()

    for l in l_mus:

        score_list = list()

        for c_mu in c_mus:

            ### Fit and score a model with the current parameters ###
            model_dict = {
                'c': [c_mu, 2],
                'a': 2,
                'l': [l, 2],
            }

            model = GPGP(model_dict)
            model.fit(y_tr, F)
            score = model.score(y_va)
            print(f'\nScore on heldout set = {score}')
            score_list.append(score)

        ### Choose the best model for the current time-scale ###
        best_id = np.argmax(score_list)
        best_score = score_list[best_id]
        best_params = c_mus[best_id]

        print(f'Best prior params for l = {l}: ', end='')
        print(f'mean = {best_params} | ', end='')
        print(f'score = {best_score}\n')

        score_per_time_scale.append(best_score)
        params_per_time_scale.append(best_params)

    ### Choose the best model overall ###
    best_id = np.argmax(score_per_time_scale)
    best_score = score_per_time_scale[best_id]
    best_params = params_per_time_scale[best_id]

    print('Best prior params overall: ', end='')
    print(f'l = {l_mus[best_id]} | ', end='')
    print(f'mean = {best_params} | ', end='')
    print(f'score = {best_score}\n')

    ### Plot best score for each timescale prior assumption ###
    perf_ax.set_title('GGP Performance vs Time-Scale Prior')
    perf_ax.set_xlabel('Time-scale prior mean')
    perf_ax.set_ylabel('Heldout log lik')
    perf_ax.plot(l_mus, score_per_time_scale, 's-')

    ### Write best model parameters to json file ###
    model = dict()
    model['c'] = [best_params, 2]
    model['a'] = 2
    model['l'] = [l_mus[best_id], 2]

    with open(output_model_file, 'w') as f:
        json.dump(model, f, indent=4)

    ### Plot heldout forecasts using best model ###
    '''
    best_model = NegBinGP(model)
    best_model.fit(y_tr, F)
    samples = best_model.forecast(1000)
    forecast_ax.set_title('GGP Forecasts')
    start = date.fromisoformat(end) - timedelta(F-1)
    plot_forecasts(samples, start, forecast_ax, y_va, future=False)
    '''


