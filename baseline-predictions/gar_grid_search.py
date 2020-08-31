import argparse
import arg_types
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pymc3 as pm
import json
from datetime import date
from datetime import timedelta

from GPAR import GPAR
from plot_forecasts import plot_forecasts

def gar_grid_search(counts, output_model_file, perf_ax, forecast_ax, end):
    T = int(.8 * len(counts))
    y_tr = counts[:T]
    y_va = counts[T:]
    F = len(y_va)

    window_sizes = [1, 2, 3, 4, 5, 6, 7]
    prior_sigmas = [(0.5, 0.1), (0.1, 0.1), (0.1, 0.01), (0.01, 0.01)]
    
    score_per_window_size = list()
    params_per_window_size = list()

    for W in window_sizes:

        score_list = list()

        for bias_sigma, beta_sigma in prior_sigmas:
            model_dict = {
                'window_size': W,
                'bias': [0, bias_sigma],
                'beta_recent': [1, beta_sigma],
                'beta': [0, beta_sigma]
            }

            model = GPAR(model_dict)
            model.fit(y_tr, F)
            score = model.score(y_va)
            score_list.append(score)
            print(f'W = {W} | bias sigma = {bias_sigma} | beta sigma = {beta_sigma} | score = {score}')

        best_id = np.argmax(score_list)
        best_score = score_list[best_id]

        print()
        print(f'Best params for W = {W}: {prior_sigmas[best_id]} | score = {best_score}\n')
        score_per_window_size.append(best_score)
        params_per_window_size.append(prior_sigmas[best_id])

    best_id = np.argmax(score_per_window_size)
    print('Best hypers overall:')
    print(f'W = {window_sizes[best_id]} | prior sigmas = {params_per_window_size[best_id]} | score = {score_per_window_size[best_id]}\n')

    perf_ax.plot(window_sizes, score_per_window_size, 's-')
    perf_ax.set_title('GAR Performance vs Window Size')
    perf_ax.set_xlabel('window size')
    perf_ax.set_ylabel('heldout log likelihood')

    model = dict()
    model['window_size'] = window_sizes[best_id]
    model['bias'] = [0, params_per_window_size[best_id][0]]
    model['beta_recent'] = [1, params_per_window_size[best_id][1]]
    model['beta'] = [0, params_per_window_size[best_id][1]]

    with open(output_model_file, 'w') as f:
        json.dump(model, f, indent=4)

    # Plot heldout forecasts using best model
    '''
    best_model = NegBinAutoregression(model)
    best_model.fit(y_tr, F)
    samples = best_model.forecast(1000)
    forecast_ax.set_title('GAR Best Model Heldout Forecasts')
    start = date.fromisoformat(end) - timedelta(F-1)
    plot_forecasts(samples, start, forecast_ax, y_va, future=False)
    '''


