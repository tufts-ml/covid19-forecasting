'''
glm_grid_search.py
------------------
Perform grid search over a set of hyperparameters for a GLM fit to the given count data.
Score model on last 20% of examples (i.e., most recent counts).
Write best model parameters to JSON file with the given filename
Plot best heldout log likelihood for each window size on the given axis.

Goal: Given counts from the past W timesteps, predict next count.
Model: Generalized linear regression with Negative Binomial likelihood

--- Parameters ---                                   --- Priors ---
intercept:   Bias weight                       Normal (mu=0, sigma=TUNED)
beta:        Array of weights on all previous  Normal (mu=0, sigma=TUNED)
             timesteps except most recent  
beta_recent: Weight on most recent timestep    Normal (mu=1, sigma=TUNED)
alpha:       NegBin dispersion parameter       TruncatedNormal (mu=TUNED, sigma=500, low=0)
'''

import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import itertools
from NegativeBinomialRegression import NegativeBinomialRegression

def glm_grid_search(counts, output_model_file, ax):
    n_valid = int(.2 * len(counts))

    ### Initialize hyperparameter spaces ###
    window_sizes = [2, 3, 4, 5, 6, 7, 8]
    intercept_sigmas = [1, 5, 10]
    beta_sigmas = [1, 5, 10]
    alpha_mus = [500, 1000, 1500, 2000, 2500, 3000]
    params_to_search = list(itertools.product(intercept_sigmas, beta_sigmas, alpha_mus))

    score_per_window_size = list()
    params_per_window_size = list()

    for W in window_sizes:

        score_list = list()

        ### Reformat data: For each count y[i], x[i] is an array of the counts from the past W timesteps ###
        N = len(counts) - W
        x = np.vstack([counts[i:i+W] for i in range(N)])
        y = counts[W:]

        ### Split data into training and validation (last 20%) ###
        x_train = x[:-n_valid]
        x_valid = x[-n_valid:]
        y_train = y[:-n_valid]
        y_valid = y[-n_valid:]

        for intercept_sigma, beta_sigma, alpha_mu in params_to_search:

            ### Fit and score a model with the current parameters ###
            model_dict = {
                "window_size": W,
                "intercept": [0, intercept_sigma],
                "beta": [0, beta_sigma],
                    "beta_recent": [1, beta_sigma],
                "alpha": [alpha_mu, 500]
            }

            print('Priors:')
            print(f'W={W} | intercept sigma={intercept_sigma} | beta sigma={beta_sigma} | alpha mu={alpha_mu}\n')
            model = NegativeBinomialRegression(model_dict)
            model.fit(x_train, y_train)
            # print(f'\nPosterior Means:')
            # for var in ['intercept', 'beta', 'beta_recent', 'alpha']:
            #     print(var, model.post_mean[var])
            # print()
            score = model.score(x_valid, y_valid)
            printf(f'\nScore on heldout set = {score}')
            score_list.append(score)

        ### Choose the best model for the current window size ###
        best_id = np.argmax(score_list)
        best_score = score_list[best_id]
        best_params = params_to_search[best_id]

        print(f'Best params for W = {W}: ', end='')
        print(f'Intercept sigma = {best_params[0]} | ', end='')
        print(f'Beta sigma = {best_params[1]} | ', end='')
        print(f'Alpha mu = {best_params[2]}')
        print(f'Score = {best_score}\n')

        score_per_window_size.append(best_score)
        params_per_window_size.append(best_params)

    ### Choose the best model overall ###
    best_id = np.argmax(score_per_window_size)
    best_score = score_per_window_size[best_id]
    best_params = params_per_window_size[best_id]

    print('Best params overall: ', end='')
    print(f'W = {window_sizes[best_id]} | ', end='')
    print(f'Intercept sigma = {best_params[0]} | ', end='')
    print(f'Beta sigma = {best_params[1]} | ', end='')
    print(f'Alpha mu = {best_params[2]} | ', end='')
    print(f'Score = {best_score}\n')

    ### Plot best score for each window size ###
    ax.set_title('GLM Performance vs Window Size')
    ax.set_xlabel('Window size')
    ax.set_ylabel('Heldout log lik')
    ax.plot(window_sizes, score_per_window_size, 's-')

    ### Write best model parameters to json file ###
    model = dict()
    model['window_size'] = window_sizes[best_id]
    model["intercept"] = [0, best_params[0]] # Normal [mu, sigma]
    model["beta"] = [0, best_params[1]] # Normal [mu, sigma]
    model["beta_recent"] = [1, best_params[1]] # Normal [mu, sigma]
    model["alpha"] = [best_params[2], 500] # TruncatedNormal [mu, sigma]

    with open(output_model_file, 'w') as f:
        json.dump(model, f, indent=4)

