'''
grid_search.py
--------------
Perform grid search over a set of hyperparameters.
Score model on last 20% of examples (i.e., most recent counts).
Produce JSON file with best model parameters.
Plot best heldout log likelihood for each window size.

Goal: Given counts from the past W timesteps, predict next count.

Model: Negative Binomial Regression
-----------------------------------
    - mu: Mean of negative binomial
            inner_prod(weights, last W timesteps)
        - intercept: Bias weight
                Prior is Normal (mu = 0, sigma = TUNED)
        - beta: Array of weights on all previous timesteps except most recent
                Prior is Normal (mu = 0, sigma = TUNED)
        - beta_recent: Weight on most recent timestep
                Prior is Normal (mu = 1, sigma = TUNED)
    - alpha: Negative binomial dispersion parameter
            Prior is TruncatedNormal (mu = TUNED, sigma = 500, lower = 0)
'''

import logging
import argparse
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import itertools
from NegativeBinomialRegression import NegativeBinomialRegression

### Silence warnings ###
logger = logging.getLogger('pymc3')
logger.setLevel(logging.ERROR)
logger = logging.getLogger('arviz')
logger.setLevel(logging.ERROR)

### Parse command line arguments ###
def csv_file(text):
    if not text.endswith('.csv'):
        raise argparse.ArgumentTypeError('Not a valid csv file name.')
    return text

def json_file(text):
    if not text.endswith('.json'):
        raise argparse.ArgumentTypeError('Not a valid json file name.')
    return text

parser = argparse.ArgumentParser()
parser.add_argument('input_csv_file', type=csv_file, help='name of input csv file')
parser.add_argument('-o', '--output_model_file', type=json_file, default='model.json',
                    help='name of JSON file to write model parameters to, default \'model.json\'')
parser.add_argument('-c', '--target_col_name', default='cases',
                    help='column of csv file with counts to make predictions on, default \'cases\'')
args = parser.parse_args()

### Read data from CSV file ###
train_df = pd.read_csv(args.input_csv_file)
counts = train_df[args.target_col_name].values

### Initialize hyperparameter spaces ###
window_sizes = [1, 2, 3, 4, 5, 6, 7, 8] # best = 4
intercept_sigmas = [1, 10] # best = 10
beta_sigmas = [1, 10] # best = 1
alpha_mus = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000] # best = 2000
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
    n_valid = int(.2 * N)
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

        model = NegativeBinomialRegression(model_dict)
        model.fit(x_train, y_train)
        score = model.score(x_valid, y_valid)

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
print(f'Score = {best_score}')

### Plot best score for each window size ###
plt.figure()
plt.plot(window_sizes, score_per_window_size, 'rs-')
plt.title('Performance vs Window Size')
plt.xlabel('Window size')
plt.ylabel('Heldout log lik')

### Write best model parameters to json file ###
model = dict()
model['window_size'] = window_sizes[best_id]
model["intercept"] = [0, best_params[0]] # Normal [mu, sigma]
model["beta"] = [0, best_params[1]] # Normal [mu, sigma]
model["beta_recent"] = [1, best_params[1]] # Normal [mu, sigma]
model["alpha"] = [best_params[2], 500] # TruncatedNormal [mu, sigma]

with open(args.output_model_file, 'w') as f:
    json.dump(model, f, indent=4)

plt.show()

