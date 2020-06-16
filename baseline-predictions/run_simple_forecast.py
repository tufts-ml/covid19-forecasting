# Learn parameters for model and make predictions for next n_days_ahead
# Make plot with predictions and error bars
# Write predictions to output csv file


# Desired interface:
# python run_simple_forecast.py
# --input_csv_file /path/to/data.csv
# --target_col_name cases
# --model_file model.json
# --output_csv_file_pattern /path/to/output-{sample}.csv
# --n_samples 5000
# --day_forecasts_start DATE
# --n_days_ahead 14

import json
import argparse
from datetime import date
from datetime import timedelta
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

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
parser.add_argument('-t', '--target_col_name', default='cases',
                    help='column of csv file with counts to make predictions on, default \'cases\'')
# parser.add_argument('-l', '--likelihood', choices=['Normal', 'Poisson', 'NegativeBinomial'],
                    # default='Normal', help='likelihood for generalized linear model, default \'Normal\'')
# parser.add_argument('-w', '--window_size', type=int, default=4,
                    # help='number of past timesteps to condition next count on')
parser.add_argument('-m', '--model_file', type=json_file, default='model.json',
                    help='json file that specifies model hyperparameters')
parser.add_argument('-o', '--output_csv_file', type=csv_file, default='simple_forecast.csv',
                    help='name of output csv file')
parser.add_argument('-n', '--n_samples', type=int, default=1000,
                    help='number of samples to draw from predictive distributions')
parser.add_argument('-s', '--day_forecasts_start', type=date.fromisoformat, metavar='YYYY-MM-DD',
                    help='default: day after last day of data')
parser.add_argument('-a', '--n_days_ahead', default=7, type=int,
                    help='number of days of predictions beyond start date')

### Parse command line arguments ###
args = parser.parse_args()
start = args.day_forecasts_start
n_predictions = args.n_days_ahead + 1
n_samples = args.n_samples
with open(args.model_file) as f:
    model = json.load(f)
W = model['window_size']
lik = model['likelihood']

### Read and reformat data ###
train_df = pd.read_csv(args.input_csv_file)
dates = train_df['date'].values
counts = train_df[args.target_col_name].values

start = args.day_forecasts_start
if start == None:
    start = date.fromisoformat(dates[-1]) + timedelta(1)

N = len(counts) - W
x_train = np.vstack([counts[i:i+W] for i in range(N)])
y_train = counts[W:]

print(f'Conditioning on last {W} time steps')
print(f'Using Bayesian linear regression with {lik} likelihood')
print('Making forecasts starting on', start)
print(f'Taking {n_samples} samples of predictions for the start date and {n_predictions-1} days ahead')

### Define model ###
with pm.Model() as model:
    intercept = pm.Normal('intercept', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=1, shape=W-1)
    beta_recent = pm.Normal('beta_recent', mu=1, sd=1)
    alpha = pm.HalfNormal('alpha', sd=1000)
    # sigma = pm.HalfNormal('sigma', sd=100)

    mu = intercept
    for i in range(W-1):
        mu += beta[i] * x_train[:,i]
    mu += beta_recent * x_train[:,W-1]

    Y_obs = pm.NegativeBinomial('Y_obs', mu=mu, alpha=alpha, observed=y_train)
    # Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y_train)

### Train and score ###
map = pm.find_MAP(model=model, maxeval=1000)
print(f'MAP estimate for W={W}')
for key, arr in map.items():
    print(key, arr)
    
score = model['Y_obs'].logp(map) / N
print(f'Per day log lik: {score}\n')


### Plot predictions for the days ahead ###

# Return mean of predictive distribution for next timestep given last W timesteps
#     inner_prod(MAP estimate of weights, history)
# Parameter: history (array of last W counts)
def get_mu(history):
    mu = 0.0
    mu += np.inner(map['beta'], history[:-1])
    mu += map['intercept'] + map['beta_recent'] * history[-1]
    return mu

observed = np.zeros(W+n_predictions)
observed[:W] = counts[-W:]
samples = np.zeros((n_samples,n_predictions))
alpha = map['alpha']

pred_dates = np.full(n_predictions,start)
for i in range(n_predictions):
    pred_dates[i] = start + timedelta(i)

for i in range(n_samples):
    print(i)
    for j in range(n_predictions):
        mu = get_mu(observed[j:j+W])
        y_t = pm.NegativeBinomial.dist(mu=mu, alpha=alpha)
        # y_t = pm.Normal.dist(mu=mu, sigma=map['sigma'])
        sample = y_t.random()
        samples[i][j] = sample
        observed[j+W] = sample


low = np.zeros(n_predictions)
high = np.zeros(n_predictions)
mean = np.zeros(n_predictions)

for i in range(n_predictions):
    low[i] = np.percentile(samples[:,i], 2.5)
    high[i] = np.percentile(samples[:,i], 97.5)
    mean[i] = np.mean(samples[:,i])

plt.figure()
plt.errorbar(np.arange(8), mean, yerr=[mean-low, high-mean], fmt='.', linewidth=1)
plt.xticks(np.arange(8), pred_dates, rotation=45)
plt.title('Predictions')
plt.xlabel('Num days into future')
plt.ylabel('95% interval estimate')


### Write results to csv file ###

output_dict = {
    'date' : pred_dates,
    'mean' : mean,
    'low' : low,
    'high' : high
}

output_df = pd.DataFrame(output_dict)
output_df.to_csv(args.output_csv_file)

plt.show()
