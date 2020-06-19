'''
run_simple_forecast.py
----------------------
Produce samples of forecasted counts using a Negative Binomial Regression model.
Takes as input CSV file of counts and JSON file that specifies model parameters.
Write predictions to CSV files, and plot summary statistics of forecasts.

Model
-----
Goal: Given counts from the past W timesteps, predict next count.
Parameters:
    mean: mu = inner_prod(weights, history of counts)
    dispersion: alpha
    window size: W
'''

import logging
import json
import argparse
from datetime import date
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
parser.add_argument('input_csv_file', type=csv_file,
                    help='name of input CSV file, assumes it has a column \'date\'')
parser.add_argument('-c', '--target_col_name', default='cases',
                    help='column of CSV file with counts to make predictions on, default \'cases\'')
parser.add_argument('-m', '--model_file', type=json_file, default='model.json',
                    help='JSON file that specifies model hyperparameters')
parser.add_argument('-o', '--output_csv_file_pattern', type=csv_file, default='samples/output-*.csv',
                    help='pathname pattern for output CSV files, default \'samples/output-*.csv\'')
parser.add_argument('-n', '--n_samples', type=int, default=1000,
                    help='number of samples to draw from predictive distributions')
parser.add_argument('-s', '--day_forecasts_start', type=date.fromisoformat, metavar='YYYY-MM-DD',
                    help='default: day after last day of data')
parser.add_argument('-d', '--n_days_ahead', default=7, type=int,
                    help='number of days of predictions beyond start date')
args = parser.parse_args()

if args.n_days_ahead < 0:
    raise argparse.ArgumentTypeError('n_days_ahead must be a non-negative integer.')
if args.n_samples <= 0:
    raise argparse.ArgumentTypeError('n_samples must be a positive integer.')

n_predictions = args.n_days_ahead + 1
n_samples = args.n_samples

with open(args.model_file) as f:
    model_dict = json.load(f)

W = model_dict['window_size']

### Read data from CSV file ###
train_df = pd.read_csv(args.input_csv_file)
dates = train_df['date'].values
counts = train_df[args.target_col_name].values

### Set start date of forecasts ###
next_day = date.fromisoformat(dates[-1]) + timedelta(1) # day after last day of data
start = args.day_forecasts_start
if start == None:
    start = next_day # default to next_day
if not date.fromisoformat(dates[0]) + timedelta(W+1) <= start <= next_day: # check that start date is within range
    raise argparse.ArgumentTypeError('day_forecasts_start must be at least (W+1) days after earliest data point and at most 1 day after last data point')

counts = counts[:(start - date.fromisoformat(dates[0])).days] # only train on data up until start
N = len(counts) - W
x_train = np.vstack([counts[i:i+W] for i in range(N)])
y_train = counts[W:]

### Fit model and run forecast ###
model = NegativeBinomialRegression(model_dict)

model.fit(x_train, y_train)
print(f'\nPosterior Means:')
for var in ['intercept', 'beta', 'beta_recent', 'alpha']:
    print(var, model.post_mean[var])
print()

score = model.score(x_train, y_train)
print(f'Per day log likelihood:\n{score}\n')

print(f'Making forecasts starting on {start}')
print(f'Taking {n_samples} samples of predictions for {start} and {n_predictions-1} days ahead')
samples = model.forecast(counts, n_samples, n_predictions, args.output_csv_file_pattern)

### Plot summary of forecasts ###
low = np.zeros(n_predictions)
high = np.zeros(n_predictions)
mean = np.zeros(n_predictions)
median = np.zeros(n_predictions)

for i in range(n_predictions):
    low[i] = np.percentile(samples[:,i], 2.5)
    high[i] = np.percentile(samples[:,i], 97.5)
    median[i] = np.percentile(samples[:,i], 50)
    mean[i] = np.mean(samples[:,i])

pred_dates = np.full(n_predictions, start)
for i in range(n_predictions):
    pred_dates[i] = start + timedelta(i)

plt.figure()
plt.errorbar(np.arange(n_predictions), median,
             yerr=[median-low, high-median],
             capsize=2, fmt='.', linewidth=1,
             label='2.5, 50, 97.5 percentiles')
plt.plot(np.arange(n_predictions), mean, 'rx', label='mean')
plt.xticks(np.arange(n_predictions), pred_dates, rotation=30)
plt.title('Predictions')
plt.ylabel('95% interval estimate')
plt.legend()
plt.show()
