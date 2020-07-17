'''
run_simple_forecast.py
----------------------
Produces samples of forecasted counts using GLM and/or GGP.
User can specify -l to use GLM only, or -g for GGP only.
Defaults to using both models and producing side-by-side plots of forecasts.
Takes as input CSV file of counts and JSON file that specifies model parameters.
Writes predictions to CSV files, and plots summary statistics of forecasts.
'''

import logging
import json
import argparse
import arg_types
from datetime import date
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glm_forecast import glm_forecast
from ggp_forecast import ggp_forecast
from plot_forecasts import plot_forecasts

if __name__ == '__main__':

    # logger = logging.getLogger('pymc3')
    # logger.setLevel(logging.ERROR)
    logger = logging.getLogger('arviz')
    logger.setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()

    parser.add_argument('input_csv_file', type=arg_types.csv_file,
                        help='name of input CSV file, assumes it has a column \'date\'')
    parser.add_argument('-c', '--target_col_name', default='cases',
                        help='column of CSV file with counts to make predictions on, default \'cases\'')

    parser.add_argument('-l', '--linear_model', action='store_true', help='only use GLM')
    parser.add_argument('-g', '--gaussian_process', action='store_true', help='only use GGP')

    parser.add_argument('-m', '--glm_model_file', type=arg_types.json_file, default='glm_model.json',
                        help='JSON file that specifies GLM hyperparameters, default \'glm_model.json\'')
    parser.add_argument('-f', '--ggp_model_file', type=arg_types.json_file, default='ggp_model.json',
                        help='JSON file that specifies GGP hyperparmeters, default \'ggp_model.json\'')

    parser.add_argument('-o', '--glm_csv_file_pattern', type=arg_types.csv_file, default='glm_samples/output-*.csv',
                        help='pathname pattern for output CSV files, default \'glm_samples/output-*.csv\'')
    parser.add_argument('-u', '--ggp_csv_file_pattern', type=arg_types.csv_file, default='ggp_samples/output-*.csv',
                        help='pathname pattern for output CSV files, default \'ggp_samples/output-*.csv\'')

    parser.add_argument('-n', '--n_samples', type=int, default=1000,
                        help='number of samples to draw from predictive distribution, default 1000')
    parser.add_argument('-s', '--day_forecasts_start', type=date.fromisoformat, metavar='YYYY-MM-DD',
                        help='default: day after last day of data')
    parser.add_argument('-d', '--n_days_ahead', default=7, type=int,
                        help='number of days of predictions (including start date), default 7')

    parser.add_argument('-p', '--plot_file', type=arg_types.png_file, default='forecasts.png',
                        help='name of PNG file to save plot as, default \'forecasts.png\'')

    args = parser.parse_args()

    if args.n_days_ahead <= 0:
        raise argparse.ArgumentTypeError('n_days_ahead must be a positive integer.')
    if args.n_samples <= 0:
        raise argparse.ArgumentTypeError('n_samples must be a positive integer.')

    n_predictions = args.n_days_ahead
    n_samples = args.n_samples

    ### Read data from CSV file ###
    train_df = pd.read_csv(args.input_csv_file)
    dates = train_df['date'].values
    counts = train_df[args.target_col_name].values

    ### Set start date of forecasts ###
    next_day = date.fromisoformat(dates[-1]) + timedelta(1) # day after last day of data
    start = args.day_forecasts_start
    if start == None:
        start = next_day # default to next_day
    if not date.fromisoformat(dates[0]) + timedelta(10) <= start <= next_day: # check that start date is within range
        raise argparse.ArgumentTypeError('day_forecasts_start must be at least 10 days after earliest data point and at most 1 day after last data point')

    counts = counts[:(start - date.fromisoformat(dates[0])).days] # only train on data up until start

    np.random.seed(42)

    if args.linear_model:
        with open(args.glm_model_file) as f:
            glm_model_dict = json.load(f)

        glm_samples = glm_forecast(glm_model_dict, counts,
                                   n_samples, n_predictions,
                                   args.glm_csv_file_pattern)

        fig,ax = plt.subplots(figsize=(8,6))
        ax.set_title('GLM Forecasts')
        plot_forecasts(glm_samples, n_predictions, start, ax)

    elif args.gaussian_process:
        with open(args.ggp_model_file) as f:
            ggp_model_dict = json.load(f)

        ggp_samples = ggp_forecast(ggp_model_dict, counts,
                                   n_samples, n_predictions,
                                   args.ggp_csv_file_pattern)

        fig,ax = plt.subplots(figsize=(8,6))
        ax.set_title('GGP Forecasts')
        plot_forecasts(ggp_samples, n_predictions, start, ax)

    else:
        fig,ax = plt.subplots(ncols=2, sharey=True, figsize=(16,6))

        with open(args.glm_model_file) as f:
            glm_model_dict = json.load(f)
        glm_samples = glm_forecast(glm_model_dict, counts,
                                   n_samples, n_predictions,
                                   args.glm_csv_file_pattern)
        ax[0].set_title('GLM Forecasts')
        plot_forecasts(glm_samples, n_predictions, start, ax[0])

        with open(args.ggp_model_file) as f:
            ggp_model_dict = json.load(f)
        ggp_samples = ggp_forecast(ggp_model_dict, counts,
                                   n_samples, n_predictions,
                                   args.ggp_csv_file_pattern)
        ax[1].set_title('GGP Forecasts')
        plot_forecasts(ggp_samples, n_predictions, start, ax[1])

    plt.savefig(args.plot_file)
    # plt.show()
