import argparse
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

# Local imports from model.py, data.py
from model import CovidModel, Compartments, LogPoissonProb, get_logging_callbacks
from data import read_data, create_warmup
from plots import make_all_plots

def run_model(warmup_start=None, warmup_end=None, train_start=None, train_end=None, test_start=None, test_end=None, state=None, state_abbrev=None,
              data_dir=None, covid_estim_date=None, hhs_date=None, owid_date=None, log_dir=None, transition_window=None,
              T_serial=None,
              vax_asymp_risk=None, alpha_bar_M_0=None,alpha_bar_M_1=None, alpha_bar_X_0=None,alpha_bar_X_1=None,
              alpha_bar_G_0=None, alpha_bar_G_1=None, beta_bar_M_0=None,beta_bar_M_1=None, beta_bar_X_0=None,beta_bar_X_1=None,
              beta_bar_G_0=None, beta_bar_G_1=None, lambda_bar_M_0=None,lambda_bar_M_1=None, lambda_bar_X_0=None,lambda_bar_X_1=None,
              lambda_bar_G_0=None, lambda_bar_G_1=None, sigma_bar_M_0=None,sigma_bar_M_1=None, sigma_bar_X_0=None,sigma_bar_X_1=None,
              sigma_bar_G_0=None, sigma_bar_G_1=None, nu_bar_M_0=None,nu_bar_M_1=None, nu_bar_X_0=None,nu_bar_X_1=None,
              nu_bar_G_0=None, nu_bar_G_1=None, tau_bar_M_0=None,tau_bar_M_1=None, tau_bar_X_0=None,tau_bar_X_1=None,
              tau_bar_G_0=None, tau_bar_G_1=None, learning_rate=None, X_warmup_multiplier=None, M_warmup_multiplier=None, A_warmup_multiplier=None,
              model_name=None):


    df = read_data(data_dir=data_dir,
                   covid_estim_date=covid_estim_date,
                   hhs_date=hhs_date,
                   owid_date=owid_date,
                   state=state, state_abbrev=state_abbrev)

    # only used to graph
    vax_mild_risk = 0.94
    vax_extreme_risk = 0.94

    # Optional, replace covidestim warmup data with fixed constants
    df.loc[:, 'extreme'] = X_warmup_multiplier * df.loc[:, 'general_ward']
    df.loc[:, 'mild'] = M_warmup_multiplier * df.loc[:, 'extreme']
    df.loc[:, 'asymp'] = A_warmup_multiplier * df.loc[:, 'mild']

    warmup_asymp, warmup_mild, warmup_extreme = create_warmup(df,
                                                              warmup_start,
                                                              warmup_end,
                                                              vax_asymp_risk,
                                                              vax_mild_risk,
                                                              vax_extreme_risk)
    print(train_start)
    training_rt = df.loc[train_start:train_end, 'Rt'].values
    training_general_ward = df.loc[train_start:train_end, 'general_ward'].values

    # Start the model from the training period so we are continuous
    testing_rt = df.loc[train_start:test_end, 'Rt'].values
    testing_general_ward = df.loc[train_start:test_end, 'general_ward'].values


    alpha_bar_M = {0:alpha_bar_M_0, 1:alpha_bar_M_1}
    beta_bar_M = {0: beta_bar_M_0, 1: beta_bar_M_1}
    lambda_bar_M = {0: lambda_bar_M_0, 1: lambda_bar_M_1}
    sigma_bar_M = {0: sigma_bar_M_0, 1: sigma_bar_M_1}
    nu_bar_M = {0: nu_bar_M_0, 1: nu_bar_M_1}
    tau_bar_M = {0: tau_bar_M_0, 1: tau_bar_M_1}
    alpha_bar_X = {0: alpha_bar_X_0, 1: alpha_bar_X_1}
    beta_bar_X = {0: beta_bar_X_0, 1: beta_bar_X_1}
    lambda_bar_X = {0: lambda_bar_X_0, 1: lambda_bar_X_1}
    sigma_bar_X = {0: sigma_bar_X_0, 1: sigma_bar_X_1}
    nu_bar_X = {0: nu_bar_X_0, 1: nu_bar_X_1}
    tau_bar_X = {0: tau_bar_X_0, 1: tau_bar_X_1}
    alpha_bar_G = {0: alpha_bar_G_0, 1: alpha_bar_G_1}
    beta_bar_G = {0: beta_bar_G_0, 1: beta_bar_G_1}
    lambda_bar_G = {0: lambda_bar_G_0, 1: lambda_bar_G_1}
    sigma_bar_G = {0: sigma_bar_G_0, 1: sigma_bar_G_1}
    nu_bar_G = {0: nu_bar_G_0, 1: nu_bar_G_1}
    tau_bar_G = {0: tau_bar_G_0, 1: tau_bar_G_1}

    model = CovidModel(transition_window, T_serial,
                       alpha_bar_M, beta_bar_M, alpha_bar_X, beta_bar_X, alpha_bar_G, beta_bar_G,
                       lambda_bar_M, sigma_bar_M, lambda_bar_X, sigma_bar_X, lambda_bar_G, sigma_bar_G,
                       nu_bar_M, tau_bar_M, nu_bar_X, tau_bar_X, nu_bar_X, tau_bar_X)

    # Define optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
    )

    loss = LogPoissonProb()

    model.compile(loss=loss, optimizer=optimizer, run_eagerly=True)
    callbacks = get_logging_callbacks(log_dir)

    # Awkwardly stuff everything into an array

    model.fit(x=(np.asarray([training_rt]),
                 np.asarray([warmup_asymp[0]]), np.asarray([warmup_asymp[1]]),
                 np.asarray([warmup_mild[0] + warmup_mild[1]]),
                 np.asarray([warmup_extreme[0] + warmup_extreme[1]]),
                 np.asarray([df.loc[warmup_start:warmup_end, 'vax_pct'].values])),
              y=np.asarray([training_general_ward]),
              epochs=500, batch_size=0, callbacks=callbacks)

    train_preds = model((training_rt,
                         warmup_asymp[0], warmup_asymp[1],
                         warmup_mild[0] + warmup_mild[1],
                         warmup_extreme[0] + warmup_extreme[1],
                         df.loc[warmup_start:warmup_end, 'vax_pct'].values))

    test_preds = model((testing_rt,
                        warmup_asymp[0], warmup_asymp[1],
                        warmup_mild[0] + warmup_mild[1],
                        warmup_extreme[0] + warmup_extreme[1],
                        df.loc[warmup_start:warmup_end, 'vax_pct'].values))

    train_loss = loss(tf.convert_to_tensor(training_general_ward, dtype=tf.float32), train_preds)
    test_loss = loss(tf.convert_to_tensor(testing_general_ward, dtype=tf.float32), test_preds)

    forecasted_fluxes = model((testing_rt,
                               warmup_asymp[0], warmup_asymp[1],
                               warmup_mild[0] + warmup_mild[1],
                               warmup_extreme[0] + warmup_extreme[1],
                               df.loc[warmup_start:warmup_end, 'vax_pct'].values), return_all=True)


    save_path = os.path.join(log_dir, f'{model_name}.png')
    csv_path = os.path.join(log_dir, f'{model_name}.csv')

    results = make_all_plots(df, model,
               alpha_bar_M, beta_bar_M,
               alpha_bar_X, beta_bar_X,
               alpha_bar_G, beta_bar_G,
                   warmup_start, warmup_end,
                   train_start, train_end,
                   test_start, test_end,
                   train_preds, test_preds,
                   vax_asymp_risk, vax_mild_risk, vax_extreme_risk,
                   forecasted_fluxes, loss, save_path=save_path)

    model_details = {'transition_window':transition_window,
              'T_serial':T_serial,
              'vax_asymp_risk':vax_asymp_risk, 'alpha_bar_M_0':alpha_bar_M_0,'alpha_bar_M_1':alpha_bar_M_1,
                'alpha_bar_X_0':alpha_bar_X_0,'alpha_bar_X_1':alpha_bar_X_1,
              'alpha_bar_G_0':alpha_bar_G_0, 'alpha_bar_G_1':alpha_bar_G_1, 'beta_bar_M_0':beta_bar_M_0,
    'beta_bar_M_1':beta_bar_M_1, 'beta_bar_X_0':beta_bar_X_0,'beta_bar_X_1':beta_bar_X_1,
              'beta_bar_G_0':beta_bar_G_0, 'beta_bar_G_1':beta_bar_G_1, 'lambda_bar_M_0':lambda_bar_M_0,
    'lambda_bar_M_1':lambda_bar_M_1, 'lambda_bar_X_0':lambda_bar_X_0,'lambda_bar_X_1':lambda_bar_X_1,
              'lambda_bar_G_0':lambda_bar_G_0, 'lambda_bar_G_1':lambda_bar_G_1, 'sigma_bar_M_0':sigma_bar_M_0,
    'sigma_bar_M_1':sigma_bar_M_1, 'sigma_bar_X_0':sigma_bar_X_0,'sigma_bar_X_1':sigma_bar_X_1,
              'sigma_bar_G_0':sigma_bar_G_0, 'sigma_bar_G_1':sigma_bar_G_1, 'nu_bar_M_0':nu_bar_M_0,'nu_bar_M_1':nu_bar_M_1, 'nu_bar_X_0':nu_bar_X_0,
    'nu_bar_X_1':nu_bar_X_1,
              'nu_bar_G_0':nu_bar_G_0, 'nu_bar_G_1':nu_bar_G_1, 'tau_bar_M_0':tau_bar_M_0,'tau_bar_M_1':tau_bar_M_1, 'tau_bar_X_0':tau_bar_X_0,
    'tau_bar_X_1':tau_bar_X_1,
              'tau_bar_G_0':tau_bar_G_0, 'tau_bar_G_1':tau_bar_G_1, 'learning_rate':learning_rate,'X_warmup_multiplier':
    X_warmup_multiplier, 'M_warmup_multiplier':M_warmup_multiplier, 'A_warmup_multiplier':A_warmup_multiplier,
              'model_name':model_name}

    results = pd.concat([results, pd.DataFrame.from_records([model_details])], axis=1)

    results.to_csv(csv_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run a covid model')

    parser.add_argument('--warmup_start', help='YYYYMMDD date to start warmup data', type=str,
                        default='20210401')
    parser.add_argument('--warmup_end', help='YYYYMMDD date to end warmup data', type=str,
                        default='20210430')
    parser.add_argument('--train_start', help='YYYYMMDD date to start train data', type=str,
                        default='20210501')
    parser.add_argument('--train_end', help='YYYYMMDD date to end train data', type=str,
                        default='20210731')
    parser.add_argument('--test_start', help='YYYYMMDD date to start test data', type=str,
                        default='20210801')
    parser.add_argument('--test_end', help='YYYYMMDD date to end test data', type=str,
                        default='20210831')
    parser.add_argument('--state', help='Capitalized state name', type=str,
                        default='Massachusetts')
    parser.add_argument('--state_abbrev', help='2-letter state abbreviation', type=str,
                        default='MA')
    parser.add_argument('--data_dir', help='Path to directory with relevant data', type=str,
                        default='./data')
    parser.add_argument('--covid_estim_date', help='YYYYMMDD date covid_estim data retrieved', type=str,
                        default='20210901')
    parser.add_argument('--hhs_date', help='YYYYMMDD date hhs data retrieved', type=str,
                        default='20210903')
    parser.add_argument('--owid_date', help='YYYYMMDD date owid data retrieved', type=str,
                        default='20210903')
    parser.add_argument('--log_dir', help='Path to directory to write logs into', type=str)
    parser.add_argument('--transition_window', help='Number of days a case can take to transition compartments', type=int,
                        default=20)
    parser.add_argument('--T_serial', help='Covid estim model parameter', type=float,
                        default=5.8)
    parser.add_argument('--vax_asymp_risk', help='Used to split A into vax/non-vax', type=float,
                        default=0.9)
    parser.add_argument('--alpha_bar_M_0', help='Beta prior A param', type=float,
                        default=5.5)
    parser.add_argument('--beta_bar_M_0', help='Beta prior B param', type=float,
                        default=3.53)
    parser.add_argument('--alpha_bar_X_0', help='Beta prior A param', type=float,
                        default=1.89)
    parser.add_argument('--beta_bar_X_0', help='Beta prior B param', type=float,
                        default=20)
    parser.add_argument('--alpha_bar_G_0', help='Beta prior A param', type=float,
                        default=28.2)
    parser.add_argument('--beta_bar_G_0', help='Beta prior B param', type=float,
                        default=162.3)
    parser.add_argument('--alpha_bar_M_1', help='Beta prior A param', type=float,
                        default=5.5)
    parser.add_argument('--beta_bar_M_1', help='Beta prior B param', type=float,
                        default=3.53)
    parser.add_argument('--alpha_bar_X_1', help='Beta prior A param', type=float,
                        default=1.89)
    parser.add_argument('--beta_bar_X_1', help='Beta prior B param', type=float,
                        default=20)
    parser.add_argument('--alpha_bar_G_1', help='Beta prior A param', type=float,
                        default=28.2)
    parser.add_argument('--beta_bar_G_1', help='Beta prior B param', type=float,
                        default=162.3)
    parser.add_argument('--lambda_bar_M_0', help='Trunc Normal prior mean param', type=float,
                        default=4.7)
    parser.add_argument('--sigma_bar_M_0', help='Trunc Normal prior scale param', type=float,
                        default=1.0)
    parser.add_argument('--nu_bar_M_0', help='Trunc Normal prior mean param', type=float,
                        default=1.7)
    parser.add_argument('--tau_bar_M_0', help='Trunc Normal prior scale param', type=float,
                        default=0.10)
    parser.add_argument('--lambda_bar_M_1', help='Trunc Normal prior mean param', type=float,
                        default=4.7)
    parser.add_argument('--sigma_bar_M_1', help='Trunc Normal prior scale param', type=float,
                        default=1.0)
    parser.add_argument('--nu_bar_M_1', help='Trunc Normal prior mean param', type=float,
                        default=1.7)
    parser.add_argument('--tau_bar_M_1', help='Trunc Normal prior scale param', type=float,
                        default=0.10)
    parser.add_argument('--lambda_bar_X_0', help='Trunc Normal prior mean param', type=float,
                        default=4)
    parser.add_argument('--sigma_bar_X_0', help='Trunc Normal prior scale param', type=float,
                        default=0.50)
    parser.add_argument('--nu_bar_X_0', help='Trunc Normal prior mean param', type=float,
                        default=18)
    parser.add_argument('--tau_bar_X_0', help='Trunc Normal prior scale param', type=float,
                        default=8.1)
    parser.add_argument('--lambda_bar_X_1', help='Trunc Normal prior mean param', type=float,
                        default=4)
    parser.add_argument('--sigma_bar_X_1', help='Trunc Normal prior scale param', type=float,
                        default=0.5)
    parser.add_argument('--nu_bar_X_1', help='Trunc Normal prior mean param', type=float,
                        default=18)
    parser.add_argument('--tau_bar_X_1', help='Trunc Normal prior scale param', type=float,
                        default=8.1)
    parser.add_argument('--lambda_bar_G_0', help='Trunc Normal prior mean param', type=float,
                        default=3.3)
    parser.add_argument('--sigma_bar_G_0', help='Trunc Normal prior scale param', type=float,
                        default=1)
    parser.add_argument('--nu_bar_G_0', help='Trunc Normal prior mean param', type=float,
                        default=9)
    parser.add_argument('--tau_bar_G_0', help='Trunc Normal prior scale param', type=float,
                        default=0.2)
    parser.add_argument('--lambda_bar_G_1', help='Trunc Normal prior mean param', type=float,
                        default=3.3)
    parser.add_argument('--sigma_bar_G_1', help='Trunc Normal prior scale param', type=float,
                        default=1)
    parser.add_argument('--nu_bar_G_1', help='Trunc Normal prior mean param', type=float,
                        default=9)
    parser.add_argument('--tau_bar_G_1', help='Trunc Normal prior scale param', type=float,
                        default=0.2)
    parser.add_argument('--learning_rate', help='Learning rate', type=float,
                        default=1e-1)
    parser.add_argument('--X_warmup_multiplier', help='X=this*G', type=float,
                        default=7)
    parser.add_argument('--M_warmup_multiplier', help='M=this*X', type=float,
                        default=10)
    parser.add_argument('--A_warmup_multiplier', help='A=this*M', type=float,
                        default=1.5)
    parser.add_argument('--model_name', type=str)

    args = parser.parse_args()
    run_model(**vars(args))
