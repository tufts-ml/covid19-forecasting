import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import theano.tensor as tt
import os
from datetime import date
from datetime import timedelta

from GenPoisson import GenPoisson
from plot_forecasts import plot_forecasts

input_files = os.listdir('../mass_dot_gov_datasets')
input_files.sort()
col_name = 'hospitalized_total_covid_patients_suspected_and_confirmed_including_icu'

W = 1
F = 14

for filename in input_files:

    print(filename)
    print()
    
    df = pd.read_csv(f'../mass_dot_gov_datasets/{filename}')
    counts = df[col_name].astype(float)
    dates = df['date'].values

    y_te = counts[-F:]
    y_va = counts[-2*F:-F]
    y_tr = counts[:-2*F]
    T = len(y_tr)

    print('STANDARD POISSON')
    print('----------------')

    with pm.Model() as model:
        bias = pm.Normal('beta[0]', mu=0, sigma=0.1)
        beta_recent = pm.Normal('beta[1]', mu=1, sigma=0.1)
        rho = [bias, beta_recent]
        for i in range(2, W+1):
            beta = pm.Normal(f'beta[{i}]', mu=0, sigma=0.1)
            rho.append(beta)
        tau = pm.HalfNormal('tau', sigma=0.1)

        f = pm.AR('f', rho, sigma=tau, constant=True, shape=T+F)
        y_past = pm.Poisson('y_past', mu=tt.exp(f[:-F]), observed=y_tr)
        y_past_logp = pm.Deterministic('y_past_logp', y_past.logpt)

        trace = pm.sample(5000, tune=2000, chains=2, cores=1, target_accept=0.99,
                          init='adapt_diag', max_treedepth=15, random_seed=42)

    summary = pm.summary(trace)['mean'].to_dict()
    for i in range(W+1):
        print(f'beta[{i}]', summary[f'beta[{i}]'])
    print('tau', summary['tau'])
    print()

    print('Training Scores:')
    print(np.log(np.mean(np.exp(trace.get_values('y_past_logp', chains=0)))) / T)
    print(np.log(np.mean(np.exp(trace.get_values('y_past_logp', chains=1)))) / T)
    print()

    with model:
        y_future = pm.Poisson('y_future', mu=tt.exp(f[-F:]), observed=y_va)
        y_logp = pm.Deterministic('y_logp', y_future.logpt)
        logp_list = pm.sample_posterior_predictive(trace, vars=[y_logp], keep_size=True)

    print(f'Heldout Scores:')
    print(np.log(np.mean(np.exp(logp_list['y_logp'][0]))) / F)
    print(np.log(np.mean(np.exp(logp_list['y_logp'][1]))) / F)
    print()

    print('GENERALIZED POISSON')
    print('-------------------')

    with pm.Model() as model:
        bias = pm.Normal('beta[0]', mu=0, sigma=0.1)
        beta_recent = pm.Normal('beta[1]', mu=1, sigma=0.1)
        rho = [bias, beta_recent]
        for i in range(2, W+1):
            beta = pm.Normal(f'beta[{i}]', mu=0, sigma=0.1)
            rho.append(beta)
        tau = pm.HalfNormal('tau', sigma=0.1)
        f = pm.AR('f', rho, sigma=tau, constant=True, shape=T+F)
        
        lam = pm.TruncatedNormal('lam', mu=0, sigma=0.3, lower=-1, upper=1)
        y_past = GenPoisson('y_past', theta=tt.exp(f[:-F]), lam=lam, observed=y_tr)
        y_past_logp = pm.Deterministic('y_past_logp', y_past.logpt)

        trace = pm.sample(5000, tune=2000, chains=2, cores=1, target_accept=0.99,
                          init='adapt_diag', max_treedepth=15, random_seed=42)

    summary = pm.summary(trace)['mean'].to_dict()
    for i in range(W+1):
        print(f'beta[{i}]', summary[f'beta[{i}]'])
    print('tau', summary['tau'])
    print('lambda', summary['lam'])
    print()

    print('Training Scores:')
    print(np.log(np.mean(np.exp(trace.get_values('y_past_logp', chains=0)))) / T)
    print(np.log(np.mean(np.exp(trace.get_values('y_past_logp', chains=1)))) / T)
    print()

    with model:
        y_future = GenPoisson('y_future', theta=tt.exp(f[-F:]), lam=lam, observed=y_va)
        y_logp = pm.Deterministic('y_logp', y_future.logpt)
        logp_list = pm.sample_posterior_predictive(trace, vars=[y_logp], keep_size=True)

    print(f'Heldout Scores:')
    print(np.log(np.mean(np.exp(logp_list['y_logp'][0]))) / F)
    print(np.log(np.mean(np.exp(logp_list['y_logp'][1]))) / F)
    print()




