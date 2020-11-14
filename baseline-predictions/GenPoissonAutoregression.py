'''
GenPoissonAutoregression.py
---------------------------
Defines a generalized autoregressive model with Generalized Poisson likelihood.
Contains fit, score, and forecast methods.
'''

import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import theano.tensor as tt
from datetime import date
from datetime import timedelta
from GenPoisson import GenPoisson

class GenPoissonAutoregression:

    '''
    init
    ----
    Takes in dictionary that specifies the model parameters.
    Arrays give mean and standard deviation for Normal priors.
    
    --- Example input ---
    {
        "window_size": 4,
        "bias": [0, 0.5],
        "beta_recent": [1, 0.1],
        "beta": [0, 0.1]
    }
    '''
    def __init__(self, model_dict=None):
        if model_dict is None:
            self.W = 1
            self.bias = [0, 0.1]
            self.beta_recent = [1, 0.1]
            self.beta = [0, 0.1]
        else:
            self.W = model_dict['window_size']
            self.bias = model_dict['bias']
            self.beta_recent = model_dict['beta_recent']
            self.beta = model_dict['beta']

    '''
    fit
    ---
    Fits a PyMC3 model for linear regression with Negative Binomial likelihood
    to the given data.
    Samples all model parameters from the posterior.
    '''
    def fit(self, y_tr, n_future):
        T = len(y_tr)
        self.F = n_future

        with pm.Model() as self.model:
            bias = pm.Normal('beta[0]', mu=self.bias[0], sigma=self.bias[1])
            beta_recent = pm.Normal('beta[1]', mu=self.beta_recent[0], sigma=self.beta_recent[1])
            rho = [bias, beta_recent]
            for i in range(2, self.W+1):
                beta = pm.Normal(f'beta[{i}]', mu=self.beta[0], sigma=self.beta[1])
                rho.append(beta)
            tau = pm.HalfNormal('tau', sigma=0.1)
            self.f = pm.AR('f', rho, sigma=tau, constant=True, shape=T+self.F)
            
            self.lam = pm.TruncatedNormal('lam', mu=0, sigma=0.3, lower=-1, upper=1)
            y_past = GenPoisson('y_past', theta=tt.exp(self.f[:T]), lam=self.lam, observed=y_tr, testval=1)
            y_past_logp = pm.Deterministic('y_past_logp', y_past.logpt)

            self.trace = pm.sample(5000, tune=2000, init='adapt_diag', max_treedepth=15,
                                   target_accept=0.99, random_seed=42, chains=2, cores=1)

        summary = pm.summary(self.trace)['mean'].to_dict()
        print('Posterior Means:')
        for i in range(self.W+1):
            print(f'beta[{i}]', summary[f'beta[{i}]'])
        print('tau', summary['tau'])
        print('lambda', summary['lam'])
        print()

        print('Training Scores:')
        print(np.log(np.mean(np.exp(self.trace.get_values('y_past_logp', chains=0)))) / T)
        print(np.log(np.mean(np.exp(self.trace.get_values('y_past_logp', chains=1)))) / T)
        print()

    '''
    score
    -----
    Returns the heldout log probability of the given dataset under the model.
    '''
    def score(self, y_va):
        assert len(y_va) == self.F

        with self.model:
            y_future = GenPoisson('y_future', theta=tt.exp(self.f[-self.F:]), lam=self.lam, observed=y_va)
            lik = pm.Deterministic('lik', y_future.logpt)
            logp_list = pm.sample_posterior_predictive(self.trace, vars=[lik], keep_size=True)

        print('Heldout Scores:')
        print(np.log(np.mean(np.exp(logp_list['lik'][0]))) / self.F)
        print(np.log(np.mean(np.exp(logp_list['lik'][1]))) / self.F)
        print()
        score = np.log(np.mean(np.exp(logp_list['lik'][0]))) / self.F
        return score

    '''
    forecast
    --------
    Takes n_samples from the joint predictive distribution for n_predictions days.
    Writes forecasted values to CSV files with the given filename pattern.
    '''
    def forecast(self, output_csv_file_pattern=None):
        with self.model:
            y_pred = GenPoisson('y_pred', theta=tt.exp(self.f[-self.F:]), lam=self.lam, shape=self.F, testval=1)
            forecasts = pm.sample_posterior_predictive(self.trace, vars=[y_pred], keep_size=True, random_seed=42)
        samples = forecasts['y_pred'][0]

        if output_csv_file_pattern != None:
            for i in range(len(samples)):
                if(i % 1000 == 0):
                    print(f'Saved {i} forecasts...')
                output_dict = {'forecast': samples[i]}
                output_df = pd.DataFrame(output_dict)
                output_df.to_csv(output_csv_file_pattern.replace('*', str(i+1)))

        return samples

