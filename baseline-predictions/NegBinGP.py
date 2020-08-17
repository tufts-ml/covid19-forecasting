'''
NegativeBinomialGP.py
---------------------
Defines a generalized Gaussian Process model with Negative Binomial likelihood.
Contains fit, score, and forecast methods.
'''

import pymc3 as pm
import numpy as np
import pandas as pd
from datetime import date
from datetime import timedelta
import theano
import theano.tensor as tt
theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"

class NegBinGP:

    '''
    init
    ----
    Takes in dictionary that specifies the model parameters.
    Each prior is a Truncated Normal dist, lower bounded at 0.
    Mean is first value in array, stddev is second value in array.
    
    --- Example input ---
    {
        "c": [4, 2],
        "a": 2,
        "l": [7, 2],
    }
    
    c: value of Constant mean fn
    a: amplitude of SqExp cov fn
    l: time-scale of SqExp cov fn
    alpha: Negative Binomial dispersion parameter
    '''
    def __init__(self, model_dict=None):
        if model_dict is None:
            self.c = [4, 2]
            self.a = 2
            self.l = [7, 2]
        else:
            self.c = model_dict['c']
            self.a = model_dict['a']
            self.l = model_dict['l']

    '''
    fit
    ---
    Fits a PyMC3 model for a latent GP with Negative Binomial likelihood
    to the given data.
    Samples all model parameters from the posterior.
    '''
    def fit(self, y_tr, n_future):
        T = len(y_tr)
        self.F = n_future
        t = np.arange(T+self.F)[:,None]

        with pm.Model() as self.model:
            c = pm.TruncatedNormal('mean', mu=self.c[0], sigma=self.c[1], lower=0)
            mean_func = pm.gp.mean.Constant(c=c)
            
            a = pm.HalfNormal('amplitude', sigma=self.a)
            l = pm.TruncatedNormal('time-scale', mu=self.l[0], sigma=self.l[1], lower=0)
            cov_func = a**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=l)
            
            self.gp = pm.gp.Latent(mean_func=mean_func, cov_func=cov_func)
            self.f = self.gp.prior('f', X=t)

            y_past = pm.Poisson('y_past', theta=tt.exp(self.f[:T]), observed=y_tr)

            self.trace = pm.sample(500, tune=1000, target_accept=.99, chains=2, random_seed=42, cores=1)

            summary = pm.summary(self.trace)['mean'].to_dict()
            for key in ['mean', 'amplitude', 'time-scale', 'lam']:
                print(key, summary[key])

    '''
    score
    -----
    Returns the heldout log probability of the given dataset under the model.
    '''
    def score(self, y_va):
        assert len(y_va) == self.F

        with self.model:
            y_future = pm.Poisson('y_future', theta=tt.exp(self.f[-self.F:]), observed=y_va)
            lik = pm.Deterministic('lik', y_future.logpt)
            logp_list = pm.sample_posterior_predictive(self.trace, vars=[lik], keep_size=True)

        score = np.log(np.mean(np.exp(logp_list['lik'][0]))) / self.F
        return score

    '''
    forecast
    --------
    Returns n_samples from the posterior predictive distribution for n_predictions days.
    Writes forecasted values to CSV files with the given filename pattern.
    '''
    def forecast(self, n_samples, output_csv_file_pattern=None):
        with self.model:
            y_future = pm.Poisson('y_future', theta=tt.exp(self.f[-self.F:]), shape=(self.F,))
            forecasts = pm.sample_posterior_predictive(self.trace, vars=[y_future], samples=n_samples, random_seed=42)
        samples = forecasts['y_future']

        if output_csv_file_pattern != None:
            for i in range(n_samples):
                if(i % 100 == 0):
                    print(f'Saved {i} forecasts...')
                output_dict = {'forecast': samples[i]}
                output_df = pd.DataFrame(output_dict)
                output_df.to_csv(output_csv_file_pattern.replace('*', str(i+1)))

        return samples

