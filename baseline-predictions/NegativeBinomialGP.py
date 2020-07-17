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

class NegativeBinomialGP:

    '''
    init
    ----
    Takes in dictionary that specifies the model parameters.
    Each prior is a Truncated Normal dist, lower bounded at 0.
    Mean is first value in array, stddev is second value in array.
    
    --- Example input ---
    {
        "c": [7, 10],
        "a": [10, 10],
        "l": [10, 5],
        "alpha": [500, 500]
    }
    
    c: value of Constant mean fn
    a: amplitude of SqExp cov fn
    l: time-scale of SqExp cov fn
    alpha: Negative Binomial dispersion parameter
    '''
    def __init__(self, model_dict=None):
        if model_dict is None:
            self.c = [7, 10]
            self.a = [10, 10]
            self.l = [10, 5]
            self.alpha = [500, 500]
        else:
            self.c = model_dict['c']
            self.a = model_dict['a']
            self.l = model_dict['l']
            self.alpha = model_dict['alpha']

    '''
    fit
    ---
    Fits a PyMC3 model for a latent GP with Negative Binomial likelihood
    to the given data.
    Samples all model parameters from the posterior.
    '''
    def fit(self, t_tr, y_tr):
        with pm.Model() as self.model:
            c = pm.TruncatedNormal('mean', mu=self.c[0], sigma=self.c[1], lower=0)
            mean_func = pm.gp.mean.Constant(c=c)
            
            a = pm.TruncatedNormal('amplitude', mu=self.a[0], sigma=self.a[1], lower=0)
            l = pm.TruncatedNormal('time-scale', mu=self.l[0], sigma=self.l[1], lower=0)
            cov_func = a**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=l)
            
            self.gp = pm.gp.Latent(mean_func=mean_func, cov_func=cov_func)
            
            f = self.gp.prior('f', X=t_tr)
            
            self.alpha = pm.TruncatedNormal('alpha', mu=self.alpha[0], sigma=self.alpha[1], lower=0)
            y_ = pm.NegativeBinomial('y', mu=tt.exp(f)+1e-6, alpha=self.alpha, observed=y_tr)

            self.trace = pm.sample(500, tune=1000, target_accept=.98, max_treedepth=15, chains=2)
            pm.traceplot(self.trace)

    '''
    score
    -----
    Returns the log probability of the given dataset under the model.
    '''
    def score(self, t_va, y_va):
        with self.model:
            f_pred = self.gp.conditional('f_pred', t_va)
            y_pred = pm.NegativeBinomial('y_pred', mu=tt.exp(f_pred)+1e-6, alpha=self.alpha, shape=(len(t_va),))
            pred_samples = pm.sample_posterior_predictive(self.trace, vars=[f_pred])

        logp_list = np.zeros(len(self.trace))

        for i in range(len(self.trace)):
            if i % 100 == 0:
                print(f'Scored {i} samples...')
            point = self.trace[i]
            point['f_pred'] = pred_samples['f_pred'][i]
            point['y_pred'] = y_va
            logp_list[i] = y_pred.logp(point)

        score = np.mean(logp_list) / len(y_va)
        return score

    '''
    forecast
    --------
    Returns n_samples from the posterior predictive distribution for n_predictions days.
    Writes forecasted values to CSV files with the given filename pattern.
    '''
    def forecast(self, counts, n_samples, n_predictions, output_csv_file_pattern):
        t_forecast = np.arange(len(counts), len(counts) + n_predictions)[:,None]

        with self.model:
            f_forecast = self.gp.conditional('f_forecast', t_forecast)
            y_forecast = pm.NegativeBinomial('y_forecast', mu=tt.exp(f_forecast)+1e-6, alpha=alpha, shape=(len(t_forecast),))
            forecast_samples = pm.sample_posterior_predictive(self.trace, vars=[y_forecast], samples=n_samples)

        samples = forecast_samples['y_forecast']

        for i in range(n_samples):   
            output_dict = {'forecast': samples[i]}
            output_df = pd.DataFrame(output_dict)
            output_df.to_csv(output_csv_file_pattern.replace('*', str(i+1)))

        return samples

