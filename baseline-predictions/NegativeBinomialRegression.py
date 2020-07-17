'''
NegativeBinomialRegression.py
-----------------------------
Defines a generalized autoregressive model with Negative Binomial likelihood.
Contains fit, score, and forecast methods.
'''

import pymc3 as pm
import numpy as np
import pandas as pd
from datetime import date
from datetime import timedelta

class NegativeBinomialRegression:

    '''
    init
    ----
    Takes in dictionary that specifies the model parameters.
    For Normal priors, mean is 1st value in array, stddev is 2nd value in array.
    
    --- Example input ---
    {
        "window_size": 4,
        "intercept": [0, 10],
        "beta": [0, 1],
        "beta_recent": [1, 1],
        "alpha": [500, 500]
    }
    
    window_size: number of previous timesteps to condition on
    intercept: bias weight (Normal prior)
    beta: array of weights on all previous timesteps except most recent (Normal prior)
    beta_recent: weight on most recent timestep (Normal prior)
    alpha: Negative Binomial disperson parameter (TruncatedNormal(low=0) prior)
    '''
    def __init__(self, model_dict=None):
        if model_dict is None:
            self.W = 4
            self.intercept = [0, 10]
            self.beta = [0, 1]
            self.beta_recent = [1, 1]
            self.alpha = [1000, 500]
        else:
            self.W = model_dict['window_size']
            self.intercept = model_dict['intercept']
            self.beta = model_dict['beta']
            self.beta_recent = model_dict['beta_recent']
            self.alpha = model_dict['alpha']

    '''
    fit
    ---
    Fits a PyMC3 model for linear regression with Negative Binomial likelihood
    to the given data.
    Samples all model parameters from the posterior.

    Args: x_train: shape (N, W)
          y_train: shape (N,)
          For each count y_train[i], x_train[i] is an array of the counts from the previous W timesteps
    '''
    def fit(self, x_train, y_train):
        with pm.Model() as model:
            intercept = pm.Normal('intercept', mu=self.intercept[0], sd=self.intercept[1])
            beta = pm.Normal('beta', mu=self.beta[0], sd=self.beta[1], shape=self.W-1)
            beta_recent = pm.Normal('beta_recent', mu=self.beta_recent[0], sd=self.beta_recent[1])
            alpha = pm.TruncatedNormal('alpha', mu=self.alpha[0], sd=self.alpha[1], lower=0)

            mu = intercept
            for i in range(self.W-1):
                mu += beta[i] * x_train[:,i]
            mu += beta_recent * x_train[:,self.W-1]

            Y_obs = pm.NegativeBinomial('Y_obs', mu=mu, alpha=alpha, observed=y_train)

            trace = pm.sample(500, init='adapt_diag', tune=1000, target_accept=.90)
                              # compute_convergence_checks=False)

        self.trace = trace
        self.post_mean = pm.summary(trace)['mean'].to_dict()
        self.post_mean['alpha_lowerbound__'] = np.log(self.post_mean['alpha'])
        beta = np.zeros(self.W-1)
        for i in range(self.W-1):
            beta[i] = self.post_mean[f'beta[{i}]']
        self.post_mean['beta'] = beta

    '''
    score
    -----
    Returns the log probability of the given dataset under the model.

    Args: x_train: shape (N, W)
          y_train: shape (N,)
          For each count y_valid[i], x_valid[i] is an array of the counts from the previous W timesteps
    '''
    def score(self, x_valid, y_valid):
        with pm.Model() as model:
            intercept = pm.Normal('intercept', mu=self.intercept[0], sd=self.intercept[1])
            beta = pm.Normal('beta', mu=self.beta[0], sd=self.beta[1], shape=self.W-1)
            beta_recent = pm.Normal('beta_recent', mu=self.beta_recent[0], sd=self.beta_recent[1])
            alpha = pm.TruncatedNormal('alpha', mu=self.alpha[0], sd=self.alpha[1], lower=0)

            mu = intercept
            for i in range(self.W-1):
                mu += beta[i] * x_valid[:,i]
            mu += beta_recent * x_valid[:,self.W-1]

            Y_obs = pm.NegativeBinomial('Y_obs', mu=mu, alpha=alpha, observed=y_valid)

        # Score by Monte Carlo integration on posterior samples
        logp_list = np.zeros(len(self.trace))
        for i in range(len(self.trace)):
            if i % 100 == 0:
                print(f'Scored {i} samples...')
            logp_list[i] = model['Y_obs'].logp(self.trace[i])
        score = np.mean(logp_list) / len(y_valid)

        # Score on mean of posterior
        # score = model['Y_obs'].logp(self.post_mean) / len(y_valid)
        return score

    '''
    forecast
    --------
    Takes n_samples from the joint predictive distribution for n_predictions days.
    Writes forecasted values to CSV files with the given filename pattern.
    '''
    def forecast(self, counts, n_samples, n_predictions, output_csv_file_pattern):
        '''
        get_mu
        ------
        Returns mean of predictive distribution for next timestep given last W timesteps:
            inner_prod(weights, history)
        Returns 1 if computed mean <= 0 (mean of Negative Binomial must be positive).
        '''
        def get_mu(history):
            mu = 0.0
            mu += np.inner(self.post_mean['beta'], history[:-1])
            mu += self.post_mean['intercept'] + self.post_mean['beta_recent'] * history[-1]
            if mu <= 0:
                return 1
            return mu

        observed = np.zeros(self.W + n_predictions)
        observed[:self.W] = counts[-self.W:] # what we've seen so far, updated at each timestep
        samples = np.zeros((n_samples,n_predictions))
        alpha = self.post_mean['alpha']

        for i in range(n_samples):
            if(i % 100 == 0):
                print(f'Collected {i} samples...')
            
            for j in range(n_predictions):
                mu = get_mu(observed[j:j+self.W])
                y_t = pm.NegativeBinomial.dist(mu=mu, alpha=alpha)
                sample = y_t.random()
                samples[i][j] = sample
                observed[j+self.W] = sample
            
            output_dict = {'forecast': samples[i]}
            output_df = pd.DataFrame(output_dict)
            output_df.to_csv(output_csv_file_pattern.replace('*', str(i+1)))

        return samples

