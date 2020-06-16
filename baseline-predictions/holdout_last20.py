# Determine best model parameters
# Hold out most recent 20% of counts as validation set

# Takes in csv file on command line
# Output is json file with best model parameters
# Right now I'm just searching a range of window sizes. The prior
# hyperparameters are the ones that I've found to be the best for
# the Middlesex data set.

import argparse
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
plt.style.use('seaborn-darkgrid')

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
                    help='name of JSON file to write model parameters to')
parser.add_argument('-t', '--target_col_name', default='cases',
                    help='column of csv file with counts to make predictions on, default \'cases\'')

args = parser.parse_args()
model_file = args.output_model_file

train_df = pd.read_csv(args.input_csv_file)
counts = train_df[args.target_col_name].values

window_sizes = range(1,9)
score_list = []
map = dict() # dictionary mapping window size to MAP estimate
best_W = 0
best_score = -100

def model_factory(history,counts):
    with pm.Model() as model:
        intercept = pm.Normal('intercept', mu=0, sd=10)
        beta = pm.Normal('beta', mu=0, sd=1, shape=W-1)
        beta_recent = pm.Normal('beta_recent', mu=1, sd=1)
        alpha = pm.HalfNormal('alpha', sd=1000)

        mu = intercept
        for i in range(W-1):
            mu += beta[i] * history[:,i]
        mu += beta_recent * history[:,W-1]

        Y_obs = pm.NegativeBinomial('Y_obs', mu=mu, alpha=alpha, observed=counts)
    return model

for W in window_sizes:

    N = len(counts) - W
    x = np.vstack([counts[i:i+W] for i in range(N)])
    y = counts[W:]
    n_valid = int(.2 * N)

    # split data into training and validation (last 20% of examples)
    x_train = x[:-n_valid]
    x_valid = x[-n_valid:]
    y_train = y[:-n_valid]
    y_valid = y[-n_valid:]
    
    model = model_factory(x_train, y_train)
    va_model = model_factory(x_valid, y_valid)

    map[W] = pm.find_MAP(model=model, maxeval=1000)
    print(f'MAP estimate for W={W}')
    for key, arr in map[W].items():
        print(key, arr)
    
    score = va_model['Y_obs'].logp(map[W]) / len(y_valid)
    score_list.append(score)
    if score > best_score:
        best_W = W
        best_score = score
    print(f'Heldout log lik: {score}\n')

print('Best window size:', best_W)
print('Best heldout log lik:', best_score)

plt.figure()
plt.plot(window_sizes, score_list, 'rs-')
plt.title('Performance vs Window Size')
plt.xlabel('Window size')
plt.ylabel('Heldout log lik')


### Write best model parameters to json file ###

model = dict()
model['likelihood'] = 'NegativeBinomial'
model['window_size'] = best_W
priors = {"intercept" : ["Normal", 0, 10],
     "beta" : ["Normal", 0, 1],
     "beta_recent" : ["Normal", 1, 1],
     "alpha" : ["HalfNormal", 1000]
}
model['priors'] = priors

with open(model_file, 'w') as f:
    json.dump(model, f, indent=4)

plt.show()

