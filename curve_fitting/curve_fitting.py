import numpy as np
import scipy
import scipy.optimize
import argparse
import sklearn.metrics
import matplotlib.pyplot as plt
import autograd.scipy
import autograd.numpy as ag_np
import autograd
import pandas as pd

## TODO:
#   - Need a mapping from timesteps to dates

################## Functions for fitting data ########################

def erf(t, alpha, beta, p):
    return 0.5*p*(scipy.special.erf(alpha*(t - beta)) + 1.0)

def ag_erf(t, alpha, beta, p):
    return 0.5*p*(autograd.scipy.special.erf(alpha*(t - beta)) + 1.0)

#####################################################################

def load_data(input_filename, fit_type):
    df = pd.read_csv(input_filename)
    if fit_type == 'cumulative':
        cumulative = []
        for i in range(df['rate'].shape[0]):
            if i == 0:
                cumulative.append(df['rate'][i])
            else:
                cumulative.append(df['rate'][i] + cumulative[i-1])
        y = np.array(cumulative)
    elif fit_type == 'log_rate':
        y = np.log(np.array(df['rate']) + 1e-13)
    
    x = ag_np.arange(y.shape[0], dtype=float)
    dates = df['date']
    return x, dates, y

def extend_forecast(x, dates, max_date):
    pass

def calc_rate(y):
    rate = []
    for i in range(y.shape[0]):
        if i == 0:
            rate.append(y[i])
        elif i > 0:
            rate.append(y[i] - y[i -1])
    return np.array(rate)

def calc_cumulative(y):
    cumulative = []
    for i in range(y.shape[0]):
        if i == 0:
            cumulative.append(y[i])
        else:
            cumulative.append(y[i] + cumulative[i-1])
    return np.array(cumulative)

def save_results(x, y_true, fun, params, fit_type, dates, outputfile):
    x_ext = ag_np.arange(x.shape[0] + 30) # extend prediction by a month
    y = fun(x_ext, *params)
    if fit_type == 'cumulative':
        rate = calc_rate(y)
        rate_true = calc_rate(y_true)
    elif fit_type == 'log_rate':
        rate = ag_np.exp(y)
        rate_true = ag_np.exp(y_true)

    plt.scatter(x, y_true, label='True data')
    plt.plot(x_ext, y, label='Prediction')
    plt.ylabel(fit_type)
    plt.show()

    plt.scatter(x, rate_true, label='True data')
    plt.plot(x_ext, rate, label='Prediction')
    plt.ylabel('rate')
    plt.show()

    # TODO: compute dates using mapping

    # save results to outputfile as csv
    df = pd.DataFrame({'date': x_ext,'rate': rate, 'cumulative': calc_cumulative(rate)})
    df.to_csv(outputfile)

    return x_ext, rate

############## loss calculation ###############

def MSE(y_true, y_hat):
    return ag_np.sum(ag_np.power(y_true - y_hat, 2)) / y_true.shape[0]

#####################################################################

FUNCTIONS = {'erf': erf, 'ag_erf': ag_erf}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--function', default='ag_erf')
    parser.add_argument('--fit_type', default='cumulative')
    parser.add_argument('--lower_bound', default='0.0')
    parser.add_argument('--inputfile', default='input_example_mass_positives.csv')
    parser.add_argument('--outputfile', default='output_example_mass_positives.csv')
    args = parser.parse_args()

    fun = FUNCTIONS[args.function]
    fit_type = args.fit_type
    lower_bound = float(args.lower_bound)

    num_params = 3
    seed_list = ag_np.arange(5)
    x, dates, y = load_data(args.inputfile, fit_type)

    best_loss = ag_np.inf
    best_seed = 0

    def calc_loss(params):
        '''
        Default loss is MSE.
        '''
        yhat = fun(x, *params)
        loss = MSE(y, yhat)
        return loss

    for seed in seed_list:
        ag_np.random.seed(seed)
        initial_guess = ag_np.random.random(num_params)

        result = scipy.optimize.minimize(
            calc_loss,
            initial_guess,
            jac=autograd.grad(calc_loss),
            method='l-bfgs-b',
            constraints={},
            # changing the lower bounds shifts the peak, we can explore
            # this for the sake of confidence intervals.
            bounds=[(0, ag_np.inf), (0, ag_np.inf), (np.max(y)*lower_bound, ag_np.inf)])
        params = result.x

        loss = calc_loss(params)
        if loss < best_loss:
            best_loss = loss
            best_seed = seed

    ag_np.random.seed(best_seed)
    initial_guess = ag_np.random.random(num_params)

    result = scipy.optimize.minimize(
                                    calc_loss,
                                    initial_guess,
                                    jac=autograd.grad(calc_loss),
                                    method='l-bfgs-b',
                                    constraints={},
                                    bounds=[(0, ag_np.inf), (0, ag_np.inf), (np.max(y)*lower_bound, ag_np.inf)])
    params = result.x
    save_results(x, y, fun, params, fit_type, dates, args.outputfile)


