'''
plot_forecasts.py
-----------------
Plots mean, median, 2.5 and 97.5 percentiles for each prediction on the given
axis. Sets xlabels to dates of predictions, starting from given start date.
'''

import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from datetime import timedelta

def plot_forecasts(samples, start, ax, observed, future=True):
    n_predictions = len(samples[0])

    if future == False:
        assert len(observed) == n_predictions

    low = np.zeros(n_predictions)
    high = np.zeros(n_predictions)
    mean = np.zeros(n_predictions)
    median = np.zeros(n_predictions)

    for i in range(n_predictions):
        low[i] = np.percentile(samples[:,i], 2.5)
        high[i] = np.percentile(samples[:,i], 97.5)
        median[i] = np.percentile(samples[:,i], 50)
        mean[i] = np.mean(samples[:,i])

    x_future = np.arange(n_predictions)
    ax.errorbar(x_future, median,
                yerr=[median-low, high-median],
                capsize=2, fmt='.', linewidth=1,
                label='2.5, 50, 97.5 percentiles')
    ax.plot(x_future, mean, 'x', label='mean')

    # Plot observed data points
    if future == True:
        x_past = np.arange(-len(observed), 0)
        ax.plot(x_past, observed, 's', label='observed')
        dates = np.full(len(observed) + n_predictions, start)
        for i in range(-len(observed), n_predictions):
            dates[i] = start + timedelta(i)
        ax.set_xticks(np.concatenate((x_past, x_future)))

    else:
        ax.plot(x_future, observed, 's', label='observed')
        dates = np.full(n_predictions, start)
        for i in range(n_predictions):
            dates[i] = start + timedelta(i)
        ax.set_xticks(x_future)

    ax.set_xticklabels(dates, rotation=30)
    ax.legend()