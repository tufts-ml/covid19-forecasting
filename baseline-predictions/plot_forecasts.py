'''
plot_forecasts.py
-----------------

Plots mean, median, 2.5 and 97.5 percentiles for each prediction on
the given axis. Sets xlabels to dates of predictions, starting from
given start date.
'''

import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from datetime import timedelta

def plot_forecasts(samples, n_predictions, start, ax):
	low = np.zeros(n_predictions)
	high = np.zeros(n_predictions)
	mean = np.zeros(n_predictions)
	median = np.zeros(n_predictions)

	for i in range(n_predictions):
	    low[i] = np.percentile(samples[:,i], 2.5)
	    high[i] = np.percentile(samples[:,i], 97.5)
	    median[i] = np.percentile(samples[:,i], 50)
	    mean[i] = np.mean(samples[:,i])

	pred_dates = np.full(n_predictions, start)
	for i in range(n_predictions):
	    pred_dates[i] = start + timedelta(i)

	xticks = np.arange(n_predictions)

	ax.errorbar(xticks, median,
	             yerr=[median-low, high-median],
	             capsize=2, fmt='.', linewidth=1,
	             label='2.5, 50, 97.5 percentiles')
	ax.plot(xticks, mean, 'rx', label='mean')
	ax.set_xticks(xticks)
	ax.set_xticklabels(pred_dates, rotation=30)
	ax.legend()