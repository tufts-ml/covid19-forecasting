'''
ggp_forecast.py
---------------
Fits a NegativeBinomialGP model with the parameters in model_dict to
the given count data. Produces n_samples of forecasts for n_predictions
days ahead. Writes forecasts to CSV files with the given filename pattern,
and returns array of forecasts.
'''

import numpy as np
from NegativeBinomialGP import NegativeBinomialGP

def ggp_forecast(model_dict, counts, n_samples, n_predictions, output_csv_file_pattern):
	T = len(counts)
	t = np.arange(T)[:,None]

	model = NegativeBinomialGP(model_dict)
	model.fit(t, counts)
	score = model.score(t, counts)
	print(f'\nScore on training data = {score}\n')

	samples = model.forecast(counts, n_samples, n_predictions, output_csv_file_pattern)
	return samples