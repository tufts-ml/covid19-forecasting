'''
glm_forecast.py
---------------
Fits a NegativeBinomialRegression model with the parameters specified in
model_dict to the given count data. Produces n_samples of forecasts for
n_predictions days ahead. Writes forecasts to CSV files with the given
filename pattern, and returns array of forecasts.
'''

import numpy as np
from NegativeBinomialRegression import NegativeBinomialRegression

def glm_forecast(model_dict, counts, n_samples, n_predictions, output_csv_file_pattern):
	W = model_dict['window_size']

	N = len(counts) - W
	x = np.vstack([counts[i:i+W] for i in range(N)])
	y = counts[W:]

	model = NegativeBinomialRegression(model_dict)
	model.fit(x, y)
	# print(f'\nPosterior Means:')
	# for var in ['intercept', 'beta', 'beta_recent', 'alpha']:
	#     print(var, model.post_mean[var])
	# print()

	score = model.score(x, y)
	print(f'\nScore on training data = {score}\n')

	samples = model.forecast(counts, n_samples, n_predictions, output_csv_file_pattern)
	return samples