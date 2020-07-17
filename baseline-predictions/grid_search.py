'''
grid_search.py
--------------
Runs grid search for GLM and/or GGP.
User can specify -l to run grid search for GLM only, or -g for GGP only.
Defaults to using both models and producing side-by-side plots of performance.
Plots heldout log likelihood vs window size for GLM, and heldout log likelihood
vs time-scale prior for GGP.
'''

import logging
import argparse
import arg_types
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glm_grid_search import glm_grid_search
from ggp_grid_search import ggp_grid_search

if __name__ == '__main__':

	# logger = logging.getLogger('pymc3')
	# logger.setLevel(logging.ERROR)
	logger = logging.getLogger('arviz')
	logger.setLevel(logging.ERROR)

	parser = argparse.ArgumentParser()

	parser.add_argument('input_csv_file', type=arg_types.csv_file, help='name of input CSV file')
	parser.add_argument('-c', '--target_col_name', default='cases',
	                    help='column of CSV file with counts to make predictions on, default \'cases\'')

	parser.add_argument('-l', '--linear_model', action='store_true', help='only use GLM')
	parser.add_argument('-g', '--gaussian_process', action='store_true', help='only use GGP')

	parser.add_argument('-m', '--glm_model_file', type=arg_types.json_file, default='glm_model.json',
						help='name of JSON file to write GLM model parameters to, default \'glm_model.json\'')
	parser.add_argument('-f', '--ggp_model_file', type=arg_types.json_file, default='ggp_model.json',
	                    help='name of JSON file to write GGP model parameters to, default \'ggp_model.json\'')

	parser.add_argument('-p', '--plot_file', type=arg_types.png_file, default='performance.png',
						help='name of PNG file to save plot as, default \'performance.png\'')
	
	args = parser.parse_args()

	train_df = pd.read_csv(args.input_csv_file)
	counts = train_df[args.target_col_name].values

	np.random.seed(42)

	if args.linear_model:
		fig,ax = plt.subplots()
		glm_grid_search(counts, args.glm_model_file, ax)

	elif args.gaussian_process:
		fig,ax = plt.subplots()
		ggp_grid_search(counts, args.ggp_model_file, ax)

	else:
		fig,ax = plt.subplots(ncols=2, sharey=True, figsize=(12,6))
		glm_grid_search(counts, args.glm_model_file, ax[0])
		ggp_grid_search(counts, args.ggp_model_file, ax[1])

	plt.savefig(args.plot_file)
	# plt.show()

