'''
gar_forecast.py
---------------
Fits a GenPoissonAutoregression model with the parameters in model_dict to
the given count data. Produces n_samples of forecasts for n_predictions
days ahead. Writes forecasts to CSV files with the given filename pattern,
and returns array of forecasts.
'''

from GenPoissonAutoregression import GenPoissonAutoregression
from plot_forecasts import plot_forecasts

def gar_forecast(model_dict, counts, n_samples, n_predictions,
                 output_csv_file_pattern, start, ax):

    T = len(counts)

    model = GenPoissonAutoregression(model_dict)
    model.fit(counts, n_predictions)

    samples = model.forecast(n_samples, output_csv_file_pattern)
    ax.set_title('GAR Forecasts')
    plot_forecasts(samples, start, ax, counts[-5:])