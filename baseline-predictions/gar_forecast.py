from GPAR import GPAR
from plot_forecasts import plot_forecasts

def gar_forecast(model_dict, counts, n_samples, n_predictions,
                 output_csv_file_pattern, start, ax):

    T = len(counts)

    model = GPAR(model_dict)
    model.fit(counts, n_predictions)

    samples = model.forecast(n_samples, output_csv_file_pattern)
    ax.set_title('GAR Forecasts')
    plot_forecasts(samples, start, ax, counts[-5:])