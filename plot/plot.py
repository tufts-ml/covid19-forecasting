import json
import pandas as pd
import re
import plotly.graph_objects as go


def load_output_data():
    with open('config.json') as f:
        config = json.load(f)

    path = config.pop('output_dir')

    dataframes = []
    for file_type, file_name in config.items():
        percentile = re.findall('\d*\.?\d+', file_name)[0]

        df_temp = pd.read_csv(path + file_name)
        df_temp['percentile'] = percentile

        dataframes.append(df_temp)

    return pd.concat(dataframes)


def make_figure(df):
    percentiles = sorted(df.percentile.unique())
    y_cols = list(df.columns)
    y_cols.remove('timestep')
    y_cols.remove('percentile')

    figures = []
    for y in y_cols:
        fig = go.Figure()

        for p in percentiles:
            df_filtered = df[df['percentile'] == p]

            fig.add_trace(go.Scatter(
                x=df_filtered.timestep,
                y=df_filtered[y],
                name='%tile: ' + str(p))
            )

            fig.update_layout(
                title=y,
                xaxis_title="Time Step",
                yaxis_title="Patient Count",
                font=dict(
                    size=16,
                    color="#7f7f7f"
                )
            )

        figures.append(fig)

    return figures


def main():
    df = load_output_data()
    return make_figure(df)



main()

