import argparse
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

def figures_to_html(figs, filename="dashboard.html"):
    dashboard = open(filename, 'w')
    dashboard.write("<html><head></head><body><h2>Tufts Medical Center COVID-19 Hospital Impact Dashboard</h2>" + "\n")
    for fig in figs:
        inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
        dashboard.write(inner_html)
    dashboard.write("</body></html>" + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dash', default=True)

    args = parser.parse_args()
    dash = args.dash
    print(dash)

    df = load_output_data()
    figures = make_figure(df)

    if dash==True:
        print("Called here.")
        return figures
    else:
        figures_to_html(figures)



main()

