# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import plot

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

children = []
children.append(html.H2(children='Tufts Medical Center COVID-19 Hospital Impact Dashboard'))

for i in plot.main():
    children.append(dcc.Graph(
        figure=i))

app.layout = html.Div(
    className='container-fluid', children=children
)

if __name__ == '__main__':
    app.run_server(debug=True)