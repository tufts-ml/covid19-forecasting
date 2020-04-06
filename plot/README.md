# covid19-forecasting plots

A simple Plotly Dash application that displays plots for various stages of hospitalization

PI: Michael C. Hughes

## Usage

Configurations are stored in plot/config.json

### Two workflows are supported:
View the charts by running a Plotly Dash app (using a local webserver)
Embed the Plotly graphs in an html page

### To run the Plotly Dash app:

```
$ cd plot
$ python app.py
```

Visit: 
[http://127.0.0.1:8050/](http://127.0.0.1:8050/)

### To embed the Plotly graphs in an html page
```
$ cd plot
$ python plot.py --dash False
```

The plots are available in plot/dashboard.html


## Summary



## Install

#### 1. Install Plotly Dash

```
pip install dash==1.10.0
```
