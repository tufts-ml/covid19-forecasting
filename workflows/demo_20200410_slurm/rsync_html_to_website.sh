#!/usr/bin/env bash

LOCAL_PATH=/cluster/tufts/hugheslab/mhughe02/forecast_results/demo_20200410/
REMOTE_PATH=tuftscs:/h/mhughes/public_html/research/c19-tmc/demo_20200410/

rsync -armPKL \
    --include="/*/dashboard.html" \
    --include="/*/summary*.csv" \
    --exclude="/*/*" \
    $LOCAL_PATH/ \
    $REMOTE_PATH

