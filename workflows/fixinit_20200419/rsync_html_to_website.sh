#!/usr/bin/env bash

nick=fixinit_20200419
LOCAL_PATH=/cluster/tufts/hugheslab/mhughe02/forecast_results/$nick/
REMOTE_PATH=tuftscs:/h/mhughes/public_html/research/c19-tmc/$nick/

rsync -armPKL \
    --include="/*/dashboard.html" \
    --include="/*/summary*.csv" \
    --exclude="/*/*" \
    $LOCAL_PATH/ \
    $REMOTE_PATH

cp params.json params.txt

rsync -armPKL \
    --include="/params.txt" \
    --include="/params.json" \
    --include="/config.json" \
    --exclude="/*" \
    ./ \
    $REMOTE_PATH

rm params.txt

