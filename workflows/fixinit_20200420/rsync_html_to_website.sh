#!/usr/bin/env bash

nick=fixinit_20200420
LOCAL_PATH=/cluster/tufts/hugheslab/mhughe02/forecast_results/$nick/
REMOTE_PATH=tuftscs:/h/mhughes/public_html/research/c19-tmc/$nick/

rsync -armPKL \
    --include="/*/dashboard.html" \
    --include="/*/summary*.csv" \
    --exclude="/*/*" \
    $LOCAL_PATH/ \
    $REMOTE_PATH

cp params.json params.txt

for scen in "scenario1" "scenario2" "scenario3"
do
    rsync -armPKL \
        --include="/params.txt" \
        --include="/params.json" \
        --include="/config.json" \
        --exclude="/*" \
        ./ \
        $REMOTE_PATH/$scen/
done

rm params.txt

