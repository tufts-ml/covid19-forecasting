#!/usr/bin/env bash

nick=20200420_demo
LOCAL_PATH=/cluster/tufts/hugheslab/mhughe02/forecast_results/$nick/
REMOTE_PATH=tuftscs:/h/mhughes/public_html/research/c19-tmc/$nick/

rsync -armPKL \
    --include="/*/dashboard.html" \
    --include="/*/summary*.csv" \
    --exclude="/*/*" \
    $LOCAL_PATH/ \
    $REMOTE_PATH

cp params.json params.txt
python ../../pprint_params.py params.json > human_readable_params.txt

for scen in 1 2 3 4
do
    cp "incoming-$scen.csv" incoming.csv
    column -s, -t "incoming-$scen.csv" > incoming.txt
    rsync -armPKL \
        --include="/*.txt" \
        --include="/params.json" \
        --include="/config.json" \
        --include="/incoming.csv" \
        --exclude="/*" \
        ./ \
        $REMOTE_PATH/"scenario$scen"/
    rm incoming.csv
    rm incoming.txt
done

rm params.txt

# Make permissions so webpage is visible
ssh tuftscs 'bash ~/make_web_content_public.sh'


