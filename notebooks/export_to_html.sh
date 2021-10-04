#!/usr/bin/env bash

for nbfile in `ls TransPrior-0*.ipynb`
do

    echo $nbfile
    outputfile=`python -c "print('$nbfile'.replace('.ipynb', '.html'))"`
    echo $outputfile

    # Automagically pass along os env vars
    jupyter nbconvert \
        --to html \
        --no-input \
        --execute --allow-errors \
        --ExecutePreprocessor.timeout=120 \
        $nbfile \
        --output $outputfile

done
