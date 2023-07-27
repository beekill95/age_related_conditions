#!/bin/sh

exp=$1

# Make sure that the experiment exists.
if [ -f "$exp" ]; then
    now=$(date +"%Y%m%d_%H_%M")
    expname=$(basename "$exp")
    dst="./exp_results/${now}_${expname}"

    echo "Record experiment at $dst"
    cp $exp $dst
    echo "Add experiment to git..."
    git add -f "$dst"
    git commit -m "Record experiment $expname at $now."
else
    echo "Experiment `$exp` doesnt exist!"
fi
