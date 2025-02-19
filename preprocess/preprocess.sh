#!/bin/bash

CSV_FILE="config/experiments.csv"

task(){
    python preprocess/read_bags.py data/$exp True
    python preprocess/process_uwb.py data/$exp
    python preprocess/cleanup_csv.py data/$exp
}

N=16
(
tail -n +2 "$CSV_FILE" | while IFS=, read -r exp _; do
    rm -rf "data/$exp/ifo001"
    rm -rf "data/$exp/ifo002"
    rm -rf "data/$exp/ifo003"
    rm -rf "data/$exp/timeshift.yaml"
    ((i=i%N)); ((i++==0)) && wait
    task $exp &
done
)