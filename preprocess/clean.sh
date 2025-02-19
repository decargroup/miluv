#!/bin/bash

CSV_FILE="config/experiments.csv"

tail -n +2 "$CSV_FILE" | while IFS=, read -r exp _; do
    rm -rf "data/$exp/ifo001"
    rm -rf "data/$exp/ifo002"
    rm -rf "data/$exp/ifo003"
    rm -rf "data/$exp/timeshift.yaml"
done