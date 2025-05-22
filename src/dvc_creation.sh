#!/bin/bash

# Stage 1: Processing
dvc stage add -n processing --force \
    -d data/raw/train.csv \
    -d src/process_data.py \
    -o data/processed/train.csv \
    -o models/column_transformer.pkl \
    uv run python src/process_data.py

# Stage 2: Training
dvc stage add -n train_model --force \
    -d src/train_model.py \
    -d data/processed/train.csv \
    -o model.pkl \
    "uv run python src/train_model.py --in data/processed/train.csv --out model.pkl"