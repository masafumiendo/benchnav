#!/bin/bash

MODEL_URL="https://github.com/masafumiendo/benchnav/releases/download/v0.0/trained_models.zip"
DATASET_URL="https://github.com/masafumiendo/benchnav/releases/download/v0.0/datasets.zip"

CURRENT_DIR=$(cd $(dirname $0) && pwd)
ROOT_DIR=$(cd $CURRENT_DIR/.. && pwd)

# Download the model and dataset
echo "Downloading the trained models..."
curl -L $MODEL_URL -o $ROOT_DIR/trained_models.zip
echo "Downloading the datasets..."
curl -L $DATASET_URL -o $ROOT_DIR/datasets.zip

unzip $ROOT_DIR/trained_models.zip -d $ROOT_DIR 
unzip $ROOT_DIR/datasets.zip -d $ROOT_DIR

rm $ROOT_DIR/trained_models.zip
rm $ROOT_DIR/datasets.zip