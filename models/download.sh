#!/bin/sh
set -ex

MODEL_DIR="$(dirname "$0")"

MODEL_DOWNLOAD_DIR="$(dirname "$0")" # Feel free to change this to your preferred location

# gdown https://drive.google.com/file/d/1qlIVBGnvjrqxHvFdp0-5WNwZb6iM3pCA/view?usp=sharing --fuzzy -O ${MODEL_DOWNLOAD_DIR}/enron_model.zip

# unzip $MODEL_DOWNLOAD_DIR/enron_model.zip -d $MODEL_DOWNLOAD_DIR

# ln -s ${MODEL_DOWNLOAD_DIR}/best_checkpoint ${MODEL_DIR}/best_checkpoint  # if using a different preferred location, you may want to create a symlink to the model directory
