#!/bin/sh
set -ex

MODEL_DIR="$(dirname "$0")"

MODEL_DOWNLOAD_DIR='/mnt/swordfish-pool2/horvitz/enron_luar_paraguide' #'/burg/nlp/users/zfh2000/enron_model' #"$(dirname "$0")" # Feel free to change this to your preferred location

#gdown https://drive.google.com/file/d/1qlIVBGnvjrqxHvFdp0-5WNwZb6iM3pCA/view?usp=sharing --fuzzy -O ${MODEL_DOWNLOAD_DIR}/enron_model.zip

gdown https://drive.google.com/file/d/1ijQxv5NjBJgnpwaVoqn49GUSISZ32S_T/view?usp=sharing --fuzzy -O ${MODEL_DOWNLOAD_DIR}/enron_model_luar_conditioned.zip

#unzip $MODEL_DOWNLOAD_DIR/enron_model.zip -d $MODEL_DOWNLOAD_DIR
unzip $MODEL_DOWNLOAD_DIR/enron_model_luar_conditioned.zip -d $MODEL_DOWNLOAD_DIR

#ln -s ${MODEL_DOWNLOAD_DIR}/best_checkpoint ${MODEL_DIR}/best_checkpoint  # if using a different preferred location, you may want to create a symlink to the model directory
ln -s ${MODEL_DOWNLOAD_DIR}/best_checkpoint_luar_conditioned ${MODEL_DIR}/best_checkpoint_luar_conditioned  # if using a different preferred location, you may want to create a symlink to the model directory
