#!/bin/bash
set -ex

DATA_DIR="$(dirname "$0")"
DATA_DOWNLOAD_DIR='/burg/nlp/users/zfh2000/enron_download' #'/mnt/swordfish-pool2/horvitz/paraphrase_data/ENRON/shards/holdin_dataset/' #"$(dirname "$0")" # Feel free to change this to your preferred location

gdown https://drive.google.com/file/d/1-IGVQDMRG9vhR9vgQcqUsg0oo36f9e3H/view?usp=sharing --fuzzy -O ${DATA_DOWNLOAD_DIR}/enron_data.zip

unzip $DATA_DOWNLOAD_DIR/enron_data.zip -d $DATA_DOWNLOAD_DIR

ln -s ${DATA_DOWNLOAD_DIR}/2023-06-22-23.15.35 ${DATA_DIR}/2023-06-22-23.15.35  # if using a different preferred location, you may want to create a symlink to the model directory
