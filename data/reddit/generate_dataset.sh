#!/bin/sh

# Data from Million Reddit User Dataset (MUD) https://arxiv.org/abs/2105.07263
# Access form: https://docs.google.com/forms/d/e/1FAIpQLSesc-0HI2DRYjFqlpPo2hTh9OJ53jtWjYQiIfAtmzSVUCxiLA/viewform

# Once the data is downloaded and extracted, rename it to data.jsonl and store it in the data folder
DATAFOLDER='data_folder'
if [ -z "$DATAFOLDER" ]
then
      echo "Please specify a location where the downloaded enron data is located."
	  exit 1
fi
PATH_TO_MUD_JSONL=$DATAFOLDER/'data.jsonl'
export CUDA_VISIBLE_DEVICES=0

# Extract samples from the MUD dataset, up to 10 comment samples for each author
python scripts/extract_reddit_samples.py $PATH_TO_MUD_JSONL

# Generate paraphrases for the extracted samples
python ../enron/scripts/generate_paraphrases.py $DATAFOLDER/'400000authors_10perauth_200maxlen.tsv'

# Fix minor issue with double periods in the paraphrases, optional
python scripts/fix_issue.py $DATAFOLDER/'paraphrased_4mil_topp0.8_tmp1.5_idx0_400000authors_10perauth_200maxlen.tsv'

# Split authors into train, val and test sets
python scripts/split_data_respect_authorship.py --path $DATAFOLDER/'paraphrased_4mil_topp0.8_tmp1.5_idx0_400000authors_10perauth_200maxlen.tsv.cleaned' --out_dir $DATAFOLDER/'para_actual_ordered_4mil_400K_author_split'

# Convert into an hf dataset for training
python ../enron/scripts/preprocess_enron_aux.py --train_path $DATAFOLDER/'para_actual_ordered_4mil_400K_author_split/train.txt' \
	--val_path  $DATAFOLDER/'para_actual_ordered_4mil_400K_author_split/val.txt' \
	--test_path  $DATAFOLDER/'para_actual_ordered_4mil_400K_author_split/test.txt' \
	--out_dir  $DATAFOLDER/reddit_training_data
