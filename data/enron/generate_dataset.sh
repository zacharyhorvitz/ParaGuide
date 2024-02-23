#/bin/bash
set -ex

# Download Enron corpus  (May 7, 2015 version of dataset https://www.cs.cmu.edu/~./enron/)
# We use emails.csv from https://www.kaggle.com/datasets/wcukierski/enron-email-dataset
# Download this file and put it in $DATAFOLDER/

DATAFOLDER=''
if [ -z "$DATAFOLDER" ]
then
      echo "Please specify a location where the downloaded enron data is located."
	  exit 1
fi

export CUDA_VISIBLE_DEVICES=0

EMAILPATH=$DATAFOLDER'/emails.csv'

# Clean data, filtering duplicates and lengthy emails, and exluding users with < 10 sent emails
python scripts/filter_senders_and_clean.py $EMAILPATH

# Generate (paraphrase, original text) pairs
python scripts/generate_paraphrases.py $DATAFOLDER/users_data_50_unique_clean_min_10_fixed_sender.tsv

# Split data into holdin and holdout authors, and split holdin into training/validation/test
python scripts/split_into_shards.py $DATAFOLDER/paraphrased_topp0.8_tmp1.5_idx0_users_data_50_unique_clean_min_10_fixed_sender.tsv

# Split reports from each holdout author into training/validation/test
python scripts/holdout_split_over_authors.py $DATAFOLDER/shards/holdout_author_data.txt

# Re-aggregate by holdout author, which will making per author evaluation easier
# Also specifies authorship-transfer evaluation assignment for each author
python scripts/split_by_holdout_author.py $DATAFOLDER/shards/holdout_shards_splits

# Classify holdout data with formality and positive classifiers
python scripts/holdout_models.py --target formal --out_dir $DATAFOLDER/holdout_attribute_splits --input_paths $DATAFOLDER'/shards/holdout_shards_splits/train.txt' $DATAFOLDER'/shards/holdout_shards_splits/val.txt'  $DATAFOLDER'/shards/holdout_shards_splits/test.txt' 
python scripts/holdout_models.py --target positive --out_dir $DATAFOLDER/holdout_attribute_splits --input_paths $DATAFOLDER'/shards/holdout_shards_splits/train.txt' $DATAFOLDER'/shards/holdout_shards_splits/val.txt'  $DATAFOLDER'/shards/holdout_shards_splits/test.txt' 

# Split data in holdout shards into formal and informal, and positive and negative, using previous scores
python scripts/split_by_score.py --input_file $DATAFOLDER/holdout_attribute_splits/formal_scored.tsv --output_dir $DATAFOLDER/shards/holdout_attribute_splits/formal_splits --min_score 0.5 --max_score 0.5 --task formal
python scripts/split_by_score.py --input_file $DATAFOLDER/holdout_attribute_splits/positive_scored.tsv --output_dir $DATAFOLDER/shards/holdout_attribute_splits/sentiment_splits --min_score 0.5 --max_score 0.5 --task sentiment

# Limit test to 500 examples for pos/neg labels each, due to slow baseline inference times

head -n 500 $DATAFOLDER/shards/holdout_attribute_splits/sentiment_splits/sentiment_0.5_0.5/test_pos.tsv >> $DATAFOLDER/shards/holdout_attribute_splits/sentiment_splits/sentiment_0.5_0.5/test_pos_500.tsv 
head -n 500 $DATAFOLDER/shards/holdout_attribute_splits/sentiment_splits/sentiment_0.5_0.5/test_neg.tsv >> $DATAFOLDER/shards/holdout_attribute_splits/sentiment_splits/sentiment_0.5_0.5/test_neg_500.tsv 
head -n 500 $DATAFOLDER/shards/holdout_attribute_splits/formal_splits/formal_0.5_0.5/test_pos.tsv >> $DATAFOLDER/shards/holdout_attribute_splits/formal_splits/formal_0.5_0.5/test_pos_500.tsv
head -n 500 $DATAFOLDER/shards/holdout_attribute_splits/formal_splits/formal_0.5_0.5/test_neg.tsv >> $DATAFOLDER/shards/holdout_attribute_splits/formal_splits/formal_0.5_0.5/test_neg_500.tsv

# Convert NON-HOLDOUT data into hf dataset for training
python scripts/preprocess_enron_aux.py --train_path $DATAFOLDER/shards/holdin_author_data_train.txt \
	--val_path  $DATAFOLDER/shards/holdin_author_data_dev.txt \
	--test_path  $DATAFOLDER/shards/holdin_author_data_test.txt \
	--out_dir  $DATAFOLDER/shards/holdin_dataset





