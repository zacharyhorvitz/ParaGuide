#!/bin/sh

POSMODEL='enron_t5_sentiment/full_positive_train/2023-08-06-13.07.23/checkpoint-184/'
# NEGATIVE_DATA='../holdout_attribute_splits/sentiment_splits/sentiment_0.5_0.5/val_neg_500.tsv'
NEGATIVE_DATA='../holdout_attribute_splits/sentiment_splits/sentiment_0.5_0.5/test_neg_500.tsv'

CUDA_VISIBLE_DEVICES=3 python t5_style_eval.py --out_dir sentiment_inferences_test_block_repeats --model_path $POSMODEL  --input_file $NEGATIVE_DATA --task positive

NEGMODEL='enron_t5_sentiment/full_negative_train/2023-08-06-13.35.05/checkpoint-90/'
# POSITIVE_DATA='../holdout_attribute_splits/sentiment_splits/sentiment_0.5_0.5/val_pos_500.tsv'
POSITIVE_DATA='../holdout_attribute_splits/sentiment_splits/sentiment_0.5_0.5/test_pos_500.tsv'

CUDA_VISIBLE_DEVICES=3 python t5_style_eval.py --out_dir sentiment_inferences_test_block_repeats --model_path $NEGMODEL  --input_file $POSITIVE_DATA --task negative


