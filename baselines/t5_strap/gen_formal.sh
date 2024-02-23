#!/bin/sh

FORMALMODEL1='enron_t5_formality/formal_train_500/2023-08-03-23.36.43/checkpoint-24/'

FORMALMODEL2='enron_t5_formality/formal_train_1000/2023-08-04-00.02.16/checkpoint-30/'

FORMALMODEL3='enron_t5_formality/full_formal_train/2023-08-04-00.18.51/checkpoint-204/'

# INFORMAL_DATA='../holdout_attribute_splits/formal_splits/formal_0.5_0.5/val_neg.tsv'
INFORMAL_DATA='../holdout_attribute_splits/formal_splits/formal_0.5_0.5/test_neg.tsv'

# python t5_style_eval.py --out_dir formal_inferences --model_path $FORMALMODEL1  --input_file $INFORMAL_DATA --task formal
# python t5_style_eval.py --out_dir formal_inferences --model_path $FORMALMODEL2  --input_file $INFORMAL_DATA --task formal

CUDA_VISIBLE_DEVICES=1 python t5_style_eval.py --out_dir formal_inferences_test_beam_2 --model_path $FORMALMODEL3  --input_file $INFORMAL_DATA --task formal

INFORMALMODEL1='enron_t5_formality/informal_train_500/2023-08-04-00.40.15/checkpoint-24/'
INFORMALMODEL2='enron_t5_formality/informal_train_1000/2023-08-04-00.55.44/checkpoint-30/'
INFORMALMODEL3='enron_t5_formality/full_informal_train/2023-08-04-11.45.15/checkpoint-70/'

# FORMAL_DATA='../holdout_attribute_splits/formal_splits/formal_0.5_0.5/val_pos.tsv'

# FORMAL_DATA='../holdout_attribute_splits/formal_splits/formal_0.5_0.5/val_pos_500.tsv'

FORMAL_DATA='../holdout_attribute_splits/formal_splits/formal_0.5_0.5/test_pos_500.tsv'


# python t5_style_eval.py --out_dir formal_inferences --model_path $INFORMALMODEL1  --input_file $FORMAL_DATA --task informal
# python t5_style_eval.py --out_dir formal_inferences --model_path $INFORMALMODEL2  --input_file $FORMAL_DATA --task informal
CUDA_VISIBLE_DEVICES=2 python t5_style_eval.py --out_dir formal_inferences_test_beam_2 --model_path $INFORMALMODEL3  --input_file $FORMAL_DATA --task informal
