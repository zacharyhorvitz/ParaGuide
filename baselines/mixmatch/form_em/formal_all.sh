#!/bin/sh
# These use the Bertscore configuration

# alpha = discriminator
# beta = ???
# delta = hamming
# gamma = bleurt score
# theta = bert score

SHARD='test' # 'test' or 'val'

INFORMAL_DATA="~/enron_eval_framework/holdout_attribute_splits/formal_splits/formal_0.5_0.5/${SHARD}_neg.tsv"

FORMAL_DATA="~/enron_eval_framework/holdout_attribute_splits/formal_splits/formal_0.5_0.5/${SHARD}_pos_500.tsv"


INTERNAL_MODEL='cointegrated/roberta-base-formality'

# Informal to Formal (Discriminator) (Roberta Finetuned)
CUDA_VISIBLE_DEVICES=0 python sample_batched_form_em_len_enron.py \
--batch_size 3 \
--n_samples 3 \
--max_iter 5 \
--shuffle_positions \
--temperature 1.0 \
--out_path ./output_samples/formality/enron_neg_to_pos_DISC_enron_FT \
--alpha 140  \
--beta 1 \
--delta 15 \
--gamma 0 \
--theta 100 \
--data_path $INFORMAL_DATA \
--data_name "${SHARD}_neg" \
--disc_name $INTERNAL_MODEL \
--target_label formal \
--disc_dir $INTERNAL_MODEL \
--model_path 'roberta_enron_finetune_128/checkpoint-47000/' &

# Informal to Formal (Hamming) (Roberta Finetuned)
CUDA_VISIBLE_DEVICES=0 python sample_batched_form_em_len_enron.py \
--batch_size 3 \
--n_samples 3 \
--max_iter 5 \
--shuffle_positions \
--temperature 1.0 \
--out_path ./output_samples/formality/enron_neg_to_pos_HAM_enron_FT \
--alpha 140  \
--beta 1 \
--delta 50 \
--gamma 0 \
--theta 300 \
--data_path $INFORMAL_DATA \
--data_name "${SHARD}_neg" \
--disc_name $INTERNAL_MODEL \
--target_label formal \
--disc_dir $INTERNAL_MODEL \
--model_path 'roberta_enron_finetune_128/checkpoint-47000/' &

# Informal to Formal (Discriminator)
CUDA_VISIBLE_DEVICES=1 python sample_batched_form_em_len_enron.py \
--batch_size 3 \
--n_samples 3 \
--max_iter 5 \
--shuffle_positions \
--temperature 1.0 \
--out_path ./output_samples/formality/enron_neg_to_pos_DISC \
--alpha 140  \
--beta 1 \
--delta 15 \
--gamma 0 \
--theta 100 \
--data_path $INFORMAL_DATA \
--data_name "${SHARD}_neg" \
--disc_name $INTERNAL_MODEL \
--target_label formal \
--disc_dir $INTERNAL_MODEL &

# Informal to Formal (Hamming)
CUDA_VISIBLE_DEVICES=1 python sample_batched_form_em_len_enron.py \
--batch_size 3 \
--n_samples 3 \
--max_iter 5 \
--shuffle_positions \
--temperature 1.0 \
--out_path ./output_samples/formality/enron_neg_to_pos_HAM \
--alpha 140  \
--beta 1 \
--delta 50 \
--gamma 0 \
--theta 300 \
--data_path $INFORMAL_DATA \
--data_name "${SHARD}_neg" \
--disc_name $INTERNAL_MODEL \
--target_label formal \
--disc_dir $INTERNAL_MODEL &


########

# Formal to Informal (Discriminator) (Roberta Finetuned)

CUDA_VISIBLE_DEVICES=2 python sample_batched_form_em_len_enron.py \
--batch_size 3 \
--n_samples 3 \
--max_iter 5 \
--shuffle_positions \
--temperature 1.0 \
--out_path ./output_samples/formality/enron_pos_to_neg_DISC_enron_FT \
--alpha 140  \
--beta 1 \
--delta 15 \
--gamma 0 \
--theta 100 \
--data_path $FORMAL_DATA \
--data_name "${SHARD}_pos" \
--disc_name $INTERNAL_MODEL \
--target_label informal \
--disc_dir $INTERNAL_MODEL \
--model_path 'roberta_enron_finetune_128/checkpoint-47000/' &

# Formal to Informal(Hamming) (Roberta Finetuned)
CUDA_VISIBLE_DEVICES=2 python sample_batched_form_em_len_enron.py \
--batch_size 3 \
--n_samples 3 \
--max_iter 5 \
--shuffle_positions \
--temperature 1.0 \
--out_path ./output_samples/formality/enron_pos_to_neg_HAM_enron_FT \
--alpha 140  \
--beta 1 \
--delta 50 \
--gamma 0 \
--theta 300 \
--data_path $FORMAL_DATA \
--data_name "${SHARD}_pos" \
--disc_name $INTERNAL_MODEL \
--target_label informal \
--disc_dir $INTERNAL_MODEL \
--model_path 'roberta_enron_finetune_128/checkpoint-47000/' &

# Formal to Informal (Discriminator)
CUDA_VISIBLE_DEVICES=3 python sample_batched_form_em_len_enron.py \
--batch_size 3 \
--n_samples 3 \
--max_iter 5 \
--shuffle_positions \
--temperature 1.0 \
--out_path ./output_samples/formality/enron_pos_to_neg_DISC \
--alpha 140  \
--beta 1 \
--delta 15 \
--gamma 0 \
--theta 100 \
--data_path $FORMAL_DATA \
--data_name "${SHARD}_pos" \
--disc_name $INTERNAL_MODEL \
--target_label informal \
--disc_dir $INTERNAL_MODEL &

# Formal to Informal (Hamming)
CUDA_VISIBLE_DEVICES=3 python sample_batched_form_em_len_enron.py \
--batch_size 3 \
--n_samples 3 \
--max_iter 5 \
--shuffle_positions \
--temperature 1.0 \
--out_path ./output_samples/formality/enron_pos_to_neg_HAM \
--alpha 140  \
--beta 1 \
--delta 50 \
--gamma 0 \
--theta 300 \
--data_path $FORMAL_DATA \
--data_name "${SHARD}_pos" \
--disc_name $INTERNAL_MODEL \
--target_label informal \
--disc_dir $INTERNAL_MODEL &
