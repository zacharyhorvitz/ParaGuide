# alpha = discriminator
# beta = ???
# delta = hamming
# gamma = bleurt score
# theta = bert score



SHARD='test' # 'test' or 'val'


NEGATIVE_DATA="~/enron_eval_framework/holdout_attribute_splits/sentiment_splits/sentiment_0.5_0.5/${SHARD}_neg_500.tsv"
POSITIVE_DATA="~/enron_eval_framework/holdout_attribute_splits/sentiment_splits/sentiment_0.5_0.5/${SHARD}_pos_500.tsv"
INTERNAL_MODEL='cardiffnlp/twitter-roberta-base-sentiment'

# Negative to Positive (Discriminator) (Roberta Finetuned)
CUDA_VISIBLE_DEVICES=0 python sample_batched_form_em_len_enron.py \
--batch_size 3 \
--n_samples 3 \
--max_iter 8 \
--shuffle_positions \
--temperature 1.0 \
--out_path ./output_samples/sentiment/enron_neg_to_pos_DISC_enron_FT \
--alpha 100  \
--beta 1 \
--delta 25 \
--gamma 0 \
--theta 0 \
--data_path $NEGATIVE_DATA \
--data_name "${SHARD}_neg" \
--disc_name $INTERNAL_MODEL \
--target_label positive \
--disc_dir $INTERNAL_MODEL \
--model_path 'roberta_enron_finetune_128/checkpoint-47000/' &

# Negative to Positive (Hamming) (Roberta Finetuned)
CUDA_VISIBLE_DEVICES=1 python sample_batched_form_em_len_enron.py \
--batch_size 3 \
--n_samples 3 \
--max_iter 8 \
--shuffle_positions \
--temperature 1.0 \
--out_path ./output_samples/sentiment/enron_neg_to_pos_HAM_enron_FT \
--alpha 100  \
--beta 1 \
--delta 50 \
--gamma 0 \
--theta 0 \
--data_path $NEGATIVE_DATA \
--data_name "${SHARD}_neg" \
--disc_name $INTERNAL_MODEL \
--target_label positive \
--disc_dir $INTERNAL_MODEL \
--model_path 'roberta_enron_finetune_128/checkpoint-47000/' &

# Negative to Positive (Discriminator)
CUDA_VISIBLE_DEVICES=0 python sample_batched_form_em_len_enron.py \
--batch_size 3 \
--n_samples 3 \
--max_iter 8 \
--shuffle_positions \
--temperature 1.0 \
--out_path ./output_samples/sentiment/enron_neg_to_pos_DISC \
--alpha 100  \
--beta 1 \
--delta 25 \
--gamma 0 \
--theta 0 \
--data_path $NEGATIVE_DATA \
--data_name "${SHARD}_neg" \
--disc_name $INTERNAL_MODEL \
--target_label positive \
--disc_dir $INTERNAL_MODEL &

# Negative to Positive (Hamming)
CUDA_VISIBLE_DEVICES=1 python sample_batched_form_em_len_enron.py \
--batch_size 3 \
--n_samples 3 \
--max_iter 8 \
--shuffle_positions \
--temperature 1.0 \
--out_path ./output_samples/sentiment/enron_neg_to_pos_HAM \
--alpha 100  \
--beta 1 \
--delta 50 \
--gamma 0 \
--theta 0 \
--data_path $NEGATIVE_DATA \
--data_name "${SHARD}_neg" \
--disc_name $INTERNAL_MODEL \
--target_label positive \
--disc_dir $INTERNAL_MODEL &


#######

# Positive to Negative (Discriminator) (Roberta Finetuned)

CUDA_VISIBLE_DEVICES=2 python sample_batched_form_em_len_enron.py \
--batch_size 3 \
--n_samples 3 \
--max_iter 8 \
--shuffle_positions \
--temperature 1.0 \
--out_path ./output_samples/sentiment/enron_pos_to_neg_DISC_enron_FT \
--alpha 100  \
--beta 1 \
--delta 25 \
--gamma 0 \
--theta 0 \
--data_path $POSITIVE_DATA \
--data_name "${SHARD}_pos" \
--disc_name $INTERNAL_MODEL \
--target_label negative \
--disc_dir $INTERNAL_MODEL \
--model_path 'roberta_enron_finetune_128/checkpoint-47000/' &

# Positive to Negative(Hamming) (Roberta Finetuned)
CUDA_VISIBLE_DEVICES=3 python sample_batched_form_em_len_enron.py \
--batch_size 3 \
--n_samples 3 \
--max_iter 8 \
--shuffle_positions \
--temperature 1.0 \
--out_path ./output_samples/sentiment/enron_pos_to_neg_HAM_enron_FT \
--alpha 100  \
--beta 1 \
--delta 50 \
--gamma 0 \
--theta 0 \
--data_path $POSITIVE_DATA \
--data_name "${SHARD}_pos" \
--disc_name $INTERNAL_MODEL \
--target_label negative \
--disc_dir $INTERNAL_MODEL \
--model_path 'roberta_enron_finetune_128/checkpoint-47000/' &

# Positive to Negative (Discriminator)
CUDA_VISIBLE_DEVICES=2 python sample_batched_form_em_len_enron.py \
--batch_size 3 \
--n_samples 3 \
--max_iter 8 \
--shuffle_positions \
--temperature 1.0 \
--out_path ./output_samples/sentiment/enron_pos_to_neg_DISC \
--alpha 100  \
--beta 1 \
--delta 25 \
--gamma 0 \
--theta 0 \
--data_path $POSITIVE_DATA \
--data_name "${SHARD}_pos" \
--disc_name $INTERNAL_MODEL \
--target_label negative \
--disc_dir $INTERNAL_MODEL &

# Positive to Negative (Hamming)
CUDA_VISIBLE_DEVICES=3 python sample_batched_form_em_len_enron.py \
--batch_size 3 \
--n_samples 3 \
--max_iter 8 \
--shuffle_positions \
--temperature 1.0 \
--out_path ./output_samples/sentiment/enron_pos_to_neg_HAM \
--alpha 100  \
--beta 1 \
--delta 50 \
--gamma 0 \
--theta 0 \
--data_path $POSITIVE_DATA \
--data_name "${SHARD}_pos" \
--disc_name $INTERNAL_MODEL \
--target_label negative \
--disc_dir $INTERNAL_MODEL &