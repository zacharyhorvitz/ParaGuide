
# CUDA_VISIBLE_DEVICES=2 python finetune_t5_hf_task.py \
# --task positive \
# --training_path ~/enron_eval_framework/holdout_attribute_splits/sentiment_splits/sentiment_0.5_0.5/train_pos_500.tsv \
# --val_path ~/enron_eval_framework/holdout_attribute_splits/sentiment_splits/sentiment_0.5_0.5/val_pos.tsv \
# --out_dir enron_t5_sentiment/sentiment_train_500 \
# --pretrained_path paraphrase_data/ENRON/shards/holdin_dataset/t5_pretraining/2023-08-02-16.48.42/best_model_t5-large_0.0001_128.pt \
# --batch_size 16 \
# --learning_rate 1e-4 \
# --accumulation_steps 4 \
# --seed 42

# CUDA_VISIBLE_DEVICES=2 python finetune_t5_hf_task.py \
# --task positive \
# --training_path ~/enron_eval_framework/holdout_attribute_splits/sentiment_splits/sentiment_0.5_0.5/train_pos_1000.tsv \
# --val_path ~/enron_eval_framework/holdout_attribute_splits/sentiment_splits/sentiment_0.5_0.5/val_pos.tsv \
# --out_dir enron_t5_sentiment/positive_train_1000 \
# --pretrained_path paraphrase_data/ENRON/shards/holdin_dataset/t5_pretraining/2023-08-02-16.48.42/best_model_t5-large_0.0001_128.pt \
# --batch_size 16 \
# --learning_rate 1e-4 \
# --accumulation_steps 4 \
# --seed 42

#CUDA_VISIBLE_DEVICES=2 python finetune_t5_hf_task.py \
#--task positive \
#--training_path ~/enron_eval_framework/holdout_attribute_splits/sentiment_splits/sentiment_0.5_0.5/train_pos.tsv \
#--val_path ~/enron_eval_framework/holdout_attribute_splits/sentiment_splits/sentiment_0.5_0.5/val_pos_500.tsv \
#--out_dir enron_t5_sentiment/full_positive_train \
#--pretrained_path paraphrase_data/ENRON/shards/holdin_dataset/t5_pretraining/2023-08-02-16.48.42/best_model_t5-large_0.0001_128.pt \
#--batch_size 16 \
#--learning_rate 1e-4 \
#--accumulation_steps 4 \
#--seed 42

# CUDA_VISIBLE_DEVICES=3 python finetune_t5_hf_task.py \
# --task positive \
# --training_path ~/enron_eval_framework/holdout_attribute_splits/sentiment_splits/sentiment_0.5_0.5/train_pos.tsv \
# --val_path ~/enron_eval_framework/holdout_attribute_splits/sentiment_splits/sentiment_0.5_0.5/val_pos_500.tsv \
# --out_dir enron_t5_sentiment/full_positive_train_lower_lr \
# --pretrained_path paraphrase_data/ENRON/shards/holdin_dataset/t5_pretraining/2023-08-02-16.48.42/best_model_t5-large_0.0001_128.pt \
# --batch_size 8 \
# --learning_rate 1e-5 \
# --accumulation_steps 8 \
# --seed 42 &


# CUDA_VISIBLE_DEVICES=2 python finetune_t5_hf_task.py \
# --task negative \
# --training_path ~/enron_eval_framework/holdout_attribute_splits/sentiment_splits/sentiment_0.5_0.5/train_neg_500.tsv \
# --val_path ~/enron_eval_framework/holdout_attribute_splits/sentiment_splits/sentiment_0.5_0.5/val_neg.tsv \
# --out_dir enron_t5_sentiment/negative_train_500 \
# --pretrained_path paraphrase_data/ENRON/shards/holdin_dataset/t5_pretraining/2023-08-02-16.48.42/best_model_t5-large_0.0001_128.pt \
# --batch_size 16 \
# --learning_rate 1e-4 \
# --accumulation_steps 4 \
# --seed 42

# CUDA_VISIBLE_DEVICES=2 python finetune_t5_hf_task.py \
# --task negative \
# --training_path ~/enron_eval_framework/holdout_attribute_splits/sentiment_splits/sentiment_0.5_0.5/train_neg_1000.tsv \
# --val_path ~/enron_eval_framework/holdout_attribute_splits/sentiment_splits/sentiment_0.5_0.5/val_neg.tsv \
# --out_dir enron_t5_sentiment/negative_train_1000 \
# --pretrained_path paraphrase_data/ENRON/shards/holdin_dataset/t5_pretraining/2023-08-02-16.48.42/best_model_t5-large_0.0001_128.pt \
# --batch_size 16 \
# --learning_rate 1e-4 \
# --accumulation_steps 4 \
# --seed 42

#CUDA_VISIBLE_DEVICES=2 python finetune_t5_hf_task.py \
#--task negative \
#--training_path ~/enron_eval_framework/holdout_attribute_splits/sentiment_splits/sentiment_0.5_0.5/train_neg.tsv \
#--val_path ~/enron_eval_framework/holdout_attribute_splits/sentiment_splits/sentiment_0.5_0.5/val_neg_500.tsv \
#--out_dir enron_t5_sentiment/full_negative_train \
#--pretrained_path paraphrase_data/ENRON/shards/holdin_dataset/t5_pretraining/2023-08-02-16.48.42/best_model_t5-large_0.0001_128.pt \
#--batch_size 16 \
#--learning_rate 1e-4 \
#--accumulation_steps 4 \
#--seed 42

# CUDA_VISIBLE_DEVICES=2 python finetune_t5_hf_task.py \
# --task negative \
# --training_path ~/enron_eval_framework/holdout_attribute_splits/sentiment_splits/sentiment_0.5_0.5/train_neg.tsv \
# --val_path ~/enron_eval_framework/holdout_attribute_splits/sentiment_splits/sentiment_0.5_0.5/val_neg_500.tsv \
# --out_dir enron_t5_sentiment/full_negative_train_lower_lr \
# --pretrained_path paraphrase_data/ENRON/shards/holdin_dataset/t5_pretraining/2023-08-02-16.48.42/best_model_t5-large_0.0001_128.pt \
# --batch_size 8 \
# --learning_rate 1e-5 \
# --accumulation_steps 8 \
# --seed 42 &

