
# CUDA_VISIBLE_DEVICES=1 python finetune_t5_hf_task.py \
# --task formal \
# --training_path ~/enron_eval_framework/holdout_attribute_splits/formal_splits/formal_0.5_0.5/train_pos_500.tsv \
# --val_path ~/enron_eval_framework/holdout_attribute_splits/formal_splits/formal_0.5_0.5/val_pos.tsv \
# --out_dir enron_t5_formality/formal_train_500 \
# --pretrained_path paraphrase_data/ENRON/shards/holdin_dataset/t5_pretraining/2023-08-02-16.48.42/best_model_t5-large_0.0001_128.pt \
# --batch_size 16 \
# --learning_rate 1e-4 \
# --accumulation_steps 4 \
# --seed 42

# CUDA_VISIBLE_DEVICES=1 python finetune_t5_hf_task.py \
# --task formal \
# --training_path ~/enron_eval_framework/holdout_attribute_splits/formal_splits/formal_0.5_0.5/train_pos_1000.tsv \
# --val_path ~/enron_eval_framework/holdout_attribute_splits/formal_splits/formal_0.5_0.5/val_pos.tsv \
# --out_dir enron_t5_formality/formal_train_1000 \
# --pretrained_path paraphrase_data/ENRON/shards/holdin_dataset/t5_pretraining/2023-08-02-16.48.42/best_model_t5-large_0.0001_128.pt \
# --batch_size 16 \
# --learning_rate 1e-4 \
# --accumulation_steps 4 \
# --seed 42

# CUDA_VISIBLE_DEVICES=0 python finetune_t5_hf_task.py \
#  --task formal \
#  --training_path ~/enron_eval_framework/holdout_attribute_splits/formal_splits/formal_0.5_0.5/train_pos.tsv \
#  --val_path ~/enron_eval_framework/holdout_attribute_splits/formal_splits/formal_0.5_0.5/val_pos.tsv \
#  --out_dir enron_t5_formality/full_formal_train_smaller_lr \
#  --pretrained_path paraphrase_data/ENRON/shards/holdin_dataset/t5_pretraining/2023-08-02-16.48.42/best_model_t5-large_0.0001_128.pt \
#  --batch_size 8 \
#  --learning_rate 1e-5 \
#  --accumulation_steps 8 \
#  --seed 42 &

# CUDA_VISIBLE_DEVICES=1 python finetune_t5_hf_task.py \
# --task informal \
# --training_path ~/enron_eval_framework/holdout_attribute_splits/formal_splits/formal_0.5_0.5/train_neg_500.tsv \
# --val_path ~/enron_eval_framework/holdout_attribute_splits/formal_splits/formal_0.5_0.5/val_neg.tsv \
# --out_dir enron_t5_formality/informal_train_500 \
# --pretrained_path paraphrase_data/ENRON/shards/holdin_dataset/t5_pretraining/2023-08-02-16.48.42/best_model_t5-large_0.0001_128.pt \
# --batch_size 16 \
# --learning_rate 1e-4 \
# --accumulation_steps 4 \
# --seed 42


#CUDA_VISIBLE_DEVICES=1 python finetune_t5_hf_task.py \
#--task informal \
#--training_path ~/enron_eval_framework/holdout_attribute_splits/formal_splits/formal_0.5_0.5/train_neg_1000.tsv \
#--val_path ~/enron_eval_framework/holdout_attribute_splits/formal_splits/formal_0.5_0.5/val_neg.tsv \
#--out_dir enron_t5_formality/informal_train_1000 \
#--pretrained_path paraphrase_data/ENRON/shards/holdin_dataset/t5_pretraining/2023-08-02-16.48.42/best_model_t5-large_0.0001_128.pt \
#--batch_size 8 \
#--learning_rate 1e-4 \
#--accumulation_steps 4 \
#--seed 42


# CUDA_VISIBLE_DEVICES=1 python finetune_t5_hf_task.py \
# --task informal \
# --training_path ~/enron_eval_framework/holdout_attribute_splits/formal_splits/formal_0.5_0.5/train_neg.tsv \
# --val_path ~/enron_eval_framework/holdout_attribute_splits/formal_splits/formal_0.5_0.5/val_neg.tsv \
# --out_dir enron_t5_formality/full_informal_train_smaller_lr \
# --pretrained_path paraphrase_data/ENRON/shards/holdin_dataset/t5_pretraining/2023-08-02-16.48.42/best_model_t5-large_0.0001_128.pt \
# --batch_size 8 \
# --learning_rate 1e-5 \
# --accumulation_steps 8 \
# --seed 42 &


