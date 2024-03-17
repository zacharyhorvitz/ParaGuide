#!/usr/bin/bash

trap "kill 0" EXIT

script_role="host"
global_seed=1234
hf_cache="~/.cache/huggingface"

# Path to pretrained checkpoint (e.g, our enron checkpoint, or our reddit checkpoint or the original SSD-LM model (https://huggingface.co/xhan77/ssdlm/tree/main))
core_lm_name='../models/best_checkpoint/'

# Where to save new checkpoints
main_log_dir='/burg/nlp/users/zfh2000/enron_model/' #"/mnt/swordfish-pool2/horvitz/test_new_paraguide_train" 

# Path to huggingface dataset
tokenized_path='../data/enron/2023-06-22-23.15.35/max_len_50_min_score_None_with_style_embeds/'

# Wandb project name
project_name='paraguide_model_training' 

config_path='/home/horvitz/.cache/huggingface/accelerate/default_config.yaml' # run accelerate config to configure
 
# retrain
retrain_num_train_epochs=10000 # just a placeholder, use max train steps
retrain_per_device_train_batch_size=128
retrain_per_device_eval_batch_size=128
retrain_learning_rate=5e-6
retrain_weight_decay=0.01
retrain_gradient_accumulation_steps=1
retrain_num_warmup_steps=2000
retrain_max_train_steps=500000

sigma_num_steps=200 # 5000 if pretraining on reddit
loss_mode="xe"
remove_noise_mode="no_dir"
pa=5
cs=50
precision="fp16" # no or fp16
noise_manual_scale=1.0
train_mode="resume"

################ START ################

HF_HOME=${hf_cache} accelerate launch   \
    --config_file ${config_path} \
    --mixed_precision ${precision} \
    --main_process_ip 'localhost' \
    --num_processes 3 --num_machines 1 --machine_rank 0 \
    --num_cpu_threads_per_process 2 \
    train.py \
    --max_seq_length -1 \
    --model_name_or_path ${core_lm_name} \
    --num_train_epochs ${retrain_num_train_epochs} \
    --per_device_train_batch_size ${retrain_per_device_train_batch_size} \
    --per_device_eval_batch_size ${retrain_per_device_eval_batch_size} \
    --learning_rate ${retrain_learning_rate} \
    --weight_decay ${retrain_weight_decay} \
    --gradient_accumulation_steps ${retrain_gradient_accumulation_steps} \
    --num_warmup_steps ${retrain_num_warmup_steps} \
    --max_train_steps ${retrain_max_train_steps} \
    --seed ${global_seed} \
    --use_slow_tokenizer \
    --output_dir ${main_log_dir}/ssd_cs_dbs${cs} \
    --loss_mode ${loss_mode} \
    --remove_noise_mode ${remove_noise_mode} \
    --hardcoded_pseudo_diralpha ${pa} \
    --context_size ${cs} \
    --sigma_num_steps ${sigma_num_steps} \
    --noise_manual_scale ${noise_manual_scale} \
    --tokenized_data_file_path ${tokenized_path} \
    --if_create_tokenized_data_file "no" \
    --train_mode ${train_mode} \
    --project_name ${project_name} \
    --use_sqrt_schedule \
    --use_style_embed \
    --ctr_embed_dim 512
