
gpu=$1
folder_file=$2

# for folder in ../holdout_author_splits/*/; do
cat $folder_file | while read line 
do

    folder=$line
    echo $folder
    authorname=$(basename $folder)
 
    CUDA_VISIBLE_DEVICES=$1 python finetune_t5_hf_task.py \
    --task authorship \
    --training_path $folder/train.txt \
    --val_path $folder/val.txt \
    --out_dir enron_t5_ft_evals/$authorname \
    --assignments_json ../holdout_author_splits/assignments.json \
    --pretrained_path paraphrase_data/ENRON/shards/holdin_dataset/t5_pretraining/2023-08-02-16.48.42/best_model_t5-large_0.0001_128.pt \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --accumulation_steps 8 \
    --seed 42

    #--batch_size 16 \
    #--batch_size 64 \
    #--accumulation_steps 1 \
done
