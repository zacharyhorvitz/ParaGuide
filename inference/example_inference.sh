MODEL_PATH='../models/best_checkpoint/'

# Informal --> Formal
python inference.py \
	--out_dir example_results_formal \
	--lr 200 \
	--top_p 0.8 \
   	--kl_loss_weight 0.0 \
	--semantic_loss_weight 0.0 \
	--size 50 \
	--ctr_embed_dim 768 \
	--model_path  $MODEL_PATH \
	--total_t 200  \
	--num_drift_steps 3  \
	--task formal  \
	--input_path '../data/enron/holdout_attribute_splits/formal_splits/formal_0.5_0.5/test_neg.tsv' \
	--temperature 3.0 \
	--use_sqrt_schedule

# Authorship style transfer with Wegmann Style Embeddings (https://huggingface.co/AnnaWegmann/Style-Embedding)
python inference.py \
	--out_dir example_results_authorship \
	--lr 2500 \
	--top_p 0.8 \
   	--kl_loss_weight 0.0 \
	--semantic_loss_weight 0.0 \
	--size 50 \
	--ctr_embed_dim 768 \
	--model_path $MODEL_PATH \
	--total_t 200  \
	--num_drift_steps 3  \
	--task style  \
	--temperature 3.0 \
	--use_sqrt_schedule \
	--author_directory ../data/enron/holdout_author_splits \
	--assignments_json ../data/enron/holdout_author_splits/assignments.json