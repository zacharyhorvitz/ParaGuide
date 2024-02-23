#!/bin/sh
set -ex

# baselines
for FILE in all_test_style_results/*/style.jsonl
do
     python eval_style.py --input_path $FILE --embed_model style --author_directory ../data/enron/holdout_author_splits/
     # python eval_style.py --input_path $FILE --embed_model luar --author_directory ../data/enron/holdout_author_splits/
done




