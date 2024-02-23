#!/bin/sh

set -ex

CUDA_VISIBLE_DEVICES=0 python style_baselines.py --out_dir baseline_eval_test --approach bert --author_directory holdout_author_splits/ --assignments_json holdout_author_splits/assignments.json  &
sleep 20

CUDA_VISIBLE_DEVICES=1 python style_baselines.py --out_dir baseline_eval_test --approach para --author_directory holdout_author_splits/ --assignments_json holdout_author_splits/assignments.json  &
sleep 20

CUDA_VISIBLE_DEVICES=2 python style_baselines.py --out_dir baseline_eval_test --approach ling --author_directory holdout_author_splits/ --assignments_json holdout_author_splits/assignments.json  &
sleep 20
