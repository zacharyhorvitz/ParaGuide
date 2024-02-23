#!/bin/sh

set -ex


python hit_openai.py --out_dir chatgpt_test --approach chatgpt --author_directory ../holdout_author_splits/ --assignments_json ../holdout_author_splits/assignments.json
