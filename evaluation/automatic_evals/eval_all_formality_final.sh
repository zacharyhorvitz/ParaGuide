#!/bin/sh
set -ex

for FILE in all_test_formality_results/*/formal.jsonl
do
    python eval_attribute.py --input_path $FILE --target formal 
done

for FILE in all_test_formality_results/*/informal.jsonl
do
    python eval_attribute.py --input_path $FILE --target informal 
done

