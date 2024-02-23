#!/bin/sh
set -ex

for FILE in all_test_sentiment_results/*/positive.jsonl
do
     python eval_attribute.py --input_path $FILE --target positive
done

for FILE in all_test_sentiment_results/*/negative.jsonl
do
    python eval_attribute.py --input_path $FILE --target negative
done


