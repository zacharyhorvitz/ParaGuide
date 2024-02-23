#!/bin/sh

#python get_results_in_correct_format.py \
#--input_file ~/enron_eval_framework/holdout_attribute_splits/formal_splits/formal_0.5_0.5/val_neg.tsv \
#--samples_file output_samples/enron_form_em_val_neg_to_formal/disc_cointegrated/roberta-base-formality_data_val_neg_max_iter_5_temp_1.0_shuffle_True_block_False_alpha_140.0_beta_1.0_delta_50.0_gamma_0.0_theta_300.0_date_04_08_2023_00_40_00/opt_samples.txt \
#--target_label formal

#python get_results_in_correct_format.py \
#--input_file ~/enron_eval_framework/holdout_attribute_splits/formal_splits/formal_0.5_0.5/val_pos_500.tsv \
#--samples_file 'output_samples/enron_form_em_pos_to_informal_max_iter_5_500/disc_cointegrated/roberta-base-formality_data_val_pos_500_max_iter_5_temp_1.0_shuffle_True_block_False_alpha_140.0_beta_1.0_delta_50.0_gamma_0.0_theta_300.0_date_05_08_2023_04_09_26/opt_samples.txt' \
#--target_label informal


# informal -> formal
for FILE in output_samples/formality/enron_neg_to_pos*/*/*/opt_samples.txt
do

	python get_results_in_correct_format.py \
		--input_file ~/enron_eval_framework/holdout_attribute_splits/formal_splits/formal_0.5_0.5/test_neg.tsv \
		--samples_file $FILE \
		--target_label  formal
	sleep 10
done


# formal -> informal
for FILE in output_samples/formality/enron_pos_to_neg*/*/*/opt_samples.txt
do

	python get_results_in_correct_format.py \
		--input_file ~/enron_eval_framework/holdout_attribute_splits/formal_splits/formal_0.5_0.5/test_pos_500.tsv \
		--samples_file $FILE \
		--target_label  informal
	sleep 10
done

# negative -> positive
for FILE in output_samples/sentiment/enron_neg_to_pos*/*/*/opt_samples.txt
do
	python get_results_in_correct_format.py \
		--input_file ~/enron_eval_framework/holdout_attribute_splits/sentiment_splits/sentiment_0.5_0.5/test_neg_500.tsv \
		--samples_file $FILE \
		--target_label  positive
	sleep 10
done


# positive -> negative
for FILE in output_samples/sentiment/enron_pos_to_neg*/*/*/opt_samples.txt
do
	python get_results_in_correct_format.py \
		--input_file ~/enron_eval_framework/holdout_attribute_splits/sentiment_splits/sentiment_0.5_0.5/test_pos_500.tsv \
		--samples_file $FILE \
		--target_label  negative
	sleep 10
done
