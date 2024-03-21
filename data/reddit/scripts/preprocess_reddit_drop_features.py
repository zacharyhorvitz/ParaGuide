''' preprocess paranmt sample data '''

import os
import sys
import json
import random
import numpy as np

# import torch
import click
from datasets import Dataset
from datasets import load_dataset
from datasets import disable_caching
from datetime import datetime
from tqdm import tqdm
import torch


from datasets import load_from_disk


# add click args
@click.command()
@click.option('--dataset_path', help='path_to_dataset', default='/burg/nlp/users/zfh2000/reddit_data/max_len_50_min_score_None')
def main(dataset_path):
    disable_caching()



    output_path = os.path.normpath(dataset_path + '_just_input_ids')

    # load dataset
    dataset = load_from_disk(dataset_path)

    #print columns 
    print(dataset.column_names)

    # Drop these columns 'score', 'length_1', 'length_2', 'attention_mask', 'style_embed'
    dataset = dataset.remove_columns(['score', 'length_1', 'length_2', 'attention_mask', 'style_embed'])

    # import pdb; pdb.set_trace()

    # save dataset
    dataset.save_to_disk(output_path)
    


    
    


if __name__ == '__main__':
    main()
