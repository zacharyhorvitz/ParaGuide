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

from transformers import RobertaTokenizer

from datasets import load_from_disk

sys.path.append('../../inference')
from classifiers import text_to_style, load_style_model
from luar import load_uar_hf_model, get_uar_embeddings





def add_embeddings(*, example, detokenizer, style_tokenizer, luar_tokenizer, style_model, luar_model, start_idx):
    '''add embeddings to example'''

    output = example['input_ids'][start_idx:]
    decoded = detokenizer.decode(output, skip_special_tokens=True)
    # add embeddings
    # print(decoded)

    if style_model is not None:
        style_embedding = text_to_style(model=style_model, tokenizer=style_tokenizer, texts=[decoded], device='cuda', model_type='style')[0]
        example['style_embedding'] = style_embedding.detach().cpu().numpy()
    
    if luar_model is not None:
        luar_embedding = get_uar_embeddings(model=luar_model, tokenizer=luar_tokenizer, texts=[decoded], device='cuda')[0]
        example['luar_embedding'] = luar_embedding.detach().cpu().numpy()


    return example


# add click args
@click.command()
@click.option('--dataset_path', help='path_to_dataset')
@click.option('--max_text_len', default=50, help='length of text, used to split paraphrase; original text')
def main(dataset_path, max_text_len):
    disable_caching()

    output_path = os.path.normpath(dataset_path + '_with_style_embeds')

    # load dataset
    dataset = load_from_disk(dataset_path)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

    # load wegman style model
    style_model, style_tokenizer, _ = load_style_model()
    style_model.to('cuda')

    # load uar model
    uar_model, uar_tokenizer = load_uar_hf_model()
    uar_model.to('cuda')


    with_embeddings = dataset.map(lambda x: add_embeddings(
        example=x,
        detokenizer=tokenizer,
        style_tokenizer=style_tokenizer,
        luar_tokenizer=uar_tokenizer,
        style_model=style_model,
        luar_model=uar_model,
        start_idx=max_text_len), batched=False)
    

    with_embeddings.save_to_disk(output_path)
    


    
    


if __name__ == '__main__':
    main()
