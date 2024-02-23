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

# from classifiers import text_to_style, load_style_model

SCORE_THRESHOLD = None
MAX_LENGTH = 50  # 25 #50 #100 #200


def get_tokenizer():
    '''get tokenizer'''
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    return tokenizer


def load_line_by_line(path):
    '''read lines from text file'''
    with open(path, 'r') as f:
        for l in f.readlines():
            yield l.strip()


def split_fn(line):
    '''split line'''
    return line.split('\t')


def filter_fn(line):
    '''filter line'''
    return True
    # assert len(line) == 3
    # if float(line[2]) < SCORE_THRESHOLD:
    # return False
    # return True


def get_date():
    '''get date'''
    return datetime.now().strftime("%Y-%m-%d-%H.%M.%S")


def make_dataset(*, train_paths, val_paths, test_paths):
    # dataset = load_dataset("text", data_files={"train": val_paths})
    dataset = load_dataset(
        "text", data_files={"train": train_paths, "val": val_paths, "test": test_paths}
    )
    return dataset


def tokenize_dataset(dataset, tokenizer):
    def tokenize_function(examples):
        # x1, x2, score = examples["text"].split('\t')
        label, x1, x2 = examples["text"].split('\t')
        score = 0  # float(score)
        result1 = {
            "1_" + k: v
            for k, v in tokenizer(
                x1,
            ).items()
        }
        result2 = {
            "2_" + k: v
            for k, v in tokenizer(
                x2,
            ).items()
        }

        length_1 = np.sum(result1["1_attention_mask"])
        length_2 = np.sum(result2["2_attention_mask"])

        return {
            **result1,
            **result2,
            "score": score,
            "length_1": length_1,
            "length_2": length_2,
            "label": label,
        }

    tokenized_dataset = dataset.map(tokenize_function, batched=False)
    return tokenized_dataset


def filter_function(example, max_len, min_score):
    return (
        example["length_1"] <= max_len and example["length_2"] <= max_len
    )  # and example["score"] >= min_score


def combine_and_pad_inputs(example, max_len, pad_token_id):
    def pad_to_max_len(x, pad):
        x = x[:max_len]
        x = x + [pad] * (max_len - len(x))
        return x

    example['input_ids'] = pad_to_max_len(
        example['1_input_ids'], pad=pad_token_id
    ) + pad_to_max_len(example['2_input_ids'], pad=pad_token_id)
    example['attention_mask'] = pad_to_max_len(
        example['1_attention_mask'], pad=0
    ) + pad_to_max_len(example['2_attention_mask'], pad=0)
    return example


def encode_labels(example, label_mapping):
    example['label'] = label_mapping[example['label']]
    return example


def save_info(path, info):
    with open(path, 'w') as f:
        json.dump(info, f)


# add click args
@click.command()
@click.option('--train_path', help='train file path')
@click.option('--val_path', help='val file path', default=None)
@click.option('--test_path', help='test file path', default=None)
@click.option('--out_dir', help='out dir')
@click.option('--add_embeds', is_flag=True, help='add style embeds')
def main(train_path, val_path, test_path, out_dir, add_embeds):
    disable_caching()
    name = f'max_len_{MAX_LENGTH}_min_score_{SCORE_THRESHOLD}'
    if val_path is None:
        print('val_path is None, using train_path as val_path')
        val_path = train_path
    if test_path is None:
        print('test_path is None, using val_path as test_path')
        test_path = val_path
    dataset = make_dataset(
        train_paths=train_path, val_paths=val_path, test_paths=test_path
    )
    tokenizer = get_tokenizer()
    tokenized = tokenize_dataset(dataset, tokenizer=tokenizer)
    filtered = tokenized.filter(
        lambda example: filter_function(
            example, max_len=MAX_LENGTH, min_score=SCORE_THRESHOLD
        )
    )

    label_mapping = {
        k: i
        for i, k in enumerate(
            sorted(
                set(
                    filtered['train']['label']
                    + filtered['val']['label']
                    + filtered['test']['label']
                )
            )
        )
    }

    print(label_mapping)
    filtered = filtered.map(lambda example: encode_labels(example, label_mapping))

    combined_inputs = filtered.map(
        lambda example: combine_and_pad_inputs(
            example, max_len=MAX_LENGTH, pad_token_id=tokenizer.pad_token_id
        )
    )
    # combined_inputs = dataset
    if add_embeds:
        raise NotImplementedError()
        # style_model, style_tokenizer, _ = load_style_model()
        # style_model.to('cuda')
        # style_model.eval()
        # def add_embeds(examples):
        #     output_text = []
        #     for text in examples['text']:
        #         _, _, o_text = text.split('\t')
        #         output_text.append(o_text)
        #     # print(output_text)
        #     with torch.no_grad():
        #         embed = text_to_style(model=style_model, tokenizer=style_tokenizer, texts=output_text, device='cuda')
        #         embed = [e.detach().cpu().numpy() for e in embed]
        #     examples['style_embed'] = embed
        #     return examples
        # combined_inputs = combined_inputs.map(add_embeds, batched=True, batch_size=32)

    combined_inputs = combined_inputs.remove_columns(
        ["1_input_ids", "1_attention_mask", "2_input_ids", "2_attention_mask", "text"]
    )
    out_dir = os.path.join(out_dir, get_date())
    print(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    info = {
        "name": name,
        "tokenizer": tokenizer.name_or_path,
        "max_length": MAX_LENGTH,
        "min_score": SCORE_THRESHOLD,
        "num_train_examples": len(combined_inputs["train"]),
        "num_val_examples": len(combined_inputs["val"]),
        "num_test_examples": len(combined_inputs["test"]),
        'train_path': train_path,
        'val_path': val_path,
        'test_path': test_path,
        'label_mapping': label_mapping,
    }
    save_info(os.path.join(out_dir, "info.json"), info)
    combined_inputs.save_to_disk(os.path.join(out_dir, name))


if __name__ == '__main__':
    main()
