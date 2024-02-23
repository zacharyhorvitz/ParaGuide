''' finetune t5 on paraphrase data with style encoder '''

#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.


from transformers import (
    T5Tokenizer,
    T5EncoderModel,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup,
    AdamW,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# trainer


import argparse
from cmath import exp
import logging
import math
from multiprocessing.sharedctypes import Value
import os
import random

from datetime import datetime
import torch

import shutil
import wandb


import argparse
import wandb

from finetunet5 import T5Style, collate_fn
from t5_style_eval import evaluate_on_assignments


def collate_fn(batch):
    inputs = [b['input_ids'][:max_len] for b in batch]
    labels = [b['labels'][:max_len] for b in batch]

    max_input_len = max([len(i) for i in inputs])
    max_label_len = max([len(l) for l in labels])

    for i in range(len(inputs)):
        inputs[i] = inputs[i] + [0] * (max_input_len - len(inputs[i]))
        labels[i] = labels[i] + [-100] * (max_label_len - len(labels[i]))

    retval = {}
    retval['input_ids'] = torch.tensor(inputs)
    retval['attention_mask'] = (retval['input_ids'] != 0).long()
    retval['labels'] = torch.tensor(labels)
    return retval


# define dataset for style transfer
class AuthorshipStyleTransferDataset(torch.utils.data.Dataset):
    # use
    def __init__(self, *, task, shard_path, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.task = task
        self.shard = shard_path
        with open(shard_path, 'r') as f:
            self.data = [l.strip().split('\t') for l in f.readlines()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if len(sample) == 2:
            para, text = sample
            author = None
        elif len(sample) == 3:
            author, para, text = sample
        else:
            raise ValueError('Invalid data format')
        task = self.task

        input_ids = self.tokenizer.encode(
            para, max_length=self.max_len, truncation=True
        )
        label_ids = self.tokenizer.encode(
            text, max_length=self.max_len, truncation=True
        )
        return {
            'input_ids': input_ids,
            'labels': label_ids,
            'author': author,
            'task': task,
            'para': para,
            'text': text,
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--task', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--pretrained_path', type=str)
    parser.add_argument('--training_path', type=str)
    parser.add_argument('--val_path', type=str)
    parser.add_argument('--assignments_json', type=str)

    # example usage
    # python finetune_t5_hf_task.py
    # --task authorship
    # --training_path $folder/train.txt
    # --val_path $folder/val.txt
    # --out_dir $authorname
    # --assignment_json ../holdout_author_splits/assignments.json
    # --pretrained_path paraphrase_data/ENRON/shards/holdin_dataset/t5_pretraining/2023-08-02-16.48.42/best_model_t5-large_0.0001_128.pt \
    # --batch_size 64 \
    # --learning_rate 1e-4 \
    # --accumulation_steps 1 \
    # --seed 42

    args = parser.parse_args()
    seed = args.seed
    random.seed(seed)

    OUTDIR = os.path.join(args.out_dir, datetime.now().strftime("%Y-%m-%d-%H.%M.%S"))
    os.makedirs(OUTDIR, exist_ok=False)

    # load data
    tokenizer = T5Tokenizer.from_pretrained('t5-large')
    max_len = 512
    train_dataset = AuthorshipStyleTransferDataset(
        task=args.task,
        shard_path=args.training_path,
        tokenizer=tokenizer,
        max_len=max_len,
    )
    val_dataset = AuthorshipStyleTransferDataset(
        task=args.task, shard_path=args.val_path, tokenizer=tokenizer, max_len=max_len
    )

    wandb.init(
        project=f't5_{args.task}_finetune',
        config={
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'accumulation_steps': args.accumulation_steps,
            'seed': seed,
            'task': args.task,
            'out_dir': args.out_dir,
            'training_path': args.training_path,
            'val_path': args.val_path,
            'max_len': max_len,
        },
    )

    # accelerator = Accelerator()
    # device = accelerator.device
    T5Style("t5-large", use_style=False, num_styles=None)
    model = T5Style("t5-large", use_style=False, num_styles=None)
    model.load_state_dict(
        torch.load(args.pretrained_path, map_location='cuda')
    )  # use pretrained fn eventually
    model.to('cuda')

    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)

    # use hugging face trainer
    training_args = TrainingArguments(
        output_dir=OUTDIR,
        num_train_epochs=30,  # 500,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_dir=OUTDIR,
        logging_steps=1,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        seed=seed,
        learning_rate=args.learning_rate,
        # save_steps=1,
        # eval_steps=1,
        report_to="wandb",
        lr_scheduler_type='constant',
    )

    trainer = Trainer(
        model.model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()

    # save model
    model.model.eval()

    if args.assignments_json is not None:
        author = os.path.dirname(args.training_path).split('/')[-1]
        if author == 'From_scott_gardner@pgn.com':
            fixed_name = 'From: scott_gardner@pgn.com'
        elif author == 'From_djenergy@dowjones.com':
            fixed_name = 'From: djenergy@dowjones.com'
        else:
            fixed_name = author.replace('From', 'From:').replace('_', ' ')

        evaluate_on_assignments(
            model=model,
            tokenizer=tokenizer,
            out_dir=args.out_dir,
            assignments_json=args.assignments_json,
            target_author=fixed_name,
            sample_key='val_samples',
            args=args,
        )
        evaluate_on_assignments(
            model=model,
            tokenizer=tokenizer,
            out_dir=args.out_dir,
            assignments_json=args.assignments_json,
            target_author=fixed_name,
            sample_key='test_samples',
            args=args,
        )

        # delete every checkpoint

        for filename in os.listdir(OUTDIR):
            if 'checkpoint-' in filename:
                shutil.rmtree(os.path.join(OUTDIR, filename))
