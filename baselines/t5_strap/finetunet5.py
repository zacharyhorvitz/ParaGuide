''' finetune t5 on paraphrase data with style encoder '''

#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.


from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup,
)
import os
import sys
import random
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import json

from accelerate import Accelerator


from datasets import load_from_disk
import json
import wandb
from accelerate import Accelerator


class T5Style(torch.nn.Module):
    def __init__(
        self, base_model, use_style=False, num_styles=11, style_extractor_args=None
    ):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(base_model)

    def forward(self, input_ids, attention_mask, labels=None, style=None):
        return self.model(input_ids, attention_mask, labels=labels)

    def generate(self, input_ids, attention_mask, labels=None, style=None, **kwargs):
        return self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )


def collate_fn(batch, max_len=512):
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
    # retval['style'] = torch.tensor([b['style'] for b in batch])
    return retval


def run_model(*, batch, model, device):
    # import pdb; pdb.set_trace()
    input_ids = batch['input_ids']  # .to(device)
    attention_mask = batch['attention_mask']  # .to(device)
    labels = batch['labels']  # .to(device)
    style = None  # batch['style'] #.to(device)
    result = model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels, style=style
    )
    loss = result.loss
    logits = result.logits
    return logits, loss


if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    max_val_batch = 200
    LEARNING_RATE = float(sys.argv[1])  # 1e-5 #5e-6 #2.5e-6 #5e-6 #1e-5
    BATCH_SIZE = 32  # 64 #128 #64 #32 #6 #32 #64
    ACCUMULATION_STEPS = 4  # 1 #2 #2 #4
    # OUTDIR = 'REDDIT_PARAPHRASES/t5_reddit_pretraining' #'strap/cds_finetuned_t5/'
    OUTDIR = 'paraphrase_data/ENRON/shards/holdin_dataset/t5_pretraining'
    device = 'cuda'
    WARMUP_STEPS = 500  # 5000 #5000
    MAX_STEPS = 500000  # 200000
    MODEL_DIM = 768
    # BOTTLENECK_DIM = 512 #128 #64
    # use_style = True
    MODEL = 't5-large'
    # "t5-base"
    # "t5-large"
    # STYLE_EXTRACTOR = 't5-base' #'t5-small'
    # tokenized_data_file_path = 'REDDIT_PARAPHRASES/max_50_para_first_4mil_auth_style_embed/2023-05-08-05.42.53/t5_max_len_50_roberta_determined_min_score_None/'

    tokenized_data_file_path = 'paraphrase_data/ENRON/shards/holdin_dataset/2023-06-22-23.15.35/t5_max_len_50_roberta_determined_min_score_None'
    CHECKPOINT = 'REDDIT_PARAPHRASES/t5_reddit_pretraining/2023-07-27-01.55.52/best_model_t5-large_1e-05_128.pt'

    # add date
    OUTDIR = os.path.join(OUTDIR, datetime.now().strftime("%Y-%m-%d-%H.%M.%S"))
    os.makedirs(OUTDIR, exist_ok=True)

    wandb.init(
        project='ssdlm-enron',  #'ssdlm-reddit-pretrain', #"cds_t5_finetune",
        config={
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "accumulation_steps": ACCUMULATION_STEPS,
            "warmup_steps": WARMUP_STEPS,
            "model": MODEL,
            # "style_extractor": STYLE_EXTRACTOR,
            "outdir": OUTDIR,
            "max_steps": MAX_STEPS,
            # "model_dim": MODEL_DIM,
            # "bottleneck_dim": BOTTLENECK_DIM,
            "max_val_batch": max_val_batch,
            # "use_style": use_style,
            "device": device,
            "tokenized_data_file_path": tokenized_data_file_path,
            "seed": seed,
        },
    )

    accelerator = Accelerator()
    device = accelerator.device
    print(device)

    tokenizer = T5Tokenizer.from_pretrained(MODEL)
    model = T5Style(
        base_model=MODEL
    )  # , use_style=use_style, style_extractor_args=style_extractor_args)

    # load from checkpoint
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))

    tokenized_datasets = load_from_disk(tokenized_data_file_path)

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["val"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        print(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate_fn
    )
    eval_dataloader = DataLoader(
        eval_dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate_fn
    )

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    # scheduler with linear warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=MAX_STEPS
    )

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    fname = f'best_model_{MODEL}_{LEARNING_RATE}_{BATCH_SIZE*ACCUMULATION_STEPS}.pt'

    best_val_loss = None
    optimizer.zero_grad()
    steps = 0
    counter = 0
    for epoch in range(100):
        model.train()
        wandb.log({"epoch": epoch})

        with tqdm(total=len(train_dataloader)) as pbar:
            for i, data in enumerate(train_dataloader):
                _, loss = run_model(batch=data, model=model, device=device)
                if steps % 100 == 0:
                    print('Epoch: ', epoch, ', Train Loss: ', loss.item())

                wandb.log({"train_loss": loss.item()})
                loss = loss / ACCUMULATION_STEPS

                accelerator.backward(loss)
                pbar.update(1)

                if (counter + 1) % ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    steps += 1

                if counter % (1000 * ACCUMULATION_STEPS) == 0:
                    model.eval()
                    losses = []
                    with torch.no_grad():
                        for j, val_data in enumerate(eval_dataloader):
                            if j > max_val_batch:
                                break
                            _, loss = run_model(
                                batch=val_data, model=model, device=device
                            )
                            losses.append(loss.item())

                    val_loss = sum(losses) / len(losses)
                    wandb.log({"val_loss": val_loss})
                    print('Epoch: ', epoch, ', Val Loss: ', val_loss)

                    if best_val_loss is None or val_loss < best_val_loss:
                        best_val_loss = val_loss
                        print(epoch, i, 'New best val loss: ', best_val_loss)
                        with open(
                            os.path.join(OUTDIR, 'checkpoint_info.json'), 'w+'
                        ) as out_:
                            json.dump(
                                {
                                    'epoch': epoch,
                                    'i': i,
                                    'counter': counter,
                                    'steps': steps,
                                    'loss': best_val_loss,
                                },
                                out_,
                            )
                        torch.save(model.state_dict(), os.path.join(OUTDIR, fname))
                        # save optimizer state, save scheduler state
                        torch.save(
                            optimizer.state_dict(), os.path.join(OUTDIR, 'optimizer.pt')
                        )
                        torch.save(
                            scheduler.state_dict(), os.path.join(OUTDIR, 'scheduler.pt')
                        )

                    model.train()

                counter += 1

                if steps >= MAX_STEPS:
                    break

    wandb.finish()
