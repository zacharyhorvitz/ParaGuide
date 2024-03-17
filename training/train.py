#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""
# You can also adapt this script on your own mlm task. Pointers for this are left as comments.

# DIFFUSION CODE ADAPTED FROM https://github.com/xhan77/ssd-lm

from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta

import argparse
import logging
import os
import random

import datasets
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
import accelerate
from accelerate import Accelerator

from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler,
)
from transformers.utils.versions import require_version

import numpy as np
from datasets import load_from_disk
import json
import wandb

from training_utils import *


logger = logging.getLogger(__name__)
require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a Masked Language Modeling task"
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--data_cap",
        type=int,
        default=2,
        help="Max number of data for which we will save graidents.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        '--ctr_embed_dim', type=int, default=768, help='CTR embedding dimension'
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
    )
    parser.add_argument(
        "--line_by_line",
        type=bool,
        default=False,
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.0,
        help="Ratio of tokens to mask for masked language modeling loss",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--hub_token", type=str, help="The token to use to push to the Model Hub."
    )
    parser.add_argument(
        "--no_save_grads",
        action="store_true",
        help="Whether to save gradients to a file.",
    )
    parser.add_argument(
        "--init_blank_language_model",
        action="store_true",
        help="Whether or not to use a completely blank LM.",
    )
    parser.add_argument(
        "--tokenized_data_file_path",
        action='append',
        nargs='+',
        help="Path of the tokenized data file.",
    )  # required=True
    parser.add_argument('-i', action='append', nargs='+')
    parser.add_argument(
        "--if_create_tokenized_data_file",
        type=str,
        default=None,
        help="Whether to create a new tokenized data file (yes or no).",
    )
    parser.add_argument(
        "--sigma_start_value",
        type=float,
        default=-1,
        help="",
    )
    parser.add_argument(
        "--sigma_end_value",
        type=float,
        default=-1,
        help="",
    )
    parser.add_argument(
        "--sigma_num_steps",
        type=int,
        default=1000,
        help="",
    )
    parser.add_argument(
        "--loss_mode",
        type=str,
        default="",
        help="",
    )
    parser.add_argument(
        "--remove_noise_mode",
        type=str,
        default="",
        help="",
    )
    parser.add_argument(
        "--hardcoded_pseudo_diralpha",
        type=float,
        default=3,
        help="",
    )  # this is the one-hot value (simplex from logits can be seen as the mean of a Dirichlet distribution)
    parser.add_argument(
        "--context_size",
        type=int,
        default=50,
        help="",
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="",
        help="",
    )
    parser.add_argument(
        "--noise_manual_scale",
        type=float,
        default=1,
        help="",
    )
    parser.add_argument(
        "--decode_ctr_lr",
        type=float,
        default=0.0,
        help="",
    )
    # have use_label flag to indicate whether to use label or not
    parser.add_argument('--use_label', action='store_true')
    parser.add_argument('--use_style_embed', action='store_true')
    parser.add_argument('--distribute_padding', action='store_true')
    parser.add_argument('--use_self_condition', action='store_true')
    parser.add_argument('--self_cond_prob', default=0.5, type=float)
    parser.add_argument('--use_sqrt_schedule', action='store_true')
    parser.add_argument('--project_name', type=str, required=True)

    args = parser.parse_args()

    return args


def yield_from_dataloader_dict(dataloader_dict, reset=True):
    names = sorted(list(dataloader_dict.keys()))
    enumerations = [iter(enumerate(dataloader_dict[k])) for k in names]
    raised_stop_iteration = [False for _ in names]
    epochs = [0 for _ in names]
    global_step = 0

    while True:
        for idx in range(len(names)):
            if raised_stop_iteration[idx]:
                continue
            name = names[idx]
            enumeration = enumerations[idx]
            epoch = epochs[idx]

            try:
                step, batch = next(enumeration)
            except StopIteration:
                raised_stop_iteration[idx] = True
                if reset:
                    enumerations[idx] = iter(enumerate(dataloader_dict[name]))
                    raised_stop_iteration[idx] = False
                    epochs[idx] += 1
                    step, batch = next(enumerations[idx])

            if all(raised_stop_iteration):
                raise StopIteration

            yield global_step, name, epoch, step, batch
            global_step += 1


def main():
    print('starting main loop')

    args = parse_args()

    assert not (
        args.use_label and args.use_style_embed
    ), "Cannot use both label and style embedding"
    if args.use_label:
        print("Using label")
    if args.use_style_embed:
        print("Using style embedding")

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    print('Initializing accelerator')

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=2500))
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    print('Done initializing accelerator')

    print('About to wait for everyone')
    accelerator.wait_for_everyone()
    print('Done waiting')

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR
    )
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    print('Created logger')

    # If passed along, set the training seed now.
    if args.seed is not None:
        # set_seed(args.seed)
        accelerate.utils.set_seed(
            args.seed, device_specific=True
        )  # differ slightly for each device

    print('Set seed')

    # HACK: we can pass in "resume" mode, but if output_dir doesn't exist, we change to "train" mode
    if args.train_mode == "resume" and not os.path.exists(args.output_dir):
        args.train_mode = "train"

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    print('About to load config')

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    print('About to load tokenizer')

    assert args.use_slow_tokenizer == True
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    print('Loading data')

    if isinstance(args.tokenized_data_file_path, str):
        args.tokenized_data_file_path = [args.tokenized_data_file_path]
    else:
        args.tokenized_data_file_path = args.tokenized_data_file_path[0]
    print(args.tokenized_data_file_path)
    tokenized_datasets = {x: load_from_disk(x) for x in args.tokenized_data_file_path}

    print('Data loaded')

    train_dataset = {k: v["train"] for k, v in tokenized_datasets.items()}
    eval_dataset = {k: v["val"] for k, v in tokenized_datasets.items()}

    # Log a few random samples from the training set:
    for k in train_dataset.keys():
        for index in random.sample(range(len(train_dataset[k])), 3):
            logger.info(
                f"Sample {k}, {index} of the training set: {train_dataset[k][index]}."
            )

    # Data collator
    # This one will take care of randomly masking the tokens.
    assert args.mlm_probability == 0  # diffusion model does not use [MASK]
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=args.mlm_probability
    )

    # DataLoaders creation:
    train_dataloader = {
        k: DataLoader(
            v,
            shuffle=True,
            collate_fn=data_collator,
            batch_size=args.per_device_train_batch_size,
        )
        for k, v in train_dataset.items()
    }
    eval_dataloader = {
        k: DataLoader(
            v,
            shuffle=True,
            collate_fn=data_collator,
            batch_size=args.per_device_eval_batch_size,  # generator=torch.Generator().manual_seed(42)
        )
        for k, v in eval_dataset.items()
    }

    if args.init_blank_language_model:
        model = AutoModelForMaskedLM.from_config(config)
    elif args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        raise ValueError("specify --init_blank_language_model")

    model.resize_token_embeddings(len(tokenizer))

    vocab_size = model.get_input_embeddings().weight.size(0)
    args.vocab_size = vocab_size
    hidden_size = model.get_input_embeddings().weight.size(1)
    embedding_sum_layer = torch.nn.Linear(vocab_size, hidden_size, bias=False)
    with torch.no_grad():
        embedding_sum_layer.weight.copy_(
            torch.transpose(model.get_input_embeddings().weight.clone(), 0, 1)
        )
    timestep_layer = torch.nn.Linear(1, hidden_size, bias=True)

    ctr_embed_projection = torch.nn.Linear(args.ctr_embed_dim, hidden_size, bias=True)

    # load in our customized modules if necessary
    if os.path.exists(args.model_name_or_path):
        print('LOADING EMBED AND TIMESTEP LAYERS')
        _stdict = torch.load(
            os.path.join(args.model_name_or_path, "embed_sum_layer.pt")
        )
        _stdict = dict(
            (_k[7:], _stdict[_k]) if _k.startswith("module.") else (_k, _stdict[_k])
            for _k in _stdict
        )
        embedding_sum_layer.load_state_dict(_stdict)
        _stdict = torch.load(os.path.join(args.model_name_or_path, "timestep_layer.pt"))
        _stdict = dict(
            (_k[7:], _stdict[_k]) if _k.startswith("module.") else (_k, _stdict[_k])
            for _k in _stdict
        )
        timestep_layer.load_state_dict(_stdict)

        # Unused, leftover from experimentations with classifier-free guidance
        ctrl_embed_path = os.path.join(
            args.model_name_or_path, "ctr_embed_projection.pt"
        )
        if os.path.exists(ctrl_embed_path):
            _stdict = torch.load(ctrl_embed_path)
            ctr_embed_projection.load_state_dict(_stdict)

    # Optimizer
    frozen = []
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (not any(fr in n for fr in frozen))
                and (not any(nd in n for nd in no_decay))
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (not any(fr in n for fr in frozen))
                and (any(nd in n for nd in no_decay))
            ],
            "weight_decay": 0.0,
        },
        {
            "params": [p for p in embedding_sum_layer.parameters()],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for p in timestep_layer.parameters()],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    assert args.max_train_steps is not None

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    print("About to prepare")
    (
        model,
        embedding_sum_layer,
        timestep_layer,
        optimizer,
        ctr_embed_projection,
    ) = accelerator.prepare(
        model, embedding_sum_layer, timestep_layer, optimizer, ctr_embed_projection
    )

    print('Done preparing')

    #  train_dataloader, eval_dataloader,
    for k in train_dataloader.keys():
        train_dataloader[k] = accelerator.prepare(train_dataloader[k])

    for k in eval_dataloader.keys():
        eval_dataloader[k] = accelerator.prepare(eval_dataloader[k])

    print('Done preparing dataloader')

    # Register the LR scheduler
    print('Register scheduler')
    accelerator.register_for_checkpointing(lr_scheduler)

    # Save accelerator state
    print('About to wait for everyone')
    accelerator.wait_for_everyone()
    print('Done waiting')
    if (
        args.train_mode == "resume"
    ):  # resuming job could still break an exact reproducibility, since we are not saving noise states
        if os.path.exists(os.path.join(args.output_dir, 'accelerate_ckpt')):
            accelerator.load_state(os.path.join(args.output_dir, 'accelerate_ckpt'))
        else:
            logger.info("accelerator state not found, starting from scratch")
        with open(os.path.join(args.output_dir, "completed_steps.txt"), 'r') as f:
            completed_steps = int(f.read())
    elif args.train_mode == "train":
        if os.path.exists(os.path.join(args.output_dir, 'accelerate_ckpt')):
            logger.info(
                "training probably interrupted, should change mode to resume for the next run"
            )
            # return 0 # just give warnings, do not interrupt
        accelerator.save_state(os.path.join(args.output_dir, 'accelerate_ckpt'))
        completed_steps = 0
    elif args.train_mode == "decode":
        pass
    else:
        raise ValueError("train_mode must be one of 'train', 'resume', 'decode'")

    model_embedding_lut = accelerator.unwrap_model(model).get_input_embeddings()

    args.remove_noise_mode = args.remove_noise_mode.split('|')
    args.noise_analysis_list = list()

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project=args.project_name,
        # track hyperparameters and run metadata
        config=vars(args),
    )
    lowest_loss = None

    if args.train_mode == "train" or args.train_mode == "resume":
        # Train!
        total_batch_size = (
            args.per_device_train_batch_size
            * accelerator.num_processes
            * args.gradient_accumulation_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        logger.info(f"  Completed optimization steps = {completed_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(
            range(args.max_train_steps - completed_steps),
            disable=not accelerator.is_local_main_process,
        )

        # batch_size = args.per_device_train_batch_size
        if accelerator.is_local_main_process:
            save_checkpoint(
                info={},
                completed_steps=completed_steps,
                folder='best_checkpoint',
                args=args,
                model=model,
                tokenizer=tokenizer,
                embedding_sum_layer=embedding_sum_layer,
                timestep_layer=timestep_layer,
                model_embedding_lut=model_embedding_lut,
                accelerator=accelerator,
                ctr_embed_projection=ctr_embed_projection,
            )

        accelerator.wait_for_everyone()
        # begin training
        train_losses = {k: [] for k in train_dataloader.keys()}
        current_dataset_epochs = {k: 0 for k in train_dataloader.keys()}
        # for epoch in range(args.num_train_epochs):
        model.train()
        # wandb.log({"epoch": epoch})

        # for step, batch in enumerate(train_dataloader):
        for (
            step,
            dataset_name,
            dataset_epoch,
            dataset_step,
            batch,
        ) in yield_from_dataloader_dict(train_dataloader, reset=True):
            # import pdb; pdb.set_trace()
            # batch = key_renames(batch)
            if args.distribute_padding:
                batch['input_ids'] = distribute_padding(
                    batch['input_ids'], padding_idx=1, after_idx=args.context_size
                )
            batch.pop('attention_mask')

            if args.use_style_embed:
                assert 'luar_embedding' in batch
                batch_ctrl_embeds = batch['luar_embedding']
            elif args.use_label:
                assert 'label' in batch
                batch_ctrl_embeds = (
                    torch.nn.functional.one_hot(
                        batch['label'], num_classes=args.ctr_embed_dim
                    )
                    .unsqueeze(1)
                    .float()
                )
            else:
                batch_ctrl_embeds = None

            t_list = list(range(1, args.sigma_num_steps + 1))
            selected_t = torch.FloatTensor(
                np.random.choice(t_list, batch['input_ids'].shape[0], replace=True)
            ).to(accelerator.device)

            if args.use_self_condition and random.random() < args.self_cond_prob:
                raise NotImplementedError()
            else:
                self_cond_logits = None

            loss, _ = do_diffusion(
                selected_t=selected_t,
                self_cond_logits=self_cond_logits,
                batch=batch,
                args=args,
                model=model,
                embedding_sum_layer=embedding_sum_layer,
                timestep_layer=timestep_layer,
                model_embedding_lut=model_embedding_lut,
                accelerator=accelerator,
                ctr_embed_projection=ctr_embed_projection,
                batch_ctrl_embeds=batch_ctrl_embeds,
            )
            train_losses[dataset_name].append(loss.item())
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            if current_dataset_epochs[dataset_name] != dataset_epoch:
                current_dataset_epochs[dataset_name] = dataset_epoch
                wandb.log({"epoch/{}".format(dataset_name): dataset_epoch})
            wandb.log({"per_dataset_step/{}".format(dataset_name): dataset_step})

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if (step + 1) % (args.gradient_accumulation_steps * 10) == 0:
                mean_losses = {
                    f"train_loss_{k}": np.mean(v) for k, v in train_losses.items()
                }
                wandb.log(mean_losses)
                wandb.log({"train": np.mean(list(mean_losses.values()))})
                train_losses = {k: [] for k in train_dataloader.keys()}

            if (step + 1) % (args.gradient_accumulation_steps * 200) == 0:
                val_steps = 0
                max_val_steps = 50
                val_losses = {k: [] for k in eval_dataloader.keys()}
                accelerator.wait_for_everyone()
                model.eval()
                embedding_sum_layer.eval()
                timestep_layer.eval()
                with torch.no_grad():
                    for (
                        _,
                        val_dataset_name,
                        _,
                        _,
                        val_batch,
                    ) in yield_from_dataloader_dict(eval_dataloader, reset=False):
                        if args.distribute_padding:
                            val_batch['input_ids'] = distribute_padding(
                                val_batch['input_ids'],
                                padding_idx=1,
                                after_idx=args.context_size,
                            )
                        val_batch.pop('attention_mask')
                        if args.use_style_embed:
                            assert 'luar_embedding' in batch
                            val_batch_ctrl_embeds = val_batch['luar_embedding']
                        elif args.use_label:
                            assert 'label' in val_batch
                            val_batch_ctrl_embeds = (
                                torch.nn.functional.one_hot(
                                    val_batch['label'], num_classes=args.ctr_embed_dim
                                )
                                .unsqueeze(1)
                                .float()
                            )
                        else:
                            val_batch_ctrl_embeds = None

                        t_list = list(range(1, args.sigma_num_steps + 1))
                        selected_t = torch.FloatTensor(
                            np.random.choice(
                                t_list, val_batch['input_ids'].shape[0], replace=True
                            )
                        ).to(accelerator.device)
                        if (
                            args.use_self_condition
                            and random.random() < args.self_cond_prob
                        ):
                            with torch.no_grad():
                                previous_t = torch.min(
                                    selected_t + 1,
                                    torch.FloatTensor([args.sigma_num_steps]).to(
                                        accelerator.device
                                    ),
                                )
                                _, self_cond_logits = do_diffusion(
                                    selected_t=previous_t,
                                    self_cond_logits=None,
                                    batch=val_batch,
                                    args=args,
                                    model=model,
                                    embedding_sum_layer=embedding_sum_layer,
                                    timestep_layer=timestep_layer,
                                    model_embedding_lut=model_embedding_lut,
                                    accelerator=accelerator,
                                    ctr_embed_projection=ctr_embed_projection,
                                    batch_ctrl_embeds=val_batch_ctrl_embeds,
                                )
                            self_cond_logits = self_cond_logits.detach()
                        else:
                            self_cond_logits = None
                        val_loss, _ = do_diffusion(
                            selected_t=selected_t,
                            self_cond_logits=self_cond_logits,
                            batch=val_batch,
                            args=args,
                            model=model,
                            embedding_sum_layer=embedding_sum_layer,
                            timestep_layer=timestep_layer,
                            model_embedding_lut=model_embedding_lut,
                            accelerator=accelerator,
                            ctr_embed_projection=ctr_embed_projection,
                            batch_ctrl_embeds=val_batch_ctrl_embeds,
                        )
                        # val_losses.append(val_loss.item())
                        val_losses[val_dataset_name].append(val_loss.item())
                        val_steps += 1
                        if val_steps >= max_val_steps:
                            break

                    mean_losses = {
                        f"val_loss_{k}": np.mean(v) for k, v in val_losses.items()
                    }
                    wandb.log(mean_losses)

                    mean_loss = np.mean(list(mean_losses.values()))
                    wandb.log({"val": mean_loss})
                    if accelerator.is_main_process:
                        if lowest_loss is None or mean_loss < lowest_loss:
                            save_info = {
                                'completed_steps': completed_steps,
                                'val_loss': mean_loss,
                            }
                            save_checkpoint(
                                info=save_info,
                                completed_steps=completed_steps,
                                folder='best_checkpoint',
                                args=args,
                                model=model,
                                tokenizer=tokenizer,
                                embedding_sum_layer=embedding_sum_layer,
                                timestep_layer=timestep_layer,
                                model_embedding_lut=model_embedding_lut,
                                accelerator=accelerator,
                                ctr_embed_projection=ctr_embed_projection,
                            )
                            lowest_loss = mean_loss
                accelerator.wait_for_everyone()
                model.train()
                embedding_sum_layer.train()
                timestep_layer.train()

            # if accelerator.is_main_process:
            wandb.log({"step": completed_steps})
            if completed_steps >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            save_info = {'completed_steps': completed_steps}
            save_checkpoint(
                info=save_info,
                folder='final_checkpoint',
                args=args,
                model=model,
                tokenizer=tokenizer,
                embedding_sum_layer=embedding_sum_layer,
                timestep_layer=timestep_layer,
                model_embedding_lut=model_embedding_lut,
                accelerator=accelerator,
                ctr_embed_projection=ctr_embed_projection,
            )
        accelerator.wait_for_everyone()
        logger.info(
            f"TRAINING FINISHED!!! Saved model at completed steps {completed_steps}"
        )
        wandb.finish()

    ##########################################


def save_checkpoint(
    *,
    completed_steps,
    info,
    folder,
    args,
    model,
    tokenizer,
    embedding_sum_layer,
    timestep_layer,
    model_embedding_lut,
    accelerator,
    ctr_embed_projection=None,
):
    ckpt_dir = os.path.join(args.output_dir, folder)
    # if accelerator.is_main_process:
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "info.json"), "w") as f:
        json.dump(info, f)

    # accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(ckpt_dir, save_function=accelerator.save)
    # if accelerator.is_main_process:
    tokenizer.save_pretrained(ckpt_dir)
    torch.save(
        accelerator.unwrap_model(embedding_sum_layer).state_dict(),
        os.path.join(ckpt_dir, "embed_sum_layer.pt"),
    )
    torch.save(
        accelerator.unwrap_model(timestep_layer).state_dict(),
        os.path.join(ckpt_dir, "timestep_layer.pt"),
    )
    torch.save(
        accelerator.unwrap_model(ctr_embed_projection).state_dict(),
        os.path.join(ckpt_dir, "ctr_embed_projection.pt"),
    )
    with open(os.path.join(ckpt_dir, "completed_steps.txt"), 'w') as f:
        f.write(f"{completed_steps}")
        accelerator.save_state(os.path.join(ckpt_dir, 'accelerate_ckpt'))


def do_diffusion(
    batch,
    args,
    model,
    embedding_sum_layer,
    timestep_layer,
    model_embedding_lut,
    accelerator,
    batch_ctrl_embeds=None,
    ctr_embed_projection=None,
    selected_t=None,
    self_cond_logits=None,
    self_cond_labels=None,
    self_cond_weight=0.5,
):
    # args.context_size = CONTEXT_SIZE #np.random.randint(low=ctx_low, high=ctx_high) # min (1 dec block size), max (max seq length - 1 dec block size)
    assert batch["input_ids"].shape[-1] == args.context_size * 2
    args.decoding_block_size = (
        batch["input_ids"].shape[-1] - args.context_size
    )  # CONTEXT_SIZE
    assert args.decoding_block_size == args.context_size
    batch_size = args.per_device_train_batch_size

    ######## NOISE ADDITION ########

    # ctx_low = 1 # using 0 would probably cause hanging issue (parallel device waiting for syncing gradients?)
    # actx_high = args.max_seq_length - args.decoding_block_size + 1

    seq_len = args.decoding_block_size

    # split the batch in to the context part and diffusion part
    diffusion_input_ids = batch['input_ids'][
        :, args.context_size : args.context_size + seq_len
    ]
    # diffusion_mask = batch['attention_mask'][:, args.context_size:args.context_size+seq_len]

    # build alpha according to a pseudo one-hot encoding
    vocab_size = args.vocab_size  # model.get_input_embeddings().weight.size(0)
    one_hot_value = (
        args.hardcoded_pseudo_diralpha
    )  # for a pseudo one-hot encoding for alpha
    # inputs_diralpha = 2 * one_hot_value * torch.nn.functional.one_hot(diffusion_input_ids, vocab_size) - one_hot_value

    perturbed_inputs_embeds = model_embedding_lut(
        diffusion_input_ids
    ).detach()  # consider detaching
    # current_batch_size = inputs_diralpha.shape[0]
    current_batch_size = perturbed_inputs_embeds.shape[0]
    if current_batch_size != batch_size:
        print("WARNING: current batch size not", batch_size, "but", current_batch_size)

    if args.context_size > 0:
        context_input_ids = batch['input_ids'][:, : args.context_size]
        context_inputs_embeds = model_embedding_lut(context_input_ids)
    else:
        context_inputs_embeds = None

    total_t = args.sigma_num_steps

    if selected_t is None:
        t_list = list(range(1, args.sigma_num_steps + 1))
        selected_t = torch.FloatTensor(
            np.random.choice(t_list, current_batch_size, replace=True)
        ).to(accelerator.device)

    # import pdb; pdb.set_trace()

    if args.use_sqrt_schedule:
        (
            alpha_t_bar,
            alpha_t_minus_bar,
            beta_t,
            beta_t_til,
            alpha_t,
        ) = get_time_variables_new_schedule(selected_t, total_t, accelerator.device)
    else:
        (
            alpha_t_bar,
            alpha_t_minus_bar,
            beta_t,
            beta_t_til,
            alpha_t,
        ) = get_time_variables_old_schedule(selected_t, total_t, accelerator.device)

    alpha_t_bar = alpha_t_bar.view(current_batch_size, 1, 1)

    # unit_noise = args.noise_manual_scale * one_hot_value * torch.normal(0, 1, size=inputs_diralpha.shape).to(accelerator.device)
    unit_noise = args.noise_manual_scale * torch.normal(
        0, 1, size=perturbed_inputs_embeds.shape
    ).to(accelerator.device)

    # if 'no_z' in args.remove_noise_mode:
    #     raise ValueError("no_z is disabled for now")
    #     unit_noise = unit_noise * 0

    # if 'biased_z' in args.remove_noise_mode:
    #     raise ValueError("biased_z is disabled for now")
    # else:
    try:
        # perturbed_inputs_diralpha_noexp = torch.sqrt(alpha_t_bar) * inputs_diralpha + torch.sqrt(1 - alpha_t_bar) * unit_noise
        perturbed_inputs_embeds = (
            torch.sqrt(alpha_t_bar) * perturbed_inputs_embeds
            + torch.sqrt(1 - alpha_t_bar) * unit_noise
        )
    except RuntimeError as e:
        print(e)
        import pdb

        pdb.set_trace()

    # pass to the model, conditioned on the timestep as well
    # perturbed_inputs_embeds = embedding_sum_layer(perturbed_inputs_simplex)
    t_progress = selected_t / total_t
    timestep_embeds = timestep_layer(
        t_progress.view(current_batch_size, 1, 1).repeat(1, seq_len, 1)
    )

    diffusion_embeds = perturbed_inputs_embeds + timestep_embeds

    if context_inputs_embeds is not None:
        diffusion_embeds = torch.cat((context_inputs_embeds, diffusion_embeds), dim=1)
    if batch_ctrl_embeds is not None:
        assert ctr_embed_projection is not None
        batch_ctrl_embeds = ctr_embed_projection(batch_ctrl_embeds).unsqueeze(1)
        diffusion_embeds = torch.cat((batch_ctrl_embeds, diffusion_embeds), dim=1)

    outputs = model(inputs_embeds=diffusion_embeds, output_hidden_states=False)
    equivalent_score = outputs.logits
    equivalent_score = equivalent_score[:, args.context_size :].contiguous()
    if batch_ctrl_embeds is not None:
        equivalent_score = equivalent_score[:, batch_ctrl_embeds.size(1) :].contiguous()

    if args.loss_mode == "xe":
        loss = torch.nn.functional.cross_entropy(
            equivalent_score.view(-1, vocab_size),
            diffusion_input_ids.contiguous().view(-1),
        )  # , reduction="none")
        loss = torch.mean(loss)

    else:
        raise ValueError("check loss_mode")

    return loss, equivalent_score


if __name__ == "__main__":
    main()
