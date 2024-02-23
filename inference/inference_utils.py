"""

Inference code helpers

"""

import os
import torch
import accelerate
import json
import logging
from argparse import Namespace
from accelerate import Accelerator
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
)

from diffusion_utils import decode
from classifiers import *


def get_setup(
    lr=20,
    total_t=500,
    use_sqrt_schedule=False,
    use_self_condition=False,
    top_p=0.9,
    kl_loss_weight=0.0,
    semantic_loss_weight=0,
    size=50,
    num_drift_steps=3,
    ctr_embed_dim=768,
    info_path=None,
    input_path=None,
    assignments_json=None,
    author_directory=None,
    model_path=None,
    temperature=1.0,
    straight_through=False,
    use_actual=False,
):
    args = Namespace()
    args.model_name_or_path = model_path

    args.ctr_embed_dim = ctr_embed_dim
    args.max_seq_length = size * 2
    args.one_hot_value = 5
    args.decoding_block_size = size
    args.decode_total_gen_len = size
    args.decode_depth = 1
    args.decode_log_interval = 100
    args.total_t = total_t
    args.projection_top_p = top_p
    args.num_drift_steps = num_drift_steps
    args.seed = 2022
    args.decode_ctr_lr = lr
    args.use_slow_tokenizer = True
    args.temperature = temperature
    args.kl_loss_weight = kl_loss_weight
    args.semantic_loss_weight = semantic_loss_weight
    args.input_path = input_path
    args.assignments_json = assignments_json
    args.author_directory = author_directory
    args.use_sqrt_schedule = use_sqrt_schedule
    args.use_self_condition = use_self_condition
    args.straight_through = straight_through
    args.use_actual = use_actual

    args.info_path = info_path
    if args.info_path is not None:
        with open(args.info_path, 'r') as f:
            args.info = json.load(f)

    accelerator = Accelerator()
    accelerate.utils.set_seed(args.seed, device_specific=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=not args.use_slow_tokenizer
    )
    model = AutoModelForMaskedLM.from_pretrained(
        args.model_name_or_path, from_tf=False, config=config
    )

    model.resize_token_embeddings(len(tokenizer))
    vocab_size = model.get_input_embeddings().weight.size(0)
    hidden_size = model.get_input_embeddings().weight.size(1)

    embedding_sum_layer = torch.nn.Linear(vocab_size, hidden_size, bias=False)
    _stdict = torch.load(os.path.join(args.model_name_or_path, "embed_sum_layer.pt"))
    _stdict = dict(
        (_k[len("module.") :], _stdict[_k])
        if _k.startswith("module.")
        else (_k, _stdict[_k])
        for _k in _stdict
    )
    embedding_sum_layer.load_state_dict(_stdict)

    timestep_layer = torch.nn.Linear(1, hidden_size, bias=True)
    _stdict = torch.load(os.path.join(args.model_name_or_path, "timestep_layer.pt"))
    _stdict = dict(
        (_k[len("module.") :], _stdict[_k])
        if _k.startswith("module.")
        else (_k, _stdict[_k])
        for _k in _stdict
    )
    timestep_layer.load_state_dict(_stdict)

    ctr_embed_projection = torch.nn.Linear(args.ctr_embed_dim, hidden_size, bias=True)
    ctrl_embed_path = os.path.join(args.model_name_or_path, "ctr_embed_projection.pt")

    if os.path.exists(ctrl_embed_path):
        _stdict = torch.load(ctrl_embed_path)
        ctr_embed_projection.load_state_dict(_stdict)
    else:
        print("WARNING: NO STYLE LINEAR FOUND")

    (
        model,
        embedding_sum_layer,
        timestep_layer,
        ctr_embed_projection,
    ) = accelerator.prepare(
        model, embedding_sum_layer, timestep_layer, ctr_embed_projection
    )

    # a bit more preparation before decoding
    model.eval()
    model_embedding_lut = accelerator.unwrap_model(model).get_input_embeddings()
    args.vocab_size = vocab_size
    args.accelerator = accelerator
    args.orig_decode_truncate_len = args.max_seq_length - args.decode_total_gen_len

    return (
        args,
        model,
        tokenizer,
        model_embedding_lut,
        embedding_sum_layer,
        timestep_layer,
        ctr_embed_projection,
    )


def batched_controlled_paraphrase(
    input_text,
    num_samples,
    args,
    model,
    tokenizer,
    model_embedding_lut,
    embedding_sum_layer,
    timestep_layer,
    batch_ctrl_embeds,
    ctr_embed_projection,
    logging=False,
):
    assert isinstance(input_text, list)

    # initial_sentence = input_text

    args.context_size = args.decode_total_gen_len

    INITIAL_IDS = tokenizer(
        input_text, max_length=args.context_size, padding='max_length', truncation=True
    )['input_ids']
    input_ids = torch.LongTensor(INITIAL_IDS).to(args.accelerator.device)
    # print(args.accelerator.device)

    # assert args.max_seq_length - args.decode_total_gen_len - args.context_size == 0, "check the length of the prompt"
    args.decode_truncate_len = args.orig_decode_truncate_len - args.context_size
    # input_ids = input_ids.unsqueeze(0)

    outputs = []
    for i in range(num_samples):
        # start sampling from SSD-LM
        _, _, _, sampled_sequences, _, _ = decode(
            args,
            input_ids,
            args.decode_depth,
            args.total_t,
            model_embedding_lut,
            embedding_sum_layer,
            timestep_layer,
            model,
            tokenizer,
            batch_ctrl_embeds=batch_ctrl_embeds,
            ctr_embed_projection=ctr_embed_projection,
            logging=logging,
        )
        # print("\n\n")

        results = []
        for result in sampled_sequences:
            if '</s>' in result:
                result = result[: result.index('</s>')]
            result = result.replace('<pad>', '').replace('<s>', '').strip()
            results.append(result.replace('<pad>', '').replace('<s>', '').strip())

        outputs.append(results)

    return outputs


def controlled_paraphrase(
    input_text,
    num_samples,
    args,
    model,
    tokenizer,
    model_embedding_lut,
    embedding_sum_layer,
    timestep_layer,
    batch_ctrl_embeds,
    ctr_embed_projection,
    logging=False,
):
    initial_sentence = [input_text]
    INITIAL_IDS = tokenizer(initial_sentence)['input_ids'][0]
    # print(initial_sentence)

    if len(INITIAL_IDS) > args.decoding_block_size:  # 25:
        print('INITIAL TOO LONG, SKIPPING')
        return []
    else:
        INITIAL_IDS = INITIAL_IDS + [tokenizer.pad_token_id] * (
            args.decoding_block_size - len(INITIAL_IDS)
        )
    input_ids = torch.LongTensor(INITIAL_IDS).to(args.accelerator.device)
    # print(args.accelerator.device)

    args.context_size = len(input_ids)
    assert (
        args.max_seq_length - args.decode_total_gen_len - args.context_size == 0
    ), "check the length of the prompt"
    args.decode_truncate_len = args.orig_decode_truncate_len - args.context_size
    input_ids = input_ids.unsqueeze(0)

    outputs = []
    for i in range(num_samples):
        _, _, _, sampled_sequences, _, _ = decode(
            args,
            input_ids,
            args.decode_depth,
            args.total_t,
            model_embedding_lut,
            embedding_sum_layer,
            timestep_layer,
            model,
            tokenizer,
            batch_ctrl_embeds=batch_ctrl_embeds,
            ctr_embed_projection=ctr_embed_projection,
            logging=logging,
        )

        result = sampled_sequences[0]
        if '</s>' in result:
            result = result[: result.index('</s>')]

        result = result.replace('<pad>', '').replace('<s>', '').strip()
        outputs.append(result)

    return outputs
