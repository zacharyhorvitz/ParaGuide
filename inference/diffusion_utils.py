# A portion of this code is adapted from https://github.com/xhan77/ssd-lm

import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

import sys

sys.path.append('../training')
from training_utils import (
    get_time_variables_old_schedule,
    get_time_variables_new_schedule,
)

def clean_str(x):
    return str(x).replace('<pad>', '').strip()


# a few helper functions
def apply_controlling_drift(args, perturbed_inputs_diralpha, lr, semantic_lr=0):
    if args.decode_ctr_lr <= 0:
        args.ctr_loss = -1
        return perturbed_inputs_diralpha

    args.ctr_loss = 0
    for _ in range(args.num_drift_steps):
        with torch.enable_grad():
            perturbed_inputs_diralpha_4ctr = perturbed_inputs_diralpha.clone()
            perturbed_inputs_diralpha_4ctr.requires_grad_()
            perturbed_inputs_simplex_4ctr = torch.nn.functional.softmax(
                perturbed_inputs_diralpha_4ctr / args.temperature, dim=-1
            )

            if args.straight_through:
                source_dist_straight_through = (
                    torch.nn.functional.one_hot(
                        torch.argmax(perturbed_inputs_simplex_4ctr, -1),
                        perturbed_inputs_simplex_4ctr.shape[-1],
                    )
                    - perturbed_inputs_diralpha_4ctr.detach()
                    + perturbed_inputs_diralpha_4ctr
                )
            else:
                source_dist_straight_through = perturbed_inputs_simplex_4ctr

            perturbed_inputs_embeds_4ctr = torch.nn.functional.linear(
                source_dist_straight_through, args.ctr_embeds.t()
            )
            attention_mask = (
                torch.argmax(perturbed_inputs_simplex_4ctr, -1)
                != args.tokenizer.pad_token_id
            )
            ctr_loss = args.loss_fn(perturbed_inputs_embeds_4ctr, attention_mask)

            if semantic_lr != 0:
                perturbed_inputs_embeds_4sem = torch.nn.functional.linear(
                    source_dist_straight_through, args.semantic_embeds.t()
                )
                semantic_loss = args.semantic_loss_fn(
                    perturbed_inputs_embeds_4sem,
                    attention_mask,
                    args.target_semantic_embed,
                )
                semantic_loss = semantic_loss
            else:
                semantic_loss = 0

            # kl loss between original dist and perturbed dist
            if args.kl_loss_weight != 0:
                raise NotImplementedError
                # kl_loss = args.kl_loss_weight * torch.nn.functional.kl_div(perturbed_inputs_simplex_4ctr, original_dist)
                # ctr_loss = ctr_loss + kl_loss
            # print(ctr_loss,lr, semantic_loss, semantic_lr)
            args.ctr_loss = ctr_loss * lr + semantic_loss * semantic_lr

            ctr_delta = -torch.autograd.grad(
                args.ctr_loss, perturbed_inputs_diralpha_4ctr
            )[0]
            # print(ctr_delta.shape)

        perturbed_inputs_diralpha = perturbed_inputs_diralpha + (ctr_delta)

    return perturbed_inputs_diralpha


def logits_sampling_projection(logits, top_p, one_hot_value):
    assert len(logits.size()) == 3

    probs = torch.nn.functional.softmax(logits, dim=-1)
    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
    cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus = cum_sum_probs < top_p
    nucleus = torch.cat(
        [nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1
    )
    valid_indices = nucleus.scatter(2, indices, nucleus)

    filtered_logits = logits.masked_fill(valid_indices == 0, -float('Inf'))
    m = torch.distributions.categorical.Categorical(logits=filtered_logits)
    selected = m.sample()
    return (
        2 * one_hot_value * torch.nn.functional.one_hot(selected, logits.size(2))
        - one_hot_value
    )


def run_model(model, diffusion_embeds):
    output = model(inputs_embeds=diffusion_embeds, output_hidden_states=False)
    return output


def get_lr(args, percent):
    return args.decode_ctr_lr * (np.sin(np.pi * percent))


def get_lr_semantic(args, percent):
    return args.semantic_loss_weight * (percent) ** 2


def decode(
    args,
    batch_input_ids,
    dec_depth,
    total_t,
    model_embedding_lut,
    embedding_sum_layer,
    timestep_layer,
    model,
    tokenizer,
    batch_ctrl_embeds=None,
    ctr_embed_projection=None,
    logging=False,
):
    batch_size = batch_input_ids.shape[0]  # 1 # for the demo
    # if args.decode_truncate_len > 0:
    # diffusion_input_ids = batch_input_ids[:, args.context_size:-args.decode_truncate_len]
    # else:
    diffusion_input_ids = batch_input_ids[:, args.context_size :]

    assert (
        args.max_seq_length - args.context_size - args.decode_truncate_len
    ) % dec_depth == 0, "check whether the total generation length is divisible by the depth of decoding"
    unit_seq_len = int(
        (args.max_seq_length - args.context_size - args.decode_truncate_len) / dec_depth
    )
    if args.context_size > 0:
        unit_context_input_ids = batch_input_ids[:, : args.context_size].clone()
    else:
        unit_context_input_ids = None
    history_decode_ids = None

    for i in range(dec_depth):
        # unit_noise = args.one_hot_value * torch.normal(0, 1, size=(batch_size, unit_seq_len, args.vocab_size)).to(args.accelerator.device)
        unit_noise = torch.normal(0, 1, size=(batch_size, unit_seq_len, 1024)).to(
            args.accelerator.device
        )

        xt = unit_noise

        if unit_context_input_ids is not None:
            context_inputs_embeds = model_embedding_lut(unit_context_input_ids)
        else:
            context_inputs_embeds = None

        t_range = list(range(1, total_t + 1))
        t_range.reverse()
        # progress_bar = tqdm(range(len(t_range)), disable=not args.accelerator.is_local_main_process)

        enumerated = list(enumerate(t_range))
        old_simplex = None
        for idx, t in enumerated:
            selected_t = (
                torch.FloatTensor([t]).repeat(batch_size).to(args.accelerator.device)
            )

            # if selected_t / total_t > 0.5:
            #    continue
            # try substituting text at around 0.25, and this point text is recoverable.
            # current problem: information is essentially lost until very late in decoding.
            # nice aspect of this: gradients probably helpful
            # initial text is just a bunch of duplicated words

            if args.use_sqrt_schedule:
                (
                    alpha_t_bar,
                    alpha_t_minus_bar,
                    beta_t,
                    beta_t_til,
                    alpha_t,
                ) = get_time_variables_new_schedule(
                    selected_t, total_t, args.accelerator.device
                )
            else:
                (
                    alpha_t_bar,
                    alpha_t_minus_bar,
                    beta_t,
                    beta_t_til,
                    alpha_t,
                ) = get_time_variables_old_schedule(
                    selected_t, total_t, args.accelerator.device
                )

            # zt = args.one_hot_value * torch.normal(0, 1, size=(batch_size, unit_seq_len, args.vocab_size)).to(args.accelerator.device)
            zt = torch.normal(0, 1, size=(batch_size, unit_seq_len, 1024)).to(
                args.accelerator.device
            )

            if old_simplex is not None and args.use_self_condition:
                raise NotImplementedError()
            #    perturbed_inputs_simplex = (0.5)*perturbed_inputs_simplex + torch.nn.functional.softmax(old_simplex, dim=-1) * (0.5)

            perturbed_inputs_embeds = xt
            # perturbed_inputs_embeds = embedding_sum_layer(perturbed_inputs_simplex)
            t_progress = selected_t / total_t
            timestep_embeds = timestep_layer(
                t_progress.view(batch_size, 1, 1).repeat(1, unit_seq_len, 1)
            )

            diffusion_embeds = perturbed_inputs_embeds + timestep_embeds
            if context_inputs_embeds is not None:
                diffusion_embeds = torch.cat(
                    (context_inputs_embeds, diffusion_embeds), dim=1
                )
            if batch_ctrl_embeds is not None:
                assert ctr_embed_projection is not None
                prepend_embed = ctr_embed_projection(batch_ctrl_embeds)
                diffusion_embeds = torch.cat((prepend_embed, diffusion_embeds), dim=1)

            outputs = run_model(model=model, diffusion_embeds=diffusion_embeds)

            equivalent_score = outputs.logits
            if unit_context_input_ids is not None:
                equivalent_score = equivalent_score[
                    :, unit_context_input_ids.size(1) :
                ].contiguous()
            if batch_ctrl_embeds is not None:
                equivalent_score = equivalent_score[
                    :, batch_ctrl_embeds.size(1) :
                ].contiguous()

            equivalent_score = apply_controlling_drift(
                args,
                equivalent_score,
                lr=get_lr(args, idx / len(t_range)),
                semantic_lr=get_lr_semantic(args, idx / len(t_range)),
            )
            old_simplex = (
                torch.nn.functional.softmax(equivalent_score, dim=-1).clone().detach()
            )
            projected_logits = logits_sampling_projection(
                equivalent_score,
                top_p=args.projection_top_p,
                one_hot_value=args.one_hot_value,
            )

            sampled_token = torch.argmax(projected_logits, -1)
            embedded_sampled = model_embedding_lut(sampled_token)
            xt = torch.sqrt(alpha_t_minus_bar).view(-1, 1, 1) * embedded_sampled
            xt = xt + torch.sqrt(1 - alpha_t_minus_bar).view(-1, 1, 1) * zt

            if t % args.decode_log_interval == 0 or t == 1:
                if unit_context_input_ids is not None:
                    context_sequences = tokenizer.batch_decode(
                        unit_context_input_ids.detach().to('cpu')
                    )

                real_token_ids_list = sampled_token.view(batch_size, unit_seq_len)
                sampled_sequences = tokenizer.batch_decode(
                    real_token_ids_list.clone().detach().to('cpu')
                )

        unit_context_input_ids = torch.cat(
            (unit_context_input_ids, real_token_ids_list), dim=1
        )
        if history_decode_ids is None:
            history_decode_ids = real_token_ids_list
        else:
            history_decode_ids = torch.cat(
                (history_decode_ids, real_token_ids_list), dim=1
            )

    if args.context_size > 0:
        init_context_input_ids = batch_input_ids[:, : args.context_size].clone()
        context_sequences = tokenizer.batch_decode(
            init_context_input_ids.detach().to('cpu')
        )
    else:
        init_context_input_ids = None
        context_sequences = None
    gold_sequences = tokenizer.batch_decode(
        diffusion_input_ids.clone().detach().to('cpu')
    )
    sampled_sequences = tokenizer.batch_decode(
        history_decode_ids.clone().detach().to('cpu')
    )

    return (
        history_decode_ids,
        init_context_input_ids,
        diffusion_input_ids,
        sampled_sequences,
        context_sequences,
        gold_sequences,
    )
