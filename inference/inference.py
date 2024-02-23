"""

Inference code originally based on SSD-LM demo notebook, from [repo](https://github.com/xhan77/ssd-lm) and corresponding paper ((https://arxiv.org/abs/2210.17432)) 

"""

import os
import sys
import torch
import time
import json
from tqdm.auto import tqdm
from datetime import datetime
import argparse


from classifiers import *
from inference_utils import get_setup, controlled_paraphrase

def get_author_data(*, author_name, author_directory, shard='train'):
    clean_name = author_name.replace(":", "").replace(" ", "_")
    with open(os.path.join(author_directory, clean_name, f'{shard}.txt'), 'r') as f:
        lines = f.readlines()
        input_data = [line.strip().split('\t') for line in lines]
    return input_data

MAX_ATTEMPTS = 2 # If we fail to generate a paraphrase, we try one more time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--lr', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=0.2)
    parser.add_argument('--kl_loss_weight', type=float, default=0.0)
    parser.add_argument('--semantic_loss_weight', type=float, default=0.0)
    parser.add_argument('--size', type=int, default=50)
    parser.add_argument('--ctr_embed_dim', type=int, default=11)
    parser.add_argument('--info_path', type=str)
    parser.add_argument('--assignments_json', type=str)
    parser.add_argument('--author_directory', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--total_t', type=int, default=500)
    parser.add_argument('--num_drift_steps', type=int, default=3)
    parser.add_argument('--use_sqrt_schedule', action='store_true')
    parser.add_argument('--use_self_condition', action='store_true')
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--straight_through', action='store_true')
    parser.add_argument('--use_actual', action='store_true')

    cmd_args = parser.parse_args()

    hparams = vars(cmd_args)
    task = hparams.pop('task')
    assert task in ['style', 'formal', 'informal', 'positive', 'negative']
    out_dir = hparams.pop('out_dir')

    (
        args,
        model,
        tokenizer,
        model_embedding_lut,
        embedding_sum_layer,
        timestep_layer,
        ctr_embed_projection,
    ) = get_setup(**hparams)

    dtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    task_folder = f"{out_dir}/{dtime}_{hparams['lr']}"
    os.makedirs(task_folder, exist_ok=False)

    with open(os.path.join(task_folder, "hparams.json"), 'w') as f:
        json.dump(hparams, f)

    with open(os.path.join(task_folder, "args.txt"), 'w') as f:
        json.dump(str(args), f)

    if task == 'style':
        args.optimizing_label_index = None
        ctr_model, tokenizer, ctr_embeds = load_style_model()

        args.ctr_model = ctr_model
        args.tokenizer = tokenizer
        args.ctr_embeds = ctr_embeds
        args.ctr_embeds = args.ctr_embeds.to(args.accelerator.device)
        args.ctr_model.to(args.accelerator.device)
        args.ctr_model.eval()

    elif task in ["positive", "negative"]:
        ctr_model, tokenizer, ctr_embeds, possible_labels = load_sentiment_model(
            "sentiment"
        )
        label = task
        args.optimizing_label_index = possible_labels.index(label)
        args.ctr_model = ctr_model
        args.ctr_embeds = ctr_embeds
        args.tokenizer = tokenizer

        args.ctr_embeds = args.ctr_embeds.to(args.accelerator.device)
        args.ctr_model.to(args.accelerator.device)
        args.ctr_model.eval()

        args.loss_fn = lambda embeds, mask: -torch.nn.functional.log_softmax(
            args.ctr_model(inputs_embeds=embeds, attention_mask=mask).logits, dim=-1
        )[:, args.optimizing_label_index].mean()

    elif task in ['formal', 'informal']:
        ctr_model, tokenizer, ctr_embeds, _ = load_formality_model()
        args.optimizing_label_index = (
            1 if task == 'formal' else 0
        ) 
        args.ctr_model = ctr_model
        args.ctr_embeds = ctr_embeds
        args.tokenizer = tokenizer
        args.ctr_embeds = args.ctr_embeds.to(args.accelerator.device)
        args.ctr_model.to(args.accelerator.device)
        args.ctr_model.eval()
        args.loss_fn = lambda embeds, mask: -torch.nn.functional.log_softmax(
            args.ctr_model(inputs_embeds=embeds, attention_mask=mask).logits, dim=-1
        )[:, args.optimizing_label_index].mean()

    else:
        raise NotImplementedError

    if task in ['style']:
        with open(hparams['assignments_json'], 'r') as f:
            assignments = json.load(f)

        # total_transfers = sum([len(assignments[source_author]['target'])*len(assignments[source_author]['val_samples']) for source_author in assignments.keys()])
        total_transfers = sum(
            [
                len(assignments[source_author]['target'])
                * len(assignments[source_author]['test_samples'])
                for source_author in assignments.keys()
            ]
        )

        with open(os.path.join(task_folder, f"{task}.jsonl"), 'w+') as out:
            ########
            with tqdm(total=total_transfers) as pbar:
                for source_author in sorted(assignments.keys()):
                    # val_examples = assignments[source_author]['val_samples']
                    val_examples = assignments[source_author]['test_samples']
                    target_authors = assignments[source_author]['target']
                    for target_author in target_authors:
                        target_author_training = [
                            x[1]
                            for x in get_author_data(
                                author_name=target_author,
                                author_directory=hparams['author_directory'],
                                shard='train',
                            )
                        ]

                        if args.decode_ctr_lr != 0.0:
                            print(target_author, target_author_training)
                            args.target_embeds = text_to_style(
                                model=args.ctr_model,
                                tokenizer=args.tokenizer,
                                texts=target_author_training,
                                device=args.accelerator.device,
                                model_type=task,
                            )
                            args.loss_fn = lambda embeds, mask: compute_style_loss(
                                embeds,
                                model=args.ctr_model,
                                target_embeds=args.target_embeds,
                                attention_mask=mask.float(),
                                model_type=task,
                            )

                        for paraphrase, original_text in val_examples:
                            # import pdb; pdb.set_trace()

                            ctr_embed_projection = None
                            batch_ctrl_embeds = None

                            for attempt in range(MAX_ATTEMPTS):
                                start = time.time()
                                print(f'{source_author} -> {target_author}')
                                outputs = controlled_paraphrase(
                                    original_text if args.use_actual else paraphrase,
                                    num_samples=1,
                                    args=args,
                                    model=model,
                                    tokenizer=tokenizer,
                                    model_embedding_lut=model_embedding_lut,
                                    embedding_sum_layer=embedding_sum_layer,
                                    timestep_layer=timestep_layer,
                                    ctr_embed_projection=ctr_embed_projection,
                                    batch_ctrl_embeds=batch_ctrl_embeds,
                                    logging=False,
                                )

                                result = dict(
                                    input_label=source_author,
                                    paraphrase=paraphrase,
                                    original_text=original_text,
                                    target_label=target_author,
                                    decoded=outputs,
                                    attempt=attempt,
                                )

                                print(f'{original_text} -> {paraphrase} -> {outputs}')
                                if len(outputs) and outputs[0].strip() != "":
                                    break
                                else:
                                    print(f'Attempt {attempt} failed, retrying')

                            out.write(json.dumps(result) + '\n')
                            print('Elapsed:', time.time() - start)
                            pbar.update(1)

    else:
        with open(args.input_path, 'r') as f:
            input_data = [l.strip().split('\t') for l in f.readlines()]

        total_transfers = len(input_data)

        with open(os.path.join(task_folder, f"{task}.jsonl"), 'w+') as out:

            with tqdm(total=total_transfers) as pbar:

                for _, paraphrase, original_text in input_data:

                    ctr_embed_projection=None
                    batch_ctrl_embeds=None

                    for attempt in range(MAX_ATTEMPTS):
                        start = time.time()
                        outputs = controlled_paraphrase( original_text if args.use_actual else paraphrase, num_samples=1, args=args, model=model, tokenizer=tokenizer, model_embedding_lut=model_embedding_lut, embedding_sum_layer=embedding_sum_layer, timestep_layer=timestep_layer, ctr_embed_projection=ctr_embed_projection, batch_ctrl_embeds=batch_ctrl_embeds, logging=False)
                        result = dict(
                            input_label=args.input_path,
                            paraphrase=paraphrase,
                            original_text=original_text,
                            target_label=task,
                            decoded=outputs,
                            attempt=attempt)
                        print(f'{original_text} -> {paraphrase} -> {outputs}')
                        if len(outputs) and outputs[0].strip() != "":
                            break
                        else:
                            print(f'Attempt {attempt} failed, retrying')
                    out.write(json.dumps(result) + '\n')
                    print('Elapsed:',time.time() - start)
                    pbar.update(1)