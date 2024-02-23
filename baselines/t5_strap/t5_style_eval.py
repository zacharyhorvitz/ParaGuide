from finetunet5 import T5Style
from transformers import AutoTokenizer
import json
import argparse
from datetime import datetime
import os
import torch
from tqdm import tqdm


def generate_inference(*, model, tokenizer, text, device):
    inputs = tokenizer(
        text, return_tensors='pt', padding=True, truncation=True, max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_length=60, num_beams=1, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_results_for_assignments(
    *,
    assignments,
    model,
    tokenizer,
    target_author,
    sample_key='val_samples',
    device='cuda',
):
    total_estimate = estimate_total(
        assignments=assignments, target_author=target_author, sample_key=sample_key
    )
    if total_estimate == 0:
        raise ValueError(f"No examples found for target author: {target_author}")

    with tqdm(total=total_estimate) as pbar:
        for source_author in sorted(assignments.keys()):
            if (
                target_author == 'all'
                or target_author in assignments[source_author]['target']
            ):
                val_examples = assignments[source_author][sample_key]
                for para, original in val_examples:
                    result = generate_inference(
                        model=model, tokenizer=tokenizer, text=para, device=device
                    )

                    pbar.update(1)
                    yield dict(
                        input_label=source_author,
                        paraphrase=para,
                        original_text=original,
                        target_label=target_author,
                        decoded=[result],
                    )


def estimate_total(*, assignments, target_author, sample_key='val_samples'):
    total = 0
    for source_author in sorted(assignments.keys()):
        if (
            target_author == 'all'
            or target_author in assignments[source_author]['target']
        ):
            total += len(assignments[source_author][sample_key])
    return total


def evaluate_on_assignments(
    *, model, tokenizer, out_dir, assignments_json, target_author, sample_key, args
):
    dtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    clean_author = target_author.replace(':', '').replace(' ', '_')
    task_folder = f"{out_dir}/{clean_author}/{sample_key}/{dtime}"
    os.makedirs(task_folder, exist_ok=False)

    with open(assignments_json, 'r') as f:
        assignments = json.load(f)

    # save args
    with open(f"{task_folder}/args.json", 'w') as f:
        json.dump(vars(args), f)

    with open(f"{task_folder}/results.jsonl", 'w') as f:
        for result in generate_results_for_assignments(
            assignments=assignments,
            model=model,
            tokenizer=tokenizer,
            target_author=target_author,
            device='cuda',
            sample_key=sample_key,
        ):
            # print(result)
            f.write(json.dumps(result) + '\n')


def evaluate_on_file(*, model, tokenizer, out_dir, input_file, task, args):
    dtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    task_folder = f"{out_dir}/{task}/{dtime}"
    os.makedirs(task_folder, exist_ok=False)

    # save args
    with open(f"{task_folder}/args.json", 'w') as f:
        json.dump(vars(args), f)

    with open(f"{task_folder}/{task}.jsonl", 'w') as f:
        with open(input_file, 'r') as in_:
            lines = list(in_.readlines())
            for l in tqdm(lines):
                cols = l.strip().split('\t')
                if len(cols) == 3:
                    author, para, original = cols
                else:
                    para, original = cols
                    author = None

                decoded = generate_inference(
                    model=model, tokenizer=tokenizer, text=para, device='cuda'
                )
                result = dict(
                    input_label=author,
                    paraphrase=para,
                    original_text=original,
                    target_label=task,
                    decoded=[decoded],
                )
                f.write(json.dumps(result) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--assignments_json', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--target_author', type=str)
    parser.add_argument('--sample_key', type=str, default='val_samples')
    parser.add_argument('--task', type=str)
    parser.add_argument('--input_file', type=str)

    args = parser.parse_args()
    out_dir = args.out_dir
    model_path = args.model_path
    task = args.task

    tokenizer = AutoTokenizer.from_pretrained("t5-large")
    model = T5Style("t5-large", use_style=False, num_styles=None)
    # model.load_state_dict(torch.load(model_path, map_location='cuda')) # use pretrained fn eventually
    model.model.from_pretrained(model_path)
    model.to('cuda')
    model.eval()

    if task == 'authorship':
        assert args.assignments_json is not None
        assignments_json = args.assignments_json
        evaluate_on_assignments(
            model=model,
            tokenizer=tokenizer,
            out_dir=out_dir,
            assignments_json=assignments_json,
            target_author=args.target_author,
            sample_key=args.sample_key,
            args=args,
        )
    else:
        assert args.input_file is not None
        in_file = args.input_file
        evaluate_on_file(
            model=model,
            tokenizer=tokenizer,
            out_dir=out_dir,
            input_file=in_file,
            task=task,
            args=args,
        )
