import openai
import argparse
import json
import os
import random
from tqdm import tqdm
from datetime import datetime
import time

import sys


def get_author_data(*, author_name, author_directory, shard='train'):
    clean_name = author_name.replace(":", "").replace(" ", "_")
    with open(os.path.join(author_directory, clean_name, f'{shard}.txt'), 'r') as f:
        lines = f.readlines()
        input_data = [line.strip().split('\t') for line in lines]
    return input_data


def hit_openai(message):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": message},
        ],
    )
    # import pdb; pdb.set_trace()
    return response['choices'][0]['message']


def do_prompted_transfer(*, original_text, examples):
    message = 'The following emails are written by a single author: \n'
    for i, text in enumerate(examples):
        message += '{"text":"' + text + '"}\n'
    message += "\n\nCan you rewrite the following email to make it look like the above author's style:\n"
    message += "{'text':'" + original_text + "'}\n"

    while True:
        try:
            return hit_openai(message)
        except (
            openai.error.ServiceUnavailableError,
            openai.error.APIConnectionError,
            openai.error.APIError,
            openai.error.Timeout,
        ) as e:
            print("Service unavailable, retrying...")
            time.sleep(20)
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--author_directory', type=str)
    parser.add_argument('--assignments_json', type=str)
    parser.add_argument('--max_examples', type=int, default=16)
    parser.add_argument('--approach', type=str)

    cmd_args = parser.parse_args()
    hparams = vars(cmd_args)
    out_dir = hparams['out_dir']
    approach = hparams['approach']

    assert approach in ['chatgpt']

    dtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    task_folder = f"{out_dir}/{dtime}"
    os.makedirs(task_folder, exist_ok=False)

    with open(os.path.join(task_folder, "hparams.json"), 'w') as f:
        json.dump(hparams, f)

    with open(hparams['assignments_json'], 'r') as f:
        assignments = json.load(f)

    total_transfers = sum(
        [
            len(assignments[source_author]['target'])
            * len(assignments[source_author]['test_samples'])
            for source_author in assignments.keys()
        ]
    )

    counter = -1
    with open(os.path.join(task_folder, f"style.jsonl"), 'w+') as out:
        with tqdm(total=total_transfers) as pbar:
            for source_author in sorted(assignments.keys()):
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
                    for paraphrase, original_text in val_examples:
                        # import pdb; pdb.set_trace()
                        counter += 1
                        if counter < 925:  # 715 + 121:
                            print('skipping', counter)
                            continue
                        target_texts = target_author_training[: hparams['max_examples']]
                        result = do_prompted_transfer(
                            original_text=original_text, examples=target_texts
                        )
                        result = dict(
                            input_label=source_author,
                            paraphrase=paraphrase,
                            original_text=original_text,
                            target_label=target_author,
                            decoded=result,
                        )

                        # print(f'{original_text} -> {paraphrase} -> {result}')
                        out.write(json.dumps(result) + '\n')
                        pbar.update(1)
