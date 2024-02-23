import os
import argparse
import random
import json
import math


def split_data(x, splits):
    assert sum(splits) == 1.0
    assert len(x) > 0

    assigned = 0
    split_data = []

    for split in splits:
        split_size = math.ceil(split * len(x))
        split_data.append(x[assigned : assigned + split_size])
        assigned += split_size
    assert sum([len(split) for split in split_data]) == len(
        x
    ), f'{sum([len(split) for split in split_data])} != {len(x)}'
    return split_data


if __name__ == '__main__':
    # take input file, output_dir, and min and max score cuttoffs
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--min_score', type=float)
    parser.add_argument('--max_score', type=float)
    parser.add_argument('--task', type=str)
    # also specify train-val-test splits
    parser.add_argument('--train_split', type=float, default=0.7)
    parser.add_argument('--test_split', type=float, default=0.15)
    parser.add_argument('--val_split', type=float, default=0.15)

    # example usage:
    # python split_by_score.py --input_file formal_scored.tsv --output_dir formal_splits --min_score 0.5 --max_score 0.5 --task formal

    cmd_args = parser.parse_args()

    task_name = f'{cmd_args.task}_{cmd_args.min_score}_{cmd_args.max_score}'

    full_out_dir = os.path.join(cmd_args.output_dir, task_name)
    os.makedirs(full_out_dir, exist_ok=False)

    with open(cmd_args.input_file, 'r') as f:
        lines = f.readlines()
        input_data = [line.strip().split('\t') for line in lines]

    random.shuffle(input_data)

    pos_examples = []
    neg_examples = []

    for author, para, text, score in input_data:
        if float(score) >= cmd_args.min_score:
            pos_examples.append((author, para, text))
        elif float(score) <= cmd_args.max_score:
            neg_examples.append((author, para, text))

    # split data
    train_split = cmd_args.train_split
    val_split = cmd_args.val_split
    test_split = cmd_args.test_split

    positive_split_data = split_data(
        x=pos_examples, splits=[train_split, val_split, test_split]
    )
    negative_split_data = split_data(
        x=neg_examples, splits=[train_split, val_split, test_split]
    )

    # write data
    assert len(positive_split_data) == len(negative_split_data) == 3

    for split_name, pos, neg in zip(
        ['train', 'val', 'test'], positive_split_data, negative_split_data
    ):
        with open(os.path.join(full_out_dir, f'{split_name}_pos.tsv'), 'w+') as f:
            for author, para, text in pos:
                f.write(f'{author}\t{para}\t{text}\n')

        with open(os.path.join(full_out_dir, f'{split_name}_neg.tsv'), 'w+') as f:
            for author, para, text in neg:
                f.write(f'{author}\t{para}\t{text}\n')

    # save args
    with open(os.path.join(full_out_dir, 'args.json'), 'w+') as f:
        json.dump(vars(cmd_args), f)
