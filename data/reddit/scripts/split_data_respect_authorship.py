''' split text file into train val and test '''

import os
import sys
import json
import click
import random
import tqdm


def get_file_length(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        return len(lines)


def assign_with_authors(*, path, train_split, val_split, test_split, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # check if out_dir is empty
    if len(os.listdir(out_dir)) > 0:
        raise ValueError('out_dir is not empty')

    all_authors = set()
    # with tqdm.tqdm(total=total_length) as pbar:
    with open(path, 'r') as f:
        for l in f:
            author = l.split('\t')[0]
            all_authors.add(author)
    all_authors = sorted(all_authors)
    random.shuffle(all_authors)

    num_authors = len(all_authors)
    num_train_authors = int(num_authors * train_split)
    num_val_authors = int(num_authors * val_split)
    num_test_authors = num_authors - num_train_authors - num_val_authors

    train_authors = all_authors[:num_train_authors]
    val_authors = all_authors[num_train_authors : num_train_authors + num_val_authors]
    test_authors = all_authors[num_train_authors + num_val_authors :]

    print('num_authors', num_authors)
    print('num_train_authors', len(train_authors))
    print('num_val_authors', len(val_authors))
    print('num_test_authors', len(test_authors))

    splits = {
        'train': set(train_authors),
        'val': set(val_authors),
        'test': set(test_authors),
    }

    # check for no overlap
    assert len(splits['train'].intersection(splits['val'])) == 0
    assert len(splits['train'].intersection(splits['test'])) == 0
    assert len(splits['val'].intersection(splits['test'])) == 0

    with open(path, 'r') as f:
        with open(os.path.join(out_dir, 'train.txt'), 'w+') as f_train:
            with open(os.path.join(out_dir, 'val.txt'), 'w+') as f_val:
                with open(os.path.join(out_dir, 'test.txt'), 'w+') as f_test:
                    for l in f:
                        author = l.split('\t')[0]
                        if author in splits['train']:
                            f_train.write(l)
                        elif author in splits['val']:
                            f_val.write(l)
                        elif author in splits['test']:
                            f_test.write(l)
                        else:
                            raise ValueError('author not in splits')


@click.command()
@click.option('--path', type=str, required=True, help='path to text file')
@click.option('--train_split', type=float, default=0.90, help='train split')
@click.option('--val_split', type=float, default=0.05, help='val split')
@click.option('--test_split', type=float, default=0.05, help='test split')
@click.option('--out_dir', type=str, required=True, help='path to text file')
def main(path, train_split, val_split, test_split, out_dir):
    random.seed(42)
    assert train_split + val_split + test_split == 1.0
    assign_with_authors(
        path=path,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        out_dir=out_dir,
    )


if __name__ == '__main__':
    main()
