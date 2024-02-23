# split into shards


import sys
import os
from collections import defaultdict
import random
import json
import math

PATH = sys.argv[1]
# TRAIN_DEV_TEST = [0.8, 0.1, 0.1]
TRAIN_DEV_TEST = [0.6, 0.2, 0.2]
SEED = 42
random.seed(SEED)

out_dir = os.path.join(os.path.dirname(PATH), 'holdout_shards_splits')
os.makedirs(out_dir, exist_ok=False)

samples = defaultdict(list)


def read_data(path=PATH):
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            print(line)
            line = line.strip().split('\t')
            if len(line) != 3:
                continue
            author, para, text = line
            yield author, para, text


for author, para, text in read_data():
    samples[author].append((author, para, text))

for author in samples:
    random.shuffle(samples[author])


def split_by_author(author_samples, splits):
    total = len(author_samples)
    assert sum(splits) == 1
    num_by_split = []
    for s in splits:
        num_by_split.append(math.ceil(total * s))

    # assert sum(num_by_split) == total, (sum(num_by_split), total)
    samples_split = []
    i = 0
    for n in num_by_split:
        samples_split.append(author_samples[i : i + n])
        i += n
    assert sum(len(x) for x in samples_split) == total, (
        sum(len(x) for x in samples_split),
        total,
    )
    return samples_split


split_samples = {'train': [], 'val': [], 'test': []}
for author, author_samples in samples.items():
    a_train, a_dev, a_test = split_by_author(author_samples, splits=TRAIN_DEV_TEST)
    split_samples['train'].extend(a_train)
    split_samples['val'].extend(a_dev)
    split_samples['test'].extend(a_test)

for split in split_samples:
    random.shuffle(split_samples[split])

for split in split_samples:
    with open(os.path.join(out_dir, f'{split}.txt'), 'w') as f:
        for sample in split_samples[split]:
            f.write('\t'.join(sample) + '\n')


with open(os.path.join(out_dir, 'info.json'), 'w') as f:
    json.dump(
        {
            'train_dev_test': TRAIN_DEV_TEST,
            'path': PATH,
            'seed': SEED,
            'holdout_authors': list(samples.keys()),
            'num_holdout_authors': len(samples),
            'num_holdout_samples': sum(len(x) for x in samples.values()),
            'num_train_samples': len(split_samples['train']),
            'num_val_samples': len(split_samples['val']),
            'num_test_samples': len(split_samples['test']),
        },
        f,
    )


# import pdb; pdb.set_trace()
