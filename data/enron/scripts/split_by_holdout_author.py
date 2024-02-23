import os
from collections import defaultdict
import random
import json
import sys

random.seed(42)

path = sys.argv[1]
outpath = os.path.join(os.path.dirname(path), 'holdout_author_splits/')

os.makedirs(outpath, exist_ok=True)

# def get_all_authors():
#     pass

author_to_sharded = defaultdict(lambda: {k: [] for k in ['train', 'val', 'test']})

for shard in ['train', 'val', 'test']:
    with open(path + f'/{shard}.txt', 'r') as f:
        lines = f.readlines()
    for l in lines:
        author, para, text = l.strip().split('\t')
        author_to_sharded[author][shard].append((para, text))

total_authors = len(author_to_sharded)
for author, author_data in author_to_sharded.items():
    clean_author = '_'.join(author.replace(':', '').split())
    author_path = os.path.join(outpath, clean_author)
    os.makedirs(author_path, exist_ok=True)
    print([len(author_data[k]) for k in ['train', 'val', 'test']])
    for split in ['train', 'val', 'test']:
        with open(os.path.join(author_path, f'{split}.txt'), 'w') as f:
            random.shuffle(author_data[split])
            for para, text in author_data[split]:
                f.write(f'{para}\t{text}\n')

author_list = sorted(list(author_to_sharded.keys()))
assignments = {}
for i, author in enumerate(author_list):
    other_authors = [a for a in author_list if a != author]
    random.shuffle(other_authors)
    val_samples = author_to_sharded[author]['val'][:5]
    test_samples = author_to_sharded[author]['test'][:5]
    assignments[author] = {
        'test_samples': test_samples,
        'val_samples': val_samples,
        'target': other_authors[:5],
    }

with open(os.path.join(outpath, 'assignments.json'), 'w') as f:
    json.dump(assignments, f, indent=2)


# For paraguide
# for each 110 authors
#  - take up to 5 (val/test) reports
#  - take the target author's training data
#  - perform guidance
#  - evaluate toward on the target author's (full val/test) data
#  - evaluate away on the source author's (full val/test) data
#  - evaluate toward on the target author's (train) data
#  - evaluate away on the source author's (train) data

# ChatGPT
# for each 110 authors
#  - take up to 5 (val/test) reports
#  - take (up to 16) of target author's training data
#  - prompt model
#  - evaluate toward on the target author's (full val/test) data
#  - evaluate away on the source author's (full val/test) data
#  - evaluate toward on the target author's (train) data
#  - evaluate away on the source author's (train) data

# STRAP
# Train model for each author on training data, choose best model based on val
# for each 110 authors
#  - take up to 5 (val/test) reports
#  - evaluate toward on the target author's (full val/test) data
#  - evaluate away on the source author's (full val/test) data
#  - evaluate toward on the target author's (train) data
#  - evaluate away on the source author's (train) data

# 110 x 5 x 3  = 1650
