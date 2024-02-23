# split into shards


import sys
import os
from collections import defaultdict
import random
import json

PATH = sys.argv[
    1
]  # 'processed_data/paraphrased_topp0.8_tmp1.5_idx0_users_data_50_unique_clean_min_10_fixed_sender.tsv'
MIN_CUTOFF = 10
HOLDOUT = 0.1
TRAIN_DEV_TEST = [0.8, 0.1, 0.1]
SEED = 42
random.seed(SEED)

# Remember, this is the list of duplicates manually surfaced for 10 min 10 examples
DUPLICATE_LIST = (
    '''
From: postmaster@enron.com From: postmaster@ftenergy.com
From: k..allen@enron.com From: kimberly.allen@enron.com
From: pyoung@cliverunnells.com From: pyoung@pdq.net
From: daphneco64@alltel.net From: daphneco64@bigplanet.com
From: mark.knippa@enron.com From: mark.koenig@enron.com
From: j..porter@enron.com From: jeffrey.porter@enron.com
From: outlook-migration-team@enron.com From: outlook.team@enron.com
From: will.smith@enron.com From: william.smith@enron.com
From: announcements.enron@enron.com From: enron.announcements@enron.com
From: capstone@ktc.com From: capstone@texas.net
From: lwbthemarine@alltel.net From: lwbthemarine@bigplanet.com
From: scott.neal@enron.com From: scott.neal@worldnet.att.net
From: mona.l.petrochko@enron.com From: mona.petrochko@enron.com
From: eldon@direcpc.com From: eldon@interx.net
From: postmaster@dowjones.com From: postmaster@ftenergy.com
From: kvanpelt@flash.net From: kvanpelt@houston.rr.com
From: l..petrochko@enron.com From: mona.l.petrochko@enron.com
From: m..taylor@enron.com From: mark.taylor@enron.com
'''.replace(
        'From: ', ''
    )
    .strip()
    .split()
)

out_dir = os.path.join(os.path.dirname(PATH), 'shards')
os.makedirs(out_dir, exist_ok=False)

counts = defaultdict(int)


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


# print('0', sum(counts.values()))
# print('10', sum([v for k,v in counts.items() if v >= 10]))
# print('20', sum([v for k,v in counts.items() if v >= 20]))
# print('50', sum([v for k,v in counts.items() if v >= 50]))

for author, _, _ in read_data():
    counts[author] += 1

valid_authors = sorted([k for k, v in counts.items() if v >= MIN_CUTOFF])

duplicate_authors = [
    author for author in valid_authors if author.replace('From: ', '') in DUPLICATE_LIST
]
print(duplicate_authors)
print('DUPLICATE AUTHORS:', len(duplicate_authors))
valid_authors = [
    author
    for author in valid_authors
    if author.replace('From: ', '') not in DUPLICATE_LIST
]

print(len(valid_authors))
random.shuffle(valid_authors)
valid_authors = (
    valid_authors + duplicate_authors
)  # put duplicate authors at the end so they are not in the holdout set
num_holdout = int(len(valid_authors) * HOLDOUT)
valid_authors_holdout = set(valid_authors[:num_holdout])
valid_authors_holdin = set(valid_authors[num_holdout:])

print('HOLDOUT AUTHORS:', len(valid_authors_holdout))
print('HOLDIN AUTHORS:', len(valid_authors_holdin))

data_splits = {'holdout': [], 'holdin': []}
for author, para, text in read_data():
    if author in valid_authors_holdout:
        data_splits['holdout'].append((author, para, text))
    else:
        data_splits['holdin'].append((author, para, text))

random.shuffle(data_splits['holdout'])
random.shuffle(data_splits['holdin'])

num_holdin_training = int(len(data_splits['holdin']) * TRAIN_DEV_TEST[0])
num_holdin_dev = int(len(data_splits['holdin']) * TRAIN_DEV_TEST[1])

data_splits['holdin_splits'] = {
    'train': data_splits['holdin'][:num_holdin_training],
    'dev': data_splits['holdin'][
        num_holdin_training : num_holdin_training + num_holdin_dev
    ],
    'test': data_splits['holdin'][num_holdin_training + num_holdin_dev :],
}

print('HOLDOUT:', len(data_splits['holdout']))
print('HOLDIN:', len(data_splits['holdin']))
print('\tTRAIN:', len(data_splits['holdin_splits']['train']))
print('\tDEV:', len(data_splits['holdin_splits']['dev']))
print('\tTEST:', len(data_splits['holdin_splits']['test']))

with open(os.path.join(out_dir, 'holdout_author_data.txt'), 'w') as f_holdout:
    for author, para, text in data_splits['holdout']:
        f_holdout.write('\t'.join([author, para, text]) + '\n')

for split in ['train', 'dev', 'test']:
    with open(
        os.path.join(out_dir, f'holdin_author_data_{split}.txt'), 'w'
    ) as f_holdin:
        for author, para, text in data_splits['holdin_splits'][split]:
            f_holdin.write('\t'.join([author, para, text]) + '\n')

with open(os.path.join(out_dir, 'info.json'), 'w') as f:
    json.dump(
        {
            'min_cutoff': MIN_CUTOFF,
            'holdout': HOLDOUT,
            'train_dev_test': TRAIN_DEV_TEST,
            'path': PATH,
            'seed': SEED,
            'holdout_authors': len(valid_authors_holdout),
            'holdin_authors': len(valid_authors_holdin),
            'num_holdout': len(data_splits['holdout']),
            'num_holdin': len(data_splits['holdin']),
            'num_holdin_train': len(data_splits['holdin_splits']['train']),
            'num_holdin_dev': len(data_splits['holdin_splits']['dev']),
            'num_holdin_test': len(data_splits['holdin_splits']['test']),
            'duplicate_authors': duplicate_authors,
        },
        f,
    )


# import pdb; pdb.set_trace()
