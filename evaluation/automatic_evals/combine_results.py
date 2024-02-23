import glob
import argparse
import pandas as pd
import os
import json


def choose_meaningful_hparams(hparam_lists):
    all_keys = set()
    for hparams in hparam_lists:
        all_keys.update(hparams.keys())

    keys_to_values = {k: [] for k in all_keys}
    for hparams in hparam_lists:
        for k, v in hparams.items():
            assert k in keys_to_values
            if v not in keys_to_values[k]:
                keys_to_values[k].append(v)

    meaningful_hparams = []
    for k, v in keys_to_values.items():
        if len(v) > 1:
            meaningful_hparams.append(k)

    return meaningful_hparams


COLUMN_ORDER = [
    'n',
    '_style_joint_fluency',
    '_luar_joint_fluency',
    'holdout_joint_gm',
    'holdout_joint_binary',
    'joint',
    'joint_fluency',
    'bleu',
    'cola',
    'accuracy',
    'holdout_accuracy',
    '_style_joint',
    '_luar_joint',
    'similarity',
    'rouge',
    '_style_away',
    '_style_confusion',
    '_style_towards',
    '_luar_away',
    '_luar_confusion',
    '_luar_towards',
]

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_dir', type=str)
    argparser.add_argument('--clean', action='store_true')
    argparser.add_argument('--task', type=str)

    args = argparser.parse_args()
    assert args.task in [
        'luar_eval',
        'detox_eval',
        'informal_eval',
        'formal_eval',
        'style_eval',
        'positive_eval',
        'negative_eval',
    ]

    print(args.input_dir)
    key = args.task

    if args.clean:
        findkey = f'{key}.clean'
    else:
        findkey = f'{key}'

    possible_paths = sorted(
        glob.glob(os.path.join(args.input_dir, f'*/*.{findkey}'))
    ) + sorted(glob.glob(os.path.join(args.input_dir, f'*.{findkey}')))

    results = {}
    for p in possible_paths:
        parent_dir = os.path.dirname(p)
        results[parent_dir] = {}

        if os.path.exists(os.path.join(parent_dir, 'hparams.json')):
            with open(os.path.join(parent_dir, 'hparams.json'), 'r') as f:
                results[parent_dir]['hparams'] = json.load(f)
        elif os.path.exists(os.path.join(parent_dir, 'args.json')):
            with open(os.path.join(parent_dir, 'args.json'), 'r') as f:
                results[parent_dir]['hparams'] = json.load(f)
        else:
            results[parent_dir]['hparams'] = {'path': p}

        with open(p, 'r') as f:
            results[parent_dir]['eval'] = json.load(f)

    hparam_lists = [r['hparams'] for r in results.values()]
    meaningful_hparams = sorted(choose_meaningful_hparams(hparam_lists))
    print(meaningful_hparams)

    metrics = sorted(
        set().union(*[set(r['eval']['decoded'].keys()) for r in results.values()]),
        key=lambda x: COLUMN_ORDER.index(x),
    )
    filtered_results = {}
    for p, r in results.items():
        filtered_results[p] = [r['hparams'].get(k, None) for k in meaningful_hparams]
        for m in metrics:
            filtered_results[p].append(r['eval']['decoded'][m])

    df = pd.DataFrame.from_dict(
        filtered_results, orient='index', columns=meaningful_hparams + metrics
    )
    df.to_csv(os.path.join(args.input_dir, f'combined_results.{findkey}.csv'))
    print(df)
