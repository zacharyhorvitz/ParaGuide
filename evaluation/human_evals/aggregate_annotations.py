import os
import pandas as pd
from collections import defaultdict, Counter
import json
import numpy as np

folder = 'annotations/'
data_path = 'all_data_for_labeling.jsonl'
dfs = []

from statsmodels.stats.weightstats import ztest
import sys

sys.path.append('krippendorff-alpha/')

from krippendorff_alpha import krippendorff_alpha, nominal_metric


results_to_names = {
    '../automatic_evals/all_test_formality_results/2023-08-09_20-13-23': 'strap',
    '../automatic_evals/all_test_formality_results/2023-08-10_12-36-11_200.0': 'paraguide_200',
    '../automatic_evals/all_test_formality_results/2023-08-11_14-26-24': 'mm_ham',
    '../automatic_evals/all_test_formality_results/2023-08-10_12-32-31_200.0': 'paraguide_200',
    '../automatic_evals/all_test_formality_results/2023-08-11_14-25-44': 'mm_ham',
    '../automatic_evals/all_test_formality_results/2023-08-09_20-04-03': 'strap',
}


labels = ['Similar', 'Well-formedness', 'Formality']

for file in os.listdir(folder):
    if file.endswith('.csv'):
        df = pd.read_csv(folder + file)
        dfs.append(df)

id_to_ratings = {}

total_annotators = len(dfs)
for i, df in enumerate(dfs):
    for index, row in df.iterrows():
        clean_row = {}
        for key in row.keys():
            for label in labels:
                if key.startswith(label):
                    try:
                        clean_row[label] = int(row[key])
                    except ValueError:
                        clean_row[label] = None

        assert len(clean_row.keys()) == 3

        if any([clean_row[label] is None for label in labels]):
            continue

        row['id'] = str(row['id'])
        if row['id'] in id_to_ratings:
            continue

        id_to_ratings[row['id']] = {label: int(clean_row[label]) for label in labels}
        id_to_ratings[row['id']]['annotator_id'] = i


model_to_idx_ratings = defaultdict(lambda: defaultdict(list))

missing_annotations = defaultdict(list)

with open(data_path, 'r') as f:
    for l in f:
        data = json.loads(l)
        key = (
            data['target_label']
            + '_'
            + results_to_names[os.path.dirname(data['data_path'])]
        )
        if str(data['annotation_id']) not in id_to_ratings:
            missing_annotations[data['annotator']].append(data['annotation_id'])
            continue
        model_to_idx_ratings[key][data['data_index']].append(
            id_to_ratings[str(data['annotation_id'])]
        )

assert len(missing_annotations) == 0

all_labels_by_annotator = {label: [] for label in labels}


for model in model_to_idx_ratings:
    for idx in model_to_idx_ratings[model]:
        ratings = model_to_idx_ratings[model][idx]
        aggregate = {}
        for label in labels:
            label_ratings = [r[label] for r in ratings]
            annotators = [r['annotator_id'] for r in ratings]

            to_fill_in = ['*' for _ in range(total_annotators)]
            for l, a in zip(label_ratings, annotators):
                to_fill_in[a] = l

            all_labels_by_annotator[label].append(to_fill_in)

            aggregate[label] = Counter(label_ratings).most_common(1)[0][0]
        model_to_idx_ratings[model][idx] = aggregate

model_to_aggregate_ratings = {model: {} for model in model_to_idx_ratings}
model_to_all_ratings = {
    model: {label: [] for label in labels} for model in model_to_idx_ratings
}

for model in model_to_idx_ratings:
    for label in labels:
        model_to_all_ratings[model][label] = np.array(
            [
                model_to_idx_ratings[model][idx][label]
                for idx in model_to_idx_ratings[model]
            ]
        )
        model_to_aggregate_ratings[model][label] = np.mean(
            model_to_all_ratings[model][label]
        )

    if model.startswith('formal'):
        model_to_all_ratings[model]['Accuracy'] = model_to_all_ratings[model][
            'Formality'
        ]

    else:
        model_to_all_ratings[model]['Accuracy'] = (
            1 - model_to_all_ratings[model]['Formality']
        )

    model_to_aggregate_ratings[model]['Accuracy'] = np.mean(
        model_to_all_ratings[model]['Accuracy']
    )

    all_correct = (
        model_to_all_ratings[model]['Similar']
        * model_to_all_ratings[model]['Well-formedness']
        * model_to_all_ratings[model]['Accuracy']
    )

    model_to_all_ratings[model]['Success (SWF)'] = all_correct
    model_to_aggregate_ratings[model]['Success (SWF)'] = np.mean(all_correct)


df = pd.DataFrame.from_dict(
    model_to_aggregate_ratings,
    orient='index',
    columns=['model'] + labels + ['Accuracy', 'Success (SWF)'],
).sort_index()

column_order = ['Accuracy', 'Similar', 'Well-formedness', 'Success (SWF)']
df = df[column_order]


informal_df = df[df.index.str.startswith('informal')]
formal_df = df[df.index.str.startswith('formal')]

print(formal_df)
print(informal_df)

# save to csv
formal_df.to_csv('formal_aggregate.csv')
informal_df.to_csv('informal_aggregate.csv')

for label in all_labels_by_annotator:
    data = np.array(all_labels_by_annotator[label])
    print(data.shape)
    print(
        "krippendorff_alpha",
        krippendorff_alpha(np.transpose(data), nominal_metric, missing_items='*'),
    )


all_models = sorted(
    set([x.replace('informal', '').replace('formal', '') for x in model_to_all_ratings])
)

for label in ['Success (SWF)', 'Accuracy', 'Similar', 'Well-formedness']:
    print(label)

    model_to_all_results = {model: [] for model in all_models}
    for model in all_models:
        for direction in ['formal', 'informal']:
            assert direction + model in model_to_all_ratings
            model_to_all_results[model].extend(
                model_to_all_ratings[direction + model][label]
            )

    top_model = model_to_all_results['_paraguide_200']
    second_model = model_to_all_results['_mm_ham']

    # z test
    z_stat, p_val = ztest(top_model, second_model, alternative='larger')
    print("z test", z_stat, p_val)
