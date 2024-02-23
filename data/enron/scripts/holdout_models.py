''' Do semantic evals on the paraphrase models '''

import os
import sys
import numpy as np
import torch
import argparse
import json
import re
from tqdm import tqdm
import math


from transformers import (
    XLMRobertaTokenizerFast,
    XLMRobertaForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from transformers import (
    RobertaTokenizer,
    RobertaModel,
    RobertaForSequenceClassification,
)


def batch_pairs(lists, batch_size=64):
    # if references is not None:
    #     assert len(references) == len(candidates)
    lengths = [len(l) for l in lists]
    assert len(set(lengths)) == 1, lengths

    num_elements = len(lists[0])
    batches = []
    idx = 0
    for i in range(math.ceil(num_elements / batch_size)):
        max_idx = idx + batch_size
        batches.append([l[idx:max_idx] for l in lists])
        idx = max_idx
    assert sum([len(x[0]) for x in batches]) == num_elements, (
        sum([len(x[0]) for x in batches]),
        num_elements,
    )
    return batches


def get_holdout_formality_model(target, device='cuda'):
    assert target in ['formal', 'informal']
    holdout_tokenizer = XLMRobertaTokenizerFast.from_pretrained(
        'SkolkovoInstitute/xlmr_formality_classifier'
    )
    holdout_model = XLMRobertaForSequenceClassification.from_pretrained(
        'SkolkovoInstitute/xlmr_formality_classifier'
    )
    holdout_optimizing_label_index = 0 if target == 'formal' else 1
    holdout_model.to(device)
    holdout_model.eval()

    return holdout_model, holdout_tokenizer, holdout_optimizing_label_index


def get_holdout_toxicity_model(target, device='cuda'):
    assert target == 'detox'
    holdout_tokenizer = AutoTokenizer.from_pretrained("martin-ha/toxic-comment-model")
    holdout_model = AutoModelForSequenceClassification.from_pretrained(
        "martin-ha/toxic-comment-model"
    )
    holdout_optimizing_label_index = 0
    holdout_model.to(device)
    holdout_model.eval()
    return holdout_model, holdout_tokenizer, holdout_optimizing_label_index


def get_holdout_sentiment_model(target, device='cuda'):
    assert target in ['positive', 'negative']
    holdout_tokenizer = AutoTokenizer.from_pretrained(
        "siebert/sentiment-roberta-large-english"
    )
    holdout_model = AutoModelForSequenceClassification.from_pretrained(
        "siebert/sentiment-roberta-large-english"
    )
    holdout_optimizing_label_index = 1 if target == 'positive' else 0
    holdout_model.to(device)
    holdout_model.eval()
    return holdout_model, holdout_tokenizer, holdout_optimizing_label_index


def get_holdout_fluency(target, device='cuda'):
    assert target == 'fluent'
    cola_tokenizer = RobertaTokenizer.from_pretrained('textattack/roberta-base-CoLA')
    cola_model = RobertaForSequenceClassification.from_pretrained(
        'textattack/roberta-base-CoLA'
    )
    holdout_optimizing_label_index = 1
    cola_model.to(device)
    cola_model.eval()
    return cola_model, cola_tokenizer, holdout_optimizing_label_index


def get_holdout_wellformed(target, device='cuda'):
    assert target == 'wellformed'
    holdout_tokenizer = AutoTokenizer.from_pretrained(
        "salesken/query_wellformedness_score"
    )
    holdout_model = AutoModelForSequenceClassification.from_pretrained(
        "salesken/query_wellformedness_score"
    )
    holdout_optimizing_label_index = 0
    holdout_model.to(device)
    holdout_model.eval()
    return holdout_model, holdout_tokenizer, holdout_optimizing_label_index


def get_holdout_irony(target, device='cuda'):
    assert target == 'ironic'
    holdout_tokenizer = AutoTokenizer.from_pretrained(
        "cardiffnlp/twitter-roberta-base-irony"
    )
    holdout_model = AutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base-irony"
    )
    holdout_optimizing_label_index = 1 if target == 'ironic' else 0
    holdout_model.to(device)
    holdout_model.eval()
    return holdout_model, holdout_tokenizer, holdout_optimizing_label_index


def get_holdout_question_statement(target, device='cuda'):
    assert target == 'question'
    holdout_tokenizer = AutoTokenizer.from_pretrained(
        "shahrukhx01/question-vs-statement-classifier"
    )
    holdout_model = AutoModelForSequenceClassification.from_pretrained(
        "shahrukhx01/question-vs-statement-classifier"
    )
    holdout_optimizing_label_index = 1
    holdout_model.to(device)
    holdout_model.eval()
    return holdout_model, holdout_tokenizer, holdout_optimizing_label_index


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_paths', type=str, nargs='+')
    argparser.add_argument('--out_dir', type=str)
    argparser.add_argument('--target', type=str)

    args = argparser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_name = os.path.join(args.out_dir, f'{args.target}_scored.tsv')

    assert args.target in [
        'formal',
        'informal',
        'detox',
        'positive',
        'negative',
        'fluent',
        'ironic',
        'wellformed',
        'question',
    ]

    if args.target in ['formal', 'informal']:
        (
            holdout_model,
            holdout_tokenizer,
            holdout_optimizing_label_index,
        ) = get_holdout_formality_model(args.target)
    elif args.target == 'detox':
        (
            holdout_model,
            holdout_tokenizer,
            holdout_optimizing_label_index,
        ) = get_holdout_toxicity_model(args.target)
    elif args.target in ['positive', 'negative']:
        (
            holdout_model,
            holdout_tokenizer,
            holdout_optimizing_label_index,
        ) = get_holdout_sentiment_model(args.target)
    elif args.target == 'fluent':
        (
            holdout_model,
            holdout_tokenizer,
            holdout_optimizing_label_index,
        ) = get_holdout_fluency(args.target)
    elif args.target == 'ironic':
        (
            holdout_model,
            holdout_tokenizer,
            holdout_optimizing_label_index,
        ) = get_holdout_irony(args.target)
    elif args.target == 'wellformed':
        (
            holdout_model,
            holdout_tokenizer,
            holdout_optimizing_label_index,
        ) = get_holdout_wellformed(args.target)
    elif args.target == 'question':
        (
            holdout_model,
            holdout_tokenizer,
            holdout_optimizing_label_index,
        ) = get_holdout_question_statement(args.target)
    else:
        raise ValueError(f'Unknown target {args.target}')

    all_data = []

    for path in sorted(args.input_paths):
        with open(path, 'r') as f:
            lines = f.readlines()
        for l in lines:
            all_data.append(l.strip().split('\t'))

    authors = [x[0] for x in all_data]
    paras = [x[1] for x in all_data]
    texts = [x[2] for x in all_data]

    batched_data = batch_pairs([authors, paras, texts], batch_size=32)

    all_scores = []

    with open(out_name, 'w+') as f:
        for a, p, t in tqdm(batched_data):
            inputs = holdout_tokenizer(t, return_tensors="pt", padding=True)
            inputs.to('cuda')
            outputs = holdout_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            scores = probs[:, holdout_optimizing_label_index].cpu().detach().numpy()
            for i in range(len(a)):
                f.write(f'{a[i]}\t{p[i]}\t{t[i]}\t{scores[i]}\n')
                # print(f'{a[i]}\t{p[i]}\t{t[i]}\t{scores[i]}\n')
                all_scores.append(scores[i])

    # TypeError: Object of type float32 is not JSON serializable

    mean_score = np.mean(all_scores)
    std_score = np.std(all_scores)
    class_greater_half = np.sum(np.array(all_scores) > 0.5)

    with open(out_name + '.stats.json', 'w') as f:
        json.dump(
            {
                'target': args.target,
                'input_paths': args.input_paths,
                'out_dir': args.out_dir,
                'mean_score': float(mean_score),
                'std_score': float(std_score),
                'class_greater_half': float(class_greater_half),
                'num_examples': len(all_scores),
            },
            f,
        )
