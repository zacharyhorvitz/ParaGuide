''' Attribute style evals '''

import os
import sys
import numpy as np
import torch
import argparse
import json
from transformers import (
    XLMRobertaTokenizerFast,
    XLMRobertaForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


sys.path.append('../inference')
# Internal Accuracies
from classifiers import load_formality_model, load_sentiment_model
from eval_style import get_raw_mis_score, get_cola_score, clean_signature

# load roberta base model


def get_attribute_acc(
    *, model, texts, target_idx, tokenizer, device='cuda', aggregate=True
):
    scores = []
    for text in texts:
        inputs = tokenizer([text], return_tensors="pt", padding=True)
        inputs.to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        scores.append(float(preds[0].item() == target_idx))
    if aggregate:
        return np.mean(scores)

    return scores


def run_formality_eval(*, references, candidates, target, device='cuda'):
    ctr_model, tokenizer, _, _ = load_formality_model()
    optimizing_label_index = 1 if target == 'formal' else 0
    ctr_model.to(device)
    ctr_model.eval()

    holdout_tokenizer = XLMRobertaTokenizerFast.from_pretrained(
        'SkolkovoInstitute/xlmr_formality_classifier'
    )
    holdout_model = XLMRobertaForSequenceClassification.from_pretrained(
        'SkolkovoInstitute/xlmr_formality_classifier'
    )
    holdout_optimizing_label_index = 0 if target == 'formal' else 1
    holdout_model.to(device)
    holdout_model.eval()

    retval = {}
    retval['cola'] = get_cola_score(candidates, aggregate=False)
    retval['accuracy'] = get_attribute_acc(
        model=ctr_model,
        texts=candidates,
        target_idx=optimizing_label_index,
        tokenizer=tokenizer,
        device=device,
        aggregate=False,
    )
    retval['holdout_accuracy'] = get_attribute_acc(
        model=holdout_model,
        texts=candidates,
        target_idx=holdout_optimizing_label_index,
        tokenizer=holdout_tokenizer,
        device=device,
        aggregate=False,
    )
    retval['similarity'] = get_raw_mis_score(
        references=references,
        candidates=candidates,
        targets=[None for _ in range(len(references))],
        aggregate=False,
    )
    retval['n'] = len(references)
    retval['holdout_joint_gm'] = np.mean(
        [
            (a * c * m) ** (1 / 3)
            for a, c, m in zip(
                retval['holdout_accuracy'], retval['cola'], retval['similarity']
            )
        ]
    )
    retval['cola'] = np.mean(retval['cola'])
    retval['accuracy'] = np.mean(retval['accuracy'])
    retval['similarity'] = np.mean(retval['similarity'])
    retval['holdout_accuracy'] = np.mean(retval['holdout_accuracy'])

    return retval


def run_sentiment_eval(*, references, candidates, target, device='cuda'):
    ctr_model, tokenizer, _, possible_labels = load_sentiment_model("sentiment")
    optimizing_label_index = possible_labels.index(target)
    ctr_model.to(device)
    ctr_model.eval()

    holdout_tokenizer = AutoTokenizer.from_pretrained(
        "siebert/sentiment-roberta-large-english"
    )
    holdout_model = AutoModelForSequenceClassification.from_pretrained(
        "siebert/sentiment-roberta-large-english"
    )
    holdout_optimizing_label_index = 1 if target == 'positive' else 0
    holdout_model.to(device)
    holdout_model.eval()

    retval = {}
    retval['cola'] = get_cola_score(candidates, aggregate=False)
    retval['accuracy'] = get_attribute_acc(
        model=ctr_model,
        texts=candidates,
        target_idx=optimizing_label_index,
        tokenizer=tokenizer,
        device=device,
        aggregate=False,
    )
    retval['holdout_accuracy'] = get_attribute_acc(
        model=holdout_model,
        texts=candidates,
        target_idx=holdout_optimizing_label_index,
        tokenizer=holdout_tokenizer,
        device=device,
        aggregate=False,
    )
    retval['similarity'] = get_raw_mis_score(
        references=references,
        candidates=candidates,
        targets=[None for _ in range(len(references))],
        aggregate=False,
    )
    retval['n'] = len(references)

    retval['holdout_joint_gm'] = np.mean(
        [
            (a * c * m) ** (1 / 3)
            for a, c, m in zip(
                retval['holdout_accuracy'], retval['cola'], retval['similarity']
            )
        ]
    )

    retval['cola'] = np.mean(retval['cola'])
    retval['accuracy'] = np.mean(retval['accuracy'])
    retval['similarity'] = np.mean(retval['similarity'])
    retval['holdout_accuracy'] = np.mean(retval['holdout_accuracy'])

    return retval


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_path', type=str)
    argparser.add_argument('--clean', action='store_true')
    argparser.add_argument('--target', type=str)

    clean_regex = r"(?<=[\.\,\!\?\-\)\/\`])(\s[a-zA-Z]+([a-zA-Z\s]?)([a-zA-Z\.\<\>\-0-9\\\:\@\s]{0,8})(?<!\.)$|[\w\s]+@[A-Za-z]+(\s\d{2}?)(\/\d{2}?)(\/\d{4}\s?)\d{2}:(\d{2}\s?)(?:AM|PM)$)"
    args = argparser.parse_args()
    out_name = args.input_path + f'.{args.target}_eval'
    if args.clean:
        out_name += '.clean'

    with open(args.input_path, 'r') as f:
        lines = f.readlines()
        input_data = [json.loads(line.strip()) for line in lines]

    references = [d['original_text'] for d in input_data]
    candidates = [d['decoded'][0] if len(d['decoded']) > 0 else '' for d in input_data]
    candidates_para = [d['paraphrase'] for d in input_data]

    if args.clean:
        references = [clean_signature(r, clean_regex) for r in references]
        candidates = [clean_signature(c, clean_regex) for c in candidates]
        candidates_para = [clean_signature(c, clean_regex) for c in candidates_para]

    # decoded eval
    if args.target in ['formal', 'informal']:
        eval_results = run_formality_eval(
            references=references, candidates=candidates, target=args.target
        )
    elif args.target == 'positive' or args.target == 'negative':
        eval_results = run_sentiment_eval(
            references=references, candidates=candidates, target=args.target
        )
    else:
        raise ValueError(f'Unknown target {args.target}')
    print('Decoded eval results:', eval_results)

    with open(out_name, 'w') as f:
        json.dump(
            {
                'input_path': args.input_path,
                'decoded': eval_results,
                'clean': args.clean,
            },
            f,
        )
