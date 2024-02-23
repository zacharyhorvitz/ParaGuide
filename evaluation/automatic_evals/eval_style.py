''' Authorship evals '''

import os
import sys
import numpy as np
import torch
import argparse
import json
import re
from tqdm import tqdm
import math
import re

##################
# Evaluation metrics from STYLL paper:
# Patel, A.; Andrews, N.; and Callison-Burch, C. 2022. LowResource Authorship Style Transfer with In-Context Learning. arXiv:2212.08986.


def sim(u, v):
    s = (
        1
        - np.arccos(
            np.clip(
                np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-10), -1, 1
            )
        )
        / math.pi
    )
    return (s + 1) / 2


def sim_c(u, v):
    return 1 - sim(u, v)


def away(*, source, transferred, target):
    return min(sim_c(transferred, source), sim_c(target, source)) / (
        sim_c(target, source) + 1e-10
    )


def towards(*, source, transferred, target):
    return max(sim(transferred, target) - sim(target, source), 0) / (
        sim_c(source, target) + 1e-10
    )


def confusion(*, source, transferred, target):
    return float(sim(transferred, target) > sim(transferred, source))


##################

# mis score
from mutual_implication_score import MIS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mis = MIS(device=device)


# load roberta base model
from transformers import RobertaTokenizer, RobertaForSequenceClassification

cola_tokenizer = RobertaTokenizer.from_pretrained('textattack/roberta-base-CoLA')
cola_model = RobertaForSequenceClassification.from_pretrained(
    'textattack/roberta-base-CoLA'
)
cola_model.to(device)

from sentence_transformers import SentenceTransformer

style_model = SentenceTransformer('AnnaWegmann/Style-Embedding')
style_model.to(device)


# sys.path.append('luar_inference/inference/')
# #from inference import LUARInference
# #luar_model_path = "luar_inference/LUAR.pth" # Requires downloading the UAR model
# #luar_model = LUARInference(luar_model_path)


def style_encode(text, embed_model):
    if embed_model == 'luar':
        raise NotImplementedError
        # return luar_model(text)[0].squeeze(0).detach().cpu().numpy()
    elif embed_model == 'style':
        return style_model.encode(text)
    else:
        raise NotImplementedError


def get_author_data(*, author_name, author_directory, shard='train'):
    clean_name = author_name.replace(":", "").replace(" ", "_")
    with open(os.path.join(author_directory, clean_name, f'{shard}.txt'), 'r') as f:
        lines = f.readlines()
        input_data = [line.strip().split('\t') for line in lines]
    return input_data


def batch_pairs(lists, batch_size=64):
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


def get_raw_mis_score(*, references, candidates, targets, aggregate=True):
    mis_score_transfer_source = []
    batched = batch_pairs([references, candidates, targets], batch_size=64)
    for b in tqdm(batched):
        refs, cands, targs = b
        mis_score_transfer_source.extend(mis.compute(refs, cands))

    if aggregate:
        return np.mean(mis_score_transfer_source)
    return mis_score_transfer_source


def get_cola_score(candidates, aggregate=True):
    # transferred
    prbs = []
    batched = batch_pairs([candidates], batch_size=64)
    for b in batched:
        inputs = cola_tokenizer(*b, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = cola_model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        prbs.extend(probs[:, 1].detach().cpu().tolist())

    if aggregate:
        return np.mean(prbs)
    return prbs


def get_author_embedding_scores(
    *,
    sources,
    transferred,
    input_labels,
    target_labels,
    author_to_style_embeds,
    embed_model,
    aggregate=True,
):
    away_scores = []
    towards_scores = []
    confusion_scores = []

    for _, transfer, s_label, t_label in zip(
        sources, transferred, input_labels, target_labels
    ):
        target_embeds = author_to_style_embeds.get(t_label, [])
        if len(target_embeds) == 0:
            print(f'Warning! Skipping {t_label} for style, no examples')
            continue

        source_embeds = author_to_style_embeds.get(s_label, [])
        if len(source_embeds) == 0:
            print(f'Warning! Skipping {s_label} for style, no examples')
            continue

        target_embeds = np.mean(np.stack(target_embeds), axis=0)
        source_embeds = np.mean(np.stack(source_embeds), axis=0)

        transfer_embed = style_encode(transfer, embed_model=embed_model)
        away_scores.append(
            away(source=source_embeds, transferred=transfer_embed, target=target_embeds)
        )
        towards_scores.append(
            towards(
                source=source_embeds, transferred=transfer_embed, target=target_embeds
            )
        )
        confusion_scores.append(
            confusion(
                source=source_embeds, transferred=transfer_embed, target=target_embeds
            )
        )

    if aggregate:
        return {
            'away': np.mean(away_scores),
            'towards': np.mean(towards_scores),
            'confusion': np.mean(confusion_scores),
        }

    return {
        'away': away_scores,
        'towards': towards_scores,
        'confusion': confusion_scores,
    }


def run_eval(
    *,
    references,
    candidates,
    embed_model,
    input_authors=None,
    target_authors=None,
    acc_model=None,
    acc_label_mapping=None,
    acc_tokenizer=None,
    author_to_style_embeds=None,
    author_to_text=None,
):
    retval = {}

    retval['cola'] = np.array(get_cola_score(candidates, aggregate=False))
    target_texts = [author_to_text[a] for a in input_authors]

    style_embed_results = get_author_embedding_scores(
        sources=references,
        transferred=candidates,
        input_labels=input_authors,
        target_labels=target_authors,
        author_to_style_embeds=author_to_style_embeds,
        embed_model=embed_model,
        aggregate=False,
    )
    retval['similarity'] = np.array(
        get_raw_mis_score(
            references=references,
            candidates=candidates,
            targets=target_texts,
            aggregate=False,
        )
    )

    retval['confusion'] = np.array(style_embed_results['confusion'])
    retval['joint_fluency'] = np.mean(
        (retval['confusion'] * retval['similarity'] * retval['cola']) ** (1 / 3)
    )
    retval['similarity'] = np.mean(retval['similarity'])
    retval['confusion'] = np.mean(retval['confusion'])
    retval['cola'] = np.mean(retval['cola'])

    target_texts = [author_to_text[a] for a in input_authors]

    retval['n'] = len(references)

    return retval


def clean_signature(text, regex):
    # not perfect, but should work for many cases
    cleaned = re.sub(regex, '', text)
    # print(text, cleaned)
    return cleaned


def match_author_to_text(*, authors, author_directory, shard, clean=False, regex=None):
    author_to_texts = {}
    for author in authors:
        author_to_texts[author] = [
            x[1]
            for x in get_author_data(
                author_name=author, author_directory=author_directory, shard=shard
            )
        ]
        if clean:
            author_to_texts[author] = [
                clean_signature(t, regex) for t in author_to_texts[author]
            ]

    return author_to_texts


def get_author_embeds(author_to_texts, embed_model='style'):
    author_to_embeds = {}
    print('Embedding author text')
    for a, texts in tqdm(list(author_to_texts.items())):
        # author_to_embeds[a] = [style_model.encode(t) for t in texts]
        author_to_embeds[a] = [style_encode(t, embed_model=embed_model) for t in texts]

    return author_to_embeds


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_path', type=str)
    argparser.add_argument('--clean', action='store_true')
    argparser.add_argument('--embed_model', type=str)
    argparser.add_argument('--author_directory', type=str)

    clean_regex = r"(?<=[\.\,\!\?\-\)\/\`])(\s[a-zA-Z]+([a-zA-Z\s]?)([a-zA-Z\.\<\>\-0-9\\\:\@\s]{0,8})(?<!\.)$|[\w\s]+@[A-Za-z]+(\s\d{2}?)(\/\d{2}?)(\/\d{4}\s?)\d{2}:(\d{2}\s?)(?:AM|PM)$)"
    args = argparser.parse_args()
    out_name = args.input_path + f'.{args.embed_model}_eval'
    if args.clean:
        out_name += '.clean'

    if os.path.exists(out_name):
        print(f"{out_name} exists, skipping.")
        exit(0)
    with open(args.input_path, 'r') as f:
        lines = f.readlines()
        input_data = [json.loads(line.strip()) for line in lines]

    references = [d['original_text'] for d in input_data]

    if isinstance(input_data[0]['decoded'], dict):
        candidates = [d['decoded']['content'] for d in input_data]
        candidates = [re.sub(r'^\{\"text\"\:(\s)?\"|\"\}$', '', c) for c in candidates]
        candidates = [re.sub(r"(^\{\'text\'\:(\s)?'|\'\}$)", '', c) for c in candidates]
        candidates = [re.sub(r"(^\{\'text\'\:(\s)?'|\'\}$)", '', c) for c in candidates]
        candidates = [re.sub(r'(^\"|\"$)', '', c) for c in candidates]
        candidates = [re.sub(r'(^\'|\'$)', '', c) for c in candidates]

    else:
        candidates = [
            d['decoded'][0] if len(d['decoded']) > 0 else '' for d in input_data
        ]
    candidates_para = [d['paraphrase'] for d in input_data]

    if args.clean:
        references = [clean_signature(r, clean_regex) for r in references]
        candidates = [clean_signature(c, clean_regex) for c in candidates]
        candidates_para = [clean_signature(c, clean_regex) for c in candidates_para]

    print(list(zip(references[:10], candidates[:16])))

    input_authors = [d['input_label'] for d in input_data]
    target_authors = [d['target_label'] for d in input_data]

    all_possible_authors = sorted(list(set(input_authors + target_authors)))

    author_to_text = match_author_to_text(
        authors=all_possible_authors,
        author_directory=args.author_directory,
        shard='train',
        clean=args.clean,
        regex=clean_regex,
    )
    author_to_style_embeds = get_author_embeds(
        author_to_text, embed_model=args.embed_model
    )

    # decoded eval
    eval_results = run_eval(
        references=references,
        candidates=candidates,
        input_authors=input_authors,
        target_authors=target_authors,
        author_to_style_embeds=author_to_style_embeds,
        author_to_text=author_to_text,
        acc_model=None,
        acc_label_mapping=None,
        acc_tokenizer=None,
        embed_model=args.embed_model,
    )
    print('Decoded eval results:', eval_results)

    # rename confusion, away, towards with embed prefix
    for k in ['confusion', 'away', 'towards', 'joint', 'joint_fluency']:
        if k in eval_results:
            eval_results[f'_{args.embed_model}_{k}'] = eval_results.pop(k)
    with open(out_name, 'w') as f:
        json.dump(
            {
                'input_path': args.input_path,
                'decoded': eval_results,
                'clean': args.clean,
            },
            f,
        )
