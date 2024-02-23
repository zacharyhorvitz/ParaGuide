import os
import argparse
from datetime import datetime

import faiss
import re
import nltk
import spacy
import random
import string

# import torch
import faiss
import lemminflect
import numpy as np

from eval_2 import match_author_to_text, get_author_data
from tqdm import tqdm
import json

# import time

# from spacy_langdetect import LanguageDetector
# from collections import defaultdict
import spacy_wordnet
from pycontractions import Contractions
from spacy_wordnet.wordnet_annotator import WordnetAnnotator
from spacy_wordnet.__utils__ import spacy2wordnet_pos
from sklearn.feature_extraction.text import TfidfVectorizer

# Load contractions, spacy, sentence transformers model
nltk.download('wordnet')
nltk.download('omw-1.4')

cont = Contractions(api_key="glove-twitter-100")
cont.load_models()
nlp = spacy.load("en_core_web_trf", disable=['ner', 'parser'])
nlp.add_pipe('sentencizer')
# nlp.add_pipe("spacy_wordnet", after='tagger')
nlp.add_pipe("spacy_wordnet", after='tagger')  # config={'lang': nlp.lang})


def casei_replace(input_str, search, replace):
    def replacement_func(match, repl_pattern):
        match_str = match.group(0)
        repl = ''.join(
            [
                r_char if m_char.islower() else r_char.upper()
                for r_char, m_char in zip(repl_pattern, match_str)
            ]
        )
        repl += repl_pattern[len(match_str) :]
        return repl

    try:
        return re.sub(
            search, lambda m: replacement_func(m, replace), input_str, flags=re.I
        )
    except re.error:
        return input_str.replace(search, replace)


def get_cap_style(texts):
    upper_count = 0
    lower_count = 0
    sent_count = 0
    none_count = 0
    for text in texts:
        token_count = 0
        upper_token_count = 0
        lower_token_count = 0
        lower_sent_count = 0
        sents = list(nlp(text).sents)
        for sent in sents:
            for token in sent:
                token_idx = token.i - sent.start
                token_count += 1
                if token.text.upper() == token.text:
                    upper_token_count += 1
                if token.text.lower() == token.text:
                    lower_token_count += 1
                if token_idx == 0 and token.text[0].islower():
                    lower_sent_count += 1
        if lower_sent_count < len(sents):
            sent_count += 1
        elif lower_sent_count == len(sents) and (lower_token_count / token_count) > 0.5:
            lower_count += 1
        elif (upper_token_count / token_count) > 0.5:
            upper_count += 1
        else:
            none_count += 1
    return ['upper', 'lower', 'sent', None], [
        upper_count,
        lower_count,
        sent_count,
        none_count,
    ]


def to_sent_case(text):
    sents = nlp(text).sents
    tokens = []
    for sent in sents:
        for token in sent:
            token_idx = token.i - sent.start
            if token_idx == 0:
                tokens.append(token.text[0].upper() + token.text[1:].lower())
            else:
                tokens.append(token.text.lower())
            if token.whitespace_:
                tokens.append(token.whitespace_)
    return ''.join(tokens)


def set_cap_style(texts, style):
    output = []
    rnd = random.Random(hash(tuple(texts)))
    styles = rnd.choices(style[0], weights=style[1], k=len(texts))
    for text, style in zip(texts, styles):
        if style == 'upper':
            output.append(text.upper())
        elif style == 'lower':
            output.append(text.lower())
        elif style == 'sent':
            output.append(to_sent_case(text))
        else:
            output.append(text)
    return output


def get_cont_style(texts):
    contract_texts = cont.contract_texts(texts)
    expand_texts = cont.expand_texts(texts, precise=True)
    contract_changed = sum([t != x for t, x in zip(texts, contract_texts)])
    expand_changed = sum([t != x for t, x in zip(texts, expand_texts)])
    return ['expand', 'contract'], [contract_changed, expand_changed]


def set_cont_style(texts, style):
    output = []
    rnd = random.Random(hash(tuple(texts)))
    if sum(style[1]) == 0:
        styles = rnd.choices(style[0], k=len(texts))
    else:
        styles = rnd.choices(style[0], weights=style[1], k=len(texts))
    for text, style in zip(texts, styles):
        if style == 'expand':
            output.append(list(cont.expand_texts([text], precise=True))[0])
        elif style == 'contract':
            output.append(list(cont.contract_texts([text]))[0])
    return output


def get_best_synset(synsets, token):
    matching_synsets = [
        synset for synset in synsets if synset.pos() == spacy2wordnet_pos(token.pos)
    ]
    if len(matching_synsets) > 0:
        return matching_synsets[0]
    else:
        return synsets[0]


def get_tfidf_top_features(documents, n_top=10):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf = tfidf_vectorizer.fit_transform(documents)
    importance = np.argsort(np.asarray(tfidf.sum(axis=0)).ravel())[::-1]
    tfidf_feature_names = np.array(tfidf_vectorizer.get_feature_names())
    return (
        tfidf_feature_names[importance[:n_top]].tolist(),
        tfidf_feature_names[importance[-n_top:]].tolist(),
    )


# Baseline tactics
def capitalize(source_texts, paragraph_texts, target_texts):
    target_style = get_cap_style(target_texts)
    return set_cap_style(source_texts, target_style)


def contraction(source_texts, paragraph_texts, target_texts):
    target_style = get_cont_style(target_texts)
    return set_cont_style(source_texts, target_style)


def synonym(source_texts, paragraph_texts, target_texts):
    vocab = {}
    for text in target_texts:
        sents = nlp(text).sents
        for sent in sents:
            for token in sent:
                synsets = token._.wordnet.synsets()
                if len(synsets) > 0:
                    vocab[get_best_synset(synsets, token).name()] = token
    output = []
    for text in source_texts:
        sents = nlp(text).sents
        tokens = []
        for sent in sents:
            for token in sent:
                synsets = token._.wordnet.synsets()
                if len(synsets) > 0:
                    synset_name = get_best_synset(synsets, token).name()
                    if synset_name in vocab:
                        replacement = vocab[synset_name]._.inflect(
                            token.tag_, inflect_oov=True, on_empty_ret_word=True
                        )
                        tokens.append(
                            casei_replace(token.text, token.text, replacement)
                        )
                    else:
                        tokens.append(token.text)
                else:
                    tokens.append(token.text)
                if token.whitespace_:
                    tokens.append(token.whitespace_)
        output.append(''.join(tokens))
    return output


def punctuation(source_texts, paragraph_texts, target_texts):
    rnd = random.Random(hash(tuple(source_texts)))
    punct_endings = []
    puncts = []
    for text in target_texts:
        sents = nlp(text).sents
        endings = []
        punct_whitespace = []
        punct_tokens = []
        for sent in sents:
            for token in sent:
                is_punct = all([c in string.punctuation for c in token.text])
                if is_punct:
                    if len(punct_tokens) == 0 or punct_whitespace[-1]:
                        punct_whitespace.append(token.whitespace_)
                        punct_tokens.append(token.text)
                    else:
                        punct_tokens[-1] += token.text
                        punct_whitespace[-1] = token.whitespace_
                    if token.i == sent.end - 1:
                        last_punct_token = punct_tokens.pop()
                        punct_whitespace.pop()
                        endings.append(last_punct_token)
        puncts.append(" ".join(punct_tokens))
        punct_endings.append(endings)

    output = []
    for text, endings in zip(source_texts, punct_endings):
        sents = nlp(text).sents
        tokens = []
        for sent in sents:
            for token in sent:
                is_punct = all([c in string.punctuation for c in token.text])
                if is_punct and token.i == sent.end - 1 and len(endings) > 0:
                    rnd.shuffle(endings)
                    tokens.append(endings[0])
                else:
                    tokens.append(token.text)
                if token.whitespace_:
                    tokens.append(token.whitespace_)
        output.append(''.join(tokens))
    return output


def emoji(source_texts, paragraph_texts, target_texts):
    emojis = []
    for text in target_texts:
        sents = nlp(text).sents
        emoji_whitespace = []
        emoji_tokens = []
        for sent in sents:
            for token in sent:
                is_emoji = all([not (c.isalpha()) for c in token.text])
                is_single_punct = (
                    len(token.text.strip()) == 1 and token.text in string.punctuation
                )
                if is_emoji and not is_single_punct:
                    if len(emoji_tokens) == 0 or emoji_whitespace[-1]:
                        emoji_whitespace.append(token.whitespace_)
                        emoji_tokens.append(token.text)
                    else:
                        emoji_tokens[-1] += token.text
                        emoji_whitespace[-1] = token.whitespace_
                    if (token.i == sent.end - 1 or not (token.whitespace_)) and all(
                        [c in string.punctuation for c in emoji_tokens[-1]]
                    ):
                        emoji_tokens.pop()
                        emoji_whitespace.pop()
        emojis.append(" ".join(emoji_tokens))

    return [(s + " " + e if len(e) > 0 else s) for s, e in zip(source_texts, emojis)]


def bert(source_texts, paragraph_texts, target_texts):
    target_tokens = []
    xb = []
    xq = []
    embs = faiss.IndexFlatIP(768)

    for text in target_texts:
        doc = nlp(text)
        for idx, token in enumerate(doc):
            tensor = doc._.trf_data.tensors[0][0]
            alignments = doc._.trf_data.align[idx].data.flatten()
            try:
                xb.append(tensor[alignments].mean(0))
            except IndexError:
                xq.append(
                    np.ones(
                        (
                            1,
                            768,
                        ),
                        dtype=np.float32,
                    )
                )
            target_tokens.append(token)
    xb = np.vstack(xb)
    xb_norm = np.linalg.norm(xb, axis=1)
    xb /= np.expand_dims(xb_norm, axis=1)
    embs.add(xb)

    for text in source_texts:
        doc = nlp(text)
        for idx, token in enumerate(doc):
            tensor = doc._.trf_data.tensors[0][0]
            alignments = doc._.trf_data.align[idx].data.flatten()
            try:
                xq.append(tensor[alignments].mean(0))
            except IndexError:
                xq.append(
                    np.ones(
                        (
                            1,
                            768,
                        ),
                        dtype=np.float32,
                    )
                )
    xq = np.vstack(xq)
    xq_norm = np.linalg.norm(xq, axis=1)
    xq /= np.expand_dims(xq_norm, axis=1)
    D, I = embs.search(xq, 10)

    output = []
    for text in source_texts:
        tokens = []
        doc = nlp(text)
        for token, idxs, sims in zip(doc, I, D):
            is_punct = all([c in string.punctuation for c in token.text])
            sim_tokens = [
                target_tokens[i]
                for i, s in zip(idxs, sims)
                if s > 0.6
                and token.pos == target_tokens[i].pos
                and token.pos_ not in ["AUX", "ADP", "PART"]
            ]
            if len(sim_tokens) > 0:
                replacement = sim_tokens[0]._.inflect(
                    token.tag_, inflect_oov=True, on_empty_ret_word=True
                )
                tokens.append(casei_replace(token.text, token.text, replacement))
            else:
                tokens.append(token.text)
            if token.whitespace_:
                tokens.append(token.whitespace_)
        output.append(''.join(tokens))
    return output


def tfidf(source_texts, paragraph_texts, target_texts):
    rnd = random.Random(hash(tuple(source_texts)))
    most, least = get_tfidf_top_features(target_texts)

    output = []
    for text in source_texts:
        rnd.shuffle(most)
        rnd.shuffle(least)
        output.append(text + " " + " ".join(most[:3]))

    return output


def inv_tfidf(source_texts, paragraph_texts, target_texts):
    rnd = random.Random(hash(tuple(source_texts)))
    most, least = get_tfidf_top_features(target_texts)

    output = []
    for text in source_texts:
        rnd.shuffle(most)
        rnd.shuffle(least)
        output.append(text + " " + " ".join(least[:3]))

    return output


def comb(source_texts, paragraph_texts, target_texts):
    output = capitalize(source_texts, paragraph_texts, target_texts)
    output = contraction(output, paragraph_texts, target_texts)
    output = synonym(output, paragraph_texts, target_texts)
    output = punctuation(output, paragraph_texts, target_texts)
    output = emoji(output, paragraph_texts, target_texts)
    return output


def do_baseline_transfer(*, approach, paraphrase, original_text, target_texts):
    if approach == 'ling':
        outputs = comb([original_text], [], target_texts)
    elif approach == 'bert':
        outputs = bert([original_text], [], target_texts)
    elif approach == 'para':
        outputs = [paraphrase]
    elif approach == 'inv_tfidf':
        outputs = inv_tfidf([original_text], [], target_texts)
    else:
        raise NotImplementedError

    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--author_directory', type=str)
    parser.add_argument('--assignments_json', type=str)
    parser.add_argument('--approach', type=str)

    cmd_args = parser.parse_args()
    hparams = vars(cmd_args)
    out_dir = hparams['out_dir']
    approach = hparams['approach']

    assert approach in ['ling', 'bert', 'para', 'inv_tfidf']

    dtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    task_folder = f"{out_dir}/{dtime}"
    os.makedirs(task_folder, exist_ok=False)

    with open(os.path.join(task_folder, "hparams.json"), 'w') as f:
        json.dump(hparams, f)

    with open(hparams['assignments_json'], 'r') as f:
        assignments = json.load(f)

    with open(os.path.join(task_folder, f"style.jsonl"), 'w+') as out:
        for source_author in tqdm(sorted(assignments.keys())):
            # val_examples = assignments[source_author]['val_samples']
            val_examples = assignments[source_author]['test_samples']
            target_authors = assignments[source_author]['target']
            for target_author in target_authors:
                target_author_training = [
                    x[1]
                    for x in get_author_data(
                        author_name=target_author,
                        author_directory=hparams['author_directory'],
                        shard='train',
                    )
                ]
                for paraphrase, original_text in val_examples:
                    # import pdb; pdb.set_trace()
                    result = do_baseline_transfer(
                        approach=approach,
                        paraphrase=paraphrase,
                        original_text=original_text,
                        target_texts=target_author_training,
                    )
                    result = dict(
                        input_label=source_author,
                        paraphrase=paraphrase,
                        original_text=original_text,
                        target_label=target_author,
                        decoded=result,
                    )

                    print(f'{original_text} -> {paraphrase} -> {result}')
                    out.write(json.dumps(result) + '\n')
