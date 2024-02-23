''' examine reddit data dump '''

import json
import sys
import os

from tqdm import tqdm

# import pickle
import random
import pysbd


def load_jsonl_data(path, n=None):
    '''load jsonl data'''
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if n is not None and i >= n:
                break
            yield json.loads(line)


if __name__ == "__main__":
    random.seed(0)
    count = 0
    max_per_author = 10  # 4
    num_authors = 400000  # 100000
    max_chars = 200
    path = sys.argv[1]
    directory = os.path.dirname(path)
    segmenter = pysbd.Segmenter(language="en", clean=False)
    with tqdm(total=num_authors) as pbar:
        with open(
            f'{directory}/{num_authors}authors_{max_per_author}perauth_{max_chars}maxlen.tsv',
            'w+',
        ) as f:
            for x in load_jsonl_data(path, n=num_authors):
                count += 1
                author_count = 0
                author = x['author_id']
                random.shuffle(x['syms'])
                for text in x['syms']:
                    sentences = [s for s in segmenter.segment(text) if s]
                    if len(sentences) == 0:
                        continue

                    rand = random.random()
                    if rand < 0.1 and len(sentences) > 2:
                        chosen_idx = random.randint(0, len(sentences) - 3)
                        chosen_text = (
                            sentences[chosen_idx]
                            + ' '
                            + sentences[chosen_idx + 1]
                            + ' '
                            + sentences[chosen_idx + 2]
                        )
                    elif rand < 0.3 and len(sentences) > 1:
                        chosen_idx = random.randint(0, len(sentences) - 2)
                        chosen_text = (
                            sentences[chosen_idx] + ' ' + sentences[chosen_idx + 1]
                        )
                    else:
                        chosen_idx = random.randint(0, len(sentences) - 1)
                        chosen_text = sentences[chosen_idx].strip()

                    chosen_text = " ".join(chosen_text.split())

                    if len(chosen_text) > max_chars or len(chosen_text) <= 1:
                        continue
                    author_count += 1

                    if author_count > max_per_author:
                        break
                    f.write(f'{author}\t{chosen_text}\n')

                if count % 10000 == 0:
                    print(count)
                pbar.update(1)
    print(count)
