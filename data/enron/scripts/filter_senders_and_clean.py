import pandas as pd
from collections import defaultdict
import re
import sys
import os

# hugging face roberta tokenizer
from transformers import RobertaTokenizer
from tqdm import tqdm

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
MAX_LEN = 50  # 100
users = defaultdict(list)


def clean_text(x):
    lines = x.split('\n')

    xfile_idx = [i for i, l in enumerate(lines) if 'X-FileName:' in l]
    assert len(xfile_idx) == 1
    xfile_idx = xfile_idx[0]

    meta_data = lines[: xfile_idx + 1]
    x = ' '.join(lines[xfile_idx + 1 :])
    x = ' '.join(x.split())

    # # replace email addresses with '[email]'
    # x = re.sub(r'[\w\.-]+@[\w\.-]+', '[email]', x)

    # # replace urls with '[url]'
    # x = re.sub(r'http\S+', '[url]', x)

    # # replace xls files with '[xls]'
    # x = re.sub(r'\w+\.xls', '[xls]', x)

    # # replace time with '[time]' incling AM/PM
    # x = re.sub(r'\d+:\d+(:\d+)?\s+(AM|PM)?', '[time]', x, flags=re.IGNORECASE)

    # # replace dates with '[date]' with different formats
    # x = re.sub(r'\d+/\d+(/\d+)?', '[date]', x)

    # # replace phone numbers (dash or space or paren) with '[number]'
    # x = re.sub(r'\d+[-\s\(\)\+]+\d+[-\s\(\)\+]+\d+', '[number]', x)

    thread_token = r'([-]*Original Message[-]*|[-]* Forwarded by| To:| From:)'

    x, *thread = re.split(thread_token, x)

    # if first token contains thread token, then it is all a thread
    if re.search(thread_token, x):
        thread = [x] + thread
        x = ''

    sender = [l for l in meta_data if l.startswith('From:')]
    assert len(sender) == 1
    sender = sender[0]

    cc = [l for l in meta_data if l.startswith('X-cc:')]
    assert len(cc) == 1
    cc = cc[0]

    bcc = [l for l in meta_data if l.startswith('X-bcc:')]
    assert len(bcc) == 1
    bcc = bcc[0]

    to = [l for l in meta_data if l.startswith('X-To:')]
    assert len(to) == 1
    to = to[0]

    date = [l for l in meta_data if l.startswith('Date:')]
    assert len(date) >= 1
    date = date[0]

    # import pdb; pdb.set_trace()

    return {
        'meta_data': meta_data,
        'text': x,
        'thread': thread,
        'sender': sender,
        'cc': cc,
        'to': to,
        'bcc': bcc,
        'date': date,
    }


def length(x):
    return len(tokenizer.encode(x))


fname = sys.argv[1]
directory = os.path.dirname(fname)

d = pd.read_csv(fname, chunksize=10000)
for chunk in d:
    user_list = [f.split('/')[0] for f in chunk['file'].values]
    for u, m in zip(user_list, chunk['message']):
        users[u].append(clean_text(m))

with open(f'{directory}/users_data_{MAX_LEN}.tsv', 'w+') as f:
    for u in tqdm(users):
        for m in users[u]:
            if not m['text']:
                continue
            text = m['text']
            if length(text) > MAX_LEN:
                continue
            f.write(
                '\t'.join([u, text, m['sender'], m['to'], m['cc'], m['bcc'], m['date']])
                + '\n'
            )


def clean_name(x):
    # remove things between <>
    x = re.sub(r'<.*?>', '', x).strip()
    return x


seen = set()
total = 0
with open(f'{directory}/users_data_{MAX_LEN}.tsv', 'r') as f:
    with open(f'{directory}/users_data_{MAX_LEN}_unique_clean.tsv', 'w+') as g:
        for l in f:
            data = l.split('\t')
            if len(data) != 7:
                print('SKIPPING', data)
                continue
            total += 1
            u, text, sender, to, cc, bcc, date = data
            clean_sender = clean_name(sender)
            clean_to = clean_name(to)
            clean_cc = clean_name(cc)
            clean_bcc = clean_name(bcc)

            to_write = [u, text, clean_sender, clean_to, clean_cc, clean_bcc, date]
            to_write = [x.strip() for x in to_write]

            key = (text, clean_sender, clean_to, date)
            if key in seen:
                print('DUPLICATE', key)
                continue
            else:
                seen.add(key)
                g.write('\t'.join(to_write) + '\n')

print(len(seen))
print(total)


senders = defaultdict(list)
MIN_COUNT = 10

with open(f'{directory}/users_data_{MAX_LEN}_unique_clean.tsv', 'r') as f:
    for l in f:
        u, text, sender, to, cc, bcc, date = l.split('\t')
        senders[sender].append(text)
        print(sender, to)


number = [(len(v), k) for k, v in senders.items()]
number = sorted(number, reverse=True)
number = [(n, k) for n, k in number if n >= MIN_COUNT]
top_senders = set([k for _, k in number])

with open(
    f'{directory}/users_data_{MAX_LEN}_unique_clean_min_{MIN_COUNT}_fixed_sender.tsv',
    'w+',
) as f:
    for l in open(f'{directory}/users_data_{MAX_LEN}_unique_clean.tsv', 'r'):
        u, text, sender, to, cc, bcc, date = l.split('\t')
        if sender in top_senders:
            f.write(l)

for n, k in number:
    print(n, k)
print('People:', len(number))
