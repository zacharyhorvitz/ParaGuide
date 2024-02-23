from tqdm import tqdm
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import sys
import os

path = sys.argv[1]
fname = os.path.basename(path)
directory = os.path.dirname(path)
model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)


def get_file_len(fname):
    with open(fname, "r") as f:
        return len(f.readlines())


def get_response(input_text, num_return_sequences, num_beams, top_p=0.9, temp=1.5):
    batch = tokenizer(
        input_text,
        truncation=True,
        padding='longest',
        max_length=60,
        return_tensors="pt",
    ).to(torch_device)
    # translated = model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=temp)
    translated = model.generate(
        **batch,
        max_length=60,
        do_sample=True,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        temperature=temp,
    )
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text


# 'users_data_50_unique_clean_min_10_fixed_sender.tsv' # '400000authors_10perauth_200maxlen.tsv'
temp = 1.5
top_p = 0.8
idx = 0  # int(sys.argv[1])
n = 1  # int(sys.argv[2])
batch_size = 64  # 32 #64 #32 #16


def process_batch(*, authors, texts, handler):
    paraphrases = get_response(texts, 1, 1, temp=temp)
    for author, text, paraphrase in zip(authors, texts, paraphrases):
        # print(f'{author}\t{text}\t{"  ".join(paraphrase.split())}')
        handler.write(f'{author}\t{" ".join(paraphrase.split())}\t{text}\n')


with open(path, 'r') as f_in:
    total = get_file_len(path) / n
    with open(
        f'{directory}/paraphrased_topp{top_p}_tmp{temp}_idx{idx}_' + fname, 'w+'
    ) as f_out:
        with tqdm(total=total) as pbar:
            texts = []
            authors = []
            for i, line in enumerate(f_in.readlines()):
                if i % n == idx:
                    _, text, author, _, _, _, _ = line.strip().split('\t')
                    texts.append(text)
                    authors.append(author)

                    if len(texts) == batch_size:
                        process_batch(authors=authors, texts=texts, handler=f_out)
                        authors = []
                        texts = []
                    pbar.update(1)
            if texts:
                process_batch(authors=authors, texts=texts, handler=f_out)
