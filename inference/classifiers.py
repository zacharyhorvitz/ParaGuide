from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

import torch
from urllib.request import urlopen
import csv


# def text_to_style(*, model, tokenizer, texts, device, model_type='style'):
#     embeds = []
#     for t in texts:
#         inputs = tokenizer(t, return_tensors='pt')
#         inputs = {k: v.to(device) for k, v in inputs.items()}
#         embeds.append(
#             get_style_embedding(
#                 model=model,
#                 input_tokens=inputs['input_ids'],
#                 attention_mask=inputs['attention_mask'],
#                 model_type=model_type,
#             )
#         )
#     return embeds

def text_to_style(*, model, tokenizer, texts, device, model_type='style'):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    embeds = get_style_embedding(
            model=model,
            input_tokens=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            model_type=model_type,
        )
    
    embeds = [x for x in embeds]
    return embeds

def load_style_model():
    tokenizer = AutoTokenizer.from_pretrained('AnnaWegmann/Style-Embedding')
    model = AutoModel.from_pretrained('AnnaWegmann/Style-Embedding')
    embeds = get_word_embeddings(model)
    return model, tokenizer, embeds


def load_uar_distill_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    model = AutoModelForSequenceClassification.from_pretrained(
        'distilroberta-base', num_labels=128
    )
    model.load_state_dict(torch.load(model_path))
    embeds = get_word_embeddings(model)
    return model, tokenizer, embeds


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )




def get_style_embedding(
    *,
    model,
    inputs_embeds=None,
    input_tokens=None,
    attention_mask=None,
    model_type='style',
):
    assert inputs_embeds is not None or input_tokens is not None
    if inputs_embeds is not None:
        if attention_mask is None:
            attention_mask = torch.ones(*inputs_embeds.shape[:-1]).to(
                inputs_embeds.device
            )  # this may be why I have issues when i insert padding tokens
        attention_mask = attention_mask.to(inputs_embeds.device)
        if model_type == 'style':
            return mean_pooling(
                model(inputs_embeds=inputs_embeds, attention_mask=attention_mask),
                attention_mask=attention_mask,
            )
        elif model_type == 'uar':
            return model.general_encode(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask
            )
        elif model_type == 'luar_distill':
            return model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)[0]
        else:
            raise ValueError(f'Unknown model type {model_type}')

    else:
        if attention_mask is None:
            attention_mask = torch.ones(*input_tokens.shape).to(input_tokens.device)
        attention_mask = attention_mask.to(input_tokens.device)
        if model_type == 'style':
            return mean_pooling(
                model(input_tokens, attention_mask=attention_mask),
                attention_mask=attention_mask,
            )
        elif model_type == 'uar':
            return model.general_encode(
                input_ids=input_tokens, attention_mask=attention_mask
            )
        elif model_type == 'luar_distill':
            return model(input_ids=input_tokens, attention_mask=attention_mask)[0]
        else:
            raise ValueError(f'Unknown model type {model_type}')


def get_word_embeddings(model):
    state_dict = model.state_dict()
    params = []
    for key in state_dict:
        if 'word_embeddings' in key:
            params.append((key, state_dict[key]))
    assert len(params) == 1, f'Found {params}'
    return params[0][1] 


def load_sentiment_model(task='sentiment'):
    # Tasks:
    # emoji, emotion, hate, irony, offensive, sentiment
    # stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    labels = []
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
    with urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    embeddings = get_word_embeddings(model)
    return model, tokenizer, embeddings, labels


def load_formality_model():
    MODEL = f"cointegrated/roberta-base-formality"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    labels = []
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    embeddings = get_word_embeddings(model)
    return model, tokenizer, embeddings, labels


def use_sentiment_model(model, tokenizer, text):
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    logits = output.logits
    return logits

def compute_style_loss(
    embeds, model, target_embeds, attention_mask=None, model_type='style'
):
    
    current = get_style_embedding(
        inputs_embeds=embeds,
        model=model,
        attention_mask=attention_mask,
        model_type=model_type,
    )
    loss = 0
    for target_embed in target_embeds:
        loss += 1 - torch.nn.CosineSimilarity()(current, target_embed)
    return loss / len(target_embeds) 