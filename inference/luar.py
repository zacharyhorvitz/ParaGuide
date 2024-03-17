from transformers import AutoModel, AutoTokenizer


def load_uar_hf_model():
    tokenizer = AutoTokenizer.from_pretrained("rrivera1849/LUAR-CRUD")
    model = AutoModel.from_pretrained("rrivera1849/LUAR-CRUD", trust_remote_code=True)
    return model, tokenizer

def get_uar_embeddings(model, tokenizer, texts, device):
    tokenized_text = tokenizer(
    texts, 
    max_length=50,
    padding="max_length", 
    truncation=True,
    return_tensors="pt",
    )
    tokenized_text = {k: v.to(device) for k, v in tokenized_text.items()}
    tokenized_text["input_ids"] = tokenized_text["input_ids"].reshape(len(texts), 1, -1)
    tokenized_text["attention_mask"] = tokenized_text["attention_mask"].reshape(len(texts), 1, -1)
    out = model(**tokenized_text)
    return out