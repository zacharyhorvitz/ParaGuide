''' convert from roberta tokenizer to t5 '''
import os
from transformers import RobertaTokenizer, T5Tokenizer
from datasets import load_from_disk


def convert_tokenized_dataset(tokenized_dataset, tokenizer1, tokenizer2):
    def convert_tokenized_sample(example):
        result = {}

        split = (
            tokenizer1.decode(example['input_ids'])
            .replace('<pad>', '')
            .replace('</s>', '')
            .split('<s>')
        )
        if len(split) > 3:
            print('WARNING:', split)
        _, i, l = split[:3]
        result['input_text'] = i
        result['label_text'] = l
        encoded_input = tokenizer2.encode(i)
        result['input_ids'] = encoded_input
        encoded_label = tokenizer2.encode(l)
        result['labels'] = encoded_label

        return result

    return tokenized_dataset.map(convert_tokenized_sample)


if __name__ == '__main__':
    for diffusion_roberta_path in [
        # 'REDDIT_PARAPHRASES/max_50_para_first_4mil_auth_style_embed/2023-05-08-05.42.53/max_len_50_min_score_None',
        'paraphrase_data/ENRON/shards/holdin_dataset/2023-06-22-23.15.35/max_len_50_min_score_None',
    ]:
        out_dir = os.path.join(
            os.path.dirname(diffusion_roberta_path),
            't5_max_len_50_roberta_determined_min_score_None',
        )

        print(out_dir)
        os.makedirs(out_dir, exist_ok=False)

        diffusion_roberta_tokenized = load_from_disk(diffusion_roberta_path)
        tokenizer1 = RobertaTokenizer.from_pretrained("roberta-base")
        tokenizer2 = T5Tokenizer.from_pretrained("t5-large")

        converted_dataset = convert_tokenized_dataset(
            diffusion_roberta_tokenized, tokenizer1, tokenizer2
        )
        converted_dataset.save_to_disk(out_dir)
