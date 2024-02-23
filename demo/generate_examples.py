"""

Inference code originally based on SSD-LM demo notebook, from [repo](https://github.com/xhan77/ssd-lm) and corresponding paper ((https://arxiv.org/abs/2210.17432)) 

"""

import os
import sys
import torch
import time
import json
from tqdm.auto import tqdm
from datetime import datetime
import random

sys.path.append('../inference')

from classifiers import load_formality_model
from inference_utils import get_setup, batched_controlled_paraphrase


from transformers import PegasusForConditionalGeneration, PegasusTokenizer

if __name__ == '__main__':

    random.seed(1234)
    torch.manual_seed(1234)


    INPUT_PATH = 'inputs.txt'
    OUT_DIR = 'outputs'
    NUM_INFERENCES_PER_INPUT = 4

    TASK = 'informal'

    hparams = {
        'size': 50,
        'lr': 200,
        'total_t': 200,
        'num_drift_steps': 3,
        'use_sqrt_schedule': True,
        'top_p': 0.8,
        'temperature': 3.0,
        'straight_through': False,
        'use_actual': False,
        'model_path': '../models/best_checkpoint/' # requires downloading model
    }

    (
        args,
        model,
        tokenizer,
        model_embedding_lut,
        embedding_sum_layer,
        timestep_layer,
        ctr_embed_projection, # unused
    ) = get_setup(**hparams)


    dtime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    task_folder = f"{OUT_DIR}/{dtime}_{hparams['lr']}"
    os.makedirs(task_folder, exist_ok=False)

    with open(os.path.join(task_folder, "hparams.json"), 'w') as f:
        json.dump(hparams, f)

    with open(os.path.join(task_folder, "args.txt"), 'w') as f:
        json.dump(str(args), f)

    # Load formality guidance model
    ctr_model, tokenizer, ctr_embeds, _ = load_formality_model()
    args.optimizing_label_index = (
        1 if TASK == 'formal' else 0
    )
    args.ctr_model = ctr_model
    args.ctr_embeds = ctr_embeds
    args.tokenizer = tokenizer
    args.ctr_embeds = args.ctr_embeds.to(args.accelerator.device)
    args.ctr_model.to(args.accelerator.device)
    args.ctr_model.eval()

    # Define a loss function to optimize that takes word embeddings and a sequence mask
    args.loss_fn = lambda embeds, mask: -torch.nn.functional.log_softmax(
        args.ctr_model(inputs_embeds=embeds, attention_mask=mask).logits, dim=-1
    )[:, args.optimizing_label_index].mean()


    # load AR paraphraser
    paraphraser_tokenizer = PegasusTokenizer.from_pretrained('tuner007/pegasus_paraphrase')
    paraphraser_model = PegasusForConditionalGeneration.from_pretrained('tuner007/pegasus_paraphrase').to(args.accelerator.device)
    
    with open(INPUT_PATH, 'r') as f:
        input_data = [l.strip() for l in f.readlines()]

    total_transfers = len(input_data)
    with open(os.path.join(task_folder, f"{TASK}.jsonl"), 'w+') as out:
        with tqdm(total=total_transfers) as pbar:
            for original_text in input_data:

                start = time.time()

                # Skip paraphrasing if using actual text
                if args.use_actual:
                    input = original_text
                    paraphrase = ''
                    print(f'Using actual: {original_text}')
                
                # Otherwise, first paraphrase the input
                else:
                    encoded = paraphraser_tokenizer([original_text], return_tensors='pt').to(args.accelerator.device)
                    paraphrase = paraphraser_model.generate(**encoded, max_length=50, do_sample=True, top_p=0.8, temperature=1.5)
                    paraphrase = paraphraser_tokenizer.batch_decode(paraphrase, skip_special_tokens=True)[0]
                    input = paraphrase
                    print(f'Paraphrased: {original_text} -> {input}')

                outputs = batched_controlled_paraphrase([input]*NUM_INFERENCES_PER_INPUT, num_samples=1, args=args, model=model, tokenizer=tokenizer, model_embedding_lut=model_embedding_lut, embedding_sum_layer=embedding_sum_layer, timestep_layer=timestep_layer, ctr_embed_projection=None, batch_ctrl_embeds=None, logging=False)
                result = dict(
                    input_label=INPUT_PATH,
                    paraphrase=paraphrase,
                    original_text=original_text,
                    target_label=TASK,
                    decoded=outputs)
                
                print(f'{original_text} -> {paraphrase} ->' + "\n\t->" + "\n\t->".join(outputs[0]))
                out.write(json.dumps(result) + '\n')
                print('Elapsed:',time.time() - start)
                pbar.update(1)