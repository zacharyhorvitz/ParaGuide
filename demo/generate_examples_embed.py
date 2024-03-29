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

from classifiers import load_formality_model, load_style_model, text_to_style, compute_style_loss
from inference_utils import get_setup, batched_controlled_paraphrase
from luar import load_uar_hf_model, get_uar_embeddings

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

if __name__ == '__main__':

    random.seed(1234)
    torch.manual_seed(1234)


    INPUT_PATH = 'inputs.txt'
    OUT_DIR = 'outputs'
    NUM_INFERENCES_PER_INPUT = 4

    TASK = 'style' # 'informal'
    CTRL_EMBED = 'luar-target' # usually None unless model is trained with control embedding

    # hparams = {
    #     'size': 50,
    #     'lr': 200, # 800
    #     'total_t': 200,
    #     'num_drift_steps': 3,
    #     'use_sqrt_schedule': True,
    #     'top_p': 0.8,
    #     'temperature': 3.0,
    #     'straight_through': False,
    #     'use_actual': False,
    #     'model_path': '../models/best_checkpoint/' # requires downloading model
    # }
    hparams = {
        'size': 50,
        'lr': 0, # 800
        'total_t': 200,
        'num_drift_steps': 3,
        'use_sqrt_schedule': True,
        'top_p': 0.8,
        'temperature': 3.0,
        'straight_through': False,
        'use_actual': False,
        'ctr_embed_dim': 512,
        'model_path': '/mnt/swordfish-pool2/horvitz/reddit_hrs_work/reddit_luar_cond/ssd_cs_dbs50/best_checkpoint_backup'
        #'/burg/nlp/users/zfh2000/enron_model/ssd_cs_dbs50/best_checkpoint', #'../models/best_checkpoint/' # requires downloading model
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


    if TASK in ['formal', 'informal']:
    
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
        )[:, args.optimizing_label_index].sum()



    elif TASK in ['style']:

        args.optimizing_label_index = None
        ctr_model, tokenizer, ctr_embeds = load_style_model()
        args.ctr_model = ctr_model
        args.tokenizer = tokenizer
        args.ctr_embeds = ctr_embeds
        args.ctr_embeds = args.ctr_embeds.to(args.accelerator.device)
        args.ctr_model.to(args.accelerator.device)
        args.ctr_model.eval()

        target_style_examples = ['I went to the #Apple store and bought an iPhone!'] #['Mmmm...I think I agree.', 'Uh...I think you are doing that wrong.', 'Ok...sounds good.']

        args.target_embeds = text_to_style(
                                model=args.ctr_model,
                                tokenizer=args.tokenizer,
                                texts=target_style_examples,
                                device=args.accelerator.device,
                                model_type='style',
                            )


        def style_loss(embeds, mask):
            # To do: move set up batching inside of loss function
            loss = 0
            for e, m in zip(embeds, mask):
                loss += compute_style_loss(
                    e.unsqueeze(0),
                    model=args.ctr_model,
                    target_embeds=args.target_embeds,
                    attention_mask=m.float().unsqueeze(0),
                    model_type='style',
                )
           
            return loss

        args.loss_fn = style_loss
    
   
    else:
        raise ValueError(f"Unknown task: {TASK}")
    

    if CTRL_EMBED in ['style-target']:
        raise NotImplementedError
    
        ## load wegman style model
        # ctr_model, tokenizer, _ = load_style_model()


    elif CTRL_EMBED in ['luar-target']:

        # load uar model
        embed_model, embed_tokenizer = load_uar_hf_model()

        args.embed_model = embed_model
        args.embed_tokenizer = embed_tokenizer
        args.embed_model.to(args.accelerator.device)
        args.embed_model.eval()

        assert len(target_style_examples) > 0

        args.embed_model_target_embeds = []

        for example in target_style_examples:
            luar_embedding = get_uar_embeddings(model=args.embed_model, tokenizer=args.embed_tokenizer, texts=[example], device='cuda')[0]
            args.embed_model_target_embeds.append(luar_embedding) #.detach().cpu().numpy())
        
        # import pdb; pdb.set_trace()




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

                if CTRL_EMBED is not None:
                    # take the first one
                    # batch_ctrl_embeds = args.embed_model_target_embeds[0].unsqueeze(0).repeat(NUM_INFERENCES_PER_INPUT, 1, 1)

                    # mean pool
                    batch_ctrl_embeds = torch.stack(args.embed_model_target_embeds).mean(dim=0).unsqueeze(0).repeat(NUM_INFERENCES_PER_INPUT, 1, 1)

                else:
                    batch_ctrl_embeds = None
                    ctr_embed_projection = None

                outputs = batched_controlled_paraphrase([input]*NUM_INFERENCES_PER_INPUT, num_samples=1, args=args, model=model, tokenizer=tokenizer, model_embedding_lut=model_embedding_lut, embedding_sum_layer=embedding_sum_layer, timestep_layer=timestep_layer, ctr_embed_projection=ctr_embed_projection, batch_ctrl_embeds=batch_ctrl_embeds, logging=False)
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
