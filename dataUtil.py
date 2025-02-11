import os
import torch
import logging 
import time 
import random
import numpy as np 
from tqdm import tqdm
from datasets import load_dataset

import gc
def cleanup():
	torch.cuda.empty_cache()
	gc.collect()

def get_dataset(tokenizer, nsamples=128, seqlen=2048, dataset='wiki', seed=0):

    if(dataset == 'wiki') :
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    elif(dataset == 'ptb') :
        testdata = load_dataset("ptb_text_only", "penn_treebank", split="validation")
        testenc = tokenizer("\n\n".join(testdata['sentence']), return_tensors='pt')
    elif(dataset == 'c4') :
        valdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
        random.seed(seed)

        testenc = []
        for _ in range(256):
            while True:
                i = random.randint(0, len(valdata) - 1)
                tmp = tokenizer(valdata[i]["text"], return_tensors="pt")
                if tmp.input_ids.shape[1] >= seqlen:
                    break
            if tmp.input_ids.shape[1] == seqlen:
                # rare case, discovered with Yi tokenizer
                testenc.append(tmp.input_ids[:, 0:seqlen])
            else:
                i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
                j = i + seqlen
                testenc.append(tmp.input_ids[:, i:j])
        testenc = torch.hstack(testenc)

        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids
        testenc = TokenizerWrapper(testenc)
        
    return testenc

def perplexity(model, tokenizer, seqlen=2048, dataset='wiki', framework="pytorch"):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Calculating Perplexity...")
    
    testloader = get_dataset(tokenizer, nsamples=128, seqlen=seqlen, dataset=dataset)
    testenc = testloader.input_ids

    model.seqlen = seqlen
    nsamples = testenc.numel() // model.seqlen

    print("nsamples: " + str(nsamples))
    
    model.eval()
    nlls = []


    for i in tqdm(range(nsamples)):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)]

        # outputs = model(batch)
        if batch.size(1) < seqlen:
            break  # Skip incomplete batch
        with torch.no_grad():
            outputs = model(batch)
        
        logits = outputs.logits[0]
        shift_logits = logits[:-1, :]

       
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][
            :, 1:]
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)).cuda(),
            shift_labels.view(-1).cuda(),
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    return ppl