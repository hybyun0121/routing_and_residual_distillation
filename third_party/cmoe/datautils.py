import numpy as np
import torch

import datasets
import json
import os


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_jsonl_text(path, nsamples, seed, seqlen, model, bsz=8):
    from transformers import AutoTokenizer
    import random

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text")
            if text:
                texts.append(str(text))
    if not texts:
        raise ValueError(f"No non-empty text records found in JSONL calibration file: {path}")

    rng = random.Random(seed)
    order = list(range(len(texts)))
    rng.shuffle(order)

    token_chunks = []
    cursor = 0
    while len(token_chunks) < nsamples:
        pieces = []
        while sum(len(p) for p in pieces) < seqlen:
            text = texts[order[cursor % len(order)]]
            cursor += 1
            ids = tokenizer(text, add_special_tokens=False).input_ids
            if tokenizer.eos_token_id is not None:
                ids = ids + [tokenizer.eos_token_id]
            if ids:
                pieces.append(ids)
        flat = [tok for piece in pieces for tok in piece][:seqlen]
        if len(flat) < seqlen:
            flat.extend([tokenizer.pad_token_id] * (seqlen - len(flat)))
        token_chunks.append(torch.tensor(flat, dtype=torch.long).unsqueeze(0))

    trainloader = []
    for start in range(0, nsamples, bsz):
        batch = torch.cat(token_chunks[start:start + bsz], dim=0)
        tar = batch.clone()
        tar[:, :-1] = -100
        trainloader.append((batch, tar))

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    test_ids = torch.cat(token_chunks, dim=1)
    return trainloader, TokenizerWrapper(test_ids)


def get_wikitext2(nsamples, seed, seqlen, model, bsz = 8):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []


    for _ in range(0, nsamples, bsz):
        batch_i = None
        tar_i = None
        for _ in range(bsz):

            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            if batch_i is None:
                batch_i = inp
                tar_i = tar
            else:
                batch_i = torch.cat((batch_i, inp), 0)
                tar = torch.cat((tar_i, tar), 0)
        trainloader.append((batch_i, tar_i))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, model, bsz = 8):

    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    import random
    random.seed(seed)
    trainloader = []

    for _ in range(0, nsamples, bsz):
        batch_i = None
        tar_i = None
        for _ in range(bsz):

            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] - seqlen - 1<= 0:
                    continue
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            if batch_i is None:
                batch_i = inp
                tar_i = tar
            else:
                batch_i = torch.cat((batch_i, inp), 0)
                tar = torch.cat((tar_i, tar), 0)

        trainloader.append((batch_i, tar_i))

    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] - seqlen - 1<= 0:
                continue
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc

def get_ptb_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )


    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model='', bsz = 8
):
    if os.path.isfile(name):
        return get_jsonl_text(name, nsamples, seed, seqlen, model, bsz=bsz)
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model, bsz = bsz)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, model)
        return get_ptb(nsamples, seed, seqlen, model)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model)
        return get_c4(nsamples, seed, seqlen, model)
    raise ValueError(f"Unknown dataset or calibration JSONL path: {name}")
