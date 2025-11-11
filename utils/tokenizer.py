import sys
sys.path.append("/home2/yyzhou/PepCCD/")
import json
import os
import torch.nn as nn
import torch
import datasets
import copy
import numpy as np
from datasets import Dataset as Dataset2
from torch.utils.data import Dataset, DataLoader

class CompoundDataset(Dataset):

    def __init__(self, text_datasets, model_emb=None):
        super().__init__()
        self.text_datasets = text_datasets
        self.length = len(self.text_datasets['train'])
        self.model_emb = model_emb

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with torch.no_grad():

            input_ids = self.text_datasets['train'][idx]['input_ids']
            if isinstance(input_ids, np.ndarray):
                input_ids = torch.tensor(input_ids)


            peptide_sequence = self.model_emb.tokenizer.decode(input_ids, skip_special_tokens=True).replace(" ", "")

            hidden_state = self.model_emb.get_last_hidden_state(peptide_sequence)

            arr = hidden_state.cpu().detach().numpy().astype(np.float32)

            out_kwargs = {}
            out_kwargs['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])
            out_kwargs['self_condition'] = np.array(self.text_datasets['train'][idx]['label'])
            return arr, out_kwargs

def load_data(encoder, seq_len, tag, args):
    pep_tokenizer = encoder.tokenizer
    if tag == 'train':
        path = './dataset/Pre_Diffusion/pre_trained_sequence.jsonl'
    else:
        raise ValueError()
    sentence = {'src': [], 'trg': []}
    with open(path, 'r', encoding='utf-8') as reader:
        for row in reader:
            prot_seq = json.loads(row)['src']
            sentence['src'].append(prot_seq)
            pep_seq = json.loads(row)['trg']
            sentence['trg'].append(pep_seq)

    raw_datasets = Dataset2.from_dict(sentence)

    def tokenize_function(examples):

        input_id_pep = pep_tokenizer(examples['trg'],
                                padding=True,
                                truncation=True,
                                return_tensors="pt",
                                max_length=256)['input_ids']
        input_id_prot = pep_tokenizer(examples['src'],
                                padding=True,
                                truncation=True,
                                return_tensors="pt",
                                max_length=256)['input_ids']
        result_dict = {'input_ids': input_id_pep, 'label': input_id_prot}

        return result_dict

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        remove_columns=['src', 'trg'],
        load_from_cache_file=True,
        desc="Running tokenizer on public_database",
    )

    def pad_function(group_lst):
        max_length = seq_len
        group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], pep_tokenizer.pad_token_id, max_length)
        group_lst['label'] = _collate_batch_helper(group_lst['label'], pep_tokenizer.pad_token_id, max_length)

        return group_lst

    lm_datasets = tokenized_datasets.map(
        pad_function,
        batched=True,
        num_proc=1,
        desc=f"padding",
    )

    raw_datasets = datasets.DatasetDict()
    raw_datasets['train'] = lm_datasets

    dataset = CompoundDataset(
        raw_datasets,
        model_emb=encoder
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    while True:
        yield from dataloader

def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False):
    result = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    mask_ = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result