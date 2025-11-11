import sys
sys.path.append("/home2/yyzhou/PepCCD")
import argparse
import os
import torch
import torch as th
import numpy as np
import time
import torch.nn.functional as F
from transformers import AutoTokenizer
from train.align_train import ESMModel
import torch.nn as nn
from utils.script_utils import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)



class PretainedTokenizer():
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("./checkpoints/ESM")
        self.vocab = self.tokenizer.get_vocab()
        self.rev_vocab = {idx: word for word, idx in self.vocab.items()}

    def batch_encode(self, seqs):
        code = self.tokenizer(
            seqs,
            padding="max_length",   
            truncation=True,
            return_tensors="pt",     
            max_length=256,
        )['input_ids']
        return code

    def batch_decode(self, indices):

        decoded_sequence = [self.tokenizer.decode(idx, skip_special_tokens=True) for idx in indices]

        return ''.join(decoded_sequence)
    
    def get_valid_length(self, indices):

        special_tokens = {self.tokenizer.cls_token_id, 
                          self.tokenizer.pad_token_id, 
                          self.tokenizer.eos_token_id,
                          self.tokenizer.mask_token_id}
        
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        
        valid_count = 0
        for token_id in indices:
            if token_id in special_tokens:
                continue
            token = self.rev_vocab.get(token_id)
            if token in "ACDEFGHIKLMNPQRSTVWY":  
                valid_count += 1
                
        return valid_count


def main():
    start_time = time.perf_counter()
    args = create_argparser().parse_args()
    device = ('cuda:4' if th.cuda.is_available() else 'cpu')

    print("creating model and diffusion...")

    myTokenizer = PretainedTokenizer()
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        th.load(args.model_path, map_location="cpu",weights_only=True)
    )
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    model_emb = th.nn.Embedding(
        num_embeddings=args.vocab_size,
        embedding_dim=args.n_embd,
        _weight=model.get_input_embeddings().weight.clone().cpu()
    ).eval().requires_grad_(False)

    model_path = "./checkpoints/ESM"
    prot_encoder = ESMModel(model_path=model_path, device=device)

    prot_encoder.load_state_dict(torch.load(args.prot_encoder_path,weights_only=True))
    prot_tokenizer = PretainedTokenizer()


    def model_fn(inputs_embeds, timesteps, reference=None):
        assert reference is not None

        return model(inputs_embeds=inputs_embeds, timesteps=timesteps, self_condition=reference[0])


    print("sampling...")

    for length in args.seq_len:
        all_peptides = []

        prot = ['GHMNVKRRTHNVLERQRRNELKRSFFALRDQIPELENNEKAPKVVILKKATAYILSVQAEEQKLISEEDLLRKRREQLKHKLEQLGGC']

        test_tokens = prot_tokenizer.batch_encode(prot)
        test_tokens = test_tokens.repeat(args.batch_size, 1).to(device)

        while len(all_peptides) < args.num_samples:
            model_kwargs = {}

            prot_z = prot_encoder(prot).to(device)

            prot_z = prot_z.repeat(args.batch_size, 1).to(device)
            ref_prot = prot_z / prot_z.norm(dim=-1, keepdim=True)

            condition = ref_prot
            condition = condition.unsqueeze(1).repeat(1, 50, 1).to(device)

            model_kwargs["reference"] = [condition, test_tokens]
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            samples = sample_fn(
                model_fn,
                noise=None,
                shape=(args.batch_size, length, args.n_embd),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                top_p=1,
                clamp_step=0,
                clamp_first=False,
                x_start=None,
                device=device,
                cond_fn=None,

            )
            sample = samples[-1]
            sample = model.get_logits(sample)
            _, indices = torch.topk(sample, k=1, dim=-1)
            indices = indices.squeeze(-1)
            valid_indices = []

            for s in indices:
                if args.min_length <= myTokenizer.get_valid_length(s) <= args.max_length:
                    valid_indices.append(s)

            all_peptides.extend([s.detach().cpu().numpy() for s in valid_indices])
            print(f"created {len(valid_indices)} valid samples")
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
        output_file = os.path.join(args.sample_path, 'samples.txt')  
        with open(output_file, 'w') as f:                          
            for i, seq in enumerate(all_peptides):
                decoded_sequence = myTokenizer.batch_decode(seq)

                f.write(f"{decoded_sequence}\n")                   

                print(decoded_sequence)

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10,
        batch_size=10,
        max_loop=50,
        use_ddim=False,
        sample_path="./samples",
        prior_path="./checkpoints/Fine_Diffusion/diffusion_model.pt",
        model_path="./checkpoints/Fine_Diffusion/diffusion_model.pt",
        pep_encoder_path="./checkpoints/Align/best_pep.pth",
        prot_encoder_path="./checkpoints/Align/best_prot.pth",
        min_length=0,    
        max_length=50,   
        classifier_scale=1.0,
        embedding_scale=0.0,
        use_fp16=False,
        classifier_use_fp16=False,
        seq_len=[50],
        vocab_size=30,
        class_cond=True,
        cls=2,

    )
    defaults.update(model_and_diffusion_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == '__main__':
    main()