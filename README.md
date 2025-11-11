# MMCD Multi-Model Contrastive Diffusion

####  Codes, datasets and appendix for AAAI-2026 paper "PepCCD: A Contrastive Conditioned Diffusion Framework for Target-Specific Peptide Generation"

## Overview of MMCD 

![model](https://github.com/ZhouYyang/PepCCD/blob/main/model.png)

## Getting Started
```
conda env create -f environment.yml
conda activate PepCCD
```
1. ESM-2 Protein Language Model
Visit https://huggingface.co/facebook/esm2_t30_150M_UR50D and download the entire repository folder to: path/to/PepCCD/checkpoints/ESM/

2. PepCCD Checkpoints
Vist https://huggingface.co/ZhouYyang/PepCCD/tree/main
```
mkdir -p /path/to/PepCCD/checkpoints/Align
```
Download best_pep.pthand best_prot.pth to /path/to/PepCCD/checkpoints/Align
```
mkdir -p /path/to/PepCCD/checkpoints/Fine_Diffusion
```
Download diffusion_model.pt to /path/to/PepCCD/checkpoints/Fine_Diffusion

3. Pre-training Dataset
Vist https://huggingface.co/ZhouYyang/PepCCD/tree/main
```
mkdir -p /path/to/PepCCD/dataset/Pre_Diffusion
```
Download pre_trained_sequence.json to /path/to/PepCCD/dataset/Pre_Diffusion

### Trian
Before running the code, configure your Python path by adding the following lines at the beginning of your script:
```
import sys
sys.path.append("/path/to/PepCCD")
```
Additionally, when executing scripts or modules, always use their absolute paths​ to avoid potential path-related issues during runtime.
#### stage 1

```
python /path/to/PepCCD/train/align_train.py
```

#### stage 2

- Adjusting datasets in __utils.tokenizer.py__ , forward_backward in __utils.train_utils.py__ and loss function in __gaussian_diffusion.py__ for adapter.  
```
def load_data(encoder, seq_len, tag, args):
    pep_tokenizer = encoder.tokenizer
    if tag == 'train':
        path = './dataset/Pre_Diffusion/pre_trained_sequence.jsonl'
    ... 
```
```
def forward_backward():  
    ...  
    if k == 'input_ids':  
        micro_cond[k] = v[i: i + self.microbatch].to(self.device)
    ...  
```
```
def training_losses_seq2seq(self, model, x_start, t, model_kwargs=None, noise=None):  
    ...  
    terms["loss"] = terms["mse"]  
    ...  
```
```
python /path/to/PepCCD/train/diffusion_train.py
```
#### stage 3

- Collecting Checkpoints of pre-trained DDPM.  

- Adjusting datasets in __utils.tokenizer.py__ , forward_backward in __utils.train_utils.py__ and loss function in __gaussian_diffusion.py__ for protein-guided fine-tuning DDPM. 
```
def load_data(encoder, seq_len, tag, args):
    pep_tokenizer = encoder.tokenizer
    if tag == 'train':
        path = './dataset/Fine_Diffusion/fine_sequence.jsonl'
    ... 
```
```
def forward_backward():  
    ...  
    if k == 'input_ids':
      micro_cond[k] = v[i: i + self.microbatch].to(self.device)      
    else:
      with torch.no_grad():
            protein_sequences = v[i: i + self.microbatch]
            decoded_sequences = [
                self.prot_encoder.tokenizer.decode(seq, skip_special_tokens=True).replace(" ", "")
                for seq in protein_sequences
            ]
            prot_features = self.prot_encoder(decoded_sequences) 
            prot_features_norm = prot_features / prot_features.norm(dim=-1,keepdim=True) 
            prot_features_norm = prot_features_norm.unsqueeze(1).repeat(1, 256, 1)  
            micro_cond['self_condition'] = prot_features_norm
  ```
  ```
  def training_losses_seq2seq(self, model, x_start, t, model_kwargs=None, noise=None):  
     terms["loss"] = terms["mse"] + decoder_nll  
  
  ```
  ```
  python /path/to/PepCCD/train/diffusion_train.py
  ```
### Sample

Adjusting target_protein in __sampling.py__

target_protein = ['your target_protein sequence']

```
python /path/to/PepCCD/samples/sampling.py
```