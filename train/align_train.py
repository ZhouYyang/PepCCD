import sys
sys.path.append("/home2/yyzhou/PepCCD")
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

class ESMModel(nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        self.device = device
        self.model = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.hidden_size = self.model.config.hidden_size
        self.embedding_layer = self.model.embeddings.word_embeddings
        self.to(device)

    def get_last_hidden_state(self, inputs):
        inputs = self.tokenizer(
            inputs, 
            padding="max_length", 
            truncation=True, 
            max_length=256,
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        last_hidden_state = outputs.hidden_states[-1]
        return last_hidden_state

    def forward(self, inputs):
        inputs = self.tokenizer(
            inputs, 
            padding="max_length", 
            truncation=True, 
            max_length=256,
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        last_hidden_state = outputs.hidden_states[-1]
        pooled_output = last_hidden_state.mean(dim=1)
        features = F.normalize(pooled_output, p=2, dim=-1)  
        return features

class ProteinPeptideDataset(Dataset):
    def __init__(self, prots, peptides):
        self.prots = prots
        self.peptides = peptides

    def __len__(self):
        return len(self.prots)

    def __getitem__(self, idx):
        return self.prots[idx], self.peptides[idx]

class OrthoInfoNCE(nn.Module):
    def __init__(self, temp=0.07, ortho_lambda=0.01):
        super().__init__()
        self.temp = temp
        self.ortho_lambda = ortho_lambda
        
    def forward(self, prot, pep):
        def _ortho_loss(x):
            corr = torch.mm(x.T, x)
            return torch.norm(corr - torch.eye(x.size(1)).to(x.device))
            
        logits = torch.mm(prot, pep.T) / self.temp
        labels = torch.arange(prot.size(0)).to(prot.device)
        loss = F.cross_entropy(logits, labels)
        ortho_loss = _ortho_loss(prot) + _ortho_loss(pep)
        return loss + self.ortho_lambda * ortho_loss

def get_prot_peptides(path):
    df = pd.read_csv(path)
    return df['prot_seq'].tolist(), df['pep_seq'].tolist()

def train(prot_encoder, pep_encoder, data_loader, optimizer, loss_fn, epoch, temp=0.07):
    prot_encoder.train()
    pep_encoder.train()
    total_loss = 0
    
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}")
    for batch_idx, (prot_batch, pep_batch) in enumerate(progress_bar):
        prot_features = prot_encoder(prot_batch)
        pep_features = pep_encoder(pep_batch)
        
        loss = loss_fn(prot_features, pep_features)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prot_encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(pep_encoder.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        intra_sim = F.cosine_similarity(prot_features, pep_features).mean().item()
        inter_sim = F.cosine_similarity(prot_features[0:1], prot_features[1:]).mean().item()
        
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'intra': f"{intra_sim:.3f}",
            'inter': f"{inter_sim:.3f}"
        })
    
    return total_loss / len(data_loader)

def validate(prot_encoder, pep_encoder, data_loader, loss_fn, temp=0.07):
    prot_encoder.eval()
    pep_encoder.eval()
    metrics = {
        'loss': 0.0,
        'partner_acc': 0.0,
        'peptide_acc': 0.0,
        'partner_mrr': 0.0,
        'peptide_mrr': 0.0,
        'partner_top10p': 0.0,
        'peptide_top10p': 0.0
    }
    
    with torch.no_grad():
        for prot_batch, pep_batch in data_loader:
            prot_features = prot_encoder(prot_batch)
            pep_features = pep_encoder(pep_batch)
            
            loss = loss_fn(prot_features, pep_features)
            metrics['loss'] += loss.item()
            
            logits = torch.mm(prot_features, pep_features.T) / temp
            batch_size = prot_features.size(0)
            labels = torch.arange(batch_size).to(prot_encoder.device)
            
            # Accuracy
            partner_preds = logits.argmax(dim=1)
            peptide_preds = logits.argmax(dim=0)
            metrics['partner_acc'] += (partner_preds == labels).float().mean().item()
            metrics['peptide_acc'] += (peptide_preds == labels).float().mean().item()
            
            # MRR
            partner_ranks = logits.argsort(dim=1, descending=True).argsort(dim=1).gather(1, labels.view(-1,1)).float() + 1
            peptide_ranks = logits.argsort(dim=0, descending=True).argsort(dim=0).gather(0, labels.view(1,-1)).float() + 1
            metrics['partner_mrr'] += (1.0 / partner_ranks).mean().item()
            metrics['peptide_mrr'] += (1.0 / peptide_ranks).mean().item()
            
            # Top10%
            k = max(1, int(batch_size * 0.1))
            _, partner_topk = logits.topk(k, dim=1)
            _, peptide_topk = logits.topk(k, dim=0)
            metrics['partner_top10p'] += (partner_topk == labels.view(-1,1)).any(dim=1).float().mean().item()
            metrics['peptide_top10p'] += (peptide_topk == labels.view(1,-1)).any(dim=0).float().mean().item()
    
    for key in metrics:
        metrics[key] /= len(data_loader)
    
    return metrics

def plot_metrics(training_stats, save_path):
    plt.figure(figsize=(15, 10))
    
    # Loss
    plt.subplot(2, 2, 1)
    plt.plot([s['train_loss'] for s in training_stats], label='Train')
    plt.plot([s['val_loss'] for s in training_stats], label='Val')
    plt.title('Training Loss')
    plt.legend()
    
    # Accuracy
    plt.subplot(2, 2, 2)
    plt.plot([s['partner_acc'] for s in training_stats], label='Partner Acc')
    plt.plot([s['peptide_acc'] for s in training_stats], label='Peptide Acc')
    plt.title('Prediction Accuracy')
    plt.legend()
    
    # Top10%
    plt.subplot(2, 2, 3)
    plt.plot([s['partner_top10p'] for s in training_stats], label='Partner')
    plt.plot([s['peptide_top10p'] for s in training_stats], label='Peptide')
    plt.title('Top10% Recall')
    plt.legend()
    
    # Learning Rate
    plt.subplot(2, 2, 4)
    plt.plot([s['lr'] for s in training_stats], label='LR')
    plt.title('Learning Rate')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    config = {
        'prot_model': './checkpoints/ESM',
        'pep_model': './checkpoints/ESM',
        'lr': 3e-5,
        'ortho_lambda': 0.03,
        'batch_size': 10,
        'epochs': 50,
        'save_dir': './checkpoints/Align',
        'patience': 15,
        'warmup_epochs': 3,
        'min_lr': 1e-6,
        'val_freq': 1,
        'main_metric': 'peptide_top10p',
        'eval_temp': 0.07
    }
    

    config['save_dir'] = os.path.normpath(config['save_dir'])
    os.makedirs(config['save_dir'], exist_ok=True)
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    

    metrics_file = os.path.join(config['save_dir'], 'eval_metrics.txt')
    if not os.path.exists(metrics_file):
        with open(metrics_file, 'w') as f:
            headers = [
                'Epoch', 'Val Loss', 'Partner Acc', 'Peptide Acc',
                'Partner MRR', 'Peptide MRR', 'Partner Top10%', 'Peptide Top10%'
            ]
            f.write('\t'.join(headers) + '\n')


    prot_encoder = ESMModel(config['prot_model'], device)
    pep_encoder = ESMModel(config['pep_model'], device)
    

    optimizer = torch.optim.AdamW([
        {'params': prot_encoder.parameters(), 'lr': config['lr']},
        {'params': pep_encoder.parameters(), 'lr': config['lr']}
    ], weight_decay=1e-5)
    

    train_prots, train_peps = get_prot_peptides('./dataset/Align/train.csv')
    val_prots, val_peps = get_prot_peptides('./dataset/Align/test.csv')
    
    train_loader = DataLoader(
        ProteinPeptideDataset(train_prots, train_peps),
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        ProteinPeptideDataset(val_prots, val_peps),
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min' if 'loss' in config['main_metric'] else 'max',
        factor=0.5,
        patience=2,
        verbose=True
    )
    
    loss_fn = OrthoInfoNCE(temp=0.07, ortho_lambda=config['ortho_lambda'])
    training_stats = []
    best_metric = -np.inf if 'top' in config['main_metric'] else np.inf
    early_stop_counter = 0
    
    for epoch in range(config['epochs']):

        if epoch < config['warmup_epochs']:
            lr_scale = (epoch + 1) / config['warmup_epochs']
            for param_group in optimizer.param_groups:
                param_group['lr'] = config['lr'] * lr_scale
        
        train_loss = train(prot_encoder, pep_encoder, train_loader, optimizer, loss_fn, epoch)
        
        if (epoch + 1) % config['val_freq'] == 0:
            val_metrics = validate(prot_encoder, pep_encoder, val_loader, loss_fn, config['eval_temp'])
            
            with open(metrics_file, 'a') as f:
                record = [
                    str(epoch+1),
                    f"{val_metrics['loss']:.4f}",
                    f"{val_metrics['partner_acc']:.4f}",
                    f"{val_metrics['peptide_acc']:.4f}",
                    f"{val_metrics['partner_mrr']:.4f}",
                    f"{val_metrics['peptide_mrr']:.4f}",
                    f"{val_metrics['partner_top10p']:.4f}",
                    f"{val_metrics['peptide_top10p']:.4f}"
                ]
                f.write('\t'.join(record) + '\n')
            
            scheduler.step(val_metrics[config['main_metric']])
            
            stats = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_metrics['loss'],
                'partner_acc': val_metrics['partner_acc'],
                'peptide_acc': val_metrics['peptide_acc'],
                'partner_top10p': val_metrics['partner_top10p'],
                'peptide_top10p': val_metrics['peptide_top10p'],
                'lr': optimizer.param_groups[0]['lr']
            }
            training_stats.append(stats)
            
            print(f"\nEpoch {epoch+1}/{config['epochs']}")
            print(f"Train Loss: {train_loss:.4f}  Val Loss: {val_metrics['loss']:.4f}")
            print(f"Partner Acc: {val_metrics['partner_acc']:.2%}  Peptide Acc: {val_metrics['peptide_acc']:.2%}")
            print(f"Partner Top10%: {val_metrics['partner_top10p']:.2%}  Peptide Top10%: {val_metrics['peptide_top10p']:.2%}")
            
            current_metric = val_metrics[config['main_metric']]
            if (current_metric > best_metric) if 'top' in config['main_metric'] else (current_metric < best_metric):
                best_metric = current_metric
                early_stop_counter = 0
                torch.save(prot_encoder.state_dict(), f"{config['save_dir']}/best_prot.pth")
                torch.save(pep_encoder.state_dict(), f"{config['save_dir']}/best_pep.pth")
                print(f"✨ New best model saved ({config['main_metric']}: {best_metric:.4f})")
            else:
                early_stop_counter += 1
                if early_stop_counter >= config['patience']:
                    print(f"⏹ Early stopping at epoch {epoch+1}")
                    break
            
            plot_metrics(training_stats, f"{config['save_dir']}/training_metrics.png")
    
    torch.save(prot_encoder.state_dict(), f"{config['save_dir']}/final_prot.pth")
    torch.save(pep_encoder.state_dict(), f"{config['save_dir']}/final_pep.pth")
    print("Training completed.")