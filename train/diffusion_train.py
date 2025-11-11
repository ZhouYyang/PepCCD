import sys
sys.path.append("/home2/yyzhou/PepCCD")

import argparse
import torch.cuda
import torch.multiprocessing as mp
from train.align_train import ESMModel
from utils.tokenizer import load_data
from utils.script_utils import (
    model_and_diffusion_defaults,
    add_dict_to_argparser,
)
from utils.train_util import TrainLoop

def main():

    mp.set_start_method('spawn', force=True)
    
    from model.resample import create_named_schedule_sampler
    from utils.script_utils import create_model_and_diffusion, args_to_dict
    

    args = create_argparser().parse_args()
    device = ('cuda:7' if torch.cuda.is_available() else 'cpu')
    print("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(device)
    #model.load_state_dict(torch.load(args.pretrained_diffusion))
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    
    
    pep_encoder = ESMModel("./checkpoints/ESM", device)
    pep_encoder.load_state_dict(torch.load(args.pretrained_pep_encoder))
    pep_encoder.to(device)
    
    prot_encoder = ESMModel("./checkpoints/ESM", device)
    prot_encoder.load_state_dict(torch.load(args.pretrained_prot_encoder))
    prot_encoder.to(device)

    print("creating data loader...")
    train_data = load_data(pep_encoder, args.seq_len, "train", args)

    print("training...")
    TrainLoop(
        model=model,
        device=device,
        diffusion=diffusion,
        save_dir=args.save_dir,
        pep_encoder=pep_encoder,
        prot_encoder=prot_encoder,
        train_data=train_data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()
    
    
def create_argparser():
    defaults = dict(
        # path of pretrained model
        save_dir='./checkpoints/Pre_Diffusion',
        pretrained_diffusion='./checkpoints/Pre_Diffusion/pre_diffusion_model.pt',
        pretrained_pep_encoder='./checkpoints/Align/best_pep.pth',
        pretrained_prot_encoder='./checkpoints/Align/best_prot.pth',
        timesteps=1000,
        train_epoches=1000,
        lr=5e-4,
        weight_decay=0.0,
        lr_anneal_steps=40000,
        batch_size=10,
        microbatch=-1,  
        ema_rate="0.9999",  
        seq_len=256,
        is_need_classifier=False,
        label_num=2,
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler="uniform",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser
    
    


if __name__ == "__main__":
    
    main()
