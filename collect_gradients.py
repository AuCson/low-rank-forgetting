from trainer.my_trainer import MyTrainer
from trainer.utils import DataCollatorWithPaddingStr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, AutoModelForCausalLM
from utils.config import merge_config_into_args, load_configs
from data_utils import load_ocl_ds_by_task_id
import os
import logging
from torch.utils.data import DataLoader
import torch
from torch import nn
import argparse
from tqdm import tqdm
from utils.analysis_tools import \
    create_past_model, load_peft_model
import numpy as np
import pickle
try:
    import trak
    from trak.projectors import CudaProjector, ProjectionType
except ImportError:
    print('Trak not installed, please install trak for gradient projection experiments')
from data_utils.utils import deterministic_random_indices
from torch.utils.data import Subset

logger = logging.getLogger('main')


class GradientProjector(nn.Module):
    def __init__(self, grads):
        grad_dim = get_grad_dimension(grads)
        proj_dim = 1024
        seed = 0
        proj_type = ProjectionType.rademacher
        device = next(iter(grads.values())).device
        self.projector = CudaProjector(
            grad_dim, proj_dim, seed, proj_type, device, max_batch_size=8
        )

        dummy_input = torch.ones(1,grad_dim).to(device)
        dummy_proj_grad = self.projector.project(dummy_input, model_id=0).view(-1)
        print('Dummy proj output: {}'.format(dummy_proj_grad))

    def do_proj(self, grads):
        grads_batch = {}
        for k in grads[0].keys():
            grads_batch[k] = torch.stack([grads[i][k] for i in range(len(grads))])
            
        model_id = 0
        proj_grad = self.projector.project(grads_batch, model_id=model_id)
        return proj_grad


def collect_param_grads(model):
    grads = {}
    for n, p in model.named_parameters():
        if p.grad is not None:
            grads[n] = p.grad
    return grads

def get_grad_dimension(grads):
    s = 0
    for n, p in grads.items():
        s += p.numel()
    return s

def flatten_grad(grads):
    params = [g.view(-1) for g in grads.values()]
    return torch.cat(params)

def get_flatten_param(model):
    params = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            params.append(p.view(-1))
    return torch.cat(params)
    

def run_stat_grad(args):
    config = load_configs(*args.config_files, templates=args.templates)

    print(config.output_dir)
    os.makedirs(config.output_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(config.model_name, trust_remote_code=True).cuda()

    if config.peft == 'lora':
        print('Creating peft model')
        model = load_peft_model(config=config, base_model=model)


    if config.load_ckpt:
        print('Loading model from {}'.format(config.load_ckpt))
        state_dict = torch.load(config.load_ckpt)
        # state_dict = {k[len('model.'):]: v for k,v in state_dict.items()}
        model.load_state_dict(state_dict)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name if config.tokenizer_name is None else config.tokenizer_name,
                                              trust_remote_code=True, use_flash_attention_2=config.use_flash_attention_2)

    if config.chat_template_name != config.tokenizer_name:
        print('Loading chat template of {}'.format(config.chat_template_name))
        dummy_tokenizer = AutoTokenizer.from_pretrained(config.chat_template_name, trust_remote_code=True)
        tokenizer.chat_template = dummy_tokenizer.chat_template
    if tokenizer.pad_token_id is None:
        if '<|padding|>' in tokenizer.vocab:
            tokenizer.pad_token = '<|padding|>'
        else:
            print('Using eos as pad token')
            tokenizer.pad_token_id = tokenizer.eos_token_id

    data_collator = DataCollatorWithPaddingStr(tokenizer, max_length=config.max_input_length)  

    training_args = TrainingArguments(output_dir=config.output_dir, bf16=True, gradient_checkpointing=True)

    merge_config_into_args(config, training_args)

    logger.setLevel(logging.INFO)
    os.makedirs(config.output_dir, exist_ok=True)
    logger.addHandler(logging.FileHandler(os.path.join(config.output_dir, 'log.txt')))

    config.save(config.output_dir, 'config.json')

    trainer = MyTrainer(
        model=model, args=training_args, train_dataset=None, eval_dataset=None,
        tokenizer=tokenizer, data_collator=data_collator, memory=None, config=config,
        past_model_creator=create_past_model
    )

    ocl_train_ds, ocl_eval_ds, task = load_ocl_ds_by_task_id(config, tokenizer, config.ocl.task_category, config.ocl.task_id, include_labels=True)
    #ocl_train_ds = SFTDataset.from_auto(config.ocl.task_category, tasks=[config.ocl.task_id] if config.ocl.task_id is not None else None,
    #                                 split='train', config=config, tokenizer=tokenizer, skip_encoding=False, include_labels=True)


    print('Working on task {}:{} / saving at {}'.format(config.ocl.task_category, config.ocl.task_id, config.output_dir))
    print(tokenizer.decode(ocl_train_ds[0]['input_ids']))

    if len(ocl_train_ds) > args.max_example:
        if args.predef_full_len > 0:
            full_len = args.predef_full_len
        else:
            full_len = len(ocl_train_ds)
        indices = deterministic_random_indices(full_len, args.max_example)
        ocl_train_ds = Subset(ocl_train_ds, indices)

    ocl_train_loader = DataLoader(ocl_train_ds, 1, shuffle=False, collate_fn=data_collator)

    all_proj_grads = []

    gradient_projector = None

    proj_every = 1
    tmp_grad_store = []
    for idx, batch in tqdm(enumerate(ocl_train_loader), total=len(ocl_train_loader)):
        model.zero_grad()
        batch_ = trainer.clean_batch(batch, phase='train')
        batch_ = trainer.batch_to_gpu(batch_)
        loss_dt = trainer.training_step(model, batch_)
        #loss_dt.backward()
        grads = collect_param_grads(model)
        
        with torch.no_grad():
            flattened_grad = flatten_grad(grads)
            grad_dim = get_grad_dimension(grads)

            if gradient_projector is None:
                gradient_projector = GradientProjector(grads)

            tmp_grad_store.append(grads)
            if len(tmp_grad_store) == proj_every:
                proj_grad = gradient_projector.do_proj(tmp_grad_store)
                tmp_grad_store = []
                all_proj_grads.append(proj_grad.cpu())

            grad_norm = torch.norm(flattened_grad)
            print(f'Grad norm is {grad_norm} / dim {grad_dim}')
            
                    
    all_proj_grads = torch.cat(all_proj_grads)
    with open(os.path.join(config.output_dir, 'grad_store.pkl'),'wb') as wf:
        pickle.dump(all_proj_grads, wf)

def run_stat_param_diff(args):
    config = load_configs(*args.config_files, templates=args.templates)

    print(config.output_dir)
    os.makedirs(config.output_dir, exist_ok=True)
    
    base_model = AutoModelForCausalLM.from_pretrained(config.model_name, trust_remote_code=True)
    trained_model = AutoModelForCausalLM.from_pretrained(config.load_model_dir, trust_remote_code=True)

    base_params = get_flatten_param(base_model).cuda()
    trained_params = get_flatten_param(trained_model).cuda()

    proj_type = ProjectionType.rademacher
    device = base_params.device
    proj_dim = 1024
    seed = 0

    print("Param size is {}".format(base_params.size(0)))

    projector = CudaProjector(
        base_params.size(0), proj_dim, seed, proj_type, device, max_batch_size=8
    )
    
    model_id = 0
    proj_base = projector.project(base_params.view(1,-1), model_id=model_id)
    proj_trained = projector.project(trained_params.view(1,-1), model_id=model_id)

    diff = proj_trained - proj_base
    diff = diff.view(-1)
    diff = diff.cpu()

    with open(os.path.join(config.output_dir, 'param_diff_store.pkl'),'wb') as wf:
        pickle.dump(diff, wf)

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', nargs='*')
    parser.add_argument("--templates", nargs='*')

    parser.add_argument("--n_gpu", default=1, type=int)
    parser.add_argument('--max_example', default=1000000, type=int)
    parser.add_argument('--predef_full_len', default=-1, type=int)

    parser.add_argument('--variant', default='grad', choices=['grad','diff'])
    args = parser.parse_args(argv)

    if args.variant == 'grad':
        run_stat_grad(args)
    elif args.variant == 'diff':
        run_stat_param_diff(args)

if __name__ == '__main__':
    main()
