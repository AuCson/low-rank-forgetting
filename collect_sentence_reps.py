from trainer.my_trainer import MyTrainer
from trainer.utils import DataCollatorWithPaddingStr
from transformers import AutoModel, AutoTokenizer, TrainingArguments, AutoModelForCausalLM
from utils.config import merge_config_into_args, load_configs
from data_utils import load_ocl_ds_by_task_id
import os
import logging
from torch.utils.data import DataLoader
import torch
import argparse
from tqdm import tqdm
from utils.analysis_tools import \
    create_past_model, load_peft_model
import numpy as np
import pickle
from trak.projectors import CudaProjector, ProjectionType
from data_utils.utils import deterministic_random_indices
from torch.utils.data import Subset
import torch.nn.functional as F

logger = logging.getLogger('main')

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def run_stat_sentence_reps(args):
    config = load_configs(*args.config_files, templates=args.templates)

    print(config.output_dir)
    os.makedirs(config.output_dir, exist_ok=True)
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

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

    logger.setLevel(logging.INFO)
    os.makedirs(config.output_dir, exist_ok=True)
    logger.addHandler(logging.FileHandler(os.path.join(config.output_dir, 'log.txt')))

    ocl_train_ds, ocl_eval_ds, task = load_ocl_ds_by_task_id(config, tokenizer, config.ocl.task_category, config.ocl.task_id, include_labels=True)

    print('Working on task {}:{} / saving at {}'.format(config.ocl.task_category, config.ocl.task_id, config.output_dir))
    print(tokenizer.decode(ocl_train_ds[0]['input_ids']))

    if len(ocl_train_ds) > args.max_example:
        if args.predef_full_len > 0:
            full_len = args.predef_full_len
        else:
            full_len = len(ocl_train_ds)
        indices = deterministic_random_indices(full_len, args.max_example)
        ocl_train_ds = Subset(ocl_train_ds, indices)
        with open(os.path.join(config.output_dir, f'subsampled_{config.ocl.task_category}_{config.ocl.task_id}'),'wb') as wf:
            pickle.dump(ocl_train_ds, wf) 
          

    ocl_train_loader = DataLoader(ocl_train_ds, 16, shuffle=False, collate_fn=data_collator)
    all_reps = []
    
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(ocl_train_loader), total=len(ocl_train_loader)):
            if idx == 0:
                print(batch)
            input_ids, attention_mask = batch['input_ids'].cuda(), batch['attention_mask'].cuda()
            model_output = model(input_ids=input_ids, attention_mask=attention_mask)
            sentence_embeddings = mean_pooling(model_output, attention_mask)
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            all_reps.append(sentence_embeddings.detach().cpu())

            if idx == 0:
                print(model_output)
                

    all_reps = torch.concatenate(all_reps, 0)

    with open(os.path.join(config.output_dir, f'all_reps{args.postfix}.pkl'),'wb') as wf:
        pickle.dump(all_reps, wf)



def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', nargs='*')
    parser.add_argument("--templates", nargs='*')

    parser.add_argument("--n_gpu", default=1, type=int)
    parser.add_argument('--max_example', default=1000000, type=int)
    parser.add_argument('--predef_full_len', default=-1, type=int)
    parser.add_argument('--postfix')

    parser.add_argument('--model_name', default='allenai/OLMo-1B-hf')

    args = parser.parse_args(argv)

    run_stat_sentence_reps(args)

if __name__ == '__main__':
    main()
