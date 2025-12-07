from trainer.my_trainer import MyTrainer
from trainer.utils import DataCollatorWithPaddingStrForLM, DataCollatorMaskedStrForLM, DataCollatorWithPadding
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, AutoModelForCausalLM
from utils.config import merge_config_into_args, load_configs
from data_utils.lm import SFTDataset
from data_utils.replay_helper import prepare_replay_mixture_ds
from data_utils import load_ocl_ds_by_task_id, shuffle_ds
import os
import logging
import torch
from torch.utils.data import Subset
import argparse
from tqdm import tqdm
from utils.analysis_tools import \
    create_past_model, load_peft_model, get_revision_name, touch_file
import numpy as np
import json
from transformers import Trainer
from trainer.cl_trainer import MyCLTrainer
import accelerate

try:
    from open_lm.hf import *
except ImportError:
    pass

logger = logging.getLogger('main')

def run_pipeline_stat_errors_in_stream(args):
    config = load_configs(*args.config_files, templates=args.templates)
    print(config.output_dir)
    os.makedirs(config.output_dir, exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(os.path.join(config.output_dir, 'log.txt')))
    config.save(config.output_dir, 'config.json')

    tokenizer = AutoTokenizer.from_pretrained(config.model_name if config.tokenizer_name is None else config.tokenizer_name,
                                              trust_remote_code=True)

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

    model_kwargs = {}
    if config.pt_revision:
        revision_name = get_revision_name(config, config.pt_revision)
        model_kwargs['revision'] = revision_name
        print('Model revision is {}'.format(model_kwargs['revision']))

    model_save_dir = None
    model = AutoModelForCausalLM.from_pretrained(config.model_name, trust_remote_code=True, **model_kwargs,
                                                    use_flash_attention_2=config.use_flash_attention_2)
    trainer = None
    if config.is_continual:
        task_seq = [int(x) for x in config.task_seqs[config.ocl.task_seq_id].split(',')]
        print('Task seq is {}'.format(task_seq))
        for task_seq_id, task_id in enumerate(task_seq):
            output_dir = os.path.join(config.output_dir, 'task_seq_{}'.format(task_seq_id))
            os.makedirs(output_dir, exist_ok=True)
            touch_file(output_dir,'task_{}'.format(task_id))
            trainer = train_single_task(config, model, tokenizer, 
                            output_dir=output_dir,
                            task_category=config.ocl.task_category, task_id=task_id, trainer=trainer)
    else:
        train_single_task(config, model, tokenizer, 
                        output_dir=config.output_dir,
                        task_category=config.ocl.task_category, task_id=config.ocl.task_id)
        

def train_single_task(config, model, tokenizer,output_dir, task_category, task_id, trainer=None):
    os.makedirs(output_dir, exist_ok=True)

    ocl_train_ds, ocl_eval_ds, task = load_ocl_ds_by_task_id(config, tokenizer, task_category, task_id, include_labels=True, ans_start_patt=config.ans_start_pattern)
    #print(ocl_train_ds[0])

    if config.shuffle_ocl:
        ocl_train_ds = shuffle_ds(ocl_train_ds, config.shuffle_ocl_seed)
    
    ocl_eval_ds = Subset(ocl_eval_ds, [_ for _ in range(min(1000, len(ocl_eval_ds)))])

    max_steps = config.ocl_steps

    # if replay, then extend the stream
    if config.replay.enabled:
        max_steps = int(max_steps * (1 + config.replay.mixture_ratio))

    max_steps_by_epoch = config.max_epoch * len(ocl_train_ds) //  (config.per_device_train_batch_size * config.n_gpu * config.gradient_accumulation_steps)

    if max_steps_by_epoch < max_steps:
        max_steps = max_steps_by_epoch
    logger.info('Training step is {} for task {}/{}'.format(max_steps, task_category, task_id))

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        evaluation_strategy="steps",
        eval_steps=config.ocl_val_step,
        save_strategy="no" if config.save_steps < 0 else "steps",
        save_steps=99999999999 if config.save_steps < 0 else config.save_steps,
        max_steps=max_steps,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        deepspeed=config.deepspeed,
        bf16=config.bf16,
        fp16=config.fp16,
        logging_strategy="steps",
        logging_steps=10,
        optim=config.optimizer,
        warmup_steps=config.lr_warmup_steps,
        ddp_timeout=18000
    )

    if config.peft == 'lora':
        print('Creating peft model')
        model = load_peft_model(config=config, base_model=model)
    if config.load_ckpt:
        print('Loading model from {}'.format(config.load_ckpt))
        state_dict = torch.load(config.load_ckpt)
        # state_dict = {k[len('model.'):]: v for k,v in state_dict.items()}
        model.load_state_dict(state_dict)

    if config.replay.enabled:
        data_collator = DataCollatorWithPadding(tokenizer, max_length=config.max_input_length)  
        ocl_train_ds = prepare_replay_mixture_ds(config, tokenizer, ocl_train_ds)
    else:
        if config.is_lm_sft:
            data_collator = DataCollatorMaskedStrForLM(tokenizer, max_length=config.max_input_length, ans_start_patt=config.ans_start_pattern)
        else:
            data_collator = DataCollatorWithPaddingStrForLM(tokenizer, max_length=config.max_input_length)

    print('Working on task {}:{} / saving at {}'.format(task_category, task, output_dir))
    print(tokenizer.decode(ocl_train_ds[0]['input_ids']))
    try:
        print(tokenizer.decode(ocl_train_ds[0]['labels'][ocl_train_ds[0]['labels'] != -100]))
    except Exception:
        pass

    if trainer is None:
        is_first_task = True
        if config.is_continual:
            trainer = MyCLTrainer(
                model=model, args=training_args, train_dataset=ocl_train_ds, eval_dataset=ocl_eval_ds,
                tokenizer=tokenizer, data_collator=data_collator
            )
        else:
            trainer = Trainer(
                model=model, args=training_args, train_dataset=ocl_train_ds, eval_dataset=ocl_eval_ds,
                tokenizer=tokenizer, data_collator=data_collator
            )
    else:
        is_first_task = False

    all_val_losses = []
    
    if is_first_task:
        trainer.train()
    else:
        trainer.do_new_task(args=training_args, train_dataset=ocl_train_ds, eval_dataset=ocl_eval_ds)
        trainer._inner_training_loop(args=trainer.args)

    if config.is_continual:
        print('DEBUG: train another')
        ocl_train_ds, ocl_eval_ds, task = load_ocl_ds_by_task_id(config, tokenizer, task_category, 18, include_labels=True)
        trainer.do_new_task(args=training_args, train_dataset=ocl_train_ds, eval_dataset=ocl_eval_ds)
        trainer._inner_training_loop(args=trainer.args)
        print('DEBUG: stop train another')


    if not config.deepspeed or 'deepspeed_3' not in config.deepspeed:
        model.save_pretrained(os.path.join(output_dir, 'model_save'))
        trainer.save_state()
    else:
        print('Saving deepspeed 3 model')
        trainer.save_model(os.path.join(output_dir, 'model_save'))
            

    return trainer


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', nargs='+')
    parser.add_argument("--templates", nargs='*')

    parser.add_argument("--ocl_task")
    parser.add_argument('--debug',action='store_true')

    args = parser.parse_args(argv)
    run_pipeline_stat_errors_in_stream(args)

if __name__ == '__main__':
    main()
