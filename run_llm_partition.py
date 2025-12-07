from trainer.my_trainer import MyTrainer
from trainer.utils import DataCollatorWithPaddingStrForLM, DataCollatorMaskedStrForLM, DataCollatorWithPadding
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, AutoModelForCausalLM
from utils.config import merge_config_into_args, load_configs
from data_utils.lm import SFTDataset
from data_utils.replay_helper import prepare_replay_mixture_with_ds, ReplayedExampleRecord
from data_utils import load_ocl_ds_by_task_id, shuffle_ds
import os
import logging
import torch
from torch.utils.data import Subset
import argparse
from tqdm import tqdm
from utils.analysis_tools import \
    create_past_model, load_peft_model, get_revision_name, touch_file
from post_process.get_ppl_arr import deterministic_random_indices
import numpy as np
import json
from transformers import Trainer
from trainer.cl_trainer import MyCLTrainer
from data_utils.online_fpd_helper import OnlineFPDHelper
import gc

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

    #ocl_train_ds, ocl_eval_ds, task = load_ocl_ds_by_task_id(config, tokenizer, config.ocl.task_category, task_id, include_labels=True, ans_start_patt="<|assistant|>")

    task_id = config.ocl.task_id
    ocl_train_ds, ocl_eval_ds, task = load_ocl_ds_by_task_id(config, tokenizer, config.ocl.task_category, task_id, include_labels=True, ans_start_patt=config.ans_start_pattern)

    max_steps = config.ocl_steps

    # if replay, then extend the stream
    if config.replay.enabled:
        max_steps = int(max_steps * (1 + config.replay.mixture_ratio))

    # avoid too long training 
    max_steps_by_epoch = config.max_epoch * len(ocl_train_ds) //  (config.per_device_train_batch_size * config.n_gpu)
    if max_steps_by_epoch < max_steps:
        max_steps = max_steps_by_epoch


    training_args = TrainingArguments(
        output_dir=config.output_dir,
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
        logging_steps=5,
        optim='paged_adamw_8bit',
        warmup_steps=10,
        #lr_scheduler_type="constant"

    )

    #if config.is_lm_sft:
    #    data_collator = DataCollatorMaskedStrForLM(tokenizer, max_length=config.max_input_length, ans_start_patt="<|assistant|>")
    #else:
    #    data_collator = DataCollatorWithPaddingStrForLM(tokenizer, max_length=config.max_input_length)
    data_collator = DataCollatorWithPadding(tokenizer, max_length=config.max_input_length)  

    trainer = MyCLTrainer(
        model=model, args=training_args, train_dataset=None, eval_dataset=None,
        tokenizer=tokenizer, data_collator=data_collator
    )

    pt_ds = SFTDataset.from_auto(config.replay.task_category, tasks=None,
                                split='train', config=config, tokenizer=tokenizer, skip_encoding=False, 
                                include_labels=True, ans_start_patt=config.ans_start_pattern)
    
    replay_example_record = ReplayedExampleRecord()
    online_fpd_helper = OnlineFPDHelper(config, pt_ds)

    model_init_state = None

    ocl_eval_ds = Subset(ocl_eval_ds, [_ for _ in range(min(1000, len(ocl_eval_ds)))])
    
    model_init_state = {k: v.clone().cpu() for k,v in model.state_dict().items()}

    phase_a_train_step = int(config.partition.phase_a_train_step_proportion * max_steps)
    phase_a_mix_ocl_ds = prepare_replay_mixture_with_ds(config.partition.phase_a_replay, tokenizer, ocl_train_ds, pt_ds)

    trainer.do_new_task(args=training_args, train_dataset=phase_a_mix_ocl_ds, eval_dataset=ocl_eval_ds)

    trainer.train(skip_steps=0, custom_max_train_step=phase_a_train_step)

    logger.info(f'Finished phase a of task id {task_id}')

    if online_fpd_helper.base_model_ppl is None:
        online_fpd_helper.evaluate_base_model_seed_ppl_from_init_state(trainer, model_init_state)
        del model_init_state
        gc.collect()
    pred_forgetting = online_fpd_helper.predict_all_forgetting(trainer)

    if trainer.is_world_process_zero():
        pred_fgt_save_dir = os.path.join(config.output_dir, f'task_{task_id}')
        os.makedirs(pred_fgt_save_dir, exist_ok=True)
        np.save(os.path.join(pred_fgt_save_dir, 'pred_forgetting.npy'), pred_forgetting)
    
    phase_b_train_step = max_steps - phase_a_train_step

    if getattr(config.partition.phase_b_replay, 'enable_mask', False):
        pred_forgetting_masked = replay_example_record.mask_forgetting_arr(pred_forgetting)
    else:
        pred_forgetting_masked = pred_forgetting
    phase_b_mix_ocl_ds = prepare_replay_mixture_with_ds(config.partition.phase_b_replay, tokenizer, ocl_train_ds, pt_ds, online_forgetting_arr=pred_forgetting_masked, seed=1000)

    replay_example_record.update(phase_b_mix_ocl_ds.get_replay_idxs())
    logging.info('Replayed example: {}'.format(replay_example_record.replayed_idxs))

    trainer.do_new_task(args=training_args, train_dataset=phase_b_mix_ocl_ds, eval_dataset=ocl_eval_ds)
    trainer._inner_training_loop(args=trainer.args, skip_steps=phase_a_train_step, custom_max_train_step=phase_b_train_step)

    if trainer.is_world_process_zero():
        logging.info(f'Finished task {task_id}')
        model_save_dir = os.path.join(config.output_dir, f'task_{task_id}', 'model_save')
        os.makedirs(model_save_dir, exist_ok=True)
        #trainer.save_model(os.path.join(output_dir, 'model_save'))
        model.save_pretrained(model_save_dir)
        trainer.state.save_to_json(os.path.join(model_save_dir, 'trainer_state.json'))

        

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', nargs='+')
    parser.add_argument("--templates", nargs='*')

    parser.add_argument("--ocl_task")
    parser.add_argument("", action='store_true')
    parser.add_argument('--debug',action='store_true')

    args = parser.parse_args(argv)
    run_pipeline_stat_errors_in_stream(args)

if __name__ == '__main__':
    main()
