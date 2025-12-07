from data_utils.online_fpd_helper import OnlineFPDHelper
from utils.config import load_configs
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from trainer.utils import DataCollatorWithPaddingStrForLM, DataCollatorMaskedStrForLM, DataCollatorWithPadding
from data_utils.lm import SFTDataset
import numpy as np

if __name__ == '__main__':
    config = load_configs('configs/defaults.yaml',
                          'configs/llm/llm_defaults.yaml',
                          'configs/llm/online/7b_ft_1kstep_online.yaml', templates=["TASK_ID=65", "TASK_CATEGORY=flan"])
    model = AutoModelForCausalLM.from_pretrained(config.model_name, trust_remote_code=True, 
                                                    use_flash_attention_2=config.use_flash_attention_2)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name if config.tokenizer_name is None else config.tokenizer_name,
                                              trust_remote_code=True)

    pt_ds = SFTDataset.from_auto(config.replay.task_category, tasks=None,
                                split='train', config=config, tokenizer=tokenizer, skip_encoding=False, 
                                include_labels=True, ans_start_patt="<|assistant|>")
    
    print('Begin loading helper')
    helper = OnlineFPDHelper(config, pt_ds)

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
    
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        evaluation_strategy="steps",
        eval_steps=config.ocl_val_step,
        save_strategy="no" if config.save_steps < 0 else "steps",
        save_steps=99999999999 if config.save_steps < 0 else config.save_steps,
        max_steps=10,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        deepspeed=config.deepspeed,
        bf16=config.bf16,
        fp16=config.fp16,
        logging_strategy="steps",
        logging_steps=10,
        optim='paged_adamw_8bit',
        warmup_steps=10,
        eval_accumulation_steps=1
    )

    #if config.is_lm_sft:
    data_collator = DataCollatorWithPadding(tokenizer, max_length=config.max_input_length)  
    #else:
    #    data_collator = DataCollatorWithPaddingStrForLM(tokenizer, max_length=config.max_input_length)

    trainer = Trainer(
        model=model, args=training_args,
        tokenizer=tokenizer, data_collator=data_collator
    )

    print('Test 1: Evaluate base forgetting')
    helper.evaluate_base_model_seed_ppl(trainer)

    del model, trainer

    print('Test 2: Predict forgetting for others')

    model = AutoModelForCausalLM.from_pretrained(config.debug_ckpt_name, trust_remote_code=True, 
                                                    use_flash_attention_2=config.use_flash_attention_2)

    trainer = Trainer(
        model=model, args=training_args,
        tokenizer=tokenizer, data_collator=data_collator
    )
    
    #query_meta_idxs, query_idxs, forgetting = helper.evaluate_seed_forgetting(trainer)
    #print(query_meta_idxs, query_idxs, forgetting)


    pred_forgetting = helper.predict_all_forgetting(trainer)
    np.save('runs/forgetting_prediction/debug.npy', pred_forgetting)
