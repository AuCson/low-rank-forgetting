from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import os, shutil
import time
import socket
from data_utils.lm import SFTExampleOnlyDataset
from data_utils import load_raw_ds as load_ds
from utils.config import load_configs
import torch
import pickle

from utils.analysis_tools import get_revision_name
from post_process.get_ppl_arr import get_ppl_from_file

try:
    from open_lm.hf import *
except ImportError:
    pass

hostname = socket.gethostname()
if hostname in ['anonymous', 'anonymous']:
    model_dtype = torch.float32
else:
    model_dtype = torch.bfloat16

def create_tmp_peft_merged_model(model_name, peft_model_dir, revision):
    from peft import AutoPeftModelForCausalLM, PeftConfig
    model = AutoPeftModelForCausalLM.from_pretrained(peft_model_dir, revision=revision,trust_remote_code=True)
    #model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision)
    #model.load_adapter(peft_model_dir)

    timestamp = time.time()
    tmp_dir = os.path.join(peft_model_dir, 'tmp_{}'.format(timestamp))
    os.makedirs(tmp_dir)
    model = model.merge_and_unload()
    model.save_pretrained(tmp_dir)
    return tmp_dir

def load_peft_ckpt(peft_model_dir, model_name, tokenizer_name, revision):
    tmp_dir = create_tmp_peft_merged_model(model_name, peft_model_dir, revision)
    llm = LLM(tmp_dir, trust_remote_code=True, dtype=model_dtype, tokenizer=tokenizer_name,
              tensor_parallel_size=args.n_gpus, gpu_memory_utilization=args.gpu_memory_utilization)
    shutil.rmtree(tmp_dir)
    return llm

def load_base_llm(model_name, tokenizer_name, enable_lora=False):
    llm = LLM(model_name, trust_remote_code=True, dtype=model_dtype, tokenizer=tokenizer_name,
              tensor_parallel_size=args.n_gpus, enable_lora=enable_lora, gpu_memory_utilization=args.gpu_memory_utilization)
    return llm

def load_ft_ckpt(model_dir, tokenizer_name):
    llm = LLM(model_dir, trust_remote_code=True, dtype=model_dtype, tokenizer=tokenizer_name,
              tensor_parallel_size=args.n_gpus, gpu_memory_utilization=args.gpu_memory_utilization)
    return llm

def run_stats_on_ds(args, config, llm, tokenizer, ds):
    results = {}

    lora_request = None
    if args.peft:
        lora_request = LoRARequest(
            'default_lora_adapter',
            1,
            config.stat.task_model_dir
        )

    if args.stat_ppl :
        sampling_params = SamplingParams(max_tokens=1, prompt_logprobs=0)
        chat_examples = ds.get_chat_or_raw_examples(include_gt=True, tokenizer=tokenizer)
        results['ppl'] = llm.generate(chat_examples, sampling_params, use_tqdm=True, lora_request=lora_request)

    if args.stat_output:
        sampling_params = SamplingParams(temperature=0, max_tokens=512)
        chat_examples = ds.get_chat_or_raw_examples(include_gt=False, tokenizer=tokenizer)
        results['output'] = llm.generate(chat_examples, sampling_params, use_tqdm=True, lora_request=lora_request)

    return results

def evaluate_llm_ocl_ds(args, config, tokenizer, llm):
    ocl_ds = load_ds(args.ocl_split, config, tokenizer, config.stat.ocl_task_category, config.stat.ocl_task_id)
    
    results = run_stats_on_ds(args, config, llm, tokenizer, ocl_ds)
    return results

def evaluate_llm_pt_ds(args, config, tokenizer, llm):
    pt_ds = load_ds(args.ocl_split, config, tokenizer, config.stat.pt_task_category, config.stat.pt_task_id)
    
    if args.subsample_pt > 0:
        print('Creating subsample of PT dataset')
        pt_ds = pt_ds.make_subsample(args.subsample_pt)

    results = run_stats_on_ds(args, config, llm, tokenizer, pt_ds)
    return results

def save_sep_results(output_dir, name,results):
    out_files = []
    for key in results:
        out_file = os.path.join(output_dir, f'{name}_{key}_results.pkl')
        with open(out_file, 'wb') as wf:
            pickle.dump(results[key], wf)
        out_files.append(out_file)
    return out_files
    

def format_name(args, ds_type):
    name = ds_type
    if args.ocl_split:
        name += '-{}'.format(args.ocl_split)
    if args.eval_base:
        name += '-base'
    if args.subsample_pt > 0:
        name += '-subsample{}'.format(args.subsample_pt)
    return name

def main(args, config):
    if config.pt_revision:
        revision_name = get_revision_name(config, config.pt_revision)
    else:
        revision_name = None
    tokenizer_name = config.model_name if config.tokenizer_name is None else config.tokenizer_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
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

    os.makedirs(config.output_dir, exist_ok=True)
    print('Output dir is {}'.format(config.output_dir))
    if args.eval_base:
        llm = load_base_llm(config.model_name, tokenizer_name)
    else:
        ckpt_dir = config.stat.task_model_dir
        if args.legacy_peft:
            llm = load_peft_ckpt(ckpt_dir, config.model_name, tokenizer_name, revision_name)
        elif args.peft:
            llm = load_base_llm(config.model_name, tokenizer_name, enable_lora=True)
        else:
            llm = load_ft_ckpt(ckpt_dir, tokenizer_name)

    if not args.skip_eval_ocl_ds:
        ocl_results = evaluate_llm_ocl_ds(args, config, tokenizer, llm)
        save_sep_results(config.output_dir, format_name(args, 'ocl'), ocl_results)
    if not args.skip_eval_pt_ds:
        pt_results = evaluate_llm_pt_ds(args, config, tokenizer, llm)
        out_files = save_sep_results(config.output_dir, format_name(args, 'pt'), pt_results)

        # assuming len out files is 1
        if args.stat_ppl and not args.stat_output:
            get_ppl_from_file(out_files[0], tokenizer, is_lm=not args.is_ins_pt)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', nargs='+')
    parser.add_argument("--templates", nargs='*')

    parser.add_argument('--eval_base', action='store_true')
    #parser.add_argument('--eval_task_model', action='store_true')

    parser.add_argument('--skip_eval_ocl_ds', action='store_true')
    parser.add_argument('--skip_eval_pt_ds', action='store_true')
    parser.add_argument('--legacy_peft', action='store_true')
    parser.add_argument('--peft', action='store_true')
    
    parser.add_argument('--subsample_pt', type=int, default=-1)

    parser.add_argument('--stat_ppl', action='store_true')
    parser.add_argument('--stat_output', action='store_true')

    parser.add_argument('--ocl_split')
    parser.add_argument('--n_gpus', type=int, default=1)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    parser.add_argument('--is_ins_pt', action='store_true')
    args = parser.parse_args()

    config = load_configs(*args.config_files, templates=args.templates)
    main(args, config)