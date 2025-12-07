import numpy as np
import argparse
import re
import os
import pickle
from transformers import AutoTokenizer
import multiprocessing
from data_utils.utils import deterministic_random_indices
import yaml
import json

def make_arr(paths):
    rows = []
    arr_len = None
    for path in paths:
        if os.path.exists(path):
            arr = np.load(path)
            rows.append(arr)
            if arr_len is None:
                arr_len = arr.shape[0]
            elif arr.shape[0] != arr_len:
                raise ValueError(arr_len, arr.shape[0])
        else:
            rows.append(None)
    rows = [np.full(arr_len, np.nan) if x is None else x for x in rows]
    rows = np.stack(rows)
    return rows
    

def find_ans_start_after(input_ids, ans_start_tokens):
    pos = None
    input_ids = np.array(input_ids)
    for i in range(len(input_ids)-1, -1, -1):
        if input_ids[i] == ans_start_tokens[0]:
            if (input_ids[i:i+len(ans_start_tokens)] == ans_start_tokens).all():
                pos = i
                break
    return pos + len(ans_start_tokens) 

def find_eos_legacy(input_ids, eos_token_id):
    pos = None
    input_ids = np.array(input_ids)
    idxs = np.arange(len(input_ids))[input_ids == eos_token_id]
    if len(idxs) > 0:
        return idxs[-2] if idxs[-2] != 0 else idxs[-1]
    else:
        return None


def find_eos(input_ids, eos_token_id):
    pos = None
    input_ids = np.array(input_ids)
    idxs = np.arange(len(input_ids))[input_ids == eos_token_id]
    if len(idxs) > 0:
        return idxs[-1]
    else:
        return None

def get_outputs(all_ret):
    outputs = []
    for ret in all_ret:
        outputs.append(ret.outputs[0].text)
    return outputs

def extract_single_ppl(ret, is_lm, ans_start_tokens, tokenizer=None, use_legacy_eos=True):
    input_ids = np.array(ret.prompt_token_ids)
    if is_lm:
        pos, eos = 1, len(input_ids)
    else:
        pos = find_ans_start_after(input_ids, ans_start_tokens)
        eos = find_eos_legacy(input_ids, tokenizer.eos_token_id) if use_legacy_eos else find_eos(input_ids, tokenizer.eos_token_id)
        #print(pos, eos)
    
    if ret.prompt_logprobs is not None:
        if len(ret.prompt_logprobs) > 1 and type(next(iter(ret.prompt_logprobs[1].values()))) is not float:
            log_probs = np.array([list(x.values())[0].logprob if x is not None else None for x in ret.prompt_logprobs])
        else:
            log_probs = np.array([list(x.values())[0] if x is not None else None for x in ret.prompt_logprobs])
        gt_log_probs = log_probs[pos:eos].astype(float)

        #all_log_probs.append(gt_log_probs)
        #all_gt_tokens.append(gt_tokens)
        #avg_log_probs.append(gt_log_probs.mean())
        avg_log_prob = gt_log_probs.mean()
    else:
        avg_log_prob = None
        #print('No answer found for {}'.format(i))
    return avg_log_prob

def get_ppls(all_ppl_ret, tokenizer, is_lm):
    ans_start_patt = '<|assistant|>'
    ans_start_tokens =  np.array(tokenizer.encode(ans_start_patt))

    avg_log_probs = []
    for i, ret in enumerate(all_ppl_ret):
        avg_log_prob = extract_single_ppl(ret, is_lm, ans_start_tokens, tokenizer=tokenizer)
        avg_log_probs.append(avg_log_prob)
    avg_log_probs = np.array(avg_log_probs).astype(float)
    
    return avg_log_probs

def get_ppls_mp(all_ppl_ret, tokenizer, is_lm):
    print('Multiprocessing')
    ans_start_patt = '<|assistant|>'
    ans_start_tokens =  np.array(tokenizer.encode(ans_start_patt))

    avg_log_probs = {}
    with multiprocessing.Pool(processes=16) as pool:
       procs = {i : pool.apply_async(extract_single_ppl, (ret,is_lm,ans_start_tokens)) for i,ret in enumerate(all_ppl_ret)}
       results = {i: proc.get() for i,proc in procs.items()}

    avg_log_probs = [results[i] for i in sorted(results.keys())]
    avg_log_probs = np.array(avg_log_probs).astype(float)
    
    return avg_log_probs
    #return all_log_probs, all_gt_tokens, invalid_idxs

def get_ppl_from_file(file, tokenizer, is_lm=True):
    print(file)
    output_file = file + '.npy'
    with open(file,'rb') as f:
        all_ppl_ret = pickle.load(f)
    #if mp:
    #    avg_log_probs = get_ppls_mp(all_ppl_ret, tokenizer, is_lm=True)
    #else:
    avg_log_probs = get_ppls(all_ppl_ret, tokenizer, is_lm)
    np.save(output_file, avg_log_probs)
    print(avg_log_probs)

def get_subset_ppls(full_arr, subsample_num):
    indices = deterministic_random_indices(full_arr.shape[0], subsample_num)
    return full_arr[indices]

def get_save_forgetting(full_arr_path, ref_file):
    base_nll = - np.load(ref_file)
    out_path = '/'.join(full_arr_path.split('/')[:-1]) + '/pt_fgt_arr.npy'
    after_nll = - np.load(full_arr_path)
    fgt = after_nll - base_nll
    np.save(out_path, fgt)

def avg_ppl(arr):
    return arr[~np.isnan(arr)].mean()

def get_ppl_mat(base_dir, arr_file_name='pt_ppl_results.pkl.npy', remove_nan=True):
    len_dir = len(os.listdir(base_dir))
    paths = []
    for task_id in range(len_dir):
        path = f'{base_dir}/task_{task_id}/{arr_file_name}'
        paths.append(path)
    if remove_nan:
        ppl_arr = remove_nan_cols(make_arr(paths))
    else:
        ppl_arr = make_arr(paths)
    return ppl_arr

def get_ppl_inc_mat(base_dir, ppl_arr_before, arr_file_name='pt_ppl_results.pkl.npy', remove_nan=True):
    ppl_arr = get_ppl_mat(base_dir,arr_file_name, remove_nan=remove_nan)
    return ppl_arr_before.reshape(1,-1) - ppl_arr

def get_avg_ppl_of_tasks_ss(base_dir, tasks, arr_file_name='pt_ppl_results.pkl.npy'):
    ppl_arr = get_ppl_mat(base_dir,arr_file_name, remove_nan=False)
    indices = deterministic_random_indices(ppl_arr.shape[1], 10000)
    ppl_arr = remove_nan_cols(ppl_arr[:,indices])
    sel_ppl_arr = ppl_arr[tasks]
    return sel_ppl_arr.mean(-1)

def remove_nan_cols(arr):
    arr_one = arr[0]
    return arr[:,~np.isnan(arr_one)]

def remove_nan_rows(arr):
    arr_one = arr[:,0]
    return arr[~np.isnan(arr_one)]

def restore_nan_cols(arr, ref_row):
    ret = np.zeros((arr.shape[0],ref_row.shape[0]))
    ret[:,~np.isnan(ref_row)] = arr
    return ret


def load_base_arr_nonan(path):
    arr = np.load(path)
    return arr[~np.isnan(arr)]

def get_either_path(path, cand_path):
    if os.path.exists(path):
        return path
    return cand_path    

def save_concat_arr(all_arrs, names, save_path):
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    task_info = yaml.safe_load(open('configs/llm/llm_defaults.yaml'))
    arr_cat = np.concatenate(all_arrs, 0)
    meta = {}
    start = 0
    for name, arr in zip(names, all_arrs):
        meta[name] = {
            'start': start,
            'stop': start + arr.shape[0],
            'tasks': task_info.get(f'{name}_tasks')
        }
        start += arr.shape[0]
    np.save(save_path, arr_cat)
    with open(save_path + '.meta','w') as wf:
        json.dump(meta, wf, indent=2)
    return arr_cat, meta

def load_concat_arr(path):
    arr_cat = np.load(path)
    meta_path = path + '.meta'
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
    else:
        meta = None
    return arr_cat, meta

    
def get_ppl_inc_arr(paths, base_path):
    ppl_base_arr_inst = np.load(base_path)
    ppl_base_arr_inst = ppl_base_arr_inst[~np.isnan(ppl_base_arr_inst)]

    ppl_arr = remove_nan_cols(make_arr(paths))
    ppl_inc = ppl_base_arr_inst.reshape(1,-1) - ppl_arr
    return ppl_inc

