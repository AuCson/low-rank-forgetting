import numpy as np
from data_utils.utils import deterministic_random_indices
from utils.config import load_configs
from data_utils import load_raw_ds
import re
import string
import pickle
import os
from data_utils.squad_f1 import compute_f1

def filter_nan(arr):
    return arr[~np.isnan(arr)]

def get_base_ppl_arr(model_name, return_ss=False):
    model_to_file = {
        'olmo-7b': 'olmo-7b_base_ppl.npy',
        'olmo-1b': 'olmo-1b_base_ppl.npy',
        'olmo-7b-ins': 'olmo-7b-ins_base_ppl.npy',
        'mpt-7b':      'mpt-7b_base_ppl.npy',
        'olmo2-7b':    'olmo2-7b_base_ppl.npy',
        'olmo2-13b':   'olmo2-13b_base_ppl.npy',
        'pythia-1b':   'pythia-1b_base_ppl.npy',
        'pythia-7b':   'pythia-7b_base_ppl.npy',
        'pythia-12b':  'pythia-12b_base_ppl.npy',
    }

    if model_name not in model_to_file:
        raise NotImplementedError(f"Model name {model_name} not supported.")

    ppl_base_arr = np.load('stats/base_ppl_files/' + model_to_file[model_name])

    if return_ss:
        rand_idxs = deterministic_random_indices(ppl_base_arr.shape[0], 10000)
        ppl_base_arr = ppl_base_arr[rand_idxs]

    return np.abs(ppl_base_arr)

def get_ft_ppl_arr(model_name, task_cat, task_id, return_ss=False):
    if model_name in ['7b-10k','olmo-7b-10k']:
        ppl_arr =  np.load(f'runs/stats/stats-olmo-7b-ft/{task_cat}-1k-lr2e-6/task_{task_id}/pt_ppl_results.pkl.npy')
        #rand_idxs = deterministic_random_indices(ppl_arr.shape[0], 10000)
        #ppl_arr_ss = ppl_arr[rand_idxs]
    elif model_name in ['1b-10k','olmo-1b-10k']:
        ppl_arr =  np.load(f'runs/stats/stats-olmo-1b-ft/{task_cat}-1k-lr2e-6/task_{task_id}/pt_ppl_results.pkl.npy')
        #rand_idxs = deterministic_random_indices(ppl_arr.shape[0], 10000)
        #ppl_arr_ss = ppl_arr[rand_idxs]
    elif model_name == '7b-ins':
        ppl_arr = np.load(f'runs/stats/stats-olmo-7b-ins-ft-test/ft/{task_cat}/task_{task_id}/pt_ppl_results.pkl.npy')
    elif model_name == 'olmo2-7b':
        ppl_arr = np.load(f'runs/stats/stats-olmo2-1124-7b-ft/{task_cat}-1k-lr2e-6/task_{task_id}/pt_ppl_results.pkl.npy')
    elif model_name == 'olmo2-13b':
        ppl_arr = np.load(f'runs/stats/stats-olmo2-1124-13b-ft/{task_cat}-1k-lr2e-6/task_{task_id}/pt_ppl_results.pkl.npy')
    elif model_name == 'pythia-1b':
        ppl_arr = np.load(f'runs/stats/stats-pythia/1b/{task_cat}-1k-lr2e-6/task_{task_id}/pt_ppl_results.pkl.npy')
    elif model_name == 'pythia-7b':
        ppl_arr = np.load(f'runs/stats/stats-pythia/6.9b/{task_cat}-1k-lr2e-6/task_{task_id}/pt_ppl_results.pkl.npy')
    elif model_name == 'pythia-12b':
        ppl_arr = np.load(f'runs/stats/stats-pythia/12b/{task_cat}-1k-lr2e-6/task_{task_id}/pt_ppl_results.pkl.npy') 
    if return_ss:
        rand_idxs = deterministic_random_indices(ppl_arr.shape[0], 10000)
        ppl_arr = ppl_arr[rand_idxs]

    return np.abs(ppl_arr)
    

def get_ft_ppl_inc_pos(model_name, task_cat, task_id):
    ft_ppl_arr = get_ft_ppl_arr(model_name, task_cat, task_id)
    base_ppl_arr = get_base_ppl_arr(model_name)
    ft_ppl_inc_arr = ft_ppl_arr - base_ppl_arr
    return ft_ppl_inc_arr > 0

def compute_ppl_where_ft_inc(model_name, task_cat, task_id, path):
    ppl_arr = np.abs(np.load(path))
    ppl_inc_pos = get_ft_ppl_inc_pos(model_name, task_cat, task_id)
    ppl_arr_filter = ppl_arr[ppl_inc_pos]
    return filter_nan(ppl_arr_filter)


def compute_ppl(path, reduce='mean'):
    ppl_arr = np.abs(np.load(path))
    #base_ppl_arr = get_base_ppl_arr(model_name)
    #ppl_inc = ppl_arr - base_ppl_arr
    #print(filter_nan(ppl_arr).mean(), filter_nan(base_ppl_arr).mean())
    if reduce == 'mean':
        return filter_nan(ppl_arr).mean()
    else:
        return ppl_arr

def compute_ppl_inc(model_name,  path, reduce='mean'):
    ppl_arr = np.abs(np.load(path))
    base_ppl_arr = get_base_ppl_arr(model_name)
    ppl_inc = ppl_arr - base_ppl_arr
    #print(filter_nan(ppl_arr).mean(), filter_nan(base_ppl_arr).mean())
    if reduce == 'mean':
        return filter_nan(ppl_inc).mean()
    else:
        return ppl_inc

def compute_reduced_forgetting_perc_verbose(model_name, task_cat, task_id, path):
    ppl_inc = compute_ppl_inc(model_name, path)
    ft_ppl = get_ft_ppl_arr(model_name, task_cat, task_id)
    #print(f'Ft ppl {ft_ppl}')
    base_ppl = get_base_ppl_arr(model_name)
    #print(f'base ppl {base_ppl}')
    ft_ppl_inc = ft_ppl - base_ppl

    #print(filter_nan(ft_ppl_inc).mean())
    fgt_perc = ppl_inc / filter_nan(ft_ppl_inc).mean()
    
    return fgt_perc, ppl_inc, ft_ppl_inc, base_ppl
    
def compute_ppl(path, reduce='mean'):
    ppl_arr = np.abs(np.load(path))
    #base_ppl_arr = get_base_ppl_arr(model_name)
    #ppl_inc = ppl_arr - base_ppl_arr
    #print(filter_nan(ppl_arr).mean(), filter_nan(base_ppl_arr).mean())
    if reduce == 'mean':
        return filter_nan(ppl_arr).mean()
    else:
        return ppl_arr
    
def compute_ppl_inc_ft_positive_only(model_name, task_cat, task_id, path, reduce='mean'):
    ppl_arr = np.abs(np.load(path))
    
    ft_ppl_arr = get_ft_ppl_arr(model_name, task_cat, task_id)
    base_ppl_arr = get_base_ppl_arr(model_name)
    
    ppl_inc = ppl_arr - base_ppl_arr
    ft_ppl_inc = ft_ppl_arr - base_ppl_arr

    filter_ppl_inc = ppl_inc[ft_ppl_inc > 0]
    if reduce == 'mean':
        return filter_nan(filter_ppl_inc).mean()
    else:
        return filter_ppl_inc


def compute_ppl_ft_positive_only(model_name, task_cat, task_id, path, reduce='mean'):
    ppl_arr = np.abs(np.load(path))
    
    ft_ppl_arr = get_ft_ppl_arr(model_name, task_cat, task_id)
    base_ppl_arr = get_base_ppl_arr(model_name)
    
    #ppl_inc = ppl_arr - base_ppl_arr
    ft_ppl_inc = ft_ppl_arr - base_ppl_arr

    filter_ppl = ppl_arr[ft_ppl_inc > 0]
    if reduce == 'mean':
        return filter_nan(filter_ppl).mean()
    else:
        return filter_ppl



def compute_relative_forgetting(model_name, arr_or_path):
    if type(arr_or_path) is str:
        arr = np.load(arr_or_path)
    else:
        arr = arr_or_path
    arr = np.abs(arr)
    base_ppl_arr = np.abs(get_base_ppl_arr(model_name))

    ppl_inc = arr - base_ppl_arr
    ppl_inc_filt = ppl_inc[~np.isnan(ppl_inc)]
    ppl_inc_nz = ppl_inc_filt.copy()

    #ppl_inc_nz[ppl_inc_nz < 0] = 0

    #print(ppl_inc_nz.mean(), ppl_inc_filt.mean())

    return ppl_inc_nz.mean(0)

def pretty_print(*numbers):
    pt_numbers = ['{:.4f}'.format(x) if x is not None else None for x in numbers]
    print(*pt_numbers)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
  return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def evaluate_flan_em_arr(lm_outputs, split='train'):
    def extract_before_two_blank_lines(text):
        return text.split('\n\n')[0]

    if split == 'train':
        config = load_configs('configs/defaults.yaml', 'configs/llm/ocl_ins/stat_7b_on_flan_v2_train.yaml')
        ds = load_raw_ds('train', config, None, 'tulu')
    elif split == 'test':
        config = load_configs('configs/defaults.yaml', 'configs/llm/ocl_ins/stat_7b_on_flan_v2.yaml')
        ds = load_raw_ds('train', config, None, 'tulu')
    gts = [x[1] for x in ds]
    if type(lm_outputs[0]) is dict:
        preds = [x['outputs'][0]['text'] for x in lm_outputs]
    else:
        preds = [x.outputs[0].text for x in lm_outputs]
    em_scores = []

    assert len(gts) == len(preds)

    for gt, pred in zip(gts, preds):
        pred_extracted = extract_before_two_blank_lines(pred)
        score = compute_exact(gt, pred_extracted)
        em_scores.append(score)
    em_scores = np.array(em_scores, dtype=int)
    return em_scores


def evaluate_flan_f1_arr(lm_outputs, split='train'):
    def extract_before_two_blank_lines(text):
        return text.split('\n\n')[0]

    if type(lm_outputs) is str:
        with open(lm_outputs, 'rb') as f:
            lm_outputs = pickle.load(f)

    if split == 'train':
        config = load_configs('configs/defaults.yaml', 'configs/llm/ocl_ins/stat_7b_on_flan_v2_1k_train.yaml')
        ds = load_raw_ds('train', config, None, 'tulu')
    elif split == 'test':
        config = load_configs('configs/defaults.yaml', 'configs/llm/ocl_ins/stat_7b_on_flan_v2_1k.yaml')
        ds = load_raw_ds('train', config, None, 'tulu')
    gts = [x[1] for x in ds]
    preds = [x.outputs[0].text for x in lm_outputs]
    f1_scores = []

    assert len(gts) == len(preds)

    for gt, pred in zip(gts, preds):
        pred_extracted = extract_before_two_blank_lines(pred)
        score = compute_f1(gt, pred_extracted)
        f1_scores.append(score)
    f1_scores = np.array(f1_scores, dtype=int)
    return f1_scores

def get_7b_ins_base_output_res(split='train'):
    if split == 'train':
        base_res_path = 'runs/stats/stats-olmo-7b-ins-ft-1k/flan-v2/ft/olmo2_sft_mix/task_0/pt-base_output_results.pkl'
    elif split == 'test':
        base_res_path = 'runs/stats/stats-olmo-7b-ins-ft-test-1k/flan-v2/ft/olmo2_sft_mix/task_0/pt-base_output_results.pkl'
    else:
        raise ValueError(split)
    with open(base_res_path,'rb') as f:
        obj = pickle.load(f)
    return obj

def get_all_7b_ins_res(filter_tasks=None, replay_method=None, trunc=False, temp=None, max_task=-1, peft=False):
    task_cat2num = {
        'mmlu': 57,
        'bbh': 27,
        'truthful_qa': 32,
        'dolly': 8
    }
    if filter_tasks is not None:
        task_cat2num = {k: v for k,v in task_cat2num.items() if k == filter_tasks}
    all_outputs = []
    for task, task_num in task_cat2num.items():
        print(task)
        for task_id in range(task_num):
            if task_id == max_task:
                break
            if replay_method is None:
                if peft:
                    output_path = f'runs/stats/stats-olmo-7b-ins-peft-test/flan-v2/ft/{task}/task_{task_id}/pt_output_results.pkl'
                else:
                    output_path = f'runs/stats/stats-olmo-7b-ins-ft-test/flan-v2/ft/{task}/task_{task_id}/pt_output_results.pkl'
            else:
                if not trunc:
                    output_path = f'runs/stats/stats-olmo-7b-ins-ft-test/replay/flan_v2/{replay_method}_mix_0.125/{task}/task_{task_id}/pt_output_results.pkl'
                else:
                    output_path = f'runs/stats/stats-olmo-7b-ins-ft-test/replay/flan_v2_truncate_replay/{replay_method}_mix_0.125/{task}/task_{task_id}/pt_output_results.pkl'
                    if temp is not None:
                        output_path =  f'runs/stats/stats-olmo-7b-ins-ft-test/replay/flan_v2_truncate_replay_t{temp}/{replay_method}_mix_0.125/{task}/task_{task_id}/pt_output_results.pkl'
            with open(output_path, 'rb')  as f:
                output_obj = pickle.load(f)
            all_outputs.append(output_obj)    
    return all_outputs    

def get_7b_ins_fgt_arr():
    base_output = get_7b_ins_base_output_res()
    base_em_arr = evaluate_flan_em_arr(base_output)
    all_outputs = get_all_7b_ins_res()
    after_em_arrs = np.array([evaluate_flan_em_arr(x) for x in all_outputs])
    base_correct_mask = base_em_arr == 1
    after_em_arrs_bc = after_em_arrs[:,base_correct_mask]
    fgt = 1 - after_em_arrs_bc
    return fgt

def get_ins_em_by_dirname(base_dir, task_ids, split):
    all_em_arrs = []
    for task_id in task_ids:
        filename = os.path.join(base_dir, f'task_{task_id}/pt_output_results.pkl')
        with open(filename,'rb') as f:
            outputs = pickle.load(f)
        em_scores = evaluate_flan_em_arr(outputs, split)
        all_em_arrs.append(em_scores)
    all_em_arrs = np.stack(all_em_arrs)
    return all_em_arrs

def get_ins_f1_by_dirname(base_dir, task_ids, split):
    all_f1_arrs = []
    for task_id in task_ids:
        filename = os.path.join(base_dir, f'task_{task_id}/pt_output_results.pkl')
        with open(filename,'rb') as f:
            outputs = pickle.load(f)
        f1_scores = evaluate_flan_f1_arr(outputs, split)
        all_f1_arrs.append(f1_scores)
    all_f1_arrs = np.stack(all_f1_arrs)
    return all_f1_arrs


def get_ins_base_em(split):
    outputs = get_7b_ins_base_output_res(split)
    em_scores = evaluate_flan_em_arr(outputs, split)
    return em_scores


def save_em_forgetting(base_dir, task_ids, split, save_abs=False):
    base_em = get_ins_base_em(split)
    ft_test_ems = get_ins_em_by_dirname(base_dir, task_ids, split)
    
    fgt = base_em.reshape((1,-1)) - ft_test_ems

    for i, task_id in enumerate(task_ids):
        arr = fgt[i]
        if save_abs:
            np.save(os.path.join(base_dir, f'task_{task_id}/pt_absfgt_em_arr.npy'), np.where(arr == 1, 1, 0))
        else:
            np.save(os.path.join(base_dir, f'task_{task_id}/pt_fgt_em_arr.npy'), arr)
        
def save_em_single(root_dir, split, output_filename='pt_output_results.pkl'):
    #base_em = get_ins_base_em(split)
    with open(os.path.join(root_dir, output_filename),'rb') as f:
        outputs = pickle.load(f)
    em_scores = evaluate_flan_em_arr(outputs, split)
   
    np.save(os.path.join(root_dir, 'pt_em_arr.npy'), em_scores)



"""
To debug:

with open('runs/stats/stats-olmo-7b-ins-ft-test-1k/flan-v2/ft/olmo2_sft_mix/task_0/pt_output_results.pkl','rb') as f:
    obj = pickle.load(f)
config = load_configs('configs/defaults.yaml', 'configs/llm/ocl_ins/stat_7b_on_flan_v2_1k.yaml')
test_ds = load_raw_ds('train', config, None, 'tulu')

"""