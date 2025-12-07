import pickle
import torch
import os
from data_utils.utils import deterministic_random_indices
from post_process.get_ppl_arr import load_concat_arr
import numpy as np

def load_pt_grads(model_size='1b'):
    filename = f'runs/olmo-{model_size}-ft/grad-store/dolma_sample-2k-fix/task_/grad_store.pkl'
    with open(filename,'rb') as f:
        obj = pickle.load(f)
    return obj

def load_ocl_grads(task_type, model_size='1b'):
    base_dir = f'runs/olmo-{model_size}-ft/grad-store/{task_type}'
    n_tasks = len(os.listdir(base_dir))
    all_vecs = []

    for task_id in range(n_tasks):
        filename = os.path.join(base_dir, f'task_{task_id}', 'grad_store.cpu.pkl')
        with open(filename,'rb') as f:
            obj = pickle.load(f)
        avg_grad_vec = obj.mean(0)
        all_vecs.append(avg_grad_vec)
    all_vecs = torch.stack(all_vecs)
    return all_vecs

def load_ocl_grads_7b(task_type, model_size='7b'):
    base_dir = f'runs/olmo-{model_size}-ft/grad-store/{task_type}'
    n_tasks = len(os.listdir(base_dir))
    all_vecs = []

    for task_id in range(n_tasks):
        filename = os.path.join(base_dir, f'task_{task_id}', 'grad_store.pkl')
        with open(filename,'rb') as f:
            obj = pickle.load(f)
        avg_grad_vec = obj.mean(0)
        all_vecs.append(avg_grad_vec)
    all_vecs = torch.stack(all_vecs)
    return all_vecs

def load_diffs(task_type, model_size='1b', postfix=''):
    base_dir = f'runs/olmo-{model_size}-ft/grad-store/{task_type}-fix{postfix}'
    n_tasks = len(os.listdir(base_dir))
    all_vecs = []

    for task_id in range(n_tasks):
        filename = os.path.join(base_dir, f'task_{task_id}', 'param_diff_store.pkl')
        with open(filename,'rb') as f:
            diff_vec = pickle.load(f)
        
        all_vecs.append(diff_vec)
    all_vecs = torch.stack(all_vecs)
    return all_vecs

def sort_to_orig_idx(arr, full_len, ss_len=10000):
    rand_idxs = deterministic_random_indices(full_len, ss_len) # [3,1,5]
    orig_mat = np.zeros((arr.shape[0], full_len))
    orig_mat[:, rand_idxs] = arr
    sort_rand_idxs = np.sort(rand_idxs)
    return orig_mat[:, sort_rand_idxs], sort_rand_idxs

def make_extended_score_mat(score_arr, rand_idxs, orig_len):
    arr = np.full((score_arr.shape[0], orig_len), -np.inf, dtype=np.float64)
    arr[:,rand_idxs] = score_arr
    return arr

def do_min_max_scale(arr):
    return arr / (arr.max() - arr.min())
    
def load_grad_and_diff_mat(model_size='1b', predef_full_len=141816):
    pt_grads = load_pt_grads(model_size)
    flan_grads = load_ocl_grads('flan', model_size)
    tulu_grads = load_ocl_grads('tulu_train', model_size)
    dolly_grads = load_ocl_grads('dolly', model_size)

    flan_diffs = load_diffs('flan', model_size)
    tulu_diffs = load_diffs('tulu_train', model_size)
    dolly_diffs = load_diffs('dolly', model_size)

    ocl_grads = torch.cat([flan_grads, tulu_grads, dolly_grads])
    ocl_diffs = torch.cat([flan_diffs, tulu_diffs, dolly_diffs])
    inner_prod_mean = torch.matmul(ocl_grads, pt_grads.transpose(0,1)) / ocl_grads.size(1)
    inner_prod_diff_mean = torch.matmul(ocl_diffs, pt_grads.transpose(0,1)) / ocl_grads.size(1)

    inner_prod_mean_sorted, rand_idxs = sort_to_orig_idx(inner_prod_mean, orig_len, 10000)
