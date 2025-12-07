import pickle
import os
import torch
import numpy as np
from utils.config import load_configs

def cosdist(a,b):
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    cosdist_res = np.matmul(a_norm, b_norm.T)
    return cosdist_res

def load_pt_reps(model_size='1b'):
    filename = f'runs/olmo-{model_size}-ft/grad-store/dolma_sample-2k-fix/task_/all_repsNone.pkl'
    with open(filename,'rb') as f:
        obj = pickle.load(f)
    return obj

def load_ocl_reps(task_type, model_size='1b'):
    base_dir = f'runs/olmo-{model_size}-ft/grad-store/{task_type}-fix'
    n_tasks = len(os.listdir(base_dir))
    all_vecs = []

    for task_id in range(n_tasks):
        filename = os.path.join(base_dir, f'task_{task_id}', 'all_repsNone.pkl')
        with open(filename,'rb') as f:
            obj = pickle.load(f)
        avg_grad_vec = obj.mean(0)
        all_vecs.append(avg_grad_vec)
    all_vecs = torch.stack(all_vecs)
    return all_vecs

def load_ocl_task_reps_v2(base_dir, task_cat, tasks):
    all_task_reps = []
    for task in tasks:
        task_vec_dir = os.path.join(base_dir, f'{task_cat}_{task}_train.pkl')
        with open(task_vec_dir,'rb') as f:
            obj = pickle.load(f)
        all_task_reps.append(obj.mean(0))
    all_task_reps = torch.stack(all_task_reps)
    return all_task_reps

def load_pt_task_rep_v2(base_dir):
    pt_vec_dir = f'{base_dir}/dolma_None_train.pkl'
    with open(pt_vec_dir,'rb') as f:
        obj = pickle.load(f)
    return obj

def load_sentence_reps_sentencebert():
    default_config = load_configs('configs/llm/llm_defaults.yaml')
    base_dir = 'stats/olmo-7b/rep_cache'
    flan_reps = load_ocl_task_reps_v2(base_dir, 'flan', default_config.flan_tasks)
    tulu_reps = load_ocl_task_reps_v2(base_dir, 'tulu_train', default_config.tulu_tasks)
    dolly_reps = load_ocl_task_reps_v2(base_dir, 'dolly', default_config.dolly_tasks)
    pt_reps = load_pt_task_rep_v2(base_dir)
    ocl_reps = torch.cat([flan_reps, tulu_reps, dolly_reps])
    return ocl_reps, pt_reps

def load_sentence_reps_openai():
    default_config = load_configs('configs/llm/llm_defaults.yaml')
    base_dir = 'stats/olmo-7b/rep_cache_openai'
    flan_reps = load_ocl_task_reps_v2(base_dir, 'flan', default_config.flan_tasks)
    tulu_reps = load_ocl_task_reps_v2(base_dir, 'tulu_train', default_config.tulu_tasks)
    dolly_reps = load_ocl_task_reps_v2(base_dir, 'dolly', default_config.dolly_tasks)
    pt_reps = load_pt_task_rep_v2(base_dir)
    ocl_reps = torch.cat([flan_reps, tulu_reps, dolly_reps])
    return ocl_reps, pt_reps

def get_cos_dist_mat(ocl_tasks, model_size='1b'):
    ocl_mats = []
    for ocl_task in ocl_tasks:
        ocl_mats.append(load_ocl_reps(ocl_task, model_size))
    ocl_mats = torch.cat(ocl_mats)

    pt_mats = load_pt_reps(model_size)
    ocl_mats, pt_mats = ocl_mats.numpy(), pt_mats.numpy()
    ocl_pt_cosdist = cosdist(ocl_mats, pt_mats)
    return ocl_pt_cosdist

def get_cos_dist_mat_direct(model):
    if model == 'sentencebert':
        ocl_mats, pt_mats = load_sentence_reps_sentencebert()
    elif model == 'openai':
        ocl_mats, pt_mats = load_sentence_reps_openai()
    else:
        raise NotImplementedError
    ocl_mats, pt_mats = ocl_mats.numpy(), pt_mats.numpy()
    ocl_pt_cosdist = cosdist(ocl_mats, pt_mats)
    return ocl_pt_cosdist