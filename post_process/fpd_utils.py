import pickle
from utils.config import load_configs

def create_flan_fpd_in_domain_split(full_mat, meta, pt_task_name='dolma'):
    start = meta['flan']['start']
    assert start == 0
    stop = meta['flan']['stop']
    test_idxs = [6, 65, 18, 8, 60, 13, 37, 22, 30, 19, 64, 50, 25, 31, 32, 61, 16, 5, 53, 49]
    train_idxs = [x for x in range(0, stop) if x not in set(test_idxs)]

    train_mat = full_mat[train_idxs]
    test_mat = full_mat[test_idxs]

    train_ocl_task_info = []
    for task_idx in train_idxs:
        train_ocl_task_info.append({
            'cat': 'flan',
            'name': meta['flan']['tasks'][task_idx],
            'split': 'train'
        })
    
    test_ocl_task_info = []
    for task_idx in test_idxs:
        test_ocl_task_info.append({
            'cat': 'flan',
            'name': meta['flan']['tasks'][task_idx],
            'split': 'train'
        })

    pt_task_info = {
        'cat': pt_task_name,
        'names': None,
        'split': 'train'
    }
    
    fpd_split = {
        'train_ocl_idxs': train_idxs,
        'test_ocl_idxs': test_idxs,
        'train_mat': train_mat,
        'test_mat': test_mat,
        'train_ocl_task_info': train_ocl_task_info,
        'test_ocl_task_info': test_ocl_task_info,
        'pt_task_info': pt_task_info
    }

    return fpd_split


def create_fpd_ood_split_ood(full_mat, meta, ood_task_name, pt_task_name='dolma'):
    start = meta['flan']['start']
    assert start == 0
    stop = meta['flan']['stop']
    id_test_idxs = [6, 65, 18, 8, 60, 13, 37, 22, 30, 19, 64, 50, 25, 31, 32, 61, 16, 5, 53, 49]
    train_idxs = [x for x in range(0, stop) if x not in set(id_test_idxs)]

    train_mat = full_mat[train_idxs]
    
    ood_start = meta[ood_task_name]['start']
    ood_stop = meta[ood_task_name]['stop']
    
    ood_test_idxs = [_ for _ in range(ood_stop - ood_start)]

    train_ocl_task_info = []
    for task_idx in train_idxs:
        train_ocl_task_info.append({
            'cat': 'flan',
            'name': meta['flan']['tasks'][task_idx],
            'split': 'train'
        })

    test_mat = full_mat[ood_start: ood_stop]

    print(ood_start, ood_stop)
    
    test_ocl_task_info = []
    for task_idx in ood_test_idxs:
        test_ocl_task_info.append({
            'cat': 'tulu_train' if ood_task_name == 'tulu' else ood_task_name,
            'name': meta[ood_task_name]['tasks'][task_idx],
            'split': 'train'
        })

    pt_task_info = {
        'cat': pt_task_name,
        'names': None,
        'split': 'train'
    }

    fpd_split = {
        'train_ocl_idxs': train_idxs,
        'test_ocl_idxs': ood_test_idxs,
        'train_mat': train_mat,
        'test_mat': test_mat,
        'train_ocl_task_info': train_ocl_task_info,
        'test_ocl_task_info': test_ocl_task_info,
        'pt_task_info': pt_task_info
    }
    return fpd_split

def create_fpd_split_for_ins(full_mat, meta, ood_task_name, in_domain_split_path):
    with open(in_domain_split_path, 'rb') as f:
        in_domain_split = pickle.load(f)
    
    task_start = meta[ood_task_name]['start']
    task_stop = meta[ood_task_name]['stop']
    task_num = task_stop - task_start
    
    test_ocl_task_info = []
    for task_idx in range(0, task_num):
        #print(task_idx)
        test_ocl_task_info.append({
            'cat': ood_task_name,
            'name': meta[ood_task_name]['tasks'][task_idx],
            'split': 'train'
        })

    fpd_split = {
        'train_ocl_idxs': in_domain_split['train_ocl_idxs'],
        'test_ocl_idxs': [_ for _ in range(task_num)],
        'train_mat': in_domain_split['train_mat'],
        'test_mat': full_mat[task_start:task_stop],
        'train_ocl_task_info': in_domain_split['train_ocl_task_info'],
        'test_ocl_task_info': test_ocl_task_info,
        'pt_task_info': in_domain_split['pt_task_info']
    }
    
    return fpd_split

def create_fpd_split_for_ins(arr):
    assert arr.shape[0] == 18
    test_ocl_idxs = [0,4,8,12,16]
    train_ocl_idxs = [x for x in range(18) if x not in test_ocl_idxs]
    
    config = load_configs('configs/llm/llm_defaults.yaml')
    
    train_ocl_task_info, test_ocl_task_info = [], []
    for task_id in train_ocl_idxs:
        train_ocl_task_info.append({
            'cat': 'olmo2_sft_mix',
            'name': config.olmo2_sft_mix_tasks[task_id],
            'split': 'train'
        })
    for task_id in test_ocl_idxs:
        test_ocl_task_info.append({
            'cat': 'olmo2_sft_mix',
            'name': config.olmo2_sft_mix_tasks[task_id],
            'split': 'test'
        })

    fpd_split = {
        'train_ocl_idxs': train_ocl_idxs,
        'test_ocl_idxs': test_ocl_idxs,
        'train_mat': arr[train_ocl_idxs],
        'test_mat': arr[test_ocl_idxs],
        'train_ocl_task_info': train_ocl_task_info,
        'test_ocl_task_info': test_ocl_task_info,
        'pt_task_info': {'cat': 'tulu', 'names': None, 'split': 'train'}
    }

    return fpd_split
    
