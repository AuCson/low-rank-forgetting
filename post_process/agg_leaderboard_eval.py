import os
import re
import json
import numpy as np
from collections import OrderedDict

PTLM_TASKS = 'arc_easy,arc_challenge,boolq,hellaswag,openbookqa,piqa,sciq,winogrande'.split(',')
INS_LM_TASKS = ['leaderboard_gpqa_diamond', 'leaderboard_gpqa_extended', 'leaderboard_gpqa_main', 'leaderboard_ifeval', 'leaderboard_musr_murder_mysteries', 'leaderboard_musr_object_placements' ,'leaderboard_musr_team_allocation']

def extract_metrics(dic):
    if 'acc_norm' in dic:
        return dic['acc_norm']
    elif 'acc' in dic:
        return dic['acc']
    elif 'prompt_level_strict_acc' in dic:
        return dic['prompt_level_strict_acc']
    else:
        raise KeyError(dic)

def default_eval_result_dir_func(model, task_type=None, task_id=None):
    if task_type is None:
        base_dir = 'runs/{model}/base/lm_eval_results_preds'.format(model=model)
        subdir_name = os.listdir(base_dir)[0]
        return os.path.join(base_dir, subdir_name)
    else:
        if model == 'olmo-7b-ft' and task_type == 'tulu_train':
            base_dir = 'runs/{model}/{task_type}-10k-full-ft-lr2e-6'.format(model=model, task_type=task_type)
        elif model == 'olmo-7b-ft-ins':
            base_dir = 'runs/{model}/{task_type}-100-full-ft-lr2e-6'.format(model=model, task_type=task_type)
        else:
            base_dir = 'runs/{model}/{task_type}-1k-full-ft-lr2e-6'.format(model=model, task_type=task_type)

        if task_id is None:
            return base_dir
        else:
            model_save_folder = 'model_save' 
            if model == 'olmo-7b-ft' and task_type == 'tulu_train':
                model_save_folder = 'model_save_1k'
            subdir_name = os.listdir(os.path.join(base_dir, 'task_{task_id}/{model_save_folder}/lm_eval_results_preds/'.format(task_id=task_id, model_save_folder=model_save_folder)))[0]
            return os.path.join(base_dir, 'task_{task_id}/{model_save_folder}/lm_eval_results_preds/{subdir_name}'.format(model=model, task_type=task_type, 
                                                                                                                          model_save_folder=model_save_folder,
                                                                                                                          subdir_name=subdir_name, task_id=task_id))

def get_eval_task_name_and_paths(base_dir):
    patt = re.compile('samples_(\w+)_.*.jsonl')
    files = os.listdir(base_dir)
    ret = {}
    for file in files:
        #print(file)
        matchobj = patt.match(file)
        if matchobj:
            task_name = matchobj.group(1)
            if task_name in PTLM_TASKS or task_name in INS_LM_TASKS:
                ret[task_name] = os.path.join(base_dir, file)
    return ret

def get_single_eval_task_scores(path):
    with open(path) as f:
        lines = f.readlines()
    all_data = [json.loads(x) for x in lines]
    metric_scores = [extract_metrics(dic) for dic in all_data]
    return metric_scores

def get_all_scores(model, task_type):
    path = default_eval_result_dir_func(model, task_type)
    task_num = len(os.listdir(path))
    all_ocl_task_eval_task_scores = []

    for task_id in range(task_num):
        all_eval_task_scores = get_single_ocl_task_all_eval_task_results(model, task_type, task_id)
        all_ocl_task_eval_task_scores.append(all_eval_task_scores)

    return all_ocl_task_eval_task_scores

def get_base_scores(model):
    eval_results_dir = default_eval_result_dir_func(model)
    eval_taskname_paths = get_eval_task_name_and_paths(eval_results_dir)
    all_eval_task_scores = {}
    for eval_task, eval_result_path in eval_taskname_paths.items():
        eval_task_scores = get_single_eval_task_scores(eval_result_path)
        all_eval_task_scores[eval_task] = eval_task_scores
    return all_eval_task_scores


def compute_offset(sample_eval_task_scores):
    current_offset = 0
    offsets= {}
    for eval_task, eval_task_scores in OrderedDict(sorted(sample_eval_task_scores.items())).items():
        offsets[eval_task] = (current_offset, current_offset + len(eval_task_scores))
        current_offset += len(eval_task_scores)

    return offsets

def get_score_array_1d(all_eval_task_scores, offsets):
    if offsets is None:
        offsets = compute_offset(all_eval_task_scores)
    maxn = max([v[1] for v in offsets.values()])
    arr = np.zeros(maxn, dtype=np.float64)
    for eval_task, eval_task_scores in all_eval_task_scores.items():
        start, stop = offsets[eval_task]
        #print(eval_task_scores)
        arr[start:stop] = np.array(eval_task_scores).astype(arr.dtype)
    return arr, offsets

def make_score_array_from_extracted_scores(all_ocl_task_eval_task_scores, offsets=None):
    n_ocl_tasks = len(all_ocl_task_eval_task_scores)

    if offsets is None:
        #offsets = {}
        sample_eval_task_scores = all_ocl_task_eval_task_scores[0]
        offsets = compute_offset(sample_eval_task_scores)

    current_offset = max([v[1] for v in offsets.values()])
        
    score_arr = np.zeros((n_ocl_tasks, current_offset), dtype=np.float64)
    for task_id, all_eval_task_scores in enumerate(all_ocl_task_eval_task_scores):
        #for eval_task, eval_task_scores in all_eval_task_scores.items():
        #    start, stop = offsets[eval_task]
        #    score_arr[task_id, start:stop] = np.array(eval_task_scores).astype(score_arr.dtype)
        arr, _ = get_score_array_1d(all_eval_task_scores, offsets)

        score_arr[task_id,:] = arr
    return score_arr, offsets

def get_single_ocl_task_all_eval_task_results(model, task_type, task_id):
    eval_results_dir = default_eval_result_dir_func(model, task_type, task_id)
    eval_taskname_paths = get_eval_task_name_and_paths(eval_results_dir)
    all_eval_task_scores = {}
    eval_tasks = []
    for eval_task, eval_result_path in eval_taskname_paths.items():
        eval_task_scores = get_single_eval_task_scores(eval_result_path)
        all_eval_task_scores[eval_task] = eval_task_scores

    return all_eval_task_scores

def summarize_scores(model, task_type, offsets=None):
    all_ocl_task_eval_task_scores = get_all_scores(model, task_type)
    score_arr, offsets = make_score_array_from_extracted_scores(all_ocl_task_eval_task_scores, offsets)
    return score_arr, offsets

def summarize_base_scores(model, offsets=None):
    base_scores = get_base_scores(model)
    score_arr_1d, offsets = get_score_array_1d(base_scores, offsets)
    return score_arr_1d, offsets