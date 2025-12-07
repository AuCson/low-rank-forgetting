import json
import os
import numpy as np

method2path = {
    'base': 'runs/olmo-7b-ft-ins/base/lm_eval_results/',
    'ft': 'runs/olmo-7b-ft-ins/dolly-100-full-ft-lr2e-6/task_{task_id}/model_save/lm_eval_results',
    'er': 'runs/olmo-7b-ft-ins/replay/random_mix_0.03125_seed_0/dolly-1k-full-ft-lr2e-6/task_{task_id}/model_save/lm_eval_results',
    'knn': 'runs/olmo-7b-ft-ins/replay/knn_pred_sample_mix_0.03125_seed_0/dolly-1k-full-ft-lr2e-6/task_{task_id}/model_save/lm_eval_results'
}

method2path_baselm = {
    'base': 'runs/olmo-7b-ft/base/lm_eval_results/allenai__OLMo-7B-hf/',
    'ft': 'runs/olmo-7b-ft/dolly-1k-full-ft-lr2e-6/task_{task_id}/model_save/lm_eval_results',
    'er': 'runs/olmo-7b-ft/replay/random_mix_0.03125_seed_0/dolly-1k-full-ft-lr2e-6/task_{task_id}/model_save/lm_eval_results',
    'knn': 'runs/olmo-7b-ft/replay/knn_pred_sample_mix_0.03125_temp_0.1_seed_0/dolly-1k-full-ft-lr2e-6/task_{task_id}/model_save/lm_eval_results'
}

method2path_ins_1k = {
    'base': 'runs/olmo-7b-ft-ins/base/lm_eval_results/',
    'ft': 'runs/olmo-7b-ft-ins/olmo2_sft_mix-1k-full-ft-lr2e-6/task_{task_id}/model_save/lm_eval_results_preds',
    'er': 'runs/olmo-7b-ft-ins/replay_long_flan/random_mix_0.03125/olmo2_sft_mix-1k-full-ft-lr2e-6/task_{task_id}/model_save/lm_eval_results_preds',
    'svd': 'runs/olmo-7b-ft-ins/replay_long_flan/rep_additive_pred_sample_mix_0.03125_temp_1.0/olmo2_sft_mix-1k-full-ft-lr2e-6/task_{task_id}/model_save/lm_eval_results_preds'
}


# method2path_ins_long = {
#     'base': 
# }

def locate_result_file(base_dir):
    json_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    json_files.sort()
    with open(json_files[-1]) as f:
        print(json_files[-1])
        data = json.load(f)
    return data

def summarize_results(data):
    results = {}
    met_names = {}
    for task_cat, task_names in data['group_subtasks'].items():
        if task_cat != 'leaderboard':
            results[task_cat] = []
            met_names[task_cat] = []
            for name in task_names:
                res_dict = data['results'][name]
                for k, v in res_dict.items():
                    if k != 'alias' and 'std' not in k:
                        results[task_cat].append(v)
                        met_names[task_cat].append(k)
    
    results['if_eval_prompt'] = [data['results']['leaderboard_ifeval']['prompt_level_strict_acc,none']]
    results['if_eval_inst'] = [data['results']['leaderboard_ifeval']['inst_level_strict_acc,none']]
    results['mmlu_pro"'] = [data['results']['leaderboard_mmlu_pro']['acc,none']]
    return results, met_names


def summarize_results_baselm(data):
    results = {}
    for task in data['results'].keys():
        if 'acc_norm,none' in data['results'][task]:
            results[task] = data['results'][task]['acc_norm,none']
        else:
            results[task] = data['results'][task]['acc,none']
    return results, None

def avg_eval_scores(results):
    return {k: np.mean(v) for k,v in results.items() if all([(type(x) in [float, int]) for x in v])}

def avg_list_of_dict(l_dics):
    res = {}
    for k in l_dics[0]:
        res[k] = np.mean([dic[k] for dic in l_dics])
    return res

if __name__ == '__main__':
    pass