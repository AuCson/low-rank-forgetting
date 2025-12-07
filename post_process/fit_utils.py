from surprise import Dataset, SVD, BaselineOnly, NMF, SVDpp, SlopeOne, KNNBaseline, KNNBasic
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise.reader import Reader
import random
import numpy as np
from sklearn.metrics import ndcg_score, f1_score, mean_squared_error
import pandas as pd
import json
import argparse
import sys
from .my_mf import LogisticMF, sigmoid

def fnorm_error(target, est):
    return ((target - est) ** 2).mean()

def r2_score(target, est):
    ssr = fnorm_error(target, est)
    sst = ((target - target.mean()) ** 2).mean()
    return 1 - ssr / sst

def get_recons(u, s, vh, rank, accum=False):
    if accum:
        start = 0
    else:
        start = rank
    stop = rank + 1
    ret = np.matmul(np.matmul(u[:,start:stop],np.diag(s[start:stop])), vh[start:stop,:])
        
    return ret

def get_uonly_recons(arr):
    umean = arr.mean(1)
    preds = np.copy(arr)
    
    for i in range(arr.shape[0]):
        preds[i] = umean[i]
    score =r2_score(arr, preds)
    return score

def get_ionly_recons(arr):
    umean = arr.mean(0)
    preds = np.copy(arr)
    
    for j in range(arr.shape[1]):
        preds[:,j] = umean[j]
    score =r2_score(arr, preds)
    return score

def get_baseline_recons(arr, full_ds, bsl_options):
    algo_svd = BaselineOnly(bsl_options=bsl_options)
    algo_svd.fit(full_ds)
    all_scores = []
    for ocl_idx in range(arr.shape[0]):
        all_scores.append(get_preds(algo_svd, ocl_idx, arr.shape[1]))
    recons_svdsp = np.stack(all_scores)
    
    score =r2_score(arr, recons_svdsp)
    return score

def get_svd_recons(arr, full_ds, svd_options):
    if svd_options is None:
        svd_options = {'n_epochs': 100, 'reg_all': 0., 'lr_all': 0.005}
    algo_svd = SVD(**svd_options)
    algo_svd.fit(full_ds)
    all_scores = []
    for ocl_idx in range(arr.shape[0]):
        all_scores.append(get_preds(algo_svd, ocl_idx, arr.shape[1]))
    recons_svdsp = np.stack(all_scores)
    score = r2_score(arr, recons_svdsp)
    return score, arr, recons_svdsp

def get_svd_recons_scipy(arr, full_ds, svd_options):
    u, s, vh = np.linalg.svd(arr)
    start = 0
    stop = svd_options['n_factors']
    recons = np.matmul(np.matmul(u[:,start:stop],np.diag(s[start:stop])), vh[start:stop,:])
    score =r2_score(arr, recons)
    return score, arr, recons

def get_logistic_mf_recons(arr, rank, logit_only=False, **hparams):
    W, H = LogisticMF(arr, rank,  **hparams)
    if logit_only:
        recons_lgmf = np.dot(W,H)
    else:
        recons_lgmf = sigmoid(np.dot(W,H))
    return recons_lgmf

def get_logistic_mf_recons_progressive(arr, max_rank, logit_only=False, **hparams):
    all_recons = []
    for rank in range(1,max_rank+1):
        print(f'Fitting rank {rank}')
        recons = get_logistic_mf_recons(arr, rank, logit_only, **hparams)
        all_recons.append(recons)
    return all_recons

def get_preds(model, ocl_idx, max_pt):
    scores = []
    for pt_idx in range(max_pt):
        scores.append(model.predict(ocl_idx, pt_idx).est)
    scores = np.array(scores)
    return scores

def mat_to_interaction_df(mat):
    data = []
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):

            data.append([i, j, mat[i, j]])

    df = pd.DataFrame(data)
    return df

def get_interaction_ds(mat):
    reader = Reader(rating_scale=(-100, 100))
    full_df = mat_to_interaction_df(mat)
    ds = Dataset.load_from_df(full_df, reader)
    full_ds = ds.build_full_trainset()
    return full_ds

def examine_all_fit(arr):
    arr_ds = get_interaction_ds(arr)
    additive_score = get_baseline_recons(arr, arr_ds,{'n_epochs': 100, 'method': 'als'} )
    svd_score, _, _ = get_svd_recons_scipy(arr, arr_ds, {'n_epochs': 100, 'reg_all': 0., 'lr_all': 0.005, 'n_factors': 1, 'biased': False})

    item_only_score = get_ionly_recons(arr)
    item_only_score = get_ionly_recons(arr)
    user_only_score = get_uonly_recons(arr)
    return {
        'additive': additive_score,
        'svd': svd_score,
        #'svd2': svd_score2,
        'item_only': item_only_score,
        'user_only': user_only_score
    }

def examine_svd_fits(arr):
    res = {}
    u, s, vh = np.linalg.svd(arr)

    for rank in range(1,10):
        start, stop = 0, rank
        recons = np.matmul(np.matmul(u[:,start:stop],np.diag(s[start:stop])), vh[start:stop,:])
        score = r2_score(arr, recons)
        res[rank] = score
    return res

def debug_scale_first_row(arr, scale=10.):
    arr = np.copy(arr)
    arr[0] = arr[0] * scale
    return arr

def normalize_rows(arr):
    arr = np.copy(arr)
    arr_norm = arr / arr.sum(1).reshape((-1,1))

    arr_norm = arr_norm * (arr.mean() / arr_norm.mean())

    return arr_norm



if __name__ == '__main__':
    from post_process.get_ppl_arr import load_concat_arr
    from data_utils.utils import deterministic_random_indices

    job = sys.argv[1]

    ptlm_exps = {
        'OLMo-1b': 'stats/olmo-1b/ftd-ppl-inc.npy',
        'OLMo-7b': 'stats/olmo-7b/ftd-ppl-inc.npy',
        'OLMo-7b-LoRA': 'stats/side_exps/olmo-7b-peft/ftd-ppl-inc.npy',
        'MPT-7B': 'stats/side_exps/mpt-7b/ftd-ppl-inc.npy',
        'OLMo-7B-Instruct': 'stats/olmo-7b-ins/mbtdo2-ppl-inc.npy',
        'OLMo-7B-Instruct-LoRA': 'stats/side_exps/olmo-7b-ins-peft/mbtd-ppl-inc.npy',
        'OLMo2-7B': 'stats/side_exps/olmo2-1124-7b/ftd-ppl-inc-full.npy',
        'OLMo2-13B': 'stats/side_exps/olmo2-1124-13b/td-ppl-inc-full.npy',
        'Pythia-1B': 'stats/side_exps/pythia-1b/td-ppl-inc.npy',
        'Pythia-3B': 'stats/side_exps/pythia-3b/td-ppl-inc.npy',
        'Pythia-6.9B': 'stats/side_exps/pythia-7b/td-ppl-inc.npy',
        'Pythia-12B': 'stats/side_exps/pythia-12b/td-ppl-inc.npy',
    }
    
    out_name = {
        'all': 'stats/goodness_of_fit_19_v2.json',
        'recons': 'stats/recons_19_v2.json',
        'avg_fgt': 'stats/avg_fgt_v2.json',
        'std_fgt': 'stats/std_fgt_v2.json',
        'non_neg_perc': 'stats/non_neg_perc_v2.json',
    }[job]
    results = {}
    for name, ptlm_arr_path in ptlm_exps.items():
        arr_cat_full, meta = load_concat_arr(ptlm_arr_path)
        
        print('Truncate bottom 19')
        arr_cat_full = arr_cat_full[-19:,:]

        if arr_cat_full.shape[1] > 10000:
            rand_idx = deterministic_random_indices(arr_cat_full.shape[1], 10000)
            rand_idx.sort()
            arr_cat = arr_cat_full[:,rand_idx]
        else:
            arr_cat = arr_cat_full
        
        if job == 'all':
            res = examine_all_fit(arr_cat)
        elif job == 'recons':
            res = examine_svd_fits(arr_cat)
        elif job == 'avg_fgt':
            res = np.mean(arr_cat)
        elif job == 'std_fgt':
            res = np.std(arr_cat)
        elif job == 'non_neg_perc':
            res = int((arr_cat >= 0).sum()) / arr_cat.size
        else:
            raise NotImplementedError(job)
        
        print(ptlm_arr_path)
        print(res)
        results[name] = res
    
    with open(out_name,'w') as wf:
        json.dump(results,wf)

