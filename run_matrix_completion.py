from surprise import Dataset, SVD, BaselineOnly, KNNBaseline, KNNBasic
from surprise.reader import Reader
import random
import numpy as np
import pickle
from sklearn.metrics import f1_score, mean_squared_error, precision_score, recall_score, roc_auc_score
import pandas as pd
import logging
import argparse
import os
import pickle
import random
import json
from post_process.my_mf import LogisticMF, LogisticMFAdditive, sigmoid, additive_pred_func
from ast import literal_eval

logger = logging.getLogger()
logger.setLevel(logging.INFO)

args = None

def create_masked_arr(arr, ocl_idx, rand_k):
    # for the given ocl example, we only assume knowing ground truth forgetting of rand_k examples.
    # very negative values like -10000.0 will be filtered out when creating the dataset of collaborative filtering

    new_arr = np.full(arr.shape[1], -10000.0) 
    rng = np.random.default_rng(args.seed if args else 0)
    idxs = rng.choice(np.arange(arr.shape[1]), rand_k)
    new_arr[idxs] = arr[ocl_idx, idxs]

    keep_mask = np.ones(arr.shape[1], dtype=bool)
    keep_mask[idxs] = 0

    return new_arr, keep_mask

def create_loo_ds(train_mat, test_mat, test_idx, topk):
    # create leave-one-out dataset by concatenating the train_mat and one row (ocl example) from test_mat
    masked_arr, keep_mask = create_masked_arr(test_mat, test_idx, topk)
    loo_mat = np.concatenate([train_mat, masked_arr.reshape(1, -1)])
    return loo_mat, keep_mask

def mat_to_interaction_df(mat, reader):
    data = []
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] > -100:
                data.append([i, j, mat[i, j]])
    df = pd.DataFrame(data)
    return df

def get_preds(model, ocl_idx, max_pt):
    scores = []
    for pt_idx in range(max_pt):
        scores.append(model.predict(ocl_idx, pt_idx).est)
    scores = np.array(scores)
    return scores

# main exp function
def evaluate_for_all_ocl_ds(train_mat, test_mat, topk, algo_cls, options):
    all_preds = []
    all_gts_masked = []
    all_preds_masked = []
    reader = Reader(rating_scale=(-100., 100.))

    for test_ocl_idx in range(test_mat.shape[0]):
        print('Training and inference for task {}'.format(test_ocl_idx))
        algo = algo_cls(**options)
        loo_mat, keep_mask = create_loo_ds(train_mat, test_mat, test_ocl_idx, topk)
        loo_df = mat_to_interaction_df(loo_mat, reader)
        loo_ds = Dataset.load_from_df(loo_df, reader)
        loo_trainset = loo_ds.build_full_trainset()
        algo.fit(loo_trainset)
        loo_preds = get_preds(algo, loo_trainset.n_users - 1, loo_mat.shape[1])
        all_preds.append(loo_preds)
        all_gts_masked.append(test_mat[test_ocl_idx, keep_mask])
        all_preds_masked.append(loo_preds[keep_mask])

        if args.debug_one:
            break

    all_preds = np.stack(all_preds)
    all_gts_masked = np.stack(all_gts_masked)
    all_preds_masked = np.stack(all_preds_masked)
    return all_preds, all_preds_masked, all_gts_masked

def matrix_factorization_direct(train_mat, test_mat, topk, is_additive=False, rank=5, seed=0, debug_one=False):
    all_preds = []
    all_gts_masked = []
    all_preds_masked = []
    
    train_true_mask = np.ones_like(train_mat, dtype=bool)

    for test_ocl_idx in range(test_mat.shape[0]):
        print('Training and inference for task {}, rank {}, seed {}'.format(test_ocl_idx, rank, seed))
        masked_arr, keep_mask = create_masked_arr(test_mat, test_ocl_idx, topk)
        
        loo_mat = np.concatenate([train_mat, masked_arr.reshape(1, -1)])
        use_for_train_mask = np.concatenate([train_true_mask, ~keep_mask.reshape(1,-1)])

        print(masked_arr, ~keep_mask)

        if is_additive:
            W, H = LogisticMFAdditive(loo_mat, use_for_train_mask, seed=seed)
            fit_mat = additive_pred_func(W, H)
        else:
            W, H = LogisticMF(loo_mat, rank, use_for_train_mask, seed=seed)
            fit_mat = np.dot(W, H)
        preds = fit_mat[-1]
        preds_masked = preds[keep_mask]
        gts = test_mat[test_ocl_idx, keep_mask]

        all_preds.append(preds)
        all_preds_masked.append(preds_masked)
        all_gts_masked.append(gts)
        
        if debug_one:
            break

    all_preds = np.stack(all_preds)
    all_gts_masked = np.stack(all_gts_masked)
    all_preds_masked = np.stack(all_preds_masked)
    return all_preds, all_preds_masked, all_gts_masked


def compute_bin_metrics(gts, preds, thres=0.):
    gts_bin = gts > args.pred_thres

    preds_bin = preds > thres
    metrics = {}
    metrics['f1'] = f1_score(gts_bin.reshape(-1), preds_bin.reshape(-1))
    metrics['prec'] = precision_score(gts_bin.reshape(-1), preds_bin.reshape(-1))
    metrics['recall'] = recall_score(gts_bin.reshape(-1), preds_bin.reshape(-1)) 
    metrics['auc_roc'] = roc_auc_score(gts_bin.reshape(-1), preds.reshape(-1))
    return metrics

def compute_rmse(gts, preds):
    preds = np.stack(preds)
    rmse = np.sqrt(mean_squared_error(gts, preds))
    return rmse

def interpret_number(s):
    try:
        # Try converting to an integer
        num = int(s)
        return num
    except ValueError:
        try:
            # If integer conversion fails, try converting to a float
            num = float(s)
            return num
        except ValueError:
            # If both conversions fail, return the original string
            return s

def overwrite_options_from_args(options, hparams):
    for kv_pair in hparams:
        k,v = kv_pair.split('=')
        v = interpret_number(v)
        options[k] = v

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('method')
    parser.add_argument('split_file')
    parser.add_argument('--hparams', nargs='*', type=str)
    parser.add_argument('--known_k', default=30, type=int)
    parser.add_argument('--pred_thres', default=0., type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--is_binary', action='store_true')
    parser.add_argument('--debug_one', action='store_true')
    args = parser.parse_args()
    
    split_filename = args.split_file.split('/')[-1]

    output_dir = 'runs/forgetting_prediction/1120/{}'.format(split_filename)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.addHandler(logging.FileHandler(os.path.join(output_dir, '{}_k{}.txt'.format(args.method, args.known_k))))
    logger.addHandler(logging.StreamHandler())

    logger.info(repr(args.__dict__))
    with open(args.split_file, 'rb') as f:
        fpd_split = pickle.load(f)
    train_mat, test_mat = fpd_split['train_mat'], fpd_split['test_mat']
    train_mat = np.nan_to_num(train_mat)
    test_mat = np.nan_to_num(test_mat)
    pt_forgets = train_mat.sum(0)
    pt_idxs = pt_forgets.argsort()
    pt_idxs = pt_idxs[::-1]

    options = {}
    if args.method == 'svd':
        algo = SVD
        svd_d = 5
        lr = 0.005
        options['n_epochs'] = 1000
        options['n_factors'] = svd_d
        options['lr_all'] = lr
        options['reg_all'] = 0
        options['init_std_dev'] = 0.001
        #options['biased'] = False
    elif args.method == 'lgmf':
        options = {}
    elif args.method == 'lgmf_additive':
        options = {}
    elif args.method == 'knn':
        algo = KNNBasic
    elif args.method == 'knn_baseline':
        algo = KNNBaseline
        options['bsl_options'] = {'n_epochs': 1000, 'method': 'sgd'}
    elif args.method == 'knn_baseline_item':
        algo = KNNBaseline
        options['bsl_options'] = {'n_epochs': 1000, 'method': 'sgd'}
        options['sim_options'] = {'user_based': False}
    elif args.method == 'additive':
        algo = BaselineOnly
        options['bsl_options'] = {'n_epochs': 1000, 'method': 'sgd'}
    else:
        raise NotImplementedError
    
    is_binary = args.is_binary
    if args.method in ['lgmf', 'lgmf_additive']:
        is_binary = True
    overwrite_options_from_args(options, args.hparams)
    res = {k:v for k,v in args.__dict__.items()}


    if args.method == 'lgmf':
        all_preds, all_preds_masked, all_gts_masked = matrix_factorization_direct(
            train_mat, test_mat, args.known_k, rank=options['rank'], seed=args.seed)
    elif args.method == 'lgmf_additive':
        all_preds, all_preds_masked, all_gts_masked = matrix_factorization_direct(
            train_mat, test_mat, args.known_k, rank=options['rank'], seed=args.seed, is_additive=True)
    else:
        all_preds, all_preds_masked, all_gts_masked = evaluate_for_all_ocl_ds(train_mat, test_mat, args.known_k, algo, options)
        
    if is_binary:
        bin_metrics = compute_bin_metrics(all_gts_masked, all_preds_masked, args.pred_thres)
        res = res | bin_metrics
    else:
        rmse = compute_rmse(all_gts_masked, all_preds_masked)
        res['rmse_score'] = rmse

    logger.info(json.dumps(res))

    res_dir = os.path.join(output_dir, 'preds_{}_k{}_results'.format(args.method, args.known_k))
    if args.method == 'svd':
        res_dir = os.path.join(output_dir, 'preds_{}_k{}_d{}_lr{}_results'.format(args.method, args.known_k, svd_d, lr))
    if args.method in ['lgmf','lgmf_additive']:
        res_dir = os.path.join(output_dir, 'preds_{}_k{}_{}_results'.format(args.method, args.known_k, '_'.join(args.hparams).replace('=','+')))
        
    os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(res_dir, 'preds_seed_{}.pkl'.format(args.seed)),'wb') as wf:
        pickle.dump(all_preds, wf)
    with open(os.path.join(res_dir, 'score_seed_{}.json'.format(args.seed)),'w') as wf:
        json.dump(res, wf)

    