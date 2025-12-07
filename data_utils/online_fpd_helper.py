import surprise
import pickle
import numpy as np
from torch.utils.data import Subset
import torch
from torch.nn.functional import log_softmax, cross_entropy
import pandas as pd

def mat_to_interaction_df(mat):
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

def restore_nan_col(row, ref_row):
    ret = np.full(ref_row.shape[0], -100.0)
    ret[~np.isnan(ref_row)] = row
    return ret


class OnlineFPDHelper:
    def __init__(self, config, pt_ds):
        self.config = config
        self.pt_ds = pt_ds
        fpd_split_file = config.fpd.fpd_split_file # has nan filtered
        base_ppl_file = config.fpd.base_ppl_path


        with open(fpd_split_file, 'rb') as f:
            self.fpd_split = pickle.load(f)
        
        self.base_ppl_arr = np.load(base_ppl_file) # just for reference; will compute its
        self.base_model_ppl = None

        self.orig_n_pt = len(pt_ds)
        self.filter_n_pt = self.fpd_split['train_mat'].shape[1]
        
        assert self.base_ppl_arr.shape[0] == self.orig_n_pt

        #self.ss_selection_rng = np.random.default_rng(seed=config.fpd.ss_seed)
        self.algo_name = config.fpd.algo_name
        self.query_k = config.fpd.query_k

        if self.algo_name== 'knn_baseline':
            self.algo_cls = surprise.KNNBaseline
            self.algo_options = {}
            self.algo_options['bsl_options'] = {'n_epochs': 100}
        elif self.algo_name == 'svd':
            self.algo_cls = surprise.SVD
            self.algo_options = {}
            self.algo_options['n_epochs'] = 1000
            self.algo_options['n_factors'] = 5
            self.algo_options['reg_all'] = 0
            self.algo_options['init_std_dev'] = 0.001
        elif self.algo_name == 'additive':
            self.algo_cls = surprise.BaselineOnly
            self.algo_options = {}
            self.algo_options['bsl_options'] = {'n_epochs': 100}

    def get_non_nan_idxs(self):
        nan_pos = np.isnan(self.base_ppl_arr)
        non_nan_idxs = np.arange(self.base_ppl_arr.shape[0])[~nan_pos]
        return non_nan_idxs

    def evaluate_seed_ppl(self, trainer, return_idxs=False):
        non_nan_idxs = self.get_non_nan_idxs()
        if non_nan_idxs.shape[0] != self.filter_n_pt:
            raise ValueError(f'{non_nan_idxs.shape[0]}, {self.filter_n_pt}')

        # sample k examples to evaluate forgetting
        query_meta_idxs = np.random.default_rng(seed=self.config.fpd.ss_seed).choice(np.arange(self.filter_n_pt) , self.query_k, replace=False)
        query_idxs = non_nan_idxs[query_meta_idxs]

        query_subset = Subset(self.pt_ds, query_idxs)
        prediction_output = trainer.predict(query_subset)
        #print(prediction_output)
        log_ppls = self.extract_avg_log_probs(prediction_output.predictions, prediction_output.label_ids) # [B]
        log_ppls = log_ppls.cpu().numpy()

        print('Avg ppl', log_ppls)

        before_log_ppls = self.base_ppl_arr[query_idxs]

        print('Ref before ppl', before_log_ppls)
        if return_idxs:
            return query_meta_idxs, query_idxs, log_ppls
        else:
            return log_ppls
    
    def evaluate_base_model_seed_ppl(self, trainer):
        query_meta_idxs, query_idxs, base_model_ppl = self.evaluate_seed_ppl(trainer, return_idxs=True)
        self.base_model_ppl = base_model_ppl
        
        reference_ppl_vllm = -self.base_ppl_arr[query_idxs]
        print(f'Base model ppl: {base_model_ppl}')
        print(f'Reference model ppl: {reference_ppl_vllm}')

    def evaluate_base_model_seed_ppl_from_init_state(self, trainer, model_init_state):
        current_state = {k:v.cpu().clone() for k,v in trainer.model.state_dict().items()}
        trainer.model.load_state_dict(model_init_state)
        query_meta_idxs, query_idxs, base_model_ppl = self.evaluate_seed_ppl(trainer, return_idxs=True)
        self.base_model_ppl = base_model_ppl
        reference_ppl_vllm = -self.base_ppl_arr[query_idxs]

        trainer.model.load_state_dict(current_state)

        #print(f'Base model ppl: {base_model_ppl}')
        #print(f'Reference model ppl: {reference_ppl_vllm}')

    def evaluate_seed_forgetting(self, trainer):
        query_meta_idxs, query_idxs, after_ppl = self.evaluate_seed_ppl(trainer, return_idxs=True)
        before_ppl = self.base_model_ppl

        forgetting = after_ppl - before_ppl
        print(f'Seed forgetting is {forgetting}')

        return query_meta_idxs, query_idxs, forgetting

    def predict_all_forgetting(self, trainer):
        query_meta_idxs, query_idxs, seed_forgetting = self.evaluate_seed_forgetting(trainer)
        seed_fgt_arr = np.full(self.filter_n_pt, -10000.0)
        seed_fgt_arr[query_meta_idxs] = seed_forgetting
        pred_forgetting_filtered = self.run_matrix_completion(seed_fgt_arr)
        pred_forgetting = restore_nan_col(pred_forgetting_filtered, self.base_ppl_arr)
        return pred_forgetting

    def extract_avg_log_probs(self, logits, labels):
        # [B,T,V], [B,T]

        logits = torch.from_numpy(logits)
        labels = torch.from_numpy(labels)

        shift_logits, shift_labels = logits[:,:-1].contiguous(), labels[:,1:].contiguous()
        losses = cross_entropy(shift_logits.view(-1, shift_logits.size(2)), shift_labels.view(-1), reduction='none')
        losses = losses.view(*shift_labels.size()) 

        mask = (shift_labels != -100).float() # [B,T]

        effective_lens = mask.sum(-1)
        sum_loss = (losses * mask).sum(-1)
        avg_loss =  sum_loss / (effective_lens + 1e-10)
        return avg_loss

    def run_matrix_completion(self, seed_arr): 
        # seed arr has size filter_n_pt
        reader = surprise.Reader(rating_scale=(-100., 100.))

        algo = self.algo_cls(**self.algo_options)
        loo_mat = np.concatenate([self.fpd_split['train_mat'], seed_arr.reshape(1,-1)], 0)
        #print('Mat', loo_mat)
        loo_df = mat_to_interaction_df(loo_mat)
        loo_ds = surprise.Dataset.load_from_df(loo_df, reader)
        loo_trainset = loo_ds.build_full_trainset()
        algo.fit(loo_trainset)
        loo_preds = get_preds(algo, loo_trainset.n_users - 1, loo_mat.shape[1])
        print(f'Matrix completion preds {loo_preds}')
        return loo_preds

    