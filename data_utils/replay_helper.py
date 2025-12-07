from torch.utils.data import Dataset, Subset
from .utils import deterministic_random_indices
import numpy as np
from scipy.special import softmax
import logging
import pickle
from .lm import SFTDataset

logger = logging.getLogger('main')

def get_sampling_weights(fgt_scores, temperature, outlier_mask=0.):
    logger.info('Raw score for computing {}'.format(fgt_scores))

    if outlier_mask > 0:
        masked_scores = np.copy(fgt_scores)
        masked_scores[np.argsort(fgt_scores)[-int(outlier_mask * fgt_scores.shape[0]):]] = 0
        weight = softmax(masked_scores / temperature)
    else:
        if temperature > 1000.0:
            logger.info('Using uniform weight for debugging purpose')
            weight = np.ones_like(fgt_scores) / fgt_scores.shape[0]
        else:
            weight = softmax(fgt_scores / temperature)
    logger.info('Final weight {}'.format(weight[:100]))
    return weight

def restore_nan_cols(arr, ref_row):
    ret = np.full((arr.shape[0],ref_row.shape[0]), -100.0)
    ret[:,~np.isnan(ref_row)] = arr
    return ret

def prepare_replay_mixture_ds(config, tokenizer, ocl_ds, seed=None):
    pt_ds = SFTDataset.from_auto(config.replay.task_category, tasks='all',
                                split='train', config=config, tokenizer=tokenizer, skip_encoding=False, 
                                include_labels=True, ans_start_patt=config.ans_start_pattern)
    # print(pt_ds[0])
    mixture_ds = MixtureDatasetForReplay(ocl_ds, pt_ds, config.replay, seed=seed)
    return mixture_ds

def prepare_replay_mixture_with_ds(replay_config, tokenizer, ocl_ds, pt_ds, online_forgetting_arr=None, seed=None):
    # print(pt_ds[0])
    mixture_ds = MixtureDatasetForReplay(ocl_ds, pt_ds, replay_config, online_forgetting_arr, seed=seed)
    return mixture_ds

def repeated_bofn_sampling(sampling_weights, sample_num, bofn):
    rng = np.random.default_rng(seed=0)
    samples_idxs = []

    idxs = rng.permutation(len(sampling_weights))

    for pass_id in range(bofn):
        for start in range(0, len(idxs), bofn):
            stop = start + bofn
            cand = np.argsort(sampling_weights[idxs[start:stop]])
            if pass_id < len(cand):
                samples_idxs.append(idxs[start + cand[pass_id]])

        if len(samples_idxs) >= sample_num:
            break
    ret = np.array(samples_idxs[:sample_num])
    print('Repeated bofn sampling: {}'.format(ret[:100]))
    return ret

class ReplayedExampleRecord:
    def __init__(self):
        self.replayed_idxs = set()

    def update(self, idxs):
        self.replayed_idxs.update(idxs)

    def mask_forgetting_arr(self, forgetting_arr):
        if self.replayed_idxs:
            replayed_idxs = np.array(list(self.replayed_idxs))
            print(f"Masing {len(replayed_idxs)} replayed examples")
            ret = np.copy(forgetting_arr)
            ret[replayed_idxs] = -100.0
            return ret
        else:
            return forgetting_arr

class MixtureDatasetForReplay(Dataset):
    def __init__(self, ocl_ds, pt_ds, replay_config, online_forgetting_arr=None, seed=None):
        self.replay_config = replay_config

        mixture_seed = 0 
        if hasattr(replay_config, 'mixture_seed'):
            logging.warning(f'Manually set mixture seed as {mixture_seed}')
            mixture_seed = replay_config.mixture_seed

        self.heldout_idxs = np.sort(deterministic_random_indices(len(pt_ds), replay_config.heldout_num, mixture_seed)) if replay_config.heldout_num > 0 else None
        if self.heldout_idxs is not None:
            self.pt_ds_val = Subset(pt_ds, self.heldout_idxs)
            train_mask = np.ones(len(pt_ds), dtype=bool)

            train_mask[self.heldout_idxs] = False
            self.train_idxs = np.arange(len(pt_ds))[train_mask]
        else:
            self.pt_ds_val = None
            self.train_idxs = np.arange(len(pt_ds))
        self.pt_ds_train = Subset(pt_ds, self.train_idxs)

        self.ocl_ds = ocl_ds
        self.use_actual_ocl_len = getattr(replay_config, 'use_actual_ocl_len', False)

        self.sampled_idxs = None
    
        if replay_config.mixture_method in ['gt_max','gt_sample','gt_mir']:
            pt_fgt_scores = np.load(replay_config.gt_fgt_arr_path)
            pt_fgt_scores = np.nan_to_num(pt_fgt_scores, nan=-1e10)
            self.pt_train_fgt_scores = pt_fgt_scores[self.train_idxs]
            print('Outlier mask is {}'.format(replay_config.outlier_mask))
            if replay_config.truncate_pt_ds > 0:
                self.pt_train_fgt_scores = self.pt_train_fgt_scores[:replay_config.truncate_pt_ds]
            self.sampling_weights = get_sampling_weights(self.pt_train_fgt_scores, replay_config.temperature, replay_config.outlier_mask)     
        elif replay_config.mixture_method in ['pred_max','pred_sample']:
            with open(replay_config.pred_fgt_arr_path,'rb') as f: # [, N]
                all_pred_fgt_ss = pickle.load(f)
            with open(replay_config.fpd_split,'rb') as f:
                fpd_split_info = pickle.load(f)
            test_task_idxs = fpd_split_info['test_ocl_idxs']
            base_fgt = np.load(replay_config.base_fgt_path)
            ocl_task_id = replay_config.ocl_task_id

            if ocl_task_id not in test_task_idxs:
                raise ValueError(f'{ocl_task_id} not in test tasks')

            all_pred_fgt = restore_nan_cols(all_pred_fgt_ss, base_fgt)
            print('All pred fgt', all_pred_fgt.shape)
            #print(test_task_idxs.index(ocl_task_id))
            fgt_arr = all_pred_fgt[test_task_idxs.index(ocl_task_id)]
            self.pt_train_fgt_scores = fgt_arr[self.train_idxs]
            if replay_config.truncate_pt_ds > 0:
                self.pt_train_fgt_scores = self.pt_train_fgt_scores[:replay_config.truncate_pt_ds]
            self.sampling_weights = get_sampling_weights(self.pt_train_fgt_scores, replay_config.temperature)
        elif replay_config.mixture_method in ['online_pred_max', 'online_pred_sample']:
            if online_forgetting_arr.shape[0] != len(pt_ds):
                raise ValueError(f'{online_forgetting_arr.shape[0]}, {len(pt_ds)}')
            self.pt_train_fgt_scores = online_forgetting_arr[self.train_idxs]
            if replay_config.truncate_pt_ds > 0:
                self.pt_train_fgt_scores = self.pt_train_fgt_scores[:replay_config.truncate_pt_ds]
            self.sampling_weights = get_sampling_weights(self.pt_train_fgt_scores, replay_config.temperature)
        elif replay_config.mixture_method in ['predef_score_max', 'predef_score_sample']:
            predef_scores = np.load(replay_config.predef_score_path)
            with open(replay_config.fpd_split,'rb') as f:
                fpd_split_info = pickle.load(f)
            test_task_idxs = fpd_split_info['test_ocl_idxs']
            base_fgt = np.load(replay_config.base_fgt_path)
            ocl_task_id = replay_config.ocl_task_id

            if ocl_task_id not in test_task_idxs:
                raise ValueError(f'{ocl_task_id} not in test tasks')
            
            all_predef_score = restore_nan_cols(predef_scores, base_fgt)
            print('All predef score', all_predef_score.shape)
            self.pt_train_fgt_scores = all_predef_score[ocl_task_id, self.train_idxs]
            print('Available score', self.pt_train_fgt_scores.shape)
            if replay_config.truncate_pt_ds > 0:
                self.pt_train_fgt_scores = self.pt_train_fgt_scores[:replay_config.truncate_pt_ds]
            self.sampling_weights = get_sampling_weights(self.pt_train_fgt_scores, replay_config.temperature)

        if seed is not None:
            print(f'Using manual seed {seed}')

        self.mir_bofn = None
        if replay_config.mixture_method in ['gt_mir']:
             self.mir_bofn = replay_config.mir_bofn

        self.mixture_examples, self.replay_idxs = self.create_mixture_examples(self.ocl_ds, self.pt_ds_train, replay_config.mixture_method,
                                                                               replay_config.mixture_ratio, replay_config.seed if seed is None else seed)

        self.pt_fgt_scores, self.sampling_weights = None, None

    def get_replay_idxs(self):
        return self.replay_idxs

    def create_mixture_examples(self, ocl_ds, pt_ds_train, mixture_method, mixture_ratio, seed):
        mixture_examples = []

        ocl_example_num = len(ocl_ds)
        if self.use_actual_ocl_len:
            ocl_example_num = min(self.replay_config.actual_train_bs * self.replay_config.actual_max_train_step, ocl_example_num)
            logger.info(f'Using train set len {ocl_example_num} for creating replay mixture')

        replay_n = int(ocl_example_num * mixture_ratio) + 1
        #if mixture_method == 'random':
        replay_examples, replay_idxs = self.sample_random(pt_ds_train, replay_n, seed)
        #else:
        #raise NotImplementedError

        wait = 0
        replay_idx = 0
        for ocl_idx in range(ocl_example_num):
            mixture_examples.append(ocl_ds[ocl_idx])
            wait += mixture_ratio
            if wait >= 1:
                wait -= 1
                # add one pt example to be replayed
                mixture_examples.append(replay_examples[replay_idx])
                replay_idx += 1
        
        return mixture_examples, replay_idxs
    
    def sample_random(self, pt_ds_train, replay_n, seed):
        sample_examples = []
        rng = np.random.default_rng(seed)

        if self.replay_config.truncate_pt_ds > 0:
            print('Truncate pt ds to {}'.format(self.replay_config.truncate_pt_ds))
            pt_ds_train = Subset(pt_ds_train, [_ for _ in range(self.replay_config.truncate_pt_ds)])
        
        if self.replay_config.mixture_method == 'random':
            sample_idxs = rng.choice(len(pt_ds_train), replay_n, replace=False)
        elif self.replay_config.mixture_method in ['gt_sample', 'pred_sample', 'online_pred_sample', 'predef_score_sample']:
            sample_idxs = rng.choice(len(pt_ds_train), replay_n, replace=False, p=self.sampling_weights)
            sample_idxs = rng.permutation(sample_idxs)
        elif self.replay_config.mixture_method in ['gt_max', 'pred_max', 'online_pred_max', 'predef_score_max']:
            sample_idxs = np.argsort(self.sampling_weights)[::-1][:replay_n]
            sample_idxs = rng.permutation(sample_idxs)
        elif self.replay_config.mixture_method in ['gt_mir']:
            sample_idxs = repeated_bofn_sampling(self.sampling_weights, replay_n, self.mir_bofn)
            sample_idxs = rng.permutation(sample_idxs)
        else:
            raise ValueError(self.replay_config.mixture_method)
        logger.info('Sampled idxs {}'.format(sample_idxs[:20]))

        for idx in sample_idxs:
            sample_examples.append(pt_ds_train[idx])

        return sample_examples, sample_idxs
    
    def __getitem__(self, idx):
        return self.mixture_examples[idx]

    def __len__(self):
        return len(self.mixture_examples)
    