import numpy as np
from data_utils.utils import deterministic_random_indices
import pickle
from collections import Counter
import math
from scipy.stats import norm

def get_ocl_tasks(meta):
    tasks = []
    for k, v in meta.items():
        tasks.extend([f'{k}/{task}' for task in v['tasks']])
    return tasks

def get_dolma_tasks_10k_sample():
    rand_idx = deterministic_random_indices(141816, 10000)
    rand_idx.sort()
    with open('data/dolma_chunked_sample/source_segments_filter.pkl','rb') as f:
        segments = pickle.load(f)
    tasks = []
    for idx in rand_idx:
        task = None
        for segs in segments:
            if idx < segs[2] and idx >= segs[1]:
                task = segs[0]
        tasks.append(task)
    return tasks

def get_sig_component(component_mat, ocl_tasks, pt_tasks, topk_row=5, topk_col=100, do_abs=False):
    if do_abs:
        component_mat_abs = np.abs(component_mat)
    else:
        component_mat_abs = component_mat
    row_abs = component_mat_abs.mean(1)
    col_abs = component_mat_abs.mean(0)

    top_row_idxs =  np.argsort(-row_abs)[:topk_row]
    top_col_idxs = np.argsort(-col_abs)[:topk_col]

    top_row_task = [ocl_tasks[x] for x in top_row_idxs]
    top_pt_tasks = [pt_tasks[x] for x in top_col_idxs]
    top_row_scores = row_abs[top_row_idxs]
    top_pt_scores = col_abs[top_col_idxs]

    return top_row_task, top_pt_tasks, top_row_scores, top_pt_scores

def get_bucket_prob(items):
    counter = Counter(items)
    ret = {k: counter[k]  for k in sorted(counter.keys())}
    
    return ret

def get_ztest(x_A, n_A, x_B, n_B):


    # Sample proportions
    p_A = x_A / n_A
    p_B = x_B / n_B

    # Pooled proportion
    p_pooled = (x_A + x_B) / (n_A + n_B)

    # Standard error
    SE = math.sqrt(p_pooled * (1 - p_pooled) * (1 / n_A + 1 / n_B))

    # Z-test statistic
    z = (p_A - p_B) / SE

    # P-value for two-tailed test
    p_value = 2 * (1 - norm.cdf(abs(z)))

    return z, p_value