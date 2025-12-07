import pickle
import numpy as np
import json
import pandas as pd
from utils.config import load_configs
import matplotlib as mpl
from matplotlib import pyplot as plt
from collections import OrderedDict
from .get_ppl_arr import load_concat_arr
from .collect_dolma_source import get_repeating_segments
from .get_sig_components import get_dolma_tasks_10k_sample


def reorder_row_for_pt_exps(arr):
    _, meta = load_concat_arr('stats/olmo-7b/ftd-ppl-inc.npy')
    mapping = load_task2cat_mapping()
    cat2idxs = get_cat2idx(mapping, meta['flan']['tasks'])
    new_arr_cat = get_new_arr(arr, cat2idxs, meta)
    return new_arr_cat

def load_task2cat_mapping():
    with open('data/flan_task_cats') as f:
        lines = f.readlines()
    mapping = {}
    for line in lines:
        task, cat = line.split('|')
        mapping[task.strip()] = cat.strip()
    return mapping

def get_cat2idx(mapping, flan_tasks):
    idxs = [_ for _ in range(len(flan_tasks))]
    cat2idxs =  {}
    for idx in idxs:
        task = flan_tasks[idx]
        task_cat = mapping[task]
        if task_cat not in cat2idxs:
            cat2idxs[task_cat] = []
        cat2idxs[task_cat].append(idx)
    cat2idxs = OrderedDict(sorted(cat2idxs.items()))
    return cat2idxs

def get_new_arr(orig_arr, cat2idxs, meta):
    new_arr = []
    stop = meta['flan']['stop']
    for task_cat, task_idxs in cat2idxs.items():
        new_arr.append(orig_arr[task_idxs])

    new_arr.append(orig_arr[stop:])
    return np.concatenate(new_arr)

def get_new_meta(cat2idxs, meta):
    new_meta = OrderedDict()
    start = 0
    for cat, idxs in cat2idxs.items():
        new_meta[cat] = {
            'start': start,
            'stop': start + len(idxs),
            'tasks': [meta['flan']['tasks'][idx] for idx in idxs]
        }
        start += len(idxs)
    for k, info in meta.items():
        if k != 'flan':
            new_meta[k] = info
    return new_meta

def get_tulu_pt_mapping():
    with open('data/tulu_sample/sample_1k.json') as f:
        examples = json.load(f)

    # filter nan cols
    nonan_idxs = get_nonan_col_idxs('runs/stats/stats-olmo-7b-peft/mmlu/task_0/pt-base_ppl_results.pkl.npy')

    mapping = []
    for idx in nonan_idxs:
        mapping.append(examples[idx]['dataset'].split('.')[0])
    return pd.Series(mapping)

def get_nonan_col_idxs(ref_arr):
    arr = np.load(ref_arr)
    idxs = np.arange(len(arr))[~np.isnan(arr)]
    return idxs

def get_mmlu_bbh_mapping():
    config = load_configs('configs/p3/p3_default.yaml','configs/llm/dolma_defaults.yaml')
    mapping = ['MMLU\n({})'.format(len(config.mmlu_tasks))] * len(config.mmlu_tasks) \
      + ['BBH\n({})'.format(len(config.bbh_tasks))] * len(config.bbh_tasks)
    return pd.Series(mapping)

def get_mmlu_bbh_truthful_mapping():
    config = load_configs('configs/llm/llm_defaults.yaml', templates=None)
    mapping = ['MMLU\n({})'.format(len(config.mmlu_tasks))] * len(config.mmlu_tasks) \
      + ['BBH\n({})'.format(len(config.bbh_tasks))] * len(config.bbh_tasks) \
      + ['Truthful\n({})'.format(len(config.truthful_qa_tasks))] * len(config.truthful_qa_tasks)
    return pd.Series(mapping)

def get_mmlu_bbh_truthful_dolly_mapping():
    config = load_configs('configs/llm/llm_defaults.yaml', templates=None)
    mapping = ['MMLU\n({})'.format(len(config.mmlu_tasks))] * len(config.mmlu_tasks) \
      + ['BBH\n({})'.format(len(config.bbh_tasks))] * len(config.bbh_tasks) \
      + ['Truthful\n({})'.format(len(config.truthful_qa_tasks))] * len(config.truthful_qa_tasks) \
      + ['Dolly\n({})'.format(len(config.dolly_tasks))] * len(config.dolly_tasks)
    return pd.Series(mapping)

def get_mapping_from_meta(meta, name_mapping):
    mapping = []
    for task_cat, info in meta.items():
        tn = len(info['tasks'])
        task_name = name_mapping.get(task_cat, task_cat)
        for i in range(tn):
            mapping.append(f'{task_name}')
    return pd.Series(mapping)

def get_mmlu_bbh_truthful_tasks():
    def cats(tasks, cat):
        return [(cat,x) for x in tasks]
    
    config = load_configs('configs/llm/llm_defaults.yaml',  templates=None)
    tasks = cats(config.mmlu_tasks,'MMLU') + cats(config.bbh_tasks,'BBH') + cats(config.truthful_qa_tasks,'TruthfulQA')
    tasks = np.array(tasks)
    return tasks


def get_nzero_cols(df):
    idxs = df.sum(axis=0) != 0
    return idxs.to_numpy()

def filter_zero_cols(df, row_mapping, col_mapping):
    idxs = df.sum(axis=0) != 0
    #print(idxs.sum())
    filter_df = df.loc[:,idxs]
    col = col_mapping[idxs]
    return filter_df, row_mapping, col

def get_intervals(arr):
    start = -1
    prev = None
    rets = []
    for idx, item in enumerate(arr):
        if item != prev:
            if start != -1:
                rets.append([start, idx, prev])
            start = idx
            prev = item

    rets.append([start, len(arr), prev])
    return rets

def map_text(s):
    if s == 'hard_coded':
        return 'hc'
    s = s.replace('_','\n')
    return s

def plot_arrows_heatmaps_olmo_peft(arr, rows, cols):
    fig, ax = plt.subplots(figsize=(20,4))
    row_ivals = get_intervals(rows)
    col_ivals = get_intervals(cols)
    mesh = ax.pcolormesh(arr, vmin=-0.3, vmax=0.3, cmap='bwr', rasterized=True, zorder=5)

    ax2= fig.add_axes([0.155, 0.90, 0.744, 0.05])
    ax3= fig.add_axes([0.905, 0.23, 0.01, 0.65])  
    ax4 = fig.add_axes([0.1, 0.23, 0.01, 0.65])  
    cbar = plt.colorbar(mesh, cax=ax4)
    ax4.yaxis.set_ticks_position('left')
    
    ax2.pcolormesh(arr.mean(0).reshape(1,-1), vmin=-0.3, vmax=0.3, cmap='bwr',rasterized=True)
    ax2.text(arr.shape[1] // 2, 1.5, 'Column Mean', fontsize=14, ha='center', va='center')
    ax2.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
 
    ax3.pcolormesh(arr.astype(float).mean(1).reshape(-1,1), vmin=-0.3, vmax=0.3, cmap='bwr',rasterized=True)
    ax3.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    ax3.text(1.5, arr.shape[0] // 2, 'Row\nMean', fontsize=14, ha='left', va='center',rotation=25)
    
    ax.set_ylim(-20, arr.shape[0] + 1)
    #print(col_ivals)
    y = -10
    for start, stop, name in col_ivals:
        ax.annotate('', xy=(start,y), xytext=(stop, y), arrowprops=dict(arrowstyle='<->', lw=1.5, shrinkA=0, shrinkB=0),
                   zorder=3)
        ax.text(int((start+stop)/ 2), y-1, map_text(name), ha='center', va='center', fontsize=14, zorder=4,
                bbox=dict(facecolor='white', edgecolor='none', boxstyle='square, pad=0.0'), rotation=25)
        ax.axvline(x=start, lw=1.5, zorder=5, color='black')
    ax.axvline(x=stop, lw=1.5, zorder=5, color='black')
    xmin = - int(arr.shape[1] / 50)
    ax.set_xlim(xmin * 2, arr.shape[1])
    for start, stop, name in row_ivals:
        print(start, stop)
        ax.annotate('', xy=(xmin, start), xytext=(xmin,stop), arrowprops=dict(arrowstyle='<->', lw=1.5, shrinkA=0, shrinkB=0),
                   zorder=3)
        ax.text(xmin, int((start+stop)/ 2), map_text(name), ha='center', va='center', fontsize=14, zorder=4,
                bbox=dict(facecolor='white', edgecolor='none', boxstyle='square, pad=0.0'), rotation=25)
        ax.axhline(y=start, lw=1.5, zorder=5, color='black')
    print(stop)
    ax.axhline(y=stop - 1 , lw=1.5, zorder=5, color='black') 

    ax.set_axis_off()
    ax.axhline(y=0, color='black')
    ax.set_xlabel('Upstream Data x')
    
    return fig, ax, mesh

def plot_arrows_heatmaps_olmo_peft_simple(arr, rows, cols):
    fig, ax = plt.subplots(figsize=(20,2))
    row_ivals = get_intervals(rows)
    col_ivals = get_intervals(cols)
    mesh = ax.pcolormesh(arr, vmin=-0.5, vmax=0.5, cmap='bwr', rasterized=True, zorder=5)

    ax4 = fig.add_axes([0.1, 0.23, 0.01, 0.65])  
    cbar = plt.colorbar(mesh, cax=ax4)
    ax4.yaxis.set_ticks_position('left')
    
    ax.set_ylim(-20, arr.shape[0] + 1)
    #print(col_ivals)
    y = -15
    for start, stop, name in col_ivals:
        ax.annotate('', xy=(start,y), xytext=(stop, y), arrowprops=dict(arrowstyle='<->', lw=1.5, shrinkA=0, shrinkB=0),
                   zorder=3)
        ax.text(int((start+stop)/ 2), y-1, map_text(name), ha='center', va='center', fontsize=12, zorder=4,
                bbox=dict(facecolor='white', edgecolor='none', boxstyle='square, pad=0.0'), rotation=0)
        ax.axvline(x=start, lw=1.5, zorder=5, color='black')
    ax.axvline(x=stop, lw=1.5, zorder=5, color='black')
    xmin = - int(arr.shape[1] / 50)
    ax.set_xlim(xmin * 2, arr.shape[1])
    for start, stop, name in row_ivals:
        print(start, stop)
        ax.annotate('', xy=(xmin, start), xytext=(xmin,stop), arrowprops=dict(arrowstyle='<->', lw=1.5, shrinkA=0, shrinkB=0),
                   zorder=3)
        ax.text(xmin, int((start+stop)/ 2), map_text(name), ha='center', va='center', fontsize=10, zorder=4,
                bbox=dict(facecolor='white', edgecolor='none', boxstyle='square, pad=0.0'), rotation=25)
        ax.axhline(y=start, lw=1.5, zorder=5, color='black')
    print(stop)
    ax.axhline(y=stop - 1 , lw=1.5, zorder=5, color='black') 
    
    ax.set_axis_off()
    ax.axhline(y=0, color='black')
    ax.set_xlabel('Upstream Data x')
    
    return fig, ax, mesh

def get_flan_tulu_mapping():
    config = load_configs('configs/llm/llm_defaults.yaml')
    tulu_tasks = ['Tulu ({})'.format(len(config.tulu_tasks)) for x in config.tulu_tasks]
    flan_tasks = ['FLAN\n({})'.format(len(config.flan_tasks)) for x in config.flan_tasks]
    tasks = np.array(flan_tasks + tulu_tasks)
    return pd.Series(tasks)

def get_dolma_mapping(arr_size):
    tasks = np.array(['Dolma' for _ in range(arr_size)])
    return pd.Series(tasks)


def get_dolma_mapping_10k_domain(arr_size, rename=False):
    assert arr_size == 10000
    dolma_domains = get_dolma_tasks_10k_sample()
    if rename:
        mapping = {
            'c4': 'c4',
            'common-crawl': 'common-crawl',
            's2': 's2',
            'stack-dedup': 'stack',
            'wikipedia': '',
            'gutenberg': '',
            'reddit': 'reddit',
        }
        dolma_domains = [mapping[x] for x in dolma_domains]
    return pd.Series(dolma_domains)

def plot_arrows_heatmaps_olmo_dolma(arr, rows, cols, scale=0.3):
    fig, ax = plt.subplots(figsize=(20,2))
    row_ivals = get_intervals(rows)
    col_ivals = get_intervals(cols)
    mesh = ax.pcolormesh(arr, vmin=-0.5, vmax=0.5, cmap='bwr', rasterized=True)

   
    ax4 = fig.add_axes([0.1, 0.272, 0.01, 0.61])  
    cbar = plt.colorbar(mesh, cax=ax4)
    ax4.yaxis.set_ticks_position('left')
    
    ax.set_ylim(-20, arr.shape[0])
    #print(col_ivals)
    y = -10
    for start, stop, name in col_ivals:
        ax.annotate('', xy=(start,y), xytext=(stop, y), arrowprops=dict(arrowstyle='<->', lw=1.5, shrinkA=0, shrinkB=0),
                   zorder=3)
        ax.text(int((start+stop)/ 2), y-1, map_text(name), ha='center', va='center', fontsize=12, zorder=4,
                bbox=dict(facecolor='white', edgecolor='none', boxstyle='square, pad=0.0'), rotation=0)
        ax.axvline(x=start, lw=1.5, zorder=5, color='black')
    ax.axvline(x=stop, lw=1.5, zorder=5, color='black')
    xmin = - int(arr.shape[1] / 50)
    ax.set_xlim(xmin * 2, arr.shape[1] + 10)
    for start, stop, name in row_ivals:
        print(start, stop)
        ax.annotate('', xy=(xmin, start), xytext=(xmin,stop), arrowprops=dict(arrowstyle='<->', lw=1.5, shrinkA=0, shrinkB=0),
                   zorder=3)
        ax.text(xmin, int((start+stop)/ 2), map_text(name), ha='center', va='center', fontsize=10, zorder=4,
                bbox=dict(facecolor='white', edgecolor='none', boxstyle='square, pad=0.0'), rotation=0)
        ax.axhline(y=start, lw=1.5, zorder=5, color='black')
    ax.axhline(y=stop, lw=1.5, zorder=5, color='black') 

    ax.set_axis_off()
    ax.axhline(y=0, color='black')
    ax.set_xlabel('Upstream Data x')
    
    return fig, ax, mesh

def get_recons(u, s, vh, rank, accum=False):
    if accum:
        start = 0
    else:
        start = rank
    stop = rank + 1
    ret = np.matmul(np.matmul(u[:,start:stop],np.diag(s[start:stop])), vh[start:stop,:])
        
    return ret

def plot_progressive(arr, u,s, vh, maxr):
    #u_ins, s_ins, vh_ins = np.linalg.svd(arr)
    fig, axes = plt.subplots(maxr,1,figsize=(10,maxr+1))
    for r in range(0,maxr):
        #print(s_ins[r-1])
        if r == 0:
            axes[r].set_ylabel('Full'.format(r-1),  rotation=30, fontsize=14, labelpad=10)
            axes[r].pcolormesh(arr,vmin=-0.5, vmax=0.5, cmap='bwr',rasterized=True)
    
        else:
            axes[r].set_ylabel('k={}'.format(r), rotation=30, fontsize=14, labelpad=10)
            recons_arr_dim_ins = get_recons(u,s,vh,r-1,accum=False) 
            #plt.figure(figsize=(10,1.5))
            axes[r].pcolormesh(recons_arr_dim_ins,vmin=-0.5, vmax=0.5, cmap='bwr',rasterized=True)

        axes[r].set_yticks([])
        axes[r].tick_params(left=False, right=False, labelbottom=False, bottom=False)
    return fig, axes

def fnorm_error(target, est):
    return ((target - est) ** 2).mean()

def r2_score(target, est):
    ssr = fnorm_error(target, est)
    sst = ((target - target.mean()) ** 2).mean()
    return 1 - ssr / sst

def get_singular_and_recons_error(arr):
    u, s, v = np.linalg.svd(arr)
    recons_errors = []
    recons = np.zeros_like(arr)
    for i in range(len(s)):
        recons += s[i] * np.matmul(u[:,i:i+1],v[i:i+1,:])
        err = r2_score(arr, recons)
        recons_errors.append(err)
    recons_errors = np.array(recons_errors)
    return s, recons_errors

def plot_svd_recons_plots(svs,recons, fig, ax, ax2, lim=30, name=''):
    #ax2.set_xlim(-1,80)
    cmap = plt.get_cmap('tab10')
    svs = svs[:lim]
    recons = recons[:lim]
    ax.bar(np.arange(len(svs)), svs, label=name)

    ax.grid(axis='y')

    ax2.plot(np.arange(len(svs)), recons, label=name, color=cmap(0.15), zorder=5, ls='--')
    ax2.set_ylabel('R$^2$')
    ax.set_ylabel('Singular Values')

    #ax.set_ylim(0, 0.03)
    #ax2.set_ylim(0,500)
    #ax.legend()

    return fig, ax

def plot_arrows_heatmaps_olmo_dolma_detailed(arr, rows, cols, scale=0.3):
    fig, ax = plt.subplots(figsize=(20,4))
    row_ivals = get_intervals(rows)
    col_ivals = get_intervals(cols)
    mesh = ax.pcolormesh(arr, vmin=-0.5, vmax=0.5, cmap='bwr', rasterized=True)

   
    ax4 = fig.add_axes([0.1, 0.252, 0.01, 0.62])  
    cbar = plt.colorbar(mesh, cax=ax4)
    ax4.yaxis.set_ticks_position('left')
    
    ax.set_ylim(-20, arr.shape[0])
    #print(col_ivals)
    y = -5
    for start, stop, name in col_ivals:
        ax.annotate('', xy=(start,y), xytext=(stop, y), arrowprops=dict(arrowstyle='<->', lw=1.5, shrinkA=0, shrinkB=0),
                   zorder=3)
        ax.text(int((start+stop)/ 2), y-1, map_text(name), ha='center', va='center', fontsize=10, zorder=4,
                bbox=dict(facecolor='white', edgecolor='none', boxstyle='square, pad=0.0'), rotation=0)
        ax.axvline(x=start,ymin=0.1,  lw=1.5, zorder=5, color='black')
    ax.axvline(x=stop, lw=1.5,ymin=0.1, zorder=5, color='black')
    xmin = - int(arr.shape[1] / 15)
    xmin2 = - 20
    ax.set_xlim(xmin * 2, arr.shape[1] + 10)
    for start, stop, name in row_ivals:
        print(start, stop)
        #ax.annotate('', xy=(xmin, start), xytext=(xmin,stop), arrowprops=dict(arrowstyle='<->', lw=1.5, shrinkA=0, shrinkB=0),
        #           zorder=3)
        ax.text(xmin2, int((start+stop)/ 2), map_text(name), ha='right', va='center', fontsize=10, zorder=4,
                bbox=dict(facecolor='white', edgecolor='none', boxstyle='square, pad=0.0'), rotation=0)
        ax.axhline(y=start, lw=1.0, zorder=5, color='black')
    ax.axhline(y=stop, lw=1.5, zorder=5, color='black') 

    ax.set_axis_off()
    ax.axhline(y=0, color='black')
    ax.set_xlabel('Upstream Data x')
    
    return fig, ax, mesh