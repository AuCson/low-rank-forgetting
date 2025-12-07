from .fit_utils import get_logistic_mf_recons_progressive
import sklearn, json
import numpy as np

if __name__ == '__main__':
    fgt = np.load('stats/olmo-7b-ins-peft/flan_fgt_bin.npy')
    all_recons = get_logistic_mf_recons_progressive(fgt, 11)
    recons_scores = []
    for recons_sample in all_recons[1:]:
        f1 = sklearn.metrics.f1_score(fgt.reshape(-1).astype(int), recons_sample.reshape(-1) > 0.5)
        prec = sklearn.metrics.precision_score(fgt.reshape(-1).astype(int), recons_sample.reshape(-1) > 0.5)
        recall = sklearn.metrics.recall_score(fgt.reshape(-1).astype(int), recons_sample.reshape(-1) > 0.5)
        auc_roc = sklearn.metrics.roc_auc_score(fgt.reshape(-1), recons_sample.reshape(-1))
        recons_scores.append({
            'f1': f1,
            'prec': prec,
            'recall': recall,
            'auc_roc': auc_roc
        })
    with open('stats/olmo-7b-ins-peft/flan_fgt_bin_fit_results.json','w') as wf:
        json.dump(recons_scores, wf)