import os
import json
import argparse
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir')
    parser.add_argument('--methods', nargs='*')
    parser.add_argument('--n_seed', type=int, default=10)
    parser.add_argument('--metrics', default='rmse_score')
    args = parser.parse_args()

    if args.methods is None:
        methods = ['additive_k30','svd_k30_d10','knn_baseline_k30']
    else:
        methods = args.methods

    for method in methods:
        scores = []
        for seed in range(args.n_seed):
            try:
                path = None
                path1 = os.path.join(args.base_dir, f'preds_{method}_results', f'score_seed_{seed}.json')
                path2 = os.path.join(args.base_dir, f'preds_{method}_results_fix', f'score_seed_{seed}.json')
                if os.path.exists(path1):
                    path = path1
                else:
                    path = path2
                
                if path:
                    with open(path) as f:
                        data = json.load(f)
                        score = data[args.metrics]
                        scores.append(score)
            except FileNotFoundError:
                pass


        print(method, np.mean(scores) * 100, np.std(scores) * 100, len(scores))