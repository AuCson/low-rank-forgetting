import numpy as np
import argparse
from .common import find_files
import os
import json


def get_ocl_loss_from_state_single(path, first=False):
    ret_loss = None
    if os.path.exists(path):
        with open(path) as f:
            state_data = json.load(f)
        items = state_data['log_history']
        itr = reversed(items) if not first else items
        for item in itr:
            if 'eval_loss' in item:
                ret_loss = item['eval_loss']
                break
    return ret_loss

def get_all_ocl_loss(root_dir, max_task_n):
    all_losses = []
    for task_id in range(max_task_n):
        path = os.path.join(root_dir, f'task_{task_id}', 'trainer_state.json')
        task_eval_loss = get_ocl_loss_from_state_single(path)
        all_losses.append(task_eval_loss)
    all_losses = np.array(all_losses).astype(np.float32)
    return all_losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir')
    parser.add_argument('max_task_n', type=int)
    args = parser.parse_args()

    all_losses = get_all_ocl_loss(args.root_dir, args.max_task_n)
    print(all_losses)

