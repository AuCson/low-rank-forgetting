from datasets import load_dataset
import random
from collections import defaultdict
import json

if __name__ == '__main__':
    ds = load_dataset('allenai/tulu-v2-sft-mixture')
    train_ds = ds['train']
    tasks = []
    task2num = defaultdict(int)
    task2idx = defaultdict(list)
    for idx,example in enumerate(train_ds):
        eid = example['id']
        task = example['dataset'].split('.')[0]
        tasks.append(task)
        task2num[task] += 1
        task2idx[task].append(idx)

    # train subsample

    train_example_idxs = {}
    for task, examples in task2idx.items():
        num = 1000 #max(int(len(examples) * 0.05), 1000)
        num = min(num, len(examples))
        print('Sample {} for {}'.format(num, task))
        
        idxs = sorted(random.Random(0).sample(examples, num))
        train_example_idxs[task] = set(idxs)

    test_example_ss = []
    max_sample_n = 1000
    for task, example_idxs in task2idx.items():
        task_train_idxs = train_example_idxs[task]
        filtered_idxs = [x for x in example_idxs if x not in task_train_idxs]

        if len(filtered_idxs) < max_sample_n:
            sampled_idxs = filtered_idxs
        else:
            sampled_idxs = random.Random(1).sample(filtered_idxs, max_sample_n)
        print('Sampled {} examples for task {}'.format(len(sampled_idxs), task))
        test_example_ss.extend(sampled_idxs)
    
    test_examples = [train_ds[x] for x in test_example_ss]
    
    with open('data/tulu_sample/sample_1k_test.json','w') as f:
        json.dump(test_examples, f)
        