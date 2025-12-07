from datasets import load_dataset
import random
from collections import defaultdict
import json

if __name__ == '__main__':
    ds = load_dataset('allenai/tulu-v2-sft-mixture')
    train_ds = ds['train']
    flan_ss = []
    for idx,example in enumerate(train_ds):
        eid = example['id']
        task = example['dataset'].split('.')[0]
        if task == 'flan_v2':
            flan_ss.append(example)
    # train subsample
    print(len(flan_ss))

    
    test_examples = random.Random(0).sample(flan_ss, 10000)
    
    with open('data/tulu_sample/sample_10k_flan.json','w') as f:
        json.dump(test_examples, f)
        