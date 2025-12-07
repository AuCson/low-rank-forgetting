from .lm import SFTDataset, SFTExampleOnlyDataset, load_raw_ds, load_ocl_ds_by_task_id, load_ocl_example_only_ds_by_task_id
from torch.utils.data import Subset
import random


def shuffle_ds(ds, seed):
    print(f'Performing shuffling with seed {seed}')
    indices = random.Random(seed).sample(range(0, len(ds)), len(ds))
    ds_shuffle = Subset(ds, indices)
    return ds_shuffle

