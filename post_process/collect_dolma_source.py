import gzip, json
from data_utils import load_raw_ds
from tqdm import tqdm
import pickle

def idx2digit(x):
    s = str(x)
    if len(s) < 4:
        s = '0' * (4 - len(s)) + s
    return s

def get_repeating_segments(lst):
    result = []
    start = 0  # Start position of the current segment
    
    for i in range(1, len(lst) + 1):
        # Check if we've reached the end of the list or the current item is different
        if i == len(lst) or lst[i] != lst[start]:
            result.append((lst[start], start, i))  # Append the tuple (item, start, stop)
            start = i  # Update the start position for the next segment

    return result

if __name__ == '__main__':
    with open('data/dolma_chunked_sample/stratified_1_100_tokenize_fix.pkl','rb') as f:
        dolma_ds = pickle.load(f)
    source_names = []

    current_opened_idx = -1
    orig_data = None
    filename_format = '/mnt/nfs1/xsjin/dolma_data/dolma_v1_6-sample/v1_5r2_sample-{}.json.gz'

    for idx in tqdm(range(len(dolma_ds)), total=len(dolma_ds)):
        example = dolma_ds[idx]
        example_id, chunk_id = example['example_id'], example['chunk_id']
        if chunk_id != current_opened_idx:
            filename = filename_format.format(idx2digit(chunk_id))
            with gzip.open(filename) as f:
                orig_data = [json.loads(x) for x in f.readlines()]
            print('Loaded {}'.format(filename))
            current_opened_idx = chunk_id
        try:
            source_names.append(orig_data[example_id]['source'])
        except KeyError:
            source_names.append('Unknown')
    
    with open('data/dolma_chunked_sample/source_names.pkl','wb') as wf:
        pickle.dump(source_names, wf)