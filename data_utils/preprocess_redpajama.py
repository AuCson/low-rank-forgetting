from datasets import load_dataset, Dataset
from collections import defaultdict
import json
from transformers import AutoTokenizer
from more_itertools import chunked
import random

key2domains = {
    ('timestamp', 'yymm', 'arxiv_id', 'language', 'url'): 'arxiv',
    ('short_book_title', 'publication_date', 'url'): 'books',
    ('title',): 'books',
    ('timestamp', 'url', 'language', 'source'): 'c4',
    ('pred_label', 'pred_label_prob', 'wiki_prob', 'source'): 'cc',
    ('content_hash',
    'timestamp',
    'source',
    'line_count',
    'max_line_length',
    'avg_line_length',
    'alnum_prop',
    'repo_name',
    'id',
    'size',
    'binary',
    'copies',
    'ref',
    'path',
    'mode',
    'license',
    'language'): 'github',
    ('language', 'url', 'timestamp', 'source', 'question_score'): 'stackexchange',
    ('title', 'url', 'language', 'timestamp'): 'wikipedia'
 }

def tokenize_into_chunks(ds, tokenizer, chunk_size=2048):
    def tokenize_map(example):
        text = example['text'] + tokenizer.eos_token
        example['token_ids'] = tokenizer.encode(text)
        return example

    ds = ds.map(tokenize_map)

    for idx in range(len(ds)):
        if idx % 10000 == 0:
            print(idx)
        item = ds[idx]
        keys = tuple(eval(item['meta']).keys())
        domain = key2domains[keys]

        for chunk_id, chunk in enumerate(chunked(item['token_ids'], chunk_size)):
            new_example = {
                'orig_idx': idx,
                'domain': domain,
                'chunk_idx': chunk_id,
                'token_ids': chunk,
                'token_ids_text': tokenizer.decode(chunk)
            }
            yield new_example

if __name__ == '__main__':
    tokenizer_name = 'mosaicml/mpt-7b'
    subsample_num = 100000
    print(f'Tokenizer is {tokenizer_name}')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    ds = load_dataset("togethercomputer/RedPajama-Data-1T-Sample")['train']
    ds = ds.select(random.Random(0).sample(range(len(ds)), subsample_num))
    
    chunked_ds = Dataset.from_generator(tokenize_into_chunks, gen_kwargs={'ds': ds, 'tokenizer': tokenizer})
    
    chunked_ds.save_to_disk(f'data/chunked_redpajama_sample_{subsample_num}')