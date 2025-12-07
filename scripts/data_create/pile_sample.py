import datasets
from datasets import load_dataset
import pickle
from transformers import AutoTokenizer

if __name__ == '__main__':
    pile_ds = load_dataset('monology/pile-uncopyrighted', split='train', streaming=True)
    N = 10000
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-6.9b')
    examples = []
    for example in pile_ds:
        if len(example['text']) > 100:
            examples.append(example)
        if len(examples) == N:
            break
    
    all_texts = [example['text'] for example in examples]
    #print(all_texts[:5])
    all_tokens = tokenizer(all_texts, truncation=True, max_length=2000)
    print('Done tokenization')
    
    for example, tokens in zip(examples, all_tokens.input_ids):
        truncated_text = tokenizer.decode(tokens)
        example['truncated_text'] = truncated_text
    
    with open('data/pile_sample/pile_first_10k.pkl','wb') as wf:
        pickle.dump(examples, wf)