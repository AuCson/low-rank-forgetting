from torch.utils.data import Dataset
import numpy as np
import os
from .utils import truncate_prefix
from .mmlu import MMLUHelper
from .bbh import BBHHelper
from .utils import apply_chat_template, apply_chat_template_for_generation
import json
import pickle
import datasets
import random
from torch.utils.data import Subset
from .utils import deterministic_split, deterministic_random_indices
from utils.config import load_configs


def load_raw_ds(split, config, tokenizer, task_category, task_id=None):
    if config is None:
        config_files = [
            'configs/defaults.yaml', 
            'configs/llm/llm_defaults.yaml'
        ]
        config = load_configs(*config_files, templates=None)
    
    is_task_name = type(task_id) is not int 

    if task_category == 'mmlu':
        if split == 'test':
            print('Using test split')
            ds = SFTExampleOnlyDataset.from_mmlu([task_id if is_task_name else config.mmlu_tasks[task_id] ], 'test', config)
        else:
            ds = SFTExampleOnlyDataset.from_mmlu([task_id if is_task_name else config.mmlu_tasks[task_id]],'val',config)
    elif task_category == 'tulu':
        ds = SFTExampleOnlyDataset.from_tulu(config)
    elif task_category == 'bbh':
        ds = SFTExampleOnlyDataset.from_bbh([task_id if is_task_name else config.bbh_tasks[task_id]],'val',config)
    elif task_category == 'dolma':
        ds = SFTExampleOnlyDataset.from_dolma(config)
    elif task_category == 'truthful_qa':
        ds = SFTExampleOnlyDataset.from_truthful_qa(config, [task_id if is_task_name else  config.truthful_qa_tasks[task_id]])
    elif task_category == 'tulu_train':
        ds = SFTExampleOnlyDataset.from_tulu_train(config, [task_id if is_task_name else  config.tulu_tasks[task_id]])
    elif task_category == 'redpajama':
        ds = SFTExampleOnlyDataset.from_redpajama_sample(config)
    elif task_category == 'flan':
        ds = SFTExampleOnlyDataset.from_flan_by_task([task_id if is_task_name else config.flan_tasks[task_id]], 'train', config)
    elif task_category == 'dolly':
        ds = SFTExampleOnlyDataset.from_dolly([task_id if is_task_name else config.dolly_tasks[task_id]], 'train', config)
    elif task_category == 'olmo2pt':
        ds = SFTExampleOnlyDataset.from_olmo2pt(config)
    elif task_category == 'pile':
        ds = SFTExampleOnlyDataset.from_pile_sample(config)
    else:
        raise NotImplementedError(task_category)
    return ds


def load_ocl_ds_by_task_id(config, tokenizer, task_cat, task_id, **kwargs):
    if task_cat == 'mmlu':
        all_tasks = config.mmlu_tasks
        task = all_tasks[task_id]
        train_ds = SFTDataset.from_mmlu([task], 'val', config, tokenizer, **kwargs)
        test_ds = SFTDataset.from_mmlu([task], 'test', config, tokenizer, **kwargs)
    elif task_cat == 'bbh':
        all_tasks = config.bbh_tasks
        task = all_tasks[task_id]
        train_ds = SFTDataset.from_bbh([task], 'train', config, tokenizer, **kwargs)
        test_ds = SFTDataset.from_bbh([task], 'eval', config, tokenizer, **kwargs)
    elif task_cat == 'flan':
        all_tasks = config.flan_tasks
        task = all_tasks[task_id]
        train_ds = SFTDataset.from_flan_by_task([task],'train',config, tokenizer, **kwargs)
        test_ds = SFTDataset.from_flan_by_task([task],'validation',config,tokenizer, **kwargs)
    elif task_cat == 'truthful_qa':
        all_tasks = config.truthful_qa_tasks
        task = all_tasks[task_id]
        train_ds = SFTDataset.from_truthful_qa([task],'validation',config, tokenizer, **kwargs)
        test_ds = SFTDataset.from_truthful_qa([task],'test',config,tokenizer, **kwargs) 
    elif task_cat == 'tulu_train':
        all_tasks = config.tulu_tasks
        task = all_tasks[task_id]
        train_ds = SFTDataset.from_tulu_train([task], 'train', config, tokenizer, **kwargs)
        test_ds = SFTDataset.from_tulu_train([task], 'dev', config, tokenizer, **kwargs)  
    elif task_cat == 'dolly':
        task = config.dolly_tasks[task_id]
        train_ds = SFTDataset.from_dolly([task], 'train', config, tokenizer, **kwargs)
        test_ds = SFTDataset.from_dolly([task], 'dev', config, tokenizer, **kwargs)  
    elif task_cat == 'dolma_sample':
        task = None
        train_ds = SFTDataset.from_dolma_sample(None, 'train', config=config, tokenizer=tokenizer, **kwargs)
        test_ds = train_ds
    else:
        raise NotImplementedError(task_cat)
    return train_ds, test_ds, task

def load_ocl_example_only_ds_by_task_id(config, tokenizer, task_cat, task_id, include_gt):
    if task_cat == 'mmlu':
        all_tasks = config.mmlu_tasks
        task = all_tasks[task_id]
        train_ds = SFTExampleOnlyDataset.from_mmlu([task], 'val', config, tokenizer, include_gt=include_gt)
        test_ds = SFTExampleOnlyDataset.from_mmlu([task], 'test', config, tokenizer, include_gt=include_gt)
    else:
        raise NotImplementedError
    return train_ds, test_ds, task



def find_ans_start(input_ids, ans_start_tokens):
    pos = None
    input_ids = np.array(input_ids)
    for i in range(len(input_ids)-1, -1, -1):
        if input_ids[i] == ans_start_tokens[0]:
            if len(input_ids[i:i+len(ans_start_tokens)]) == len(ans_start_tokens) and \
            (input_ids[i:i+len(ans_start_tokens)] == ans_start_tokens).all():
                pos = i
                break
    return pos

def make_label(input_ids, ans_start_tokens, pad_token_id):
    input_ids = np.array(input_ids)
    pos = find_ans_start(input_ids, ans_start_tokens)
    labels = np.copy(input_ids)
    if pos is None:
        print('No label found')
        labels[:] = -100
    else:
        labels[:pos + len(ans_start_tokens)] = -100
    labels[labels == pad_token_id] = -100
    return labels

def make_label_lm(input_ids, pad_token_id):
    input_ids = np.array(input_ids)
    labels = np.copy(input_ids)
    labels[labels == pad_token_id] = -100
    return labels


class SFTDataset(Dataset):
    def __init__(self, config, tokenier, input_texts, task_names, indexes, is_lm=False, include_labels=False, ans_start_patt=None):
        self.config = config
        self.tokenizer = tokenier
        self.input_texts = input_texts
        self.input_encoding = truncate_prefix(tokenier, input_texts, self.config.max_input_length)
        self.indexes = indexes
        self.task_names = task_names
        self.is_lm = is_lm

        self.include_labels = include_labels

        if ans_start_patt is None:
            ans_start_patt = config.ans_start_pattern
            print(f'Warning: ans start pattern set to {ans_start_patt}')

        if include_labels:
            self.ans_start_tokens = np.array(self.tokenizer.encode(ans_start_patt))
            print('Ans start tokens', self.ans_start_tokens)
            self.labels = []
            for idx in range(len(self.input_encoding.input_ids)):
                self.labels.append(
                    make_label(self.input_encoding.input_ids[idx], self.ans_start_tokens, self.tokenizer.pad_token_id) 
                    if not is_lm else make_label_lm(self.input_encoding.input_ids[idx], self.tokenizer.pad_token_id)
                )

    @classmethod
    def from_mmlu(cls, tasks, split, config, tokenizer, skip_encoding=False, **kwargs):
        all_examples = []
        all_task_names = []
        for task in tasks:
            mmlu_helper = MMLUHelper(config, task)
            answer_type = config.mmlu.answer_type
            cot = config.mmlu.is_cot
            few_shot = config.mmlu.is_few_shot
            prompt = mmlu_helper.get_prompt(task, cot=cot, answer_type=answer_type, is_few_shot=few_shot)
            examples = mmlu_helper.create_examples(split, prompt, cot=cot, answer_type=answer_type, example_format='lm')
            task_names = [example[-1] for example in examples]

            all_examples.extend(examples)
            all_task_names.extend(task_names)
        input_texts = apply_chat_template(all_examples, tokenizer)
        indexes = [_ for _ in range(len(all_examples))]
        ds = cls(config, tokenizer, input_texts, all_task_names, indexes, **kwargs)
        return ds

    @classmethod
    def from_bbh(cls, tasks, split, config, tokenizer, skip_encoding=False, **kwargs):
        all_examples = []
        all_task_names = []
        for task in tasks:
            mmlu_helper = BBHHelper(config, task)
            cot = config.bbh.is_cot
            examples = mmlu_helper.create_examples(split, cot=cot, example_format='lm')
            task_names = [example[-1] for example in examples]

            all_examples.extend(examples)
            all_task_names.extend(task_names)
        input_texts = apply_chat_template(all_examples, tokenizer)
        indexes = [_ for _ in range(len(all_examples))]
        ds = cls(config, tokenizer, input_texts, all_task_names, indexes, **kwargs)
        return ds

    @classmethod
    def from_flan_by_task(cls, tasks, split, config, tokenizer, skip_encoding=False, **kwargs):
        all_examples = []
        all_task_names = []
        if split == 'dev':
            split_name = 'validation'
        else:
            split_name = split

        for task in tasks:
            with open(os.path.join(config.flan_by_task_dir, '{}_{}.json'.format(task, split_name))) as f:
                data = json.load(f)
            examples = [[x['inputs'], x['targets'], x['task']] for x in data]
            all_examples.extend(examples)
            all_task_names.extend([example[-1] for example in examples])

        if getattr(config, 'max_flan_example', -1) > 0:
            print('Max flan example is {} for FPD'.format(config.max_flan_example))

            all_examples = all_examples[:config.max_flan_example]
            all_task_names = all_task_names[:config.max_flan_example]
        input_texts = apply_chat_template(all_examples, tokenizer)
        indexes = [_ for _ in range(len(all_examples))]
        ds = cls(config, tokenizer, input_texts, all_task_names, indexes, **kwargs)
        return ds

    @classmethod
    def from_tulu_train(cls, tasks, split, config, tokenizer, skip_encoding=False, **kwargs):
        ds = datasets.load_dataset("allenai/tulu-v2-sft-mixture")
        all_examples = []
        for example in ds['train']:
            task = example['dataset'].split('.')[0]
            if task in tasks:
                if example['messages'][0]['role'] == 'system':
                    all_examples.append([
                        example['messages'][1]['content'],
                        example['messages'][2]['content'],
                        task
                    ])
                else:
                    all_examples.append([
                        example['messages'][0]['content'],
                        example['messages'][1]['content'],
                        task
                    ])

        all_task_names = [_[-1] for _ in all_examples]
        if getattr(config, 'max_tulu_train_example', -1) > 0:
            print('Max tulu_train example is {}'.format(config.max_tulu_train_example))
            all_examples = all_examples[:config.max_flan_example]
            all_task_names = all_task_names[:config.max_flan_example]

        input_texts = apply_chat_template(all_examples, tokenizer)
        indexes = [_ for _ in range(len(all_examples))]
        ds = cls(config, tokenizer, input_texts, all_task_names, indexes, **kwargs)
        return ds

    @classmethod
    def from_tulu(cls, tasks, split, config, tokenizer, skip_encoding=False, **kwargs):
        with open(config.tulu.path) as f:
            raw_examples = json.load(f)
        all_examples = []

        for example in raw_examples:
            task = example['dataset'].split('.')[0]
            if example['messages'][0]['role'] == 'system':
                all_examples.append([
                    example['messages'][1]['content'],
                    example['messages'][2]['content'],
                    task
                ])
            else:
                all_examples.append([
                    example['messages'][0]['content'],
                    example['messages'][1]['content'],
                    task
                ])

        input_texts = apply_chat_template(all_examples, tokenizer)
        indexes = [_ for _ in range(len(all_examples))]
        all_task_names = [_[-1] for _ in all_examples]
        ds = cls(config, tokenizer, input_texts, all_task_names, indexes, **kwargs)
        return ds

    @classmethod
    def from_truthful_qa(cls, tasks, split, config, tokenizer, skip_encoding=False, **kwargs):
        ds = datasets.load_dataset('truthful_qa', 'generation')
        all_examples = []
        for example in ds['validation']:
            task = example['category'].split(':')[0]
            if task in tasks:
                all_examples.append([
                    example['question'],
                    example['best_answer'],
                    example['category']
                ])
        all_task_names = [_[-1] for _ in all_examples]
        input_texts = apply_chat_template(all_examples, tokenizer)
        indexes = [_ for _ in range(len(all_examples))]
        ds = cls(config, tokenizer, input_texts, all_task_names, indexes, **kwargs)
        return ds

    @classmethod
    def from_dolma_sample(cls, tasks, split, config, tokenizer, skip_encoding=False, **kwargs):
        with open(config.dolma.sample_path,'rb') as f:
            raw_examples = pickle.load(f)
        all_examples = [
            [example['text'], '{}_{}'.format(example['example_id'], example['chunk_id'])] for example in raw_examples
        ]
        input_texts = [x[0] for x in all_examples]
        all_task_names = [x[-1] for x in all_examples]
        indexes = [_ for _ in range(len(all_examples))]
        ds = cls(config, tokenizer, input_texts, all_task_names, indexes, is_lm=True, **kwargs)
        return ds
    
    @classmethod
    def from_dolly(cls, tasks, split, config, tokenizer, skip_encoding=False, **kwargs):
        full_ds = datasets.load_dataset('databricks/databricks-dolly-15k')['train']
        assert len(tasks) == 1
        task = tasks[0]

        task_ds = full_ds.filter(lambda x: x['category'] == task)
        train_idxs, test_idxs = deterministic_split(len(task_ds), 0.8)
        train_task_ds, test_task_ds = task_ds.select(train_idxs), task_ds.select(test_idxs)

        hf_ds = train_task_ds if split == 'train' else test_task_ds
        all_examples = []
        for example in hf_ds:
            if example['context']:
                inst = 'Context: {}\n\n{}'.format(example['context'], example['instruction'])
            else:
                inst = example['instruction']
            all_examples.append([
                inst,
                example['response'],
                -1
            ])
        input_texts = apply_chat_template(all_examples, tokenizer)
        all_task_names = [task for _ in range(len(all_examples))]
        indexes = train_idxs if split == 'train' else test_idxs

        ds = cls(config, tokenizer, input_texts, all_task_names, indexes, **kwargs)
        return ds

    @classmethod
    def from_auto(cls, ds_category, **kwargs):
        if ds_category == 'mmlu':
            return cls.from_mmlu(**kwargs)
        elif ds_category == 'bbh':
            return cls.from_bbh(**kwargs)
        elif ds_category == 'tulu':
            return cls.from_tulu(**kwargs)
        elif ds_category == 'truthful_qa':
            return cls.from_truthful_qa(**kwargs)
        elif ds_category == 'tulu_train':
            return cls.from_tulu_train(**kwargs)
        elif ds_category in ['dolma_sample','dolma']:
            return cls.from_dolma_sample(**kwargs)
        elif ds_category == 'flan':
            return cls.from_flan_by_task(**kwargs)
        elif ds_category == 'dolly':
            return cls.from_dolly(**kwargs)
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        input_ids = self.input_encoding['input_ids'][idx]
        task_name = self.task_names[idx]

        example = {
            'input_ids': input_ids,
            'task_name': task_name,
        }

        if self.include_labels:
            example['labels'] = self.labels[idx]

        return example

    def __len__(self):
        return len(self.input_encoding['input_ids'])


class SFTExampleOnlyDataset(Dataset):
    def __init__(self, examples, is_lm=False):
        self.examples = examples
        self.is_lm = is_lm

    def shuffle(self, seed):
        random.Random(seed).shuffle(self.examples)

    def make_subsample(self, n):
        indices = deterministic_random_indices(len(self.examples), n)
        examples = [self.examples[x] for x in indices]
        ss = SFTExampleOnlyDataset(examples, is_lm=self.is_lm)
        return ss

    @classmethod
    def from_mmlu(cls, tasks, split, config):
        all_examples = []
        for task in tasks:
            mmlu_helper = MMLUHelper(config, task)
            answer_type = config.mmlu.answer_type
            cot = config.mmlu.is_cot
            few_shot = config.mmlu.is_few_shot
            prompt = mmlu_helper.get_prompt(task, cot=cot, answer_type=answer_type, is_few_shot=few_shot)
            examples = mmlu_helper.create_examples(split, prompt, cot=cot, answer_type=answer_type, example_format='lm')
            task_names = [example[-1] for example in examples]
            all_examples.extend(examples)
        ds = cls(all_examples)
        return ds

    @classmethod
    def from_bbh(cls, tasks, split, config):
        all_examples = []
        for task in tasks:
            bbh_helper = BBHHelper(config, task)

            cot = config.bbh.is_cot
            examples = bbh_helper.create_examples(split,  cot=cot, example_format='lm')
            task_names = [example[-1] for example in examples]
            all_examples.extend(examples)
        ds = cls(all_examples)
        return ds

    @classmethod
    def from_tulu(cls, config):
        with open(config.tulu.path) as f:
            raw_examples = json.load(f)
        # all_examples = [
        #     [example['messages'][0]['content'],
        #      example['messages'][1]['content'],
        #      example['dataset'].split('.')[0]] for example in raw_examples
        # ]

        all_examples = []
        for example in raw_examples:
            task = example['dataset'].split('.')[0]

            if example['messages'][0]['role'] == 'system':
                all_examples.append([
                    example['messages'][1]['content'],
                    example['messages'][2]['content'],
                    task
                ])
            else:
                all_examples.append([
                    example['messages'][0]['content'],
                    example['messages'][1]['content'],
                    task
                ])

        ds = cls(all_examples)
        return ds

    @classmethod
    def from_dolma(cls, config):
        with open(config.dolma.sample_path,'rb') as f:
            raw_examples = pickle.load(f)
        all_examples = [
            [example['text'], '{}_{}'.format(example['example_id'], example['chunk_id'])] for example in raw_examples
        ]
        ds = cls(all_examples, is_lm=True)
        return ds

    @classmethod
    def from_truthful_qa(cls, config, tasks):
        ds = datasets.load_dataset('truthful_qa', 'generation')
        all_examples = []
        for example in ds['validation']:
            task = example['category'].split(':')[0]
            if task in tasks:
                all_examples.append([
                    example['question'],
                    example['best_answer'],
                    example['category']
                ])
        ds = cls(all_examples)
        return ds

    @classmethod
    def from_tulu_train(cls, config, tasks):
        ds = datasets.load_dataset("allenai/tulu-v2-sft-mixture")
        all_examples = []
        for example in ds['train']:
            task = example['dataset'].split('.')[0]
            if task in tasks:
                if example['messages'][0]['role'] == 'system':
                    all_examples.append([
                        example['messages'][1]['content'],
                        example['messages'][2]['content'],
                        task
                    ])
                else:
                    all_examples.append([
                        example['messages'][0]['content'],
                        example['messages'][1]['content'],
                        task
                    ])
        ds = cls(all_examples)
        return ds

    @classmethod
    def from_redpajama_sample(cls, config):
        hf_ds = datasets.load_from_disk('data/chunked_redpajama_sample_100000')
        all_examples = []
        for example in hf_ds:
            all_examples.append([
                example['token_ids_text'],
                example['orig_idx'],
                example['domain']
            ])
        ds = cls(all_examples, is_lm=True)
        return ds

    @classmethod
    def from_flan_by_task(cls, tasks, split, config):
        all_examples = []
        all_task_names = []
        if split == 'dev':
            split_name = 'validation'
        else:
            split_name = split

        for task in tasks:
            with open(os.path.join(config.flan_by_task_dir, '{}_{}.json'.format(task, split_name))) as f:
                data = json.load(f)
            examples = [[x['inputs'], x['targets'], x['task']] for x in data]
            all_examples.extend(examples)
            all_task_names.extend([example[-1] for example in examples])

        if getattr(config, 'max_flan_example', -1) > 0:
            print('Max flan example is {} for FPD'.format(config.max_flan_example))
            all_examples = all_examples[:config.max_flan_example]
            all_task_names = all_task_names[:config.max_flan_example]

        ds = cls(all_examples)
        return ds
    
    @classmethod
    def from_dolly(cls, tasks, split, config):
        full_ds = datasets.load_dataset('databricks/databricks-dolly-15k')['train']
        assert len(tasks) == 1
        task = tasks[0]

        task_ds = full_ds.filter(lambda x: x['category'] == task)
        train_idxs, test_idxs = deterministic_split(len(task_ds), 0.8)
        train_task_ds, test_task_ds = task_ds.select(train_idxs), task_ds.select(test_idxs)

        hf_ds = train_task_ds if split == 'train' else test_task_ds
        all_examples = []
        for example in hf_ds:
            if example['context']:
                inst = 'Context: {}\n\n{}'.format(example['context'], example['instruction'])
            else:
                inst = example['instruction']
            all_examples.append([
                inst,
                example['response'],
                -1
            ])
       
        ds = cls(all_examples)
        return ds

    @classmethod
    def from_olmo2pt(cls, config):
        with open(config.olmo2pt.sample_path,'rb') as f:
            raw_examples = pickle.load(f)
        all_examples = [
            [example['truncated_text'],  '{}_{}'.format(example['domain'], idx)] for idx,example in enumerate(raw_examples)
        ]
        ds = cls(all_examples, is_lm=True)
        return ds

    @classmethod
    def from_pile_sample(cls, config):
        with open(config.pile.sample_path, 'rb') as f:
            raw_examples = pickle.load(f)
        all_examples = [
            [example['truncated_text'],  '{}_{}'.format(example['meta']['pile_set_name'], idx)] for idx,example in enumerate(raw_examples)
        ]
        ds = cls(all_examples, is_lm=True)
        return ds

    @classmethod
    def from_auto(cls, ds_category, **kwargs):
        if ds_category == 'mmlu':
            return cls.from_mmlu(**kwargs)
        elif ds_category == 'bbh':
            return cls.from_bbh(**kwargs)
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)

    def get_chat_or_raw_examples(self, include_gt, tokenizer):
        if self.is_lm:
            input_texts = [x[0] for x in self.examples]
        else:
            if include_gt:
                input_texts = apply_chat_template(self.examples, tokenizer)
            else:
                input_texts = apply_chat_template_for_generation(self.examples, tokenizer)
        return input_texts
    
    def get_gt_answers(self):
        return [example[1] for example in self.examples]
    