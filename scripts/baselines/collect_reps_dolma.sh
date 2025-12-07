#!/bin/bash


start_task_id=0
task_category="dolma_sample"

python collect_sentence_reps.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml configs/grads/1b_flan_2k.yaml --templates task_id=${task_id} task_category=${task_category} max_input_length=2048 --max_example 10000 --predef_full_len 141816
