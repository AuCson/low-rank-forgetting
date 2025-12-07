#!/bin/bash


start_task_id=0
task_category="dolma_sample"

accelerate launch --num_processes 1 collect_gradients.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml configs/grads/7b_flan.yaml --templates task_id=${task_id} task_category=${task_category} max_input_length=2048 --max_example 10000 --predef_full_len 141816
