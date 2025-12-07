#!/bin/bash

task_id=0
task_category="mmlu"

python vllm_inference.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/ocl_ins/stat_7b_on_flan_v2.yaml --templates ocl_task_id=${task_id} ocl_task_category=${task_category} \
--stat_output --skip_eval_ocl_ds --n_gpus 1 --is_ins_pt --gpu_memory_utilization 0.85 --eval_base