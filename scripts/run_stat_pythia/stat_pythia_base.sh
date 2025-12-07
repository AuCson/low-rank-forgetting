#!/bin/bash

task_id=0
ocl_task_category="tulu_train"
ocl_step=1k

for size in 12b
do

python vllm_inference.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/pythia/stat_pythia.yaml --templates ocl_task_id=${task_id} ocl_task_category=${ocl_task_category} ocl_step=${ocl_step} size=${size} \
--stat_ppl --skip_eval_ocl_ds --n_gpus 1 --eval_base --gpu_memory_utilization 0.7

done