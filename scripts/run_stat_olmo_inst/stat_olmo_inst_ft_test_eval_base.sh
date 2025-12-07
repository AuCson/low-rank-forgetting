#!/bin/bash

task_category=dolly
tasks="0"


for task_id in ${tasks[@]}
do

echo "${task_id}"

python vllm_inference.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/ocl_ins/stat_7b_on_tulu.yaml --templates ocl_task_id=${task_id} ocl_task_category=${task_category} \
--stat_ppl --skip_eval_ocl_ds --n_gpus 1 --is_ins_pt --gpu_memory_utilization 0.85 --eval_base


done
 

