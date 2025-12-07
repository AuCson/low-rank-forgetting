#!/bin/bash

stepn=100
task_category=mmlu
start_task_id=0
stop_task_id=57

for ((task_id = start_task_id ; task_id < stop_task_id ; task_id++ ))
do
echo "${task_id}"
python vllm_inference.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/ocl_ins/stat_7b_on_flan_v2.yaml --templates ocl_task_id=${task_id} ocl_task_category=${task} \
--stat_output --skip_eval_ocl_ds --n_gpus 1 --is_ins_pt --gpu_memory_utilization 0.85
done
 
task_category=bbh
start_task_id=0
stop_task_id=27


for ((task_id = start_task_id ; task_id < stop_task_id ; task_id++ ))
do
echo "${task_id}"
python vllm_inference.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/ocl_ins/stat_7b_on_flan_v2.yaml --templates ocl_task_id=${task_id} ocl_task_category=${task} \
--stat_output --skip_eval_ocl_ds --n_gpus 1 --is_ins_pt --gpu_memory_utilization 0.85
done

task_category=truthful_qa
start_task_id=0
stop_task_id=32

for ((task_id = start_task_id ; task_id < stop_task_id ; task_id++ ))
do
echo "${task_id}"
python vllm_inference.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/ocl_ins/stat_7b_on_flan_v2.yaml --templates ocl_task_id=${task_id} ocl_task_category=${task} \
--stat_output --skip_eval_ocl_ds --n_gpus 1 --is_ins_pt --gpu_memory_utilization 0.85
done

task_category=dolly
start_task_id=0
stop_task_id=8

for ((task_id = start_task_id ; task_id < stop_task_id ; task_id++ ))
do
echo "${task_id}"
python vllm_inference.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/ocl_ins/stat_7b_on_flan_v2.yaml --templates ocl_task_id=${task_id} ocl_task_category=${task} \
--stat_output --skip_eval_ocl_ds --n_gpus 1 --is_ins_pt --gpu_memory_utilization 0.85
done