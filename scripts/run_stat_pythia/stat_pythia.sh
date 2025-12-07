#!/bin/bash

size=${1}
task="tulu_train"
start_task_id=0
stop_task_id=11

for ((task_id = start_task_id ; task_id < stop_task_id ; task_id++ ))
do
python vllm_inference.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/pythia/stat_pythia.yaml --templates ocl_task_id=${task_id} ocl_task_category=${task} ocl_step=1k size=${size} \
--stat_ppl --skip_eval_ocl_ds --n_gpus 1 --gpu_memory_utilization 0.8
done


task="dolly"
start_task_id=0
stop_task_id=8

for ((task_id = start_task_id ; task_id < stop_task_id ; task_id++ ))
do
python vllm_inference.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/pythia/stat_pythia.yaml --templates ocl_task_id=${task_id} ocl_task_category=${task} ocl_step=1k size=${size} \
--stat_ppl --skip_eval_ocl_ds --n_gpus 1 --gpu_memory_utilization 0.8 
done