#!/bin/bash

task_category=tulu_train
start_task_id=0
stop_task_id=11
stepn=1k

for ((task_id = start_task_id ; task_id < stop_task_id ; task_id++ ))
do
echo "Current task id ${task_id}"
python vllm_inference.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/olmo2/stat_13b_olmo2pt.yaml --templates ocl_task_id=${task_id} ocl_task_category=${ocl_task_category} ocl_step=${ocl_step} \
--stat_ppl --skip_eval_ocl_ds --n_gpus 1 --gpu_memory_utilization 0.65
done

task_category=dolly
start_task_id=0
stop_task_id=8
stepn=1k


for ((task_id = start_task_id ; task_id < stop_task_id ; task_id++ ))
do
echo "Current task id ${task_id}"
python vllm_inference.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/olmo2/stat_13b_olmo2pt.yaml --templates ocl_task_id=${task_id} ocl_task_category=${ocl_task_category} ocl_step=${ocl_step} \
--stat_ppl --skip_eval_ocl_ds --n_gpus 1 --gpu_memory_utilization 0.65
done