#!/bin/bash


start_task_id=0
stop_task_id=8
task_category="dolly"

for ((task_id = start_task_id ; task_id < stop_task_id ; task_id++ ))
do
python collect_gradients.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml configs/grads/1b_flan.yaml --templates task_id=${task_id} task_category=${task_category} --max_example 1000 --variant diff
done