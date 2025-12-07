#!/bin/bash


start_task_id=${2}
stop_task_id=${3}
task_category=${1}

for ((task_id = start_task_id ; task_id < stop_task_id ; task_id++ ))
do
python collect_sentence_reps.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml configs/grads/1b_flan.yaml --templates task_id=${task_id} task_category=${task_category} --max_example 1000
done