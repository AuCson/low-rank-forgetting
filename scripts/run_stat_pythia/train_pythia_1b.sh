#!/bin/bash

task_category=${1}
start_task_id=${2}
stop_task_id=${3}
stepn=1k


for ((task_id = start_task_id ; task_id < stop_task_id ; task_id++ ))
do
echo "Current task id ${task_id}"

accelerate launch --num_processes 2 run_llm_ocl.py \
--config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/pythia/pythia_1b_ft.yaml \
--ocl_task ${task_category} --templates TASK_ID=${task_id} TASK_CATEGORY=${task_category}

done
