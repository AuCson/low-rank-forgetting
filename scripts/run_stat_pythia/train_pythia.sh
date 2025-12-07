#!/bin/bash


model_size=${1}
stepn=1k


if (( ${model_size} == "1b" || ${model_size} == "3b" )); then
    num_processes=2
else
    num_processes=4
fi

task_category="tulu_train"
start_task_id=0
stop_task_id=11

for ((task_id = start_task_id ; task_id < stop_task_id ; task_id++ ))
do
  accelerate launch --num_processes ${num_processes} --main_process_port 29501 run_llm_ocl.py \
--config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/pythia/pythia_${model_size}_ft.yaml \
--ocl_task ${task_category} --templates TASK_ID=${task_id} TASK_CATEGORY=${task_category}
done


task_category="dolly"
start_task_id=0
stop_task_id=8

for ((task_id = start_task_id ; task_id < stop_task_id ; task_id++ ))
do
  accelerate launch --num_processes ${num_processes} --main_process_port 29501 run_llm_ocl.py \
--config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/pythia/pythia_${model_size}_ft.yaml \
--ocl_task ${task_category} --templates TASK_ID=${task_id} TASK_CATEGORY=${task_category}
done


