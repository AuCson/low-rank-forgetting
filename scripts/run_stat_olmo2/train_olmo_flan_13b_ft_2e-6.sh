#!/bin/bash

task_category=tulu_train
start_task_id=0
stop_task_id=11
stepn=1k


for ((task_id = start_task_id ; task_id < stop_task_id ; task_id++ ))
do
echo "Current task id ${task_id}"

accelerate launch --num_processes 4 run_llm_ocl.py \
--config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/olmo2/olmo2_13b_ft_${stepn}step_lr2e-6.yaml \
--ocl_task ${task_category} --templates TASK_ID=${task_id} TASK_CATEGORY=${task_category}

done


task_category=tulu_train
start_task_id=0
stop_task_id=11
stepn=1k

for ((task_id = start_task_id ; task_id < stop_task_id ; task_id++ ))
do
echo "Current task id ${task_id}"

accelerate launch --num_processes 4 run_llm_ocl.py \
--config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/olmo2/olmo2_13b_ft_${stepn}step_lr2e-6.yaml \
--ocl_task ${task_category} --templates TASK_ID=${task_id} TASK_CATEGORY=${task_category}

done