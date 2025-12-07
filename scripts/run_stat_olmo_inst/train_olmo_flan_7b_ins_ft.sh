#!/bin/bash

stepn=100
task_category=mmlu
start_task_id=0
stop_task_id=57

for ((task_id = start_task_id ; task_id < stop_task_id ; task_id++ ))
do
echo "Current task id ${task_id} of task category ${task_category}"
accelerate launch --num_processes 4 run_llm_ocl.py \
--config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/ocl_ins/7b_ft_${stepn}step_lr2e-6_full_ft.yaml \
--ocl_task ${task_category} --templates TASK_ID=${task_id} TASK_CATEGORY=${task_category} 
done


task_category=bbh
start_task_id=0
stop_task_id=27

for ((task_id = start_task_id ; task_id < stop_task_id ; task_id++ ))
do
echo "Current task id ${task_id} of task category ${task_category}"
accelerate launch --num_processes 4 run_llm_ocl.py \
--config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/ocl_ins/7b_ft_${stepn}step_lr2e-6_full_ft.yaml \
--ocl_task ${task_category} --templates TASK_ID=${task_id} TASK_CATEGORY=${task_category} 
done


task_category=truthful_qa
start_task_id=0
stop_task_id=32

for ((task_id = start_task_id ; task_id < stop_task_id ; task_id++ ))
do
echo "Current task id ${task_id} of task category ${task_category}"
accelerate launch --num_processes 4 run_llm_ocl.py \
--config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/ocl_ins/7b_ft_${stepn}step_lr2e-6_full_ft.yaml \
--ocl_task ${task_category} --templates TASK_ID=${task_id} TASK_CATEGORY=${task_category} 
done


task_category=dolly
start_task_id=0
stop_task_id=8

for ((task_id = start_task_id ; task_id < stop_task_id ; task_id++ ))
do
echo "Current task id ${task_id} of task category ${task_category}"
accelerate launch --num_processes 4 run_llm_ocl.py \
--config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/ocl_ins/7b_ft_${stepn}step_lr2e-6_full_ft.yaml \
--ocl_task ${task_category} --templates TASK_ID=${task_id} TASK_CATEGORY=${task_category} 
done
