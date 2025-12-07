#!/bin/bash


stepn=1k
task_category="flan"
start_task_id=0
stop_task_id=66


for ((task_id = start_task_id ; task_id < stop_task_id ; task_id++ ))
do
echo "Fine-tuning model for task ${task_id} in ${task_category}"

accelerate launch --num_processes 2 run_llm_ocl.py \
--config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/ocl/1b_ft_${stepn}step_lr2e-6.yaml \
--ocl_task ${task_category} --templates TASK_ID=${task_id} TASK_CATEGORY=${task_category}

done



task_category="tulu_train"
start_task_id=0
stop_task_id=11


for ((task_id = start_task_id ; task_id < stop_task_id ; task_id++ ))
do
echo "Fine-tuning model for task ${task_id} in ${task_category}"

accelerate launch --num_processes 2 run_llm_ocl.py \
--config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/ocl/1b_ft_${stepn}step_lr2e-6.yaml \
--ocl_task ${task_category} --templates TASK_ID=${task_id} TASK_CATEGORY=${task_category}

done


task_category="dolly"
start_task_id=0
stop_task_id=8


for ((task_id = start_task_id ; task_id < stop_task_id ; task_id++ ))
do
echo "Fine-tuning model for task ${task_id} in ${task_category}"

accelerate launch --num_processes 2 run_llm_ocl.py \
--config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/ocl/1b_ft_${stepn}step_lr2e-6.yaml \
--ocl_task ${task_category} --templates TASK_ID=${task_id} TASK_CATEGORY=${task_category}

done
