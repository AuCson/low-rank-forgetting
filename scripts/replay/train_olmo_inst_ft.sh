#!/bin/bash

task_category="dolly"
stepn=100

tasks="0 1 2 3 4 5 6 7"

for task_id in ${tasks[@]}
do
echo "Current task id ${task_id} of ${task_category}"

# random
accelerate launch --num_processes 4 run_llm_ocl.py \
--config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/ocl_ins/7b_ft_${stepn}step_er.yaml \
--ocl_task ${task_category} --templates TASK_ID=${task_id} TASK_CATEGORY=${task_category} mixture_ratio=0.125

# additive-offline
accelerate launch --num_processes 4 run_llm_ocl.py \
--config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/ocl_ins/7b_ft_${stepn}step_pred_sample_additive.yaml \
--ocl_task ${task_category} --templates TASK_ID=${task_id} TASK_CATEGORY=${task_category} mixture_ratio=0.125

# knn-offline
accelerate launch --num_processes 4 run_llm_ocl.py \
--config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/ocl_ins/7b_ft_${stepn}step_pred_sample_knn.yaml \
--ocl_task ${task_category} --templates TASK_ID=${task_id} TASK_CATEGORY=${task_category} mixture_ratio=0.125

# mf-offline
accelerate launch --num_processes 4 run_llm_ocl.py \
--config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/ocl_ins/7b_ft_${stepn}step_pred_sample_svd.yaml \
--ocl_task ${task_category} --templates TASK_ID=${task_id} TASK_CATEGORY=${task_category} mixture_ratio=0.125

# ground-truth forgetting
accelerate launch --num_processes 4 run_llm_ocl.py \
--config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/ocl_ins/7b_ft_${stepn}step_gt_sample.yaml \
--ocl_task ${task_category} --templates TASK_ID=${task_id} TASK_CATEGORY=${task_category} mixture_ratio=0.125

done

