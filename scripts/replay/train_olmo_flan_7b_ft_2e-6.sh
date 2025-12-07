#!/bin/bash

task_category=${1}
stepn="1k"

mixture_ratio=0.03125
temperature=0.1

if task_category == "flan"; then
  tasks="65 18 8 60 6 13 37 22 30 19 64 50 25 31 32 61 16 5 53 49"
  yaml_postfix=""
elif task_category == "tulu_train"; then
  tasks="0 1 2 3 4 5 6 7 8 9 10"
  yaml_postfix="_tulu"
elif task_category == "dolly"; then
  tasks="0 1 2 3 4 5 6 7"
    yaml_postfix="_dolly"
fi


for task_id in ${tasks[@]}
do
echo "Current task id ${task_id}"

# random
accelerate launch --num_processes 4 --main_process_port 29501 run_llm_ocl.py \
--config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/replay_ocl/7b_ft_1kstep_er.yaml \
--ocl_task ${task_category} --templates TASK_ID=${task_id} TASK_CATEGORY=${task_category} mixture_ratio=${mixture_ratio}

# additive-offline
accelerate launch --num_processes 4 --main_process_port 29501 run_llm_ocl.py \
--config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/replay_ocl/7b_ft_1kstep_additive${yaml_postfix}.yaml \
--ocl_task ${task_category} --templates TASK_ID=${task_id} TASK_CATEGORY=${task_category} mixture_ratio=${mixture_ratio} mixture_method=pred_sample temperature=${temperature}

# knn-offline
accelerate launch --num_processes 4 --main_process_port 29501 run_llm_ocl.py \
--config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/replay_ocl/7b_ft_1kstep_knn${yaml_postfix}.yaml \
--ocl_task ${task_category} --templates TASK_ID=${task_id} TASK_CATEGORY=${task_category} mixture_ratio=${mixture_ratio} mixture_method=pred_sample temperature=${temperature}

# mf-offline

accelerate launch --num_processes 4 --main_process_port 29501 run_llm_ocl.py \
--config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/replay_ocl/7b_ft_1kstep_svd${yaml_postfix}.yaml \
--ocl_task ${task_category} --templates TASK_ID=${task_id} TASK_CATEGORY=${task_category} mixture_ratio=${mixture_ratio} mixture_method=pred_sample temperature=${temperature}

# ground-truth forgetting
accelerate launch --num_processes 4 --main_process_port 29505 run_llm_ocl.py \
--config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/replay_ocl/7b_ft_1kstep_gt.yaml \
--ocl_task ${task_category} --templates TASK_ID=${task_id} TASK_CATEGORY=${task_category} \
mixture_ratio=${mixture_ratio} mixture_method=gt_sample temperature=${temperature}
done

