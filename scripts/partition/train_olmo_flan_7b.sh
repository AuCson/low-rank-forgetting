#!/bin/bash

ocl_task_category="flan"


for fpd_method in "additive" "knn" "svd"
do
for task_id in "65 18 8 60 6 13 37 22 30 19 64 50 25 31 32 61 16 5 53 49"
do

accelerate launch --num_processes 4 run_llm_partition.py \
--config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/partition/7b_ft_1kstep_partition_${fpd_method}.yaml \
--ocl_task flan --templates TASK_ID=${task_id} TASK_CATEGORY=${ocl_task_category}

done


ocl_task_category="tulu_train"

for task_id in "0 1 2 3 4 5 6 7 8 9 10"
do

accelerate launch --num_processes 4 --main_process_port 29506 run_llm_partition.py \
--config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/partition/7b_ft_1kstep_partition_${fpd_method}.yaml \
--ocl_task ${ocl_task_category} --templates TASK_ID=${task_id} TASK_CATEGORY=${ocl_task_category}

done


ocl_task_category="dolly"

for task_id in "0 1 2 3 4 5 6 7"
do

accelerate launch --num_processes 4 --main_process_port 29506 run_llm_partition.py \
--config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/partition/7b_ft_1kstep_partition_${fpd_method}.yaml \
--ocl_task ${ocl_task_category} --templates TASK_ID=${task_id} TASK_CATEGORY=${ocl_task_category}

done
done
