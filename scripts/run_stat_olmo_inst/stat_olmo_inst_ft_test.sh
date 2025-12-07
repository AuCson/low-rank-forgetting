#!/bin/bash

# task_category=mmlu
# tasks="30 21 43 54 9"


# for task_id in ${tasks[@]}
# do

# echo "${task_id}"

# python vllm_inference.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
# configs/llm/ocl_ins/stat_7b_on_tulu.yaml --templates ocl_task_id=${task_id} ocl_task_category=${task_category} \
# --stat_ppl --skip_eval_ocl_ds --n_gpus 1 --is_ins_pt --gpu_memory_utilization 0.85


# done
 


# task_category="bbh"
# stepn=100

# tasks="0 11 14 7 4"

# for task_id in ${tasks[@]}
# do
# echo "Current task id ${task_id} of ${task_category}"

# python vllm_inference.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
# configs/llm/ocl_ins/stat_7b_on_tulu.yaml --templates ocl_task_id=${task_id} ocl_task_category=${task_category} \
# --stat_ppl --skip_eval_ocl_ds --n_gpus 1 --is_ins_pt --gpu_memory_utilization 0.85


# done

# task_category="truthful_qa"
# stepn=100


# tasks="0 5 10 15 20"

# for task_id in ${tasks[@]}
# do
# echo "Current task id ${task_id} of ${task_category}"

# python vllm_inference.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
# configs/llm/ocl_ins/stat_7b_on_tulu.yaml --templates ocl_task_id=${task_id} ocl_task_category=${task_category} \
# --stat_ppl --skip_eval_ocl_ds --n_gpus 1 --is_ins_pt --gpu_memory_utilization 0.85


# done


task_category="dolly"
stepn=100

tasks="0 1 2 3 4 5 6 7"

for task_id in ${tasks[@]}
do
echo "Current task id ${task_id} of ${task_category}"

python vllm_inference.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/ocl_ins/stat_7b_on_tulu.yaml --templates ocl_task_id=${task_id} ocl_task_category=${task_category} \
--stat_ppl --skip_eval_ocl_ds --n_gpus 1 --is_ins_pt --gpu_memory_utilization 0.85


done