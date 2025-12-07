#!/bin/bash

task_category="dolly"
stepn=100

tasks="0 1 2 3 4 5 6 7"

for task_id in ${tasks[@]}
do
echo "Current task id ${task_id} of ${task_category}"

python vllm_inference.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/ocl_ins/stat_7b_on_tulu_replay.yaml --templates ocl_task_id=${task_id} ocl_task_category=${task_category} mixture_ratio=0.125 \
--stat_ppl --skip_eval_ocl_ds --n_gpus 1 --is_ins_pt --gpu_memory_utilization 0.80

python vllm_inference.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/ocl_ins/stat_7b_on_tulu_replay.yaml --templates ocl_task_id=${task_id} ocl_task_category=${task_category} mixture_ratio=0.125 mixture_method=additive_pred_sample \
--stat_ppl --skip_eval_ocl_ds --n_gpus 1 --is_ins_pt --gpu_memory_utilization 0.80

python vllm_inference.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/ocl_ins/stat_7b_on_tulu_replay.yaml --templates ocl_task_id=${task_id} ocl_task_category=${task_category} mixture_ratio=0.125 mixture_method=knn_pred_sample \
--stat_ppl --skip_eval_ocl_ds --n_gpus 1 --is_ins_pt --gpu_memory_utilization 0.80

python vllm_inference.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/ocl_ins/stat_7b_on_tulu_replay.yaml --templates ocl_task_id=${task_id} ocl_task_category=${task_category} mixture_ratio=0.125 mixture_method=svd_pred_sample \
--stat_ppl --skip_eval_ocl_ds --n_gpus 1 --is_ins_pt --gpu_memory_utilization 0.80

python vllm_inference.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/ocl_ins/stat_7b_on_tulu_replay.yaml --templates ocl_task_id=${task_id} ocl_task_category=${task_category} mixture_ratio=0.125 mixture_method=gt_sample \
--stat_ppl --skip_eval_ocl_ds --n_gpus 1 --is_ins_pt --gpu_memory_utilization 0.80


python vllm_inference.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/ocl_ins/stat_7b_on_flan_v2_replay.yaml --templates ocl_task_id=${task_id} ocl_task_category=${task_category} mixture_ratio=0.125 \
--stat_output --skip_eval_ocl_ds --n_gpus 1 --is_ins_pt --gpu_memory_utilization 0.80

python vllm_inference.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/ocl_ins/stat_7b_on_flan_v2_replay.yaml --templates ocl_task_id=${task_id} ocl_task_category=${task_category} mixture_ratio=0.125 mixture_method=additive_pred_sample \
--stat_output --skip_eval_ocl_ds --n_gpus 1 --is_ins_pt --gpu_memory_utilization 0.80

python vllm_inference.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/ocl_ins/stat_7b_on_flan_v2_replay.yaml --templates ocl_task_id=${task_id} ocl_task_category=${task_category} mixture_ratio=0.125 mixture_method=knn_pred_sample \
--stat_output --skip_eval_ocl_ds --n_gpus 1 --is_ins_pt --gpu_memory_utilization 0.80

python vllm_inference.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/ocl_ins/stat_7b_on_flan_v2_replay.yaml --templates ocl_task_id=${task_id} ocl_task_category=${task_category} mixture_ratio=0.125 mixture_method=svd_pred_sample \
--stat_output --skip_eval_ocl_ds --n_gpus 1 --is_ins_pt --gpu_memory_utilization 0.80

python vllm_inference.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/ocl_ins/stat_7b_on_flan_v2_replay.yaml --templates ocl_task_id=${task_id} ocl_task_category=${task_category} mixture_ratio=0.125 mixture_method=gt_sample \
--stat_output --skip_eval_ocl_ds --n_gpus 1 --is_ins_pt --gpu_memory_utilization 0.80


done

