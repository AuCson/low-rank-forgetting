#!/bin/bash

stepn="1k"


task_category=flan
start_task_id=0
stop_task_id=66


for ((task_id = start_task_id ; task_id < stop_task_id ; task_id++ ))
do
python vllm_inference.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/stats/olmo_ft/1b_flan_dolma_lr2e-6.yaml --templates ocl_task_id=${task_id} ocl_task_category=${task_category} ocl_step=${stepn} \
--stat_ppl --skip_eval_ocl_ds --n_gpus 1
done

task_category=tulu_train
start_task_id=0
stop_task_id=11


for ((task_id = start_task_id ; task_id < stop_task_id ; task_id++ ))
do
python vllm_inference.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/stats/olmo_ft/1b_flan_dolma_lr2e-6.yaml --templates ocl_task_id=${task_id} ocl_task_category=${task_category} ocl_step=${stepn} \
--stat_ppl --skip_eval_ocl_ds --n_gpus 1
done

task_category=dolly
start_task_id=0
stop_task_id=8


for ((task_id = start_task_id ; task_id < stop_task_id ; task_id++ ))
do
python vllm_inference.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/stats/olmo_ft/1b_flan_dolma_lr2e-6.yaml --templates ocl_task_id=${task_id} ocl_task_category=${task_category} ocl_step=${stepn} \
--stat_ppl --skip_eval_ocl_ds --n_gpus 1
done
