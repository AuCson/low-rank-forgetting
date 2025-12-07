#!/bin/bash

for fpd_method in "additive" "knn" "svd"
do

ocl_task_category="flan"
ocl_step=1k
method_postfix="pred_sample_temp0.1_${fpd_method}"

for task_id in "65 18 8 60 6 13 37 22 30 19 64 50 25 31 32 61 16 5 53 49"
do

echo "debug"

python vllm_inference.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/stats/olmo_partition/1b_dolma_lr2e-6_pred_sample_partition.yaml \
--templates ocl_task_id=${task_id} ocl_task_category=${ocl_task_category} ocl_step=${ocl_step} method_postfix=${method_postfix} \
--stat_ppl --skip_eval_ocl_ds --n_gpus 1 --subsample_pt 10000

done


ocl_task_category="tulu_train"

for task_id in "0 1 2 3 4 5 6 7 8 9 10"
do

echo "debug"

python vllm_inference.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/stats/olmo_partition/1b_dolma_lr2e-6_pred_sample_partition.yaml \
--templates ocl_task_id=${task_id} ocl_task_category=${ocl_task_category} ocl_step=${ocl_step} method_postfix=${method_postfix} \
--stat_ppl --skip_eval_ocl_ds --n_gpus 1 --subsample_pt 10000

done


ocl_task_category="dolly"

for task_id in "0 1 2 3 4 5 6 7"
do

echo "debug"

python vllm_inference.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/stats/olmo_partition/1b_dolma_lr2e-6_pred_sample_partition.yaml \
--templates ocl_task_id=${task_id} ocl_task_category=${ocl_task_category} ocl_step=${ocl_step} method_postfix=${method_postfix} \
--stat_ppl --skip_eval_ocl_ds --n_gpus 1 --subsample_pt 10000

done
done