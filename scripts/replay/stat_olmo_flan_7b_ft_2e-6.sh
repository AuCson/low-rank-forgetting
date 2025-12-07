#!/bin/bash

stepn=1k
mixture_ratio=0.03125
temperature=0.1
n_gpus=1

if task_category == "flan"; then
  tasks="65 18 8 60 6 13 37 22 30 19 64 50 25 31 32 61 16 5 53 49"
elif task_category == "tulu_train"; then
  tasks="0 1 2 3 4 5 6 7 8 9 10"
elif task_category == "dolly"; then
  tasks="0 1 2 3 4 5 6 7"
fi


for task_id in ${tasks[@]}
do

echo "Current task id ${task_id}"
echo "Mixture ratio is ${mixture_ratio}"


python vllm_inference.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/replay_ocl/stat_7b_ft_1kstep_er.yaml --templates ocl_task_id=${task_id} ocl_task_category=${task_category} ocl_step=1k \
mixture_ratio=${mixture_ratio} temperature=${temperature}  \
--stat_ppl --skip_eval_ocl_ds --n_gpus ${n_gpus} --subsample_pt 10000

python vllm_inference.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/replay_ocl/stat_7b_ft_1kstep_pred_sample.yaml --templates ocl_task_id=${task_id} ocl_task_category=${task_category} ocl_step=1k \
mixture_ratio=${mixture_ratio} temperature=${temperature} fpd_method=additive mixture_method=pred_sample \
--stat_ppl --skip_eval_ocl_ds --n_gpus ${n_gpus} --subsample_pt 10000

python vllm_inference.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/replay_ocl/stat_7b_ft_1kstep_pred_sample.yaml --templates ocl_task_id=${task_id} ocl_task_category=${task_category} ocl_step=1k \
mixture_ratio=${mixture_ratio} temperature=${temperature} fpd_method=knn mixture_method=pred_sample \
--stat_ppl --skip_eval_ocl_ds --n_gpus ${n_gpus} --subsample_pt 10000

python vllm_inference.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/replay_ocl/stat_7b_ft_1kstep_pred_sample.yaml --templates ocl_task_id=${task_id} ocl_task_category=${task_category} ocl_step=1k \
mixture_ratio=${mixture_ratio} temperature=${temperature} fpd_method=svd mixture_method=pred_sample \
--stat_ppl --skip_eval_ocl_ds --n_gpus ${n_gpus} --subsample_pt 10000

python vllm_inference.py --config_files configs/defaults.yaml configs/llm/llm_defaults.yaml \
configs/llm/replay_ocl/stat_7b_ft_1kstep_gt_sample.yaml --templates ocl_task_id=${task_id} ocl_task_category=${task_category} ocl_step=1k \
mixture_ratio=${mixture_ratio} temperature=${temperature} mixture_method=gt_sample \
--stat_ppl --skip_eval_ocl_ds --n_gpus ${n_gpus} --subsample_pt 10000


done


