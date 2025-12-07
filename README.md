# Supplementary material for NeurIPS 2025 submission 15520 - Demystifying Language Model Forgetting with Low-rank Example Associations

## Environments

See `requirement.txt` for python environment requirements.

We used flash-attention 2 in our experiments. To install the package, please follow the instructions under https://github.com/Dao-AILab/flash-attention

We used 4 Quadro RTX A6000 GPUs to run 7B-13B LLM fine-tuning, and used 2 GPUs of the same type for 1B LLM fine-tuning (while keeping the effective batch size the same). 

After training the models, we collect forgetting statistics over upstream data. Each run of LLM inference requires 1 GPU of the same type.

## Reproducing the results

### Step 1: Preparing the datasets

The datasets (e.g. Dolma subsample, Tulu subsample) can be downloaded from this public s3 bucket: https://neurips-submission-15520.s3.us-east-1.amazonaws.com/data.zip and extracted under PROJECT_ROOT/data/.



### Step 2: Fine-tuning the LLMs (Skippable with Step 4)

The following scripts fine-tune LLMs on unseen new tasks and create model checkpoints under `runs`

| Model | Script | 
| --- | --- |
| OLMo-1B | scripts/run_stat_olmo/train_olmo_flan_1b_ft_2e-6.sh |
| OLMo-7B | scripts/run_stat_olmo/train_olmo_flan_7b_ft_2e-6.sh |
| OLMo-7B-Instruct | scripts/run_stat_olmo_inst/train_olmo_flan_7b_ins_ft.sh | 
| OLMo2-7B | scripts/run_stat_olmo2/train_olmo_flan_7b_ft_2e-6.sh | 
| OLMo2-13B | scripts/run_stat_olmo2/train_olmo_flan_13b_ft_2e-6.sh | 
| MPT-7B | scripts/run_stat_mpt/train_flan_7b_ft_2e-6.sh | 
| Pythia-1B | scripts/run_stat_pythia/train_pythia.sh 1b | 
| Pythia-6.9B | scripts/run_stat_pythia/train_pythia.sh 7b | 
| Pythia-12B | scripts/run_stat_pythia/train_pythia.sh 12b |


### Step 3: Evaluating forgetting (Skippable with Step 4)

The following scripts load the saved model checkpoints, and evaluate log perplexity or model outputs over the upstream data.


| Model | Script |
| ---   | ---   |
| OLMo-1B | scripts/run_stat_olmo/stat_olmo_1b_ft_2e-6.sh |
| OLMo-7B | scripts/run_stat_olmo/stat_olmo_7b_ft_2e-6.sh |
| OLMo-7B-Instruct | scripts/run_stat_olmo_inst/stat_olmo_inst_ft.sh |
| OLMo2-7B | scripts/run_stat_olmo2/stat_olmo2_flan_7b_ft.sh | 
| OLMo2-13B | scripts/run_stat_olmo2/stat_olmo2_flan_13b_ft.sh |
| MPT-7B | scripts/run_stat_mpt/train_flan_7b_ft_2e-6.sh | 
| Pythia-1B | scripts/run_stat_pythia/stat_pythia.sh 1b | 
| Pythia-6.9B | scripts/run_stat_pythia/stat_pythia.sh 7b | 
| Pythia-12B | scripts/run_stat_pythia/stat_pythia.sh 12b |


### Step 4: Obtaining matrices of forgetting

The perplexity arrays obtained in Step 3 can be summarized into M * N association matrices Z. 

Alternatively, the statistics can be downloaded from the public s3 bucket: https://neurips-submission-15520.s3.us-east-1.amazonaws.com/stats.zip and extracted under PROJECT_ROOT/stats/.

We visualized these matrics in Figure 2, 6, 8 in the submitted paper.

### Step 5: Creating data splits for forgetting prediction

The script `post_process/fpd_utils.py` splits the rows of association matrices as train and test splits of forgetting prediction.

The statistics downloaded from the public s3 bucket: https://neurips-submission-15520.s3.us-east-1.amazonaws.com/stats.zip already involve train-test splits of forgetting prediction.

After unzipping, rename `stats_release/` to `stats/`

### Step 6: Predicting forgetting

The script `scripts/mat_completion/run_all.sh` runs matrix completion algorithms for the setups included in Table 3. The RMSE or F1 scores print on the screen and are logged.

### Step 7: Mitigating forgetting with random replay or targeted replay

**Training of Random Replay, Replay w/ offline additive, offline KNN, offline MF**

| Model | Script |
| ---   | ---   |
| OLMo-1B | scripts/replay/train_olmo_flan_1b_ft_2e-6.sh [flan\|tulu_train\|dolly] |
| OLMo-7B | scripts/replay/train_olmo_flan_7b_ft_2e-6.sh [flan\|tulu_train\|dolly] |
| OLMo-7B-Instruct |  scripts/replay/train_olmo_inst_ft.sh |

**Evalution of log perplexity over upstream examples**

| Model | Script |
| ---   | ---   |
| OLMo-1B | scripts/replay/stat_olmo_flan_1b_ft_2e-6.sh [flan\|tulu_train\|dolly] |
| OLMo-7B | scripts/replay/stat_olmo_flan_7b_ft_2e-6.sh [flan\|tulu_train\|dolly] |
| OLMo-7B-Instruct |  scripts/replay/stat_olmo_inst_ft.sh | 

**Training of online additive, online KNN, online MF**

| Model | Script |
| ---   | ---   |
| OLMo-1B | scripts/partition/train_olmo_flan_1b.sh |
| OLMo-7B | scripts/partition/train_olmo_flan_7b.sh |


**Evaluation of online additive, online KNN, online MF**

| Model | Script |
| ---   | ---   |
| OLMo-1B | scripts/partition/stat_partition_1b_fpd.sh |
| OLMo-7B | scripts/partition/stat_partition_7b_fpd.sh |

Each evaluation run saves an array representing log perplexity of each upstream example. We visualize the averaged log perplexity in Figure 5 of the submission.