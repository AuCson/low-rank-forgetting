#!/bin/bash

for ((seed = 0 ; seed < 10 ; seed++ ))
do
for method in additive knn_baseline svd
do

python run_matrix_completion.py ${method} stats/olmo-1b/fpd-split-olmo-1b-id.pkl --known_k 30 --seed ${seed}
python run_matrix_completion.py ${method} stats/olmo-1b/fpd-split-olmo-1b-ood-tulu.pkl --known_k 30 --seed ${seed}
python run_matrix_completion.py ${method} stats/olmo-1b/fpd-split-olmo-1b-ood-dolly.pkl --known_k 30 --seed ${seed}


python run_matrix_completion.py ${method} stats/olmo-7b/fpd-split-olmo-7b-id.pkl --known_k 30 --seed ${seed}
python run_matrix_completion.py ${method} stats/olmo-7b/fpd-split-olmo-7b-ood-tulu.pkl --known_k 30 --seed ${seed}
python run_matrix_completion.py ${method} stats/olmo-7b/fpd-split-olmo-7b-ood-dolly.pkl --known_k 30 --seed ${seed}

python run_matrix_completion.py ${method} stats/mpt-7b/fpd-split-mpt-7b-id.pkl --known_k 30 --seed ${seed}
python run_matrix_completion.py ${method} stats/mpt-7b/fpd-split-mpt-7b-ood-tulu.pkl --known_k 30 --seed ${seed}
python run_matrix_completion.py ${method} stats/mpt-7b/fpd-split-mpt-7b-ood-dolly.pkl --known_k 30 --seed ${seed}

python run_matrix_completion.py ${method} stats/olmo-7b-ins/fpd-split-olmo-7b-ins-id.pkl --known_k 30 --seed ${seed}
python run_matrix_completion.py ${method} stats/olmo-7b-ins/fpd-split-olmo-7b-ins-ood-dolly.pkl --known_k 30 --seed ${seed}
python run_matrix_completion.py ${method} stats/olmo-7b-ins/fpd-split-olmo-7b-ins-ood-truthful_qa.pkl --known_k 30 --seed ${seed}


python run_matrix_completion.py ${method} stats/olmo-7b-ins/fpd-split-olmo-7b-ins-id.pkl --known_k 30 --seed ${seed}
python run_matrix_completion.py ${method} stats/olmo-7b-ins/fpd-split-olmo-7b-ins-ood-dolly.pkl --known_k 30 --seed ${seed}
python run_matrix_completion.py ${method} stats/olmo-7b-ins/fpd-split-olmo-7b-ins-ood-truthful_qa.pkl --known_k 30 --seed ${seed}



done

method="lgmf_additive"
python run_matrix_completion.py ${method} stats/olmo-7b-ins/fpd-split-flan-bin-olmo-7b-ins-id.pkl --known_k 30 --seed ${seed} --hparams rank=5
method="lgmf"
python run_matrix_completion.py ${method} stats/olmo-7b-ins/fpd-split-flan-bin-olmo-7b-ins-id.pkl --known_k 30 --seed ${seed} --hparams rank=5
method="lgmf_additive"
python run_matrix_completion.py ${method} stats/olmo-7b-ins/fpd-split-flan-bin-olmo-7b-ins-ood.pkl --known_k 30 --seed ${seed} --hparams rank=5
method="lgmf"
python run_matrix_completion.py ${method} stats/olmo-7b-ins/fpd-split-flan-bin-olmo-7b-ins-ood.pkl --known_k 30 --seed ${seed} --hparams rank=5


done