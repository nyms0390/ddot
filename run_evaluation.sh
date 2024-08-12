#!/bin/bash

# CONDA_BASE=$(conda info --base)
# source $CONDA_BASE/etc/profile.d/conda.sh
# conda activate odeformer39

MODELS=(
    "odeformer" "ddot"
)
    #

datasets=("odebench") # "oscillators""strogatz" 
eval_noise_gamma=("0" "0.01" "0.02" "0.03" "0.04" "0.05") #  
beam_size=("50")
eval_task=("y0_generalization") #"interpolation" 
# hyper_opt="True"
# eval_noise_type="additive"
# baseline_hyper_opt_eval_fraction="0.3"
# baseline_to_sympy="True"

# for subsample_ratio in "0.0" "0.25" "0.5";
# do
for model in "${MODELS[@]}";
do
    for dataset in "${datasets[@]}";
    do
        for task in "${eval_task[@]}";
        do
            for noise in "${eval_noise_gamma[@]}"; 
            do
                for bs in "${beam_size[@]}";
                do
                    echo "Evaluting ${model} with eval_noise_gamma=${eval_noise_gamma}"
                    python evaluate.py \
                        --baseline_model="${model}" \
                        --eval_noise_gamma="${noise}" \
                        --dataset="${dataset}" \
                        --beam_type="sampling" \
                        --beam_size="${bs}" \
                        --beam_temperature=0.1 \
                        --eval_size=1000 \
                        --eval_subsample_ratio=0.5 \
                        --batch_size_eval=16 \
                        --evaluation_task="${task}"
                        # --beam_selection_metric="divergence"
                done
            done
        done
    done
done
# done