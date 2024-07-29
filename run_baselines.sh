#!/bin/bash

# CONDA_BASE=$(conda info --base)
# source $CONDA_BASE/etc/profile.d/conda.sh
# conda activate odeformer39

MODELS=(
    # "afp"
    # "feafp"
    # "eplex"
    # "ehc"
    # "proged"
    # "proged_poly"
    # "ffx"
    # "pysr"
    # "pysr_poly"
    # "sindy"
    # "sindy_all"
    # "sindy_full"
    # "sindy_save"
    # "sindy_poly3"
    # "sindy_poly6"
    # "sindy_poly10"
    "odeformer"
)

dataset="odebench"
eval_noise_gamma=("0" "0.01" "0.02" "0.03" "0.04" "0.05") # 
beam_size=("1")
# hyper_opt="True"
# eval_noise_type="additive"
# baseline_hyper_opt_eval_fraction="0.3"
# baseline_to_sympy="True"

# for subsample_ratio in "0.0" "0.25" "0.5";
# do
for model in "${MODELS[@]}";
do
    for noise in "${eval_noise_gamma[@]}"; 
    do
        for bs in "${beam_size[@]}";
        do
            echo "Evaluting ${model} with eval_noise_gamma=${eval_noise_gamma}"
            python run_baselines.py \
                --baseline_model="${model}" \
                --eval_noise_gamma="${noise}" \
                --dataset="${dataset}" \
                --beam_type="sampling" \
                --beam_size="${bs}" \
                --beam_temperature=0.1 \
                --eval_size=10000
                # --eval_subsample_ratio=0.5 \
        done
    done
done
# done