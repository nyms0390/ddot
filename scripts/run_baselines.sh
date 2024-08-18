#!/bin/bash

# CONDA_BASE=$(conda info --base)
# source $CONDA_BASE/etc/profile.d/conda.sh
# conda activate odeformer39

MODELS=(
    # "proged"
    # "proged_poly"
    # "pysr"
    # "pysr_poly"
    "sindy"
    # "sindy_all"
    # "sindy_full"
    # "sindy_save"
    # "sindy_poly3"
    # "sindy_poly6"
    # "sindy_poly10"
    # "odeformer"
)

datasets=("odebench") # "strogatz"
eval_noise_gamma=("0" "0.01" "0.02" "0.03" "0.04" "0.05") # 
beam_size=("10")
eval_task=("interpolation" "y0_generalization") #


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
                    python run_baselines.py \
                        --baseline_model="${model}" \
                        --eval_noise_gamma="${noise}" \
                        --dataset="${dataset}" \
                        --beam_type="sampling" \
                        --beam_size="${bs}" \
                        --beam_temperature=0.1 \
                        --eval_size=1000 \
                        --eval_subsample_ratio=0.5 \
                        --e_task="${task}" \
                        --convert_prediction_to_tree=True \
                        --validation_metrics="r2_zero,accuracy_l1_biggio,snmse,divergence"
                done
            done
        done
    done
done
# done
