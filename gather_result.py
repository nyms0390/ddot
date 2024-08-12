# %%
from pathlib import Path
import pandas as pd
from prettytable import PrettyTable

# %%
def evaluate(filepath):
    df = pd.read_csv(filepath)
    df['r2_tol'] = df['r2_zero']>.9
    # df['test_r2_tol'] = df['test_r2_zero']>.8

    r2 = df['r2_zero'].mean()
    acc = df['accuracy_l1_biggio'].mean()
    acc_r2_09 = df['r2_tol'].sum() / df['r2_tol'].count()
    time = df['duration_fit'].mean()
    div = df['divergence'].mean()

    return {"R2": r2, "P(R2>.9)": acc_r2_09, "Inference time": time, "Divergence": div}

# %%
def create_table(files, title):
    result = {}

    for file in files: 
        matrics = evaluate(file)
        for k in matrics.keys():
            if k not in result.keys():
                result[k] = [matrics[k]]
            else:
                result[k].append(matrics[k])

    table = PrettyTable()
    table.title = title
    table.add_column("Noise", noise_gamma)
    for k, v in result.items():
        table.add_column(k, v)
    table.float_format = ".4"
    return table

# %%
root = "/home/310553058/odeformer/experiments"
baselines = ["ddot", "odeformer"] # 
datasets = ["odebench"] #, "strogatz"
noise_gamma = ["0.0", "0.01", "0.02", "0.03", "0.04", "0.05"]
beam_size = ["50"] # "1", "10", "20", "100"
eval_task = "y0_generalization" #"interpolation"

# for bs in beam_size:
#     for baseline in baselines:
#         files = [Path(root) / baseline / dataset / f"eval_gamma_noise_{noise}" / f"evaluation_task_{eval_task}" / f"beam_size_{bs}"/ "eval_pmlb.csv" for noise in noise_gamma]
#         title = f"{dataset} / {baseline} / beam size={bs}"
#         table = create_table(files, title)
#         print(table)

for baseline in baselines:
    for dataset in datasets:
        suffix = "pmlb" if dataset == "strogatz" else "strogatz_extended"
        files = [Path(root) / baseline / dataset / f"eval_gamma_noise_{noise}" / f"evaluation_task_{eval_task}" / "new_evals" / f"eval_{suffix}.csv" for noise in noise_gamma]
        title = f"{dataset} / {baseline}"
        table = create_table(files, title)
        print(table)