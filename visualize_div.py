# %%
import os

from itertools import chain
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pickle
from copy import deepcopy
from timeit import default_timer as timer
from typing import List
from collections import OrderedDict, defaultdict

from odeformer.envs import build_env
from parsers import get_parser
from odeformer.slurm import init_signal_handler, init_distributed_mode
from odeformer.model.__init__ import build_modules
from odeformer.trainer import Trainer
from odeformer.envs.environment import EnvDataset
from odeformer.envs.generators import integrate_ode
from evaluate import setup_odeformer, Evaluator


batch_results = defaultdict(list)

# %%
parser = get_parser()
parser.add_argument(
    "--fp16", type=bool, default=False, help="Run model with float16"
)
params = parser.parse_args(args=[
    # '--reload_data', "/home/310553058/odeformer/experiments/datagen_ftraj/datagen_use_sympy_True",
    '--reload_model', "/home/nyms/ddot/ddot.pth",
    '--use_wandb', 'False',
    '--collate_queue_size', '1000',
    #'n_steps_per_epoch':1000,
    '--print_freq', '30',
    '--ode_integrator', 'solve_ivp',
    '--num_workers', '1',
    '--tokens_per_batch', '500',
    '--min_dimension', '1',
    '--max_dimension', '6',
    '--float_descriptor_length', '3',
    '--enc_emb_dim', '512',
    '--dec_emb_dim', '512',
    #'subsample_ratio':0.5,
    '--max_points', '200',
    '--train_noise_gamma', '.01',
    '--train_subsample_ratio', '.5',
    # '--debug',
    '--eval_only', 'True',
    # '--from_pretrained', 'True',
    '--eval_dump_path', '/home/nyms/ddot/experiments',
    '--beam_size', '50',
    '--use_ft_decoder', 'True',
    '--validation_metrics', 'r2_zero,divergence,accuracy_l1_biggio,snmse',
])
env = build_env(params)

# %%
init_distributed_mode(params)
modules = build_modules(env, params)
embedder, encoder, decoder = modules['embedder'], modules['encoder'], modules['decoder']

trainer = Trainer(modules, env, params)
model = setup_odeformer(trainer)
evaluator = Evaluator(trainer, model)

# %%
from odeformer.envs.generators import tree_to_numexpr_fn

path = "datasets/strogatz.pkl" #"datasets/strogatz_extended.json"
if path.endswith(".pkl"):
    with open(path, "rb") as fpickle:
        iterator = pickle.load(fpickle)
elif path.endswith(".json"):
    iterator = evaluator.read_equations_from_json_file(path=path, save=False)

# %%
def create_nd_grid(limits, num_points):
    grids = [np.linspace(lim[0], lim[1], num_points) for lim in limits]
    mesh = np.meshgrid(*grids)
    return mesh

def flatten_grid(mesh):
    flat_coord = np.array([g.flatten() for g in mesh]).T
    return flat_coord

def compute_vector_field(func, flat_coord, grid_shape):
    out = func(flat_coord, [0])
    reshaped_out = [out[:, i].reshape(grid_shape) for i in range(out.shape[1])]
    return np.array(reshaped_out)

def compute_divergence(f, sp):
    num_dims = len(f)
    return np.ufunc.reduce(np.add, [np.gradient(f[i], sp[i], axis=i) for i in range(num_dims)])

def plot_vector_field(ax, title, mesh, vector_field, dims=(0, 1)):
    slice_idx = [0] * (len(mesh) - 2)
    slices = [slice(None) if i in dims else slice_idx.pop(0) for i in range(len(mesh))]

    X, Y = mesh[dims[0]][tuple(slices)], mesh[dims[1]][tuple(slices)]
    U = vector_field[dims[0]][tuple(slices)]
    V = vector_field[dims[1]][tuple(slices)]
    
    ax.quiver(X, Y, U, V)
    ax.set_title(title)

def plot_divergence(ax, title, mesh, divergence_field, dims=(0, 1)):
    slice_idx = [0] * (len(mesh) - 2)
    slices = [slice(None) if i in dims else slice_idx.pop(0) for i in range(len(mesh))]

    X, Y = mesh[dims[0]][tuple(slices)], mesh[dims[1]][tuple(slices)]
    div = divergence_field[tuple(slices)]
    
    ax.imshow(div, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap='RdBu')
    ax.set_title(title, size=20)

# %%
def corrupt_training_data(times, trajectories):
    for i, (time, trajectory) in enumerate(zip(times, trajectories)):
        if params.eval_noise_gamma:
            noise, gamma = env._create_noise(
                train=False,
                trajectory=trajectory,
                gamma=params.eval_noise_gamma,
                seed=params.test_env_seed,
            )
            trajectory += noise

        if params.eval_subsample_ratio:
            time, trajectory, subsample_ratio = env._subsample_trajectory(
                time,
                trajectory,
                subsample_ratio=params.eval_subsample_ratio,
                seed=params.test_env_seed,
            )
        times[i] = time
        trajectories[i] = trajectory

    return times, trajectories

def fit(samples):
    if not "test" in samples.keys():
        samples = evaluator.prepare_test_trajectory(samples, evaluation_task=params.evaluation_task)
    times, trajectories = samples["train"]["times"], samples["train"]["trajectories"]
    
    if "tree" in samples.keys():
        trees = [env.simplifier.simplify_tree(tree, expand=True) for tree in samples["tree"]]
        batch_results["trees"].extend(
            [None if tree is None else tree.infix() for tree in trees]
        )
    else:
        trees = [None]*len(times)

    original_times, original_trajectories = deepcopy(times), deepcopy(trajectories)
    times, trajectories = corrupt_training_data(original_times, original_trajectories)

    # fit
    start_time_fit = timer()
    all_candidates = model.fit(times, trajectories, verbose=False, sort_candidates=True)
    all_duration_fit = [timer() - start_time_fit] * len(times)

    # evaluate on train data
    best_results, best_candidates = evaluator._evaluate(
        original_times, original_trajectories, trees, all_candidates, all_duration_fit, params.validation_metrics
    )
    predicted_trees = [tree.infix() if hasattr(tree, 'infix') else tree for tree in best_candidates]
    predicted_trees = evaluator.str_to_tree(predicted_trees[0])
    return best_results, best_candidates, predicted_trees

# %%
def main():
    # fig_vec, axes_vec = plt.subplots(2, 7, figsize=(15, 5), sharex=True, sharey=True)
    fig_div, axes_div = plt.subplots(1, 5, figsize=(15, 5), sharex=True, sharey=True)
    # fig_grad, axes_grad = plt.subplots(6, 7, figsize=(15, 15), sharex=True, sharey=True)
    count = 0

    for idx, sample in enumerate(iterator):
        # Define the limits for each dimension and the number of points
        dim = sample["infos"]["dimension"][0]
        limits = [(-2, 2)] * dim
        num_points = 21
        
        # Create the grid and flatten the coordinates
        mesh = create_nd_grid(limits, num_points)
        flat_coord = flatten_grid(mesh)
        spacing = [(lim[1] - lim[0]) / (num_points - 1) for lim in limits]
        grid_shape = [num_points] * len(limits)

        # 
        org_tree = sample["tree"][0]
        org_np_fn = tree_to_numexpr_fn(org_tree)

        org_vector_field = compute_vector_field(org_np_fn, flat_coord, grid_shape)
        org_div = compute_divergence(org_vector_field, spacing)

        #
        results, _, pred_tree = fit(sample)
        
        pred_np_fn = tree_to_numexpr_fn(pred_tree)
        pred_vector_field = compute_vector_field(pred_np_fn, flat_coord, grid_shape)
        pred_div = compute_divergence(pred_vector_field, spacing)
        div = results['divergence'][0]

        # Plot the vector field for the first two dimensions
        if dim > 1 and idx % 4 == 0 and idx / 4 > 1:
            print(f'idx: {idx}, div: {div:.2f}')
            # plot_vector_field(axes_vec[count//7][count%7], f"ODE: {idx//4+1}", mesh, org_vector_field, dims=(0, 1))
            # plot_vector_field(axes_vec[count//7+1][count%7], f"ODE: {idx//4+1}", mesh, pred_vector_field, dims=(0, 1))
            
            plot_divergence(axes_div[count], f"DIV-diff: {div:.2f}", mesh, pred_div, dims=(0, 1))
            # plot_divergence(axes_div[count//7+1][count%7], f"ODE: {idx//4+1}", mesh, pred_div, dims=(0, 1))

            count += 1
    # plt.tight_layout()
    plt.show()
    plt.savefig('vis_div_ddot.png', bbox_inches='tight')

# %%
if __name__ == "__main__":
    main()