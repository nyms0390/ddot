# %%
import numpy as np
import matplotlib.pyplot as plt

from odeformer.envs.generators import tree_to_numexpr_fn
from .inference import setup, fit

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

# Define the arguments for the parser
params_args = [
    '--reload_model', "/home/nyms/ddot/ddot.pth",
    '--use_wandb', 'False',
    '--collate_queue_size', '1000',
    '--print_freq', '30',
    '--ode_integrator', 'solve_ivp',
    '--num_workers', '1',
    '--tokens_per_batch', '500',
    '--min_dimension', '1',
    '--max_dimension', '6',
    '--float_descriptor_length', '3',
    '--enc_emb_dim', '512',
    '--dec_emb_dim', '512',
    '--max_points', '200',
    '--train_noise_gamma', '.01',
    '--train_subsample_ratio', '.5',
    '--eval_only', 'True',
    '--eval_dump_path', '/home/nyms/ddot/experiments',
    '--beam_size', '50',
    '--use_ft_decoder', 'True',
    '--validation_metrics', 'r2_zero,divergence,accuracy_l1_biggio,snmse',
]

# Path to the dataset
path = "datasets/strogatz.pkl"  # or "datasets/strogatz_extended.json"

# Set up inference
evaluator, model, iterator, env, params = setup(params_args, path)

def main():
    # fig_vec, axes_vec = plt.subplots(2, 7, figsize=(15, 5), sharex=True, sharey=True)
    fig_div, axes_div = plt.subplots(1, 5, figsize=(15, 5), sharex=True, sharey=True)
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
        results, _, pred_tree = fit(sample, evaluator, model, env, params)
        
        pred_np_fn = tree_to_numexpr_fn(pred_tree)
        pred_vector_field = compute_vector_field(pred_np_fn, flat_coord, grid_shape)
        pred_div = compute_divergence(pred_vector_field, spacing)
        div = results['divergence'][0]

        # Plot the vector field for the first two dimensions
        if dim > 1 and idx % 4 == 0 and idx / 4 > 1:
            print(f'idx: {idx}, div: {div:.2f}')
            plot_divergence(axes_div[count], f"DIV-diff: {div:.2f}", mesh, pred_div, dims=(0, 1))
            count += 1
    plt.show()
    plt.savefig('vis_div_ddot.png', bbox_inches='tight')

# %%
if __name__ == "__main__":
    main()