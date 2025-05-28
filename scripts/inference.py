from copy import deepcopy
from timeit import default_timer as timer
from collections import defaultdict
import pickle
import numpy as np

from odeformer.envs import build_env
from odeformer.model.__init__ import build_modules
from odeformer.trainer import Trainer
from odeformer.slurm import init_distributed_mode

from .evaluate import setup_odeformer, Evaluator
from .parsers import get_parser

def setup(params_args, path):
    """
    Set up the environment, model, and iterator for inference tasks.

    Args:
        params_args (list): List of arguments for the parser.
        path (str): Path to the dataset file (e.g., .pkl or .json).

    Returns:
        tuple: (evaluator, model, iterator, env, params)
    """
    # Parse arguments
    parser = get_parser()
    params, unknown = parser.parse_known_args(args=params_args)

    # Build environment and modules
    env = build_env(params)
    init_distributed_mode(params)
    modules = build_modules(env, params)
    trainer = Trainer(modules, env, params)

    # Set up the model and evaluator
    model = setup_odeformer(trainer)
    evaluator = Evaluator(trainer, model)

    # Determine the iterator based on the file type
    if path.endswith(".pkl"):
        with open(path, "rb") as fpickle:
            iterator = pickle.load(fpickle)
    elif path.endswith(".json"):
        iterator = evaluator.read_equations_from_json_file(path=path, save=False)
    else:
        raise ValueError("Unsupported file type. Use .pkl or .json.")

    return evaluator, model, iterator, env, params

def fit(samples, evaluator, model, env, params):
    """
    Fit the model to the given samples and evaluate the results.

    Args:
        samples (dict): The input samples containing times, trajectories, and trees.
        evaluator (Evaluator): The evaluator object for evaluating the model.
        model: The trained model used for fitting.
        env: The environment object for handling noise and subsampling.
        params: The configuration parameters.

    Returns:
        tuple: Best results, best candidates, and predicted trees.
    """
    if not "test" in samples.keys():
        samples = evaluator.prepare_test_trajectory(samples, evaluation_task=params.evaluation_task)
    times, trajectories = samples["train"]["times"], samples["train"]["trajectories"]

    if "tree" in samples.keys():
        trees = [env.simplifier.simplify_tree(tree, expand=True) for tree in samples["tree"]]
    else:
        trees = [None] * len(times)

    original_times, original_trajectories = deepcopy(times), deepcopy(trajectories)
    times, trajectories = corrupt_training_data(original_times, original_trajectories, env, params)

    # Fit the model
    start_time_fit = timer()
    all_candidates = model.fit(times, trajectories, verbose=False, sort_candidates=True)
    all_duration_fit = [timer() - start_time_fit] * len(times)
    
    # Evaluate on train data
    best_results, best_candidates = evaluator._evaluate(
        original_times, original_trajectories, trees, all_candidates, all_duration_fit, params.validation_metrics
    )
    predicted_trees = [tree.infix() if hasattr(tree, 'infix') else tree for tree in best_candidates]
    predicted_trees = evaluator.str_to_tree(predicted_trees[0])
    return best_results, best_candidates, predicted_trees

def corrupt_training_data(times, trajectories, env, params):
    """
    Corrupt the training data with noise and subsampling.

    Args:
        times (list): List of time arrays.
        trajectories (list): List of trajectory arrays.
        env: The environment object for handling noise and subsampling.
        params: The configuration parameters.

    Returns:
        tuple: Corrupted times and trajectories.
    """
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

def create_sample(evaluator, equations):
    """Create a single sample from the solution of a specific ODE."""
    # Extract the first solution
    solution = equations["solutions"][0][0]
    times = np.array(solution["t"])
    trajectory = np.array(solution["y"]).T

    # Create the sample
    sample = {
        "train": {
            "times": [times],
            "trajectories": [trajectory],
        },
        "infos": {
            "dimension": [trajectory.shape[1]],
            "n_unary_ops": [np.nan],
            "n_input_points": [len(times)],
            "name": [equations["eq_description"]],
            "dataset": ["strogatz_extended"],
        },
        "tree": [
            evaluator.str_to_tree(" | ".join(map(str, equations["substituted"][0])))
        ],
    }
    return sample