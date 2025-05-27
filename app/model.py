from scripts.inference import setup, fit

params_args = [
    '--reload_model', "/home/nyms/ddot/weights/ddot.pth",
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
    '--fp16', 'False',
]

# Path to the dataset
path = "datasets/strogatz.pkl"  # or "datasets/strogatz_extended.json"

# Set up inference
evaluator, model, iterator, env, params = setup(params_args, path)

def predict(sample):
    _, _, pred_tree = fit(sample, evaluator, model, env, params)
    return pred_tree
