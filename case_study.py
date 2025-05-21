# %%
import matplotlib.pyplot as plt
import pickle
import heapq
from visualize_div import fit

# %%
from odeformer.envs import build_env
from odeformer.slurm import init_signal_handler, init_distributed_mode
from odeformer.model.__init__ import build_modules
from odeformer.trainer import Trainer
from evaluate import setup_odeformer, Evaluator
from parsers import get_parser

parser = get_parser()
parser.add_argument(
    "--fp16", type=bool, default=False, help="Run model with float16"
)
params = parser.parse_args(args=[
    '--eval_dump_path', '/home/nyms/ddot/experiments',
])
env = build_env(params)

init_distributed_mode(params)
modules = build_modules(env, params)
embedder, encoder, decoder = modules['embedder'], modules['encoder'], modules['decoder']

trainer = Trainer(modules, env, params)
model = setup_odeformer(trainer)
evaluator = Evaluator(trainer, model)

# %%
path = "/home/nyms/ddot/datasets/strogatz_extended.json"
iterator = evaluator.read_equations_from_json_file(path, True)

# %%
best_records = []
all_results = []
metric = 'r2_zero'

for idx, sample in enumerate(iterator):
    #
    sample['test'] = sample['train']
    best_results, best_candidates, pred_tree = fit(sample)
    all_results.append([best_results, pred_tree])
    
    metric_value = best_results[metric][0]
    r2, acc, snmse = best_results['r2_zero'], best_results['accuracy_l1_biggio'], best_results['snmse']
    div = best_results['divergence']
    print(f'r2_zero: {r2}, acc: {acc}, div: {div}')
    print(f'predicted: {pred_tree}')

    if len(best_records) < 10:
        heapq.heappush(best_records, (metric_value, idx))
    else:
        if metric_value > best_records[0][0]:
            heapq.heappushpop(best_records, (metric_value, idx))

# %%
for _, idx in best_records:
    print(f"=====Record {idx}=====")
    r2, acc, div = all_results[idx][0]['r2_zero'], all_results[idx][0]['accuracy_l1_biggio'], all_results[idx][0]['divergence']
    print(f'r2_zero: {r2}, acc: {acc}, div: {div}')
    pred_tree = all_results[idx][1]
    print(f'pred tree: {pred_tree}')

with open('odebench_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)

# %%
