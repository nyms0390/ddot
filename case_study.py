# %%
import matplotlib.pyplot as plt
import pickle
import heapq
from intro_experiment import fit

path = 'datasets/anesthesia.pkl'
with open(path, "rb") as fpickle:
    iterator = pickle.load(fpickle)

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
    print(f'r2_zero: {r2}, acc: {acc}, snmse: {snmse}')

    if len(best_records) < 10:
        heapq.heappush(best_records, (metric_value, idx))
    else:
        if metric_value > best_records[0][0]:
            heapq.heappushpop(best_records, (metric_value, idx))

for _, idx in best_records:
    print(f"=====Record {idx}=====")
    r2, acc, snmse = all_results[idx][0]['r2_zero'], all_results[idx][0]['accuracy_l1_biggio'], all_results[idx][0]['snmse']
    print(f'r2_zero: {r2}, acc: {acc}, snmse: {snmse}')

with open('anes_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)

# %%
