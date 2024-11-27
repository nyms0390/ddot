# %%
import random

def generate_samples(num_states, x_range, y_range, z_range):
    states = [
        [
            random.uniform(*x_range), 
            random.uniform(*y_range), 
            random.uniform(*z_range)
        ]
        for _ in range(num_states)
    ]
    return states

lorenz_format = {
    'id': 1,
    'eq': 'c_0 * (x_1 - x_0) | c_1 * x_0 - x_1 - x_0 * x_2 | x_0 * x_1 - c_2 * x_2',
    'dim': 3,
    'consts': [],
    'init': [],
    'init_constraints': 'x_0 > 0, x_1 > 0, x_2 > 0',
    'const_constraints': 'c_0 > 0, c_1 > 0, c_2 > 0',
    'eq_description': 'Lorenz equations in well-behaved periodic regime',
    'const_description': 'c_0: Prandtl number (sigma), c_1: Rayleigh number (r), c_2: unnamed parameter (b)',
    'var_description': 'x_0: x, x_1: y, x_2: z',
    'source': 'strogatz p.319'
}
# %%
equations = []

n_sample = 50
init_state = [[2.3, 8.1, 12.4], [10., 20., 30.]]
consts = generate_samples(n_sample, (5.0, 15.0), (10.0, 50.0), (1.5, 5.0))
for i in range(50):
    sample = lorenz_format.copy()
    sample['id'] = i + 1
    sample['consts'] = [consts[i]]
    sample['init'] = init_state
    equations.append(sample)

n_sample = 50
consts = generate_samples(n_sample, (5.0, 15.0), (10.0, 50.0), (1.5, 5.0))
for i in range(50):
    init_state = generate_samples(2, (0, 50), (0, 50), (0, 100))
    sample = lorenz_format.copy()
    sample['id'] = n_sample + i + 1
    sample['consts'] = [consts[i]]
    sample['init'] = init_state
    equations.append(sample)

# {
#     'id': 4,
#     'eq': '-x_1 - x_2 | x_0 + c_0 * x_1 | c_1 + x_2 * (x_0 - c_2)',
#     'dim': 3,
#     'consts': [[0.1, 0.1, 14.0]],
#     'init': [[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]],
#     'init_constraints': 'x_0 > 0, x_1 > 0, x_2 > 0',
#     'const_constraints': 'c_0 > 0, c_1 > 0, c_2 > 0',
#     'eq_description': 'Rössler system in near-periodic regime',
#     'const_description': 'c_0: coupling parameter a, c_1: coupling parameter b, c_2: coupling parameter c',
#     'var_description': 'x_0: x, x_1: y, x_2: z',
#     'source': 'adapted from literature'
# },
# {
#     'id': 5,
#     'eq': '-x_1 - x_2 | x_0 + c_0 * x_1 | c_1 + x_2 * (x_0 - c_2)',
#     'dim': 3,
#     'consts': [[0.2, 0.2, 5.7]],
#     'init': [[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]],
#     'init_constraints': 'x_0 > 0, x_1 > 0, x_2 > 0',
#     'const_constraints': 'c_0 > 0, c_1 > 0, c_2 > 0',
#     'eq_description': 'Rössler system in intermediate chaotic regime',
#     'const_description': 'c_0: coupling parameter a, c_1: coupling parameter b, c_2: coupling parameter c',
#     'var_description': 'x_0: x, x_1: y, x_2: z',
#     'source': 'adapted from literature'
# },
# {
#     'id': 6,
#     'eq': '-x_1 - x_2 | x_0 + c_0 * x_1 | c_1 + x_2 * (x_0 - c_2)',
#     'dim': 3,
#     'consts': [[0.2, 0.2, 10.0]],
#     'init': [[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]],
#     'init_constraints': 'x_0 > 0, x_1 > 0, x_2 > 0',
#     'const_constraints': 'c_0 > 0, c_1 > 0, c_2 > 0',
#     'eq_description': 'Rössler system in strongly chaotic regime',
#     'const_description': 'c_0: coupling parameter a, c_1: coupling parameter b, c_2: coupling parameter c',
#     'var_description': 'x_0: x, x_1: y, x_2: z',
#     'source': 'adapted from literature'
# },
# {
#     'id': 7,
#     'eq': 'x_2 | x_3 | -x_0 - c_0 * x_0 * x_1 | -x_1 - x_0^2 + x_1^2',
#     'dim': 4,
#     'consts': [[2]],
#     'init': [[0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5]],
#     'init_constraints': 'x_0 > 0, x_1 > 0, x_2 > 0, x_3 > 0',
#     'const_constraints': 'None',
#     'eq_description': 'Henon-Heiles equations describing chaotic dynamics in a potential field',
#     'const_description': 'No constants in this system',
#     'var_description': 'x_0: position x, x_1: position y, x_2: momentum p_x, x_3: momentum p_y',
#     'source': 'adapted from literature'
# },

# %%
