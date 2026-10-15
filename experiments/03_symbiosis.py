"""Grow a composition graph from intermediate module outputs."""

import numpy as np

from lab import fit_ridge, mse


def sample(rng, n, together):
    blocks = []
    while sum(map(len, blocks)) < n:
        x = rng.uniform(-1, 1, (n*4, 2))
        mask = np.all(np.abs(x) > .65, axis=1) if together else np.abs(x[:, 0]*x[:, 1]) < .35
        blocks.append(x[mask])
    return np.concatenate(blocks)[:n]


def atomic(x):
    return np.column_stack([np.ones(len(x)), x, x*x, np.tanh(1.3*x)])


def execute(z, operations):
    values = [np.ones(len(z)), z[:, 0], z[:, 1]]
    for left, right in operations:
        values.append(values[left] * values[right])
    return np.column_stack(values)


def grow_partnerships(z, y, vz, vy, max_joins=4):
    operations, powers = [], [(0, 0), (1, 0), (0, 1)]
    best = (float('inf'), [], None)
    trace = []
    for step in range(max_joins+1):
        x, vx = execute(z, operations), execute(vz, operations)
        coef = fit_ridge(x, y, 1e-4)
        error = mse(vy, vx @ coef)
        trace.append({'joins': [list(p) for p in operations], 'validation_mse': error})
        if error < best[0]:
            best = error, operations.copy(), coef.copy()
        if step == max_joins:
            break
        candidates = []
        for j in range(1, len(powers)):
            for k in range(j, len(powers)):
                power = tuple(a+b for a, b in zip(powers[j], powers[k]))
                if sum(power) <= 4 and power not in powers:
                    extended = np.column_stack([x, x[:, j]*x[:, k]])
                    w = fit_ridge(extended, y, 1e-4)
                    candidates.append((mse(y, extended @ w), j, k, power))
        if not candidates:
            break
        _, j, k, power = min(candidates)
        operations.append((j, k))
        powers.append(power)
    return best[1], best[2], trace


def all_interactions(z):
    return np.column_stack([z[:, 0]**i * z[:, 1]**j for total in range(5) for i in range(total+1) for j in [total-i]])


def run(seed):
    rng = np.random.default_rng(seed)
    train, valid, test = [sample(rng, n, both) for n, both in [(240, False), (120, False), (300, True)]]
    functions = [lambda a: .7*np.tanh(1.3*a)+.2*a, lambda b: .6*b+.15*b*b]
    measured = np.column_stack([f(train[:, i]) for i, f in enumerate(functions)]) + rng.normal(0, .01, (len(train), 2))
    atom_coefs = [fit_ridge(atomic(train[:, i]), measured[:, i]) for i in range(2)]
    encode = lambda x: np.column_stack([atomic(x[:, i]) @ atom_coefs[i] for i in range(2)])
    response = lambda x: (lambda a, b: a+b+.7*a*b+.4*(a*b)**2)(functions[0](x[:, 0]), functions[1](x[:, 1]))
    z, vz, tz = map(encode, [train, valid, test])
    y = response(train) + rng.normal(0, .015, len(train))
    operations, coef, trace = grow_partnerships(z, y, vz, response(valid))
    independent = fit_ridge(execute(z, []), y)
    full = fit_ridge(all_interactions(z), y, 1e-3)
    return {'metrics': {
        'partnership_mse': mse(response(test), execute(tz, operations) @ coef),
        'independent_modules_mse': mse(response(test), execute(tz, []) @ independent),
        'fixed_interactions_mse': mse(response(test), all_interactions(tz) @ full),
        'retained_joins': len(operations), 'fixed_dictionary_terms': 15,
    }, 'details': {'operations': [list(p) for p in operations], 'coefficients': coef.tolist(),
                   'selection_trace': trace, 'intermediate_training_labels_available_to_all_controls': True,
                   'train_constraint': 'abs(x1*x2) < 0.35', 'test_constraint': 'abs(x1),abs(x2) > 0.65'}}
