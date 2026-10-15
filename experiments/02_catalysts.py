"""Select conditional production/consumption rules from a supplied chemistry."""

import numpy as np
from scipy.optimize import nnls

from lab import fit_ridge, mse

RULES = ['background', 'a', 'b', 'consume_c', 'a*b', 'a*b/(1+c)',
         'a*c', 'b*c', 'positive(a-b)', 'a_squared', 'b_squared', 'c_squared']


def ingredients(rng, n, joint):
    chunks = []
    while sum(len(x) for x in chunks) < n:
        x = rng.uniform(0, 1, (n*3, 3))
        total = x[:, 0] + x[:, 1]
        chunks.append(x[total > 1.5] if joint else x[total < 1.1])
    return np.concatenate(chunks)[:n]


def chemistry(x):
    a, b, c = x.T
    return np.column_stack([np.ones(len(x)), a, b, -c, a*b, a*b/(1+c),
                            a*c, b*c, np.maximum(a-b, 0), a*a, b*b, c*c])


def select_catalysts(x, y, vx, vy):
    active = [0]
    best = (float('inf'), None, None)
    trace = []
    for _ in range(7):
        coef = nnls(x[:, active], y)[0]
        error = mse(vy, vx[:, active] @ coef)
        trace.append({'active_rules': active.copy(), 'validation_mse': error})
        if error < best[0]:
            best = error, active.copy(), coef.copy()
        residual = y - x[:, active] @ coef
        unused = [i for i in range(x.shape[1]) if i not in active]
        scores = x[:, unused].T @ residual / np.maximum(np.linalg.norm(x[:, unused], axis=0), 1e-8)
        active.append(unused[int(np.argmax(scores))])
    return best[1], best[2], trace


def run(seed):
    rng = np.random.default_rng(seed)
    train, valid, test = [ingredients(rng, n, joint) for n, joint in [(250, False), (120, False), (300, True)]]
    planted = np.zeros(len(RULES))
    planted[[0, 1, 3, 5, 8]] = [.1, .6, .3, .8, .4] * rng.uniform(.8, 1.2, 5)
    x, vx, tx = map(chemistry, [train, valid, test])
    noisy = x @ planted + rng.normal(0, .015, len(train))
    active, coef, trace = select_catalysts(x, noisy, vx, vx @ planted)
    full = nnls(x, noisy)[0]
    linear = fit_ridge(x[:, :4], noisy)
    selected = {i for i, w in zip(active, coef) if w > 1e-4}
    true = set(np.flatnonzero(planted))
    f1 = 2*len(selected & true) / (len(selected)+len(true))
    return {'metrics': {
        'catalyst_mse': mse(tx @ planted, tx[:, active] @ coef),
        'linear_mse': mse(tx @ planted, tx[:, :4] @ linear),
        'full_chemistry_mse': mse(tx @ planted, tx @ full),
        'nonzero_catalysts': len(selected), 'planted_rule_f1': f1,
    }, 'details': {
        'learned_rules': dict(zip([RULES[i] for i in active], coef.tolist())),
        'planted_rules': dict(zip([RULES[i] for i in sorted(true)], planted[sorted(true)].tolist())),
        'selection_trace': trace, 'train_constraint': 'a+b < 1.1', 'test_constraint': 'a+b > 1.5',
        'dictionary_contains_generating_rules': True,
    }}
