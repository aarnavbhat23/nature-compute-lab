"""Learn a local growth program on short cascades, then instantiate longer ones."""

import numpy as np

from lab import choose_terms, fit_ridge, mse

TERMS = ['constant', 'identity', 'square', 'cube', 'tanh(1.4x)', 'tanh(0.7x)']


def features(x):
    return np.column_stack([np.ones_like(x), x, x*x, x*x*x, np.tanh(1.4*x), np.tanh(.7*x)])


def grow(seed_values, stages, transfer):
    """The same local production instruction creates each successive stage."""
    states = [np.asarray(seed_values).copy()]
    for _ in range(stages):
        states.append(transfer(states[-1]))
    return np.asarray(states)


def run(seed):
    rng = np.random.default_rng(seed)
    a, b, c = rng.uniform(.2, .35), rng.uniform(.4, .6), rng.uniform(-.03, .03)
    truth = lambda x: a*x + b*np.tanh(1.4*x) + c
    train = grow(rng.uniform(-1.2, 1.2, 32), 3, truth)
    valid = grow(rng.uniform(-1.2, 1.2, 16), 4, truth)
    x, y = train[:-1].ravel(), train[1:].ravel()
    noisy_y = y + rng.normal(0, .01, y.shape)
    vx, vy = valid[:-1].ravel(), valid[1:].ravel()
    active, coef, trace = choose_terms(features(x), noisy_y, features(vx), vy, max_terms=6)
    linear = fit_ridge(features(x)[:, :2], noisy_y)
    full = fit_ridge(features(x), noisy_y, penalty=1e-3)
    initial = rng.uniform(-1.2, 1.2, 64)
    expected = grow(initial, 12, truth)[1:]
    local = lambda z: features(z)[:, active] @ coef
    predicted = grow(initial, 12, local)[1:]
    return {
        'metrics': {
            'grown_program_mse': mse(expected, predicted),
            'shared_linear_mse': mse(expected, grow(initial, 12, lambda z: features(z)[:, :2] @ linear)[1:]),
            'fixed_full_dictionary_mse': mse(expected, grow(initial, 12, lambda z: features(z) @ full)[1:]),
            'active_terms': len(active), 'available_terms': len(TERMS),
        },
        'details': {'selected_program': dict(zip([TERMS[i] for i in active], coef.tolist())),
                    'growth_trace': trace, 'train_stages': 3, 'test_stages': 12,
                    'test_chains': 64, 'test_contains_training_chains': False},
    }
