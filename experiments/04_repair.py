"""After one interaction changes, learn a small patch without refitting everything."""

import numpy as np

from lab import choose_terms, fit_ridge, mse

TERMS = ['constant', 'a', 'b', 'c', 'a*b', 'b*c', 'sin(a)', 'a_squared', 'c_squared']


def features(x):
    a, b, c = x.T
    return np.column_stack([np.ones(len(x)), a, b, c, a*b, b*c, np.sin(a), a*a, c*c])


def tuned_fit(x, y, vx, vy):
    trials = []
    for penalty in [1e-6, 1e-3, .1, 1., 10.]:
        coef = fit_ridge(x, y, penalty)
        trials.append((mse(vy, vx @ coef), penalty, coef))
    _, penalty, coef = min(trials, key=lambda t: t[0])
    return coef, penalty


def repair(original, x, y, vx, vy):
    residual = y - x @ original
    valid_residual = vy - vx @ original
    active, delta, trace = choose_terms(x, residual, vx, valid_residual, max_terms=4)
    patch = np.zeros_like(original)
    patch[active] = delta
    return original+patch, active, trace


def run(seed):
    rng = np.random.default_rng(seed)
    x = features(rng.uniform(-1, 1, (1000, 3)))
    original_truth = np.array([.2, .5, -.3, .25, 0, .3, .2, -.1, .15]) * rng.uniform(.8, 1.2, 9)
    original = fit_ridge(x, x @ original_truth + rng.normal(0, .03, len(x)))
    changed_truth = original_truth.copy()
    changed_truth[4] = rng.uniform(.6, 1.)
    ax, vx, tx = [features(rng.uniform(-1, 1, (n, 3))) for n in [12, 12, 500]]
    ay, vy = [z @ changed_truth + rng.normal(0, .05, len(z)) for z in [ax, vx]]
    patched, active, trace = repair(original, ax, ay, vx, vy)
    refit, refit_penalty = tuned_fit(ax, ay, vx, vy)
    delta, delta_penalty = tuned_fit(ax, ay-ax@original, vx, vy-vx@original)
    expected = tx @ changed_truth
    return {'metrics': {
        'sparse_repair_mse': mse(expected, tx @ patched),
        'frozen_mse': mse(expected, tx @ original),
        'full_refit_mse': mse(expected, tx @ refit),
        'regularized_full_patch_mse': mse(expected, tx @ (original+delta)),
        'patch_terms': len(active),
        'changed_interaction_selected': float(4 in active),
    }, 'details': {'patch': dict(zip([TERMS[i] for i in active], (patched-original)[active].tolist())),
                   'trace': trace, 'adaptation_fit_labels': 12, 'adaptation_validation_labels': 12,
                   'full_refit_penalty': refit_penalty, 'full_patch_penalty': delta_penalty,
                   'changed_term': 'a*b', 'all_adapting_controls_receive_same_examples': True}}
