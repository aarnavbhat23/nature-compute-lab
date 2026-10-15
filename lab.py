"""Small numerical helpers shared by the experiments; no hidden training framework."""

import numpy as np


def fit_ridge(x, y, penalty=1e-6):
    """Feature scaling is learned from x only; the constant column is unpenalized."""
    scale = np.maximum(np.sqrt(np.mean(x * x, axis=0)), 1e-8)
    z = x / scale
    reg = np.eye(z.shape[1]) * penalty
    reg[0, 0] = 0
    weights = np.linalg.solve(z.T @ z + reg, z.T @ y)
    return weights / scale


def mse(y, pred):
    return float(np.mean((np.asarray(y) - np.asarray(pred)) ** 2))


def choose_terms(train_x, train_y, valid_x, valid_y, max_terms=8):
    """Greedy train-residual proposals, with model size selected on validation."""
    active = [0]
    trace = []
    best = (float('inf'), active.copy(), None)
    for _ in range(min(max_terms, train_x.shape[1])):
        weights = fit_ridge(train_x[:, active], train_y)
        error = mse(valid_y, valid_x[:, active] @ weights)
        trace.append({'terms': active.copy(), 'validation_mse': error})
        if error < best[0]:
            best = (error, active.copy(), weights.copy())
        residual = train_y - train_x[:, active] @ weights
        candidates = [i for i in range(train_x.shape[1]) if i not in active]
        if not candidates:
            break
        norm = np.maximum(np.linalg.norm(train_x[:, candidates], axis=0), 1e-8)
        scores = np.abs(train_x[:, candidates].T @ residual) / norm
        active.append(candidates[int(np.argmax(scores))])
    return best[1], best[2], trace
