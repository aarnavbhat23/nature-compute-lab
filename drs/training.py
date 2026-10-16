"""Adam through closed-loop rollouts; validation selection never receives test data."""

import time
import numpy as np


def nmse(target, prediction, variance):
    return float(np.mean((target-prediction)**2/variance))


def fit(model, train, valid, variance, learning_rate, steps, batch_size, seed):
    rng = np.random.default_rng(seed)
    m, v = np.zeros_like(model.theta), np.zeros_like(model.theta)
    best, best_weights, trace = float('inf'), model.theta.copy(), []
    started = time.perf_counter()
    for step in range(1, steps+1):
        horizon = 2 if step <= steps//3 else (4 if step <= 2*steps//3 else 8)
        batch = rng.choice(len(train['initial']), batch_size, replace=False)
        loss, gradient = model.loss_gradient(train['initial'][batch], train['inputs'][batch, :horizon],
                                            train['target'][batch, :horizon], variance)
        norm = np.linalg.norm(gradient)
        if norm > 5:gradient *= 5/norm
        m = .9*m+.1*gradient; v = .999*v+.001*gradient**2
        model.theta -= learning_rate*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
        if step % 20 == 0 or step == steps:
            error = nmse(valid['target'], model.rollout(valid['initial'], valid['inputs']), variance)
            trace.append({'step':step,'horizon':horizon,'fit_batch_nmse':loss,'validation_nmse':error})
            if error < best:best, best_weights = error, model.theta.copy()
    model.theta[:] = best_weights
    return {'validation_nmse':best,'selected_step':min(trace,key=lambda r:r['validation_nmse'])['step'],
            'seconds':time.perf_counter()-started,'trace':trace}


def affine_fit(train, valid, variance):
    previous = np.concatenate([train['initial'][:, None], train['target'][:, :-1]], axis=1)
    x = np.column_stack([np.ones(previous.shape[0]*previous.shape[1]),previous.reshape(-1,previous.shape[2]),train['inputs'].reshape(-1,2)])
    y = train['target'].reshape(-1,previous.shape[2])
    choices = []
    started = time.perf_counter()
    for penalty in [1e-6, .001, .1]:
        reg = np.eye(x.shape[1])*penalty; reg[0, 0] = 0
        w = np.linalg.solve(x.T@x+reg, x.T@y)
        pred = affine_rollout(w, valid['initial'], valid['inputs'])
        choices.append((nmse(valid['target'], pred, variance),penalty,w))
    _, penalty, weights = min(choices,key=lambda z:z[0])
    return weights, {'penalty':penalty,'seconds':time.perf_counter()-started}


def affine_rollout(weights, initial, inputs):
    h, output = initial.copy(), []
    for t in range(inputs.shape[1]):
        h = np.column_stack([np.ones(len(h)),h,inputs[:,t]])@weights
        output.append(h.copy())
    return np.stack(output,axis=1)
