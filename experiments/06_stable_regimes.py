"""Learn history-dependent switching boundaries and evaluate closed-loop memory."""

import numpy as np

from lab import fit_ridge


def stimuli(rng, n, steps):
    time = np.arange(steps)[None, :]
    period = rng.uniform(35, 110, (n, 1))
    phase = rng.uniform(0, 2*np.pi, (n, 1))
    return np.sin(2*np.pi*time/period + phase) + rng.uniform(-.15, .15, (n, 1)) + rng.normal(0, .04, (n, steps))


def hysteresis(inputs, low, high):
    state = np.zeros(len(inputs), dtype=bool)
    history = []
    for u in inputs.T:
        state = np.where(u > high, True, np.where(u < low, False, state))
        history.append(state.copy())
    return np.asarray(history).T


def grid_errors(inputs, labels, lows, highs):
    state = np.zeros((len(inputs), len(lows)), dtype=bool)
    errors = np.zeros(len(lows))
    for u, target in zip(inputs.T, labels.T):
        state = np.where(u[:, None] > highs, True, np.where(u[:, None] < lows, False, state))
        errors += np.sum(state != target[:, None], axis=0)
    return errors / labels.size


def ar_features(u, previous_u, previous_state):
    return np.column_stack([np.ones(len(u)), u, previous_u, previous_state])


def autoregressive(inputs, coef):
    state, last_u = np.zeros(len(inputs)), np.zeros(len(inputs))
    outputs = []
    for u in inputs.T:
        state = ar_features(u, last_u, state) @ coef > .5
        outputs.append(state.copy())
        last_u = u
    return np.asarray(outputs).T


def fit_table(inputs, labels):
    edges = np.linspace(-1.5, 1.5, 31)
    counts, sums = np.ones((30, 2)), np.full((30, 2), .5)
    previous = np.zeros(len(inputs), dtype=int)
    for u, target in zip(inputs.T, labels.T):
        bins = np.clip(np.searchsorted(edges, u, side='right')-1, 0, 29)
        np.add.at(counts, (bins, previous), 1)
        np.add.at(sums, (bins, previous), target)
        previous = target.astype(int)
    return edges, sums/counts


def table_predict(inputs, edges, table):
    state = np.zeros(len(inputs), dtype=int)
    outputs = []
    for u in inputs.T:
        bins = np.clip(np.searchsorted(edges, u, side='right')-1, 0, 29)
        state = (table[bins, state] > .5).astype(int)
        outputs.append(state.copy())
    return np.asarray(outputs).T


def run(seed):
    rng = np.random.default_rng(seed)
    low, high = rng.uniform(-.5, -.2), rng.uniform(.2, .55)
    train, valid, test = [stimuli(rng, n, steps) for n, steps in [(24, 120), (12, 120), (32, 240)]]
    labels = hysteresis(train, low, high)
    observed = np.logical_xor(labels, rng.random(labels.shape) < .02)
    vy, ty = hysteresis(valid, low, high), hysteresis(test, low, high)
    lows, highs = np.meshgrid(np.linspace(-.7, -.1, 25), np.linspace(.1, .7, 25))
    lows, highs = lows.ravel(), highs.ravel()
    shortlist = np.argsort(grid_errors(train, observed, lows, highs), kind='stable')[:12]
    best = shortlist[np.argmin(grid_errors(valid, vy, lows[shortlist], highs[shortlist]))]
    predictions = hysteresis(test, lows[best], highs[best])
    thresholds = np.linspace(-.8, .8, 65)
    train_errors = np.mean((train[..., None] > thresholds) != observed[..., None], axis=(0, 1))
    short = np.argsort(train_errors, kind='stable')[:5]
    threshold = thresholds[short[np.argmin(np.mean((valid[..., None] > thresholds[short]) != vy[..., None], axis=(0, 1)))]]
    previous_u = np.column_stack([np.zeros(len(train)), train[:, :-1]])
    previous_state = np.column_stack([np.zeros(len(train)), observed[:, :-1]])
    coef = fit_ridge(ar_features(train.ravel(), previous_u.ravel(), previous_state.ravel()), observed.ravel())
    edges, table = fit_table(train, observed)
    return {'metrics': {
        'hysteresis_error_rate': float(np.mean(predictions != ty)),
        'memoryless_error_rate': float(np.mean((test > threshold) != ty)),
        'autoregressive_error_rate': float(np.mean(autoregressive(test, coef) != ty)),
        'transition_table_error_rate': float(np.mean(table_predict(test, edges, table) != ty)),
    }, 'details': {'learned_low': float(lows[best]), 'learned_high': float(highs[best]),
                  'true_low': low, 'true_high': high, 'test_sequences': 32,
                  'test_steps': 240, 'training_label_flip_probability': .02,
                  'test_previous_labels_visible': False}}
