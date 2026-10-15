"""Retain competing executable explanations and recall them after regime changes."""

from collections import deque

import numpy as np

from lab import fit_ridge, mse


class OnlineRidge:
    def __init__(self, x, y, forgetting=1.):
        self.gram = x.T @ x
        self.rhs = x.T @ y
        self.forgetting = forgetting

    def predict(self, x):
        return float(x @ np.linalg.solve(self.gram + np.eye(len(x))*1e-4, self.rhs))

    def observe(self, x, y):
        self.gram = self.forgetting*self.gram + np.outer(x, x)
        self.rhs = self.forgetting*self.rhs + x*y


class Repertoire:
    def __init__(self, x, y, window=24, max_experts=8):
        self.experts = [fit_ridge(x, y)]
        self.history = deque(zip(x[-window:], y[-window:]), maxlen=window)
        self.window = window
        self.max_experts = max_experts
        self.since_birth = window
        self.births = []
        self.steps = 0

    def losses(self):
        x = np.asarray([p[0] for p in self.history])
        y = np.asarray([p[1] for p in self.history])
        return [mse(y, x @ w) for w in self.experts]

    def predict(self, x):
        chosen = int(np.argmin(self.losses()))
        return float(x @ self.experts[chosen])

    def observe(self, x, y):
        self.history.append((x.copy(), y))
        self.since_birth += 1
        self.steps += 1
        if len(self.history) < self.window or self.since_birth < self.window or len(self.experts) >= self.max_experts:
            return
        old_loss = min(self.losses())
        if old_loss > .03:
            bx = np.asarray([p[0] for p in self.history])
            by = np.asarray([p[1] for p in self.history])
            candidate = fit_ridge(bx, by, .01)
            if mse(by, bx @ candidate) < .5*old_loss:
                self.experts.append(candidate)
                self.births.append(self.steps)
                self.since_birth = 0


def predict_then_learn(model, x, y):
    predictions = []
    for row, label in zip(x, y):
        predictions.append(model.predict(row))
        model.observe(row, label)
    return np.asarray(predictions)


def run(seed):
    rng = np.random.default_rng(seed)
    rules = np.array([[.2, .8, -.3], [-.3, -.8, .4], [.4, .3, .9]]) * rng.uniform(.9, 1.1, (3, 3))
    raw = rng.uniform(-1, 1, 80+6*150)
    x = np.column_stack([np.ones(len(raw)), raw, raw*raw])
    schedule = np.repeat([0, 1, 0, 2, 1, 0], 150)
    states = np.concatenate([np.zeros(80, dtype=int), schedule])
    expected = np.sum(x * rules[states], axis=1)
    observed = expected + rng.normal(0, .04, len(x))
    initial_x, initial_y = x[:80], observed[:80]
    models = {'repertoire': Repertoire(initial_x, initial_y),
              'cumulative': OnlineRidge(initial_x, initial_y),
              'forgetting': OnlineRidge(initial_x, initial_y, .94)}
    return_indices = np.concatenate([np.arange(block*150, block*150+20) for block in [2, 4, 5]])
    metrics, predictions = {}, {}
    for name, model in models.items():
        pred = predict_then_learn(model, x[80:], observed[80:])
        predictions[name] = pred
        metrics[name+'_mse'] = mse(expected[80:], pred)
        metrics[name+'_return_window_mse'] = mse(expected[80:][return_indices], pred[return_indices])
    metrics['retained_experts'] = len(models['repertoire'].experts)
    return {'metrics': metrics, 'details': {'regime_labels_visible_to_models': False,
        'warmup_labels': 80, 'online_labels': 900, 'prediction_precedes_own_label': True,
        'birth_steps': models['repertoire'].births,
        'rules': [w.tolist() for w in models['repertoire'].experts],
        'model_hyperparameters_fixed_before_stream': {'window': 24, 'birth_loss': .03, 'forgetting': .94},
        'selected_trace': [{'step': i, 'expected': float(expected[80+i]),
                            **{name: float(p[i]) for name, p in predictions.items()}}
                           for i in range(0, 900, 15)]}}
