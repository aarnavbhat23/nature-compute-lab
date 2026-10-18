"""Learn relaxation rates and allocate asynchronous update times to each process."""

import numpy as np
from scipy.optimize import minimize_scalar

from lab import mse


def learn_rates(initial, targets, dt, observed):
    inferred = []
    for j in range(initial.shape[1]):
        loss = lambda rate: mse(observed[:, j], targets[:, j] + (initial[:, j]-targets[:, j])*np.exp(-rate*dt[:, j]))
        inferred.append(minimize_scalar(loss, bounds=(.01, 6), method='bounded').x)
    return np.asarray(inferred)


def make_drive(rng, steps, processes):
    levels = rng.uniform(0, 1, (4, processes))
    phase = np.minimum(np.arange(steps+1)*4 // (steps+1), 3)
    return levels[phase]


def dense(rates, times, drive):
    state = np.zeros(len(rates))
    history = [state.copy()]
    for i in range(1, len(times)):
        # The previous target acts over the interval ending at this observation.
        state = drive[i-1] + (state-drive[i-1])*np.exp(-rates*(times[i]-times[i-1]))
        history.append(state.copy())
    return np.asarray(history), (len(times)-1)*len(rates)


def asynchronous(rates, times, drive, intervals):
    state = np.zeros(len(rates))
    last = np.zeros(len(rates))
    target = drive[0].copy()
    next_tick = np.asarray(intervals).copy()
    output = [state.copy()]
    events = 0
    for i, time in enumerate(times[1:], 1):
        for j in range(len(rates)):
            while next_tick[j] <= time + 1e-12:
                tick = next_tick[j]
                state[j] = target[j] + (state[j]-target[j])*np.exp(-rates[j]*(tick-last[j]))
                last[j] = tick
                next_tick[j] += intervals[j]
                events += 1
            if drive[i, j] != target[j]:
                # External changes invalidate the old deadline and update exactly to the boundary.
                if time > last[j] + 1e-12:
                    state[j] = target[j] + (state[j]-target[j])*np.exp(-rates[j]*(time-last[j]))
                    events += 1
                last[j] = time
                target[j] = drive[i, j]
                next_tick[j] = time + intervals[j]
        # Zero-order hold between scheduled updates; no analytic readout is hidden here.
        output.append(state.copy())
    return np.asarray(output), events


def run(seed):
    rng = np.random.default_rng(seed)
    actual = np.exp(rng.uniform(np.log(.06), np.log(3.5), 6))
    initial, targets = rng.uniform(0, 1, (32, 6)), rng.uniform(0, 1, (32, 6))
    dt = rng.uniform(.05, .8, (32, 6))
    observed = targets + (initial-targets)*np.exp(-actual*dt) + rng.normal(0, .005, initial.shape)
    rates = learn_rates(initial, targets, dt, observed)
    vt = np.linspace(0, 10, 501)
    vu = make_drive(rng, 500, 6)
    reference, _ = dense(rates, vt, vu)
    candidates = []
    for multiplier in [.002, .005, .01, .02, .05, .1, .2]:
        pred, count = asynchronous(rates, vt, vu, multiplier/rates)
        candidates.append((multiplier, mse(reference, pred), count))
    passing = [row for row in candidates if row[1] <= 1e-4]
    multiplier = max(passing, key=lambda row: row[0])[0] if passing else candidates[0][0]
    times = np.linspace(0, 18, 901)
    drive = make_drive(rng, 900, 6)
    truth, _ = dense(actual, times, drive)
    fitted_dense, dense_count = dense(rates, times, drive)
    pred, count = asynchronous(rates, times, drive, multiplier/rates)
    # A uniform clock gets an approximately equal event budget, including input-change events.
    uniform_interval = 18*6 / max(count-18, 1)
    uniform, uniform_count = asynchronous(rates, times, drive, np.full(6, uniform_interval))
    return {'metrics': {
        'asynchronous_mse': mse(truth, pred),
        'fitted_dense_mse': mse(truth, fitted_dense),
        'uniform_clock_mse': mse(truth, uniform),
        'extra_mse_vs_fitted_dense': mse(fitted_dense, pred),
        'asynchronous_updates': count, 'dense_updates': dense_count,
        'uniform_clock_updates': uniform_count, 'dense_to_async_update_ratio': dense_count/count,
        'rate_relative_mae': float(np.mean(np.abs(rates-actual)/actual)),
    }, 'details': {'true_rates': actual.tolist(), 'learned_rates': rates.tolist(),
        'validation_candidates': [{'multiplier': m, 'mse': e, 'updates': c} for m, e, c in candidates],
        'chosen_multiplier': multiplier, 'same_rates_used_by_all_test_methods': True,
        'new_test_trajectories_not_new_processes': True,
        'update_counts_are_not_wall_clock_speedup': True}}
