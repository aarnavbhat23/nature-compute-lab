import importlib.util
from pathlib import Path
import unittest

import numpy as np


def load_clocks():
    path = Path(__file__).resolve().parents[1] / 'experiments' / '07_clocks.py'
    spec = importlib.util.spec_from_file_location('clocks', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ClockChecks(unittest.TestCase):
    def test_dense_relaxation_matches_independent_solution(self):
        module = load_clocks()
        times = np.linspace(0, 2, 101)
        predicted, _ = module.dense(np.array([.7]), times, np.ones((len(times), 1)))
        np.testing.assert_allclose(predicted[:, 0], 1-np.exp(-.7*times), atol=1e-12)

    def test_asynchronous_handles_a_changed_input_without_future_values(self):
        module = load_clocks()
        times = np.linspace(0, 2, 101)
        drive = np.ones((len(times), 1))
        a, _ = module.asynchronous(np.array([.7]), times, drive, np.array([.1]))
        drive[60:] = 0
        b, _ = module.asynchronous(np.array([.7]), times, drive, np.array([.1]))
        np.testing.assert_array_equal(a[:60], b[:60])
        self.assertLess(b[-1, 0], a[-1, 0])

    def test_learning_recovers_rates_without_noise(self):
        module = load_clocks()
        rng = np.random.default_rng(19)
        x, u, dt = [rng.uniform(.1, 1, (50, 3)) for _ in range(3)]
        rates = np.array([.2, .9, 2.])
        measured = u+(x-u)*np.exp(-rates*dt)
        np.testing.assert_allclose(module.learn_rates(x, u, dt, measured), rates, atol=1e-4)


if __name__ == '__main__':
    unittest.main()
