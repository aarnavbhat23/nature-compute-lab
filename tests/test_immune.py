import unittest

import numpy as np

from test_development import load


class ImmuneChecks(unittest.TestCase):
    def test_future_labels_do_not_change_earlier_predictions(self):
        module = load('05_immune_memory.py')
        rng = np.random.default_rng(6)
        x = np.column_stack([np.ones(120), rng.normal(size=(120, 2))])
        y = x @ np.array([.1, .7, -.2])
        altered = y[40:].copy()
        altered[30:] += 10
        original = module.predict_then_learn(module.Repertoire(x[:40], y[:40]), x[40:], y[40:])
        changed = module.predict_then_learn(module.Repertoire(x[:40], y[:40]), x[40:], altered)
        np.testing.assert_array_equal(original[:31], changed[:31])


if __name__ == '__main__':
    unittest.main()
