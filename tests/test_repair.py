import unittest

import numpy as np

from test_development import load


class RepairChecks(unittest.TestCase):
    def test_single_missing_interaction_can_be_recovered(self):
        module = load('04_repair.py')
        rng = np.random.default_rng(91)
        x = module.features(rng.uniform(-1, 1, (200, 3)))
        original = np.arange(9)/10
        expected = original.copy()
        expected[4] += .7
        repaired, active, _ = module.repair(original, x[:100], x[:100] @ expected, x[100:], x[100:] @ expected)
        self.assertIn(4, active)
        self.assertLess(np.max(np.abs(x[100:] @ (repaired-expected))), 1e-5)
        np.testing.assert_array_equal(original, np.arange(9)/10)


if __name__ == '__main__':
    unittest.main()
