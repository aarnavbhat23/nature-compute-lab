import unittest

import numpy as np

from test_development import load


class RegimeChecks(unittest.TestCase):
    def test_identical_current_input_can_retain_different_histories(self):
        module = load('06_stable_regimes.py')
        inputs = np.array([[1., 0., 0.], [-1., 0., 0.]])
        output = module.hysteresis(inputs, -.4, .4)
        np.testing.assert_array_equal(output[:, -1], [True, False])

    def test_reset_overrides_memory(self):
        module = load('06_stable_regimes.py')
        output = module.hysteresis(np.array([[1., 0., -1., 0.]]), -.4, .4)
        np.testing.assert_array_equal(output, [[True, True, False, False]])


if __name__ == '__main__':
    unittest.main()
