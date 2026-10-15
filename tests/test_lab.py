import unittest

import numpy as np

from lab import choose_terms, fit_ridge, mse


class RegressionChecks(unittest.TestCase):
    def test_known_coefficients(self):
        rng = np.random.default_rng(2)
        x = np.column_stack([np.ones(100), rng.normal(size=(100, 3))])
        coef = np.array([2., -1., 0., 3.])
        np.testing.assert_allclose(fit_ridge(x, x @ coef), coef, atol=1e-5)

    def test_sparse_selection_has_no_test_input(self):
        rng = np.random.default_rng(3)
        x = np.column_stack([np.ones(300), rng.normal(size=(300, 4))])
        y = 1 + 2 * x[:, 2]
        active, coef, _ = choose_terms(x[:200], y[:200], x[200:], y[200:], 4)
        self.assertIn(2, active)
        self.assertLess(mse(y[200:], x[200:, active] @ coef), 1e-10)


if __name__ == '__main__':
    unittest.main()
