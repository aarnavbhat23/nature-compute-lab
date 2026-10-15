import unittest

import numpy as np

from test_development import load


class CatalystChecks(unittest.TestCase):
    def test_heldout_ingredient_regions_are_disjoint(self):
        model = load('02_catalysts.py')
        rng = np.random.default_rng(4)
        train = model.ingredients(rng, 100, False)
        test = model.ingredients(rng, 100, True)
        self.assertTrue(np.all(train[:, :2].sum(axis=1) < 1.1))
        self.assertTrue(np.all(test[:, :2].sum(axis=1) > 1.5))

    def test_catalyst_requires_both_participants(self):
        model = load('02_catalysts.py')
        matrix = model.chemistry(np.array([[0., 1., .5], [1., 0., .5]]))
        np.testing.assert_array_equal(matrix[:, 5], [0., 0.])


if __name__ == '__main__':
    unittest.main()
