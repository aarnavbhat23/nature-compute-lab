import unittest

import numpy as np

from test_development import load


class PartnershipChecks(unittest.TestCase):
    def test_composite_output_can_feed_a_new_module(self):
        module = load('03_symbiosis.py')
        z = np.array([[2., 3.]])
        values = module.execute(z, [(1, 2), (3, 3)])
        np.testing.assert_array_equal(values, [[1., 2., 3., 6., 36.]])

    def test_joint_test_region_is_not_in_training_region(self):
        module = load('03_symbiosis.py')
        x = module.sample(np.random.default_rng(8), 100, True)
        self.assertTrue(np.all(np.abs(x[:, 0]*x[:, 1]) > .35))


if __name__ == '__main__':
    unittest.main()
