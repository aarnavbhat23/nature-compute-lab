import importlib.util
from pathlib import Path
import unittest

import numpy as np


def load(name):
    p = Path(__file__).resolve().parents[1] / 'experiments' / name
    spec = importlib.util.spec_from_file_location(p.stem, p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DevelopmentChecks(unittest.TestCase):
    def test_growth_composes_a_local_instruction(self):
        model = load('01_development.py')
        actual = model.grow(np.array([1., 2.]), 4, lambda x: x + 1)
        np.testing.assert_array_equal(actual[-1], [5., 6.])
        self.assertEqual(actual.shape, (5, 2))


if __name__ == '__main__':
    unittest.main()
