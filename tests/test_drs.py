"""Check derivatives, numerical support, and independent reference trajectories."""

import unittest
import numpy as np

from drs.models import Substrate, NeuralRates
from drs.systems import System, dataset


class SubstrateTests(unittest.TestCase):
    def test_unrolled_parameter_gradients(self):
        rng = np.random.default_rng(44)
        for model in [Substrate(q=2, hidden=1, seed=3), NeuralRates(q=2, hidden=1, seed=3, budget=80)]:
            initial = rng.uniform(.1,.9,(2,2))
            inputs = rng.uniform(.1,.9,(2,3,2))
            target = rng.uniform(.1,.9,(2,3,2))
            variance = np.array([.1,.3])
            _, analytic = model.loss_gradient(initial,inputs,target,variance)
            # Includes zero-/one-/two-source rates, inhibition, K, and hidden starts.
            indices = np.unique(np.r_[np.linspace(0,len(model.theta)-1,25,dtype=int),len(model.theta)-1])
            for i in indices:
                original = model.theta[i];epsilon=1e-5
                model.theta[i]=original+epsilon
                plus=np.mean((model.rollout(initial,inputs)-target)**2/variance)
                model.theta[i]=original-epsilon
                minus=np.mean((model.rollout(initial,inputs)-target)**2/variance)
                model.theta[i]=original
                self.assertAlmostEqual(analytic[i],(plus-minus)/(2*epsilon),delta=2e-7)

    def test_state_jacobian_vector_product(self):
        rng=np.random.default_rng(92)
        for model in [Substrate(q=2,hidden=1),NeuralRates(q=2,hidden=1,budget=80)]:
            h=rng.uniform(.1,.9,(2,3));u=rng.uniform(.1,.9,(2,2))
            gp=rng.normal(size=h.shape);gd=rng.normal(size=h.shape)
            _,_,cache=model.rates(h,u)
            analytic,_=model.rate_vjp(gp,gd,cache)
            for row in range(2):
                for col in range(3):
                    epsilon=1e-6;plus=h.copy();minus=h.copy()
                    plus[row,col]+=epsilon;minus[row,col]-=epsilon
                    pp,pd,_=model.rates(plus,u);mp,md,_=model.rates(minus,u)
                    numerical=(np.sum(gp*(pp-mp))+np.sum(gd*(pd-md)))/(2*epsilon)
                    self.assertAlmostEqual(analytic[row,col],numerical,delta=1e-7)

    def test_long_rollout_is_bounded_at_extreme_parameters(self):
        model=Substrate(q=2,hidden=1)
        model.theta[:]=30
        predicted=model.rollout(np.array([[0.,1.]]),np.ones((1,500,2)),delta=10,substeps=1)
        self.assertTrue(np.isfinite(predicted).all())
        self.assertTrue(((predicted>=0)&(predicted<=1)).all())

    def test_invalid_measurements_fail(self):
        model=Substrate(q=2,hidden=1)
        for value in [-.1,np.nan,1.1]:
            with self.assertRaises(ValueError):
                model.rollout(np.array([[value,.5]]),np.ones((1,2,2)))

    def test_resource_cap(self):
        model=Substrate()
        self.assertEqual(model.gates,252)
        self.assertEqual(len(model.theta),736)
        self.assertLess(abs(len(NeuralRates(budget=736).theta)/736-1),.1)
        with self.assertRaises(ValueError):Substrate(q=16)

    def test_reference_resolution_and_split_independence(self):
        for family in ['saturating','polynomial','forced_sinusoidal']:
            system=System(family,q=3,seed=5)
            a=dataset(system,20,2,5)
            b=dataset(system,21,2,5)
            self.assertFalse(np.array_equal(a['initial'],b['initial']))
            fine=system.trajectory(a['initial'],a['inputs'],reference_step=.005)
            self.assertLess(np.max(np.abs(fine-a['target'])),1e-7)


if __name__=='__main__':unittest.main()
