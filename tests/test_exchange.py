"""Conservation, positivity, gradients, and approximation error of exchange blocks."""

import unittest
import numpy as np
from scipy.linalg import expm

from components.exchange import exchange_step,exchange_vjp


class ExchangeTests(unittest.TestCase):
    def test_conservation_and_positivity_under_extreme_rates(self):
        rng=np.random.default_rng(93)
        x=rng.uniform(0,10,(5,6))
        edges=np.array([(i,(i+1)%6) for i in range(6)]+[(0,3),(2,5)])
        rates=10**rng.uniform(-3,3,(5,len(edges)))
        for delta in [.001,.1,10]:
            y=exchange_step(x,rates,edges,delta)
            self.assertGreaterEqual(y.min(),-1e-12)
            np.testing.assert_allclose(y.sum(axis=1),x.sum(axis=1),rtol=1e-11,atol=1e-11)

    def test_state_and_rate_gradients(self):
        rng=np.random.default_rng(17)
        x=rng.uniform(.1,1,(2,3));rates=rng.uniform(.1,1,(2,4))
        edges=np.array([[0,1],[1,2],[2,0],[0,2]])
        g=rng.normal(size=x.shape)
        _,cache=exchange_step(x,rates,edges,.2,True)
        dx,dk=exchange_vjp(g,cache)
        for values,analytic,is_state in [(x,dx,True),(rates,dk,False)]:
            for index in np.ndindex(values.shape):
                plus=values.copy();minus=values.copy();plus[index]+=1e-6;minus[index]-=1e-6
                yp=exchange_step(plus if is_state else x,rates if is_state else plus,edges,.2)
                ym=exchange_step(minus if is_state else x,rates if is_state else minus,edges,.2)
                self.assertAlmostEqual(analytic[index],np.sum(g*(yp-ym))/2e-6,delta=1e-8)

    def test_convergence_to_exact_linear_dynamics(self):
        edges=np.array([[0,1],[1,2],[2,0]])
        rates=np.array([[.7,.3,.5]]);x=np.array([[.6,.1,.3]])
        generator=np.zeros((3,3))
        for rate,(source,dest) in zip(rates[0],edges):
            generator[source,source]-=rate;generator[dest,source]+=rate
        truth=expm(generator)@x[0]
        errors=[]
        for steps in [5,10,20]:
            y=x.copy()
            for _ in range(steps):y=exchange_step(y,rates,edges,1/steps)
            errors.append(np.linalg.norm(y[0]-truth))
        self.assertTrue(errors[2]<errors[1]<errors[0])

    def test_empty_exchange_is_identity(self):
        x=np.array([[.2,.8]])
        y,cache=exchange_step(x,np.empty((1,0)),np.empty((0,2),dtype=int),1,True)
        np.testing.assert_array_equal(y,x)
        dx,dk=exchange_vjp(np.ones_like(x),cache)
        np.testing.assert_array_equal(dx,np.ones_like(x))
        self.assertEqual(dk.shape,(1,0))

    def test_invalid_inputs_are_rejected(self):
        for rates in [np.array([[-1.]]),np.array([[np.inf]])]:
            with self.assertRaises(ValueError):
                exchange_step(np.array([[1.,0.]]),rates,np.array([[0,1]]),.1)


if __name__=='__main__':unittest.main()
