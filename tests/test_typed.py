"""Complete chain-rule checks for coupled rates, physical pools, and memory."""

import unittest
import numpy as np
from scipy.integrate import solve_ivp
from drs.typed import TypedDynamics
from drs.resource_systems import ResourceSystem, resource_dataset
from drs.repair_systems import ChangedResourceSystem


class TypedTests(unittest.TestCase):
    def test_full_unrolled_gradient(self):
        rng = np.random.default_rng(43)
        initial = rng.uniform(.1, .9, (2, 5))
        inputs = rng.uniform(.1, .9, (2, 3, 2))
        target = rng.uniform(.1, .9, (2, 3, 5))
        variance = np.arange(1, 6)*.1
        for head in ['saturating', 'neural']:
            model = TypedDynamics(head, 7)
            _, gradient = model.loss_gradient(initial, inputs, target, variance)
            for i in range(len(model.theta)):
                value = model.theta[i]
                model.theta[i] = value+1e-5
                plus = np.mean((model.rollout(initial, inputs)-target)**2/variance)
                model.theta[i] = value-1e-5
                minus = np.mean((model.rollout(initial, inputs)-target)**2/variance)
                model.theta[i] = value
                self.assertAlmostEqual(gradient[i], (plus-minus)/2e-5, delta=2e-8)

    def test_complete_step_state_gradient(self):
        rng = np.random.default_rng(53)
        state = rng.uniform(.1, .9, (2, 7));u = rng.uniform(.1, .9, (2, 2))
        g = rng.normal(size=state.shape)
        for head in ['saturating', 'neural']:
            model = TypedDynamics(head, 17)
            _, cache = model.step(state, u, .2)
            analytic, _ = model.step_vjp(g, cache)
            for index in np.ndindex(state.shape):
                plus = state.copy();minus = state.copy()
                plus[index] += 1e-6;minus[index] -= 1e-6
                yp, _ = model.step(plus, u, .2);ym, _ = model.step(minus, u, .2)
                self.assertAlmostEqual(analytic[index], np.sum(g*(yp-ym))/2e-6, delta=1e-8)

    def test_invariants_and_numerical_convergence(self):
        initial = np.array([[.3, .8, .2, .2, .7]])
        inputs = np.array([[[.8, .3]]])
        for head in ['saturating', 'neural']:
            model = TypedDynamics(head, 22)
            state = np.column_stack([initial, np.full((1, 2), .5)])
            errors = []
            def derivative(t, y):
                r, _ = model.rates(y[None], inputs[:, 0])
                dx = np.zeros(3)
                for rate, (source, dest) in zip(r[0, :4], model.edges):
                    flow = rate*y[source];dx[source] -= flow;dx[dest] += flow
                return np.r_[dx, (1-y[3:])*r[0, 4:8]-y[3:]*r[0, 8:12]]
            truth = solve_ivp(derivative, [0, 1], state[0], rtol=1e-11, atol=1e-12).y[:, -1]
            for steps in [5, 10, 20]:
                y = state.copy()
                for _ in range(steps):y, _ = model.step(y, inputs[:, 0], 1/steps)
                errors.append(np.linalg.norm(y[0]-truth))
            self.assertTrue(errors[2] < errors[1] < errors[0])
            prediction = model.rollout(initial, np.ones((1, 200, 2)), delta=10, substeps=1)
            np.testing.assert_allclose(prediction[:, :, :3].sum(axis=2), 1.3, atol=1e-11)
            self.assertTrue((prediction >= 0).all())
            self.assertTrue((prediction[:, :, 3:] <= 1).all())

    def test_independent_reference_and_splits(self):
        for mechanism in ['none', 'direct', 'resource', 'mixed']:
            system = ResourceSystem(mechanism, 9)
            data = resource_dataset(system, 33, 2, 4)
            fine = system.trajectory(data['initial'], data['inputs'], reference_step=.005)
            np.testing.assert_allclose(data['target'], fine, atol=1e-9)
            np.testing.assert_allclose(data['target'][:, :, :3].sum(axis=2),
                np.broadcast_to(data['initial'][:, :3].sum(axis=1)[:, None], (2, 4)), atol=1e-12)
            self.assertFalse(np.array_equal(data['initial'], resource_dataset(system, 34, 2, 4)['initial']))

    def test_types_and_inputs(self):
        model = TypedDynamics()
        for value in [-.1, np.nan, np.inf]:
            with self.assertRaises(ValueError):model.rollout(np.full((1, 5), value), np.ones((1, 2, 2)))
        self.assertEqual(len(model.theta), 230)
        self.assertEqual(len(TypedDynamics('neural').theta), 234)

    def test_legal_edits_preserve_rate_functions_and_have_complete_gradients(self):
        rng = np.random.default_rng(26)
        initial = rng.uniform(.1,.9,(2,5));u = rng.uniform(.1,.9,(2,2,2))
        state = rng.uniform(.1,.9,(2,7))
        for head in ['saturating','neural']:
            base = TypedDynamics(head,17)
            expanded = base.add_exchange_edges([[1,2],[2,1]])
            old, _ = base.rates(state,u[:,0]);new, _ = expanded.rates(state,u[:,0])
            np.testing.assert_array_equal(old[:,:4],new[:,:4])
            np.testing.assert_array_equal(old[:,4:],new[:,6:])
            target = rng.uniform(.1,.9,(2,2,5))
            _, g = expanded.loss_gradient(initial,u,target,np.ones(5))
            for i in np.unique(np.r_[np.linspace(0,len(g)-1,35,dtype=int),len(g)-1]):
                v = expanded.theta[i]
                expanded.theta[i] = v+1e-5
                plus = np.mean((expanded.rollout(initial,u)-target)**2)
                expanded.theta[i] = v-1e-5
                minus = np.mean((expanded.rollout(initial,u)-target)**2)
                expanded.theta[i] = v
                self.assertAlmostEqual(g[i],(plus-minus)/2e-5,delta=2e-8)
            predicted = expanded.rollout(initial,u)
            np.testing.assert_allclose(predicted[:,:,:3].sum(axis=2),
                np.broadcast_to(initial[:,:3].sum(axis=1)[:,None],(2,2)),atol=1e-12)
        for invalid in [[[0,3]],[[1,2],[1,2]],[[1.,2.]],[[0,1]]]:
            with self.assertRaises(ValueError):TypedDynamics().add_exchange_edges(invalid)

    def test_changed_reference_conserves_the_same_pool(self):
        data = resource_dataset(ResourceSystem('mixed',21),55,2,4)
        for change in ['none','parameter','edge_ab','edge_ba','both','noise']:
            system = ChangedResourceSystem(21,change)
            prediction = system.trajectory(data['initial'],data['inputs'])
            fine = system.trajectory(data['initial'],data['inputs'],reference_step=.005)
            np.testing.assert_allclose(prediction,fine,atol=1e-9)
            np.testing.assert_allclose(prediction[:,:,:3].sum(axis=2),
                np.broadcast_to(data['initial'][:,:3].sum(axis=1)[:,None],(2,4)),atol=1e-12)
            if change in ['none','noise']:np.testing.assert_array_equal(prediction,data['target'])


if __name__ == '__main__':unittest.main()
