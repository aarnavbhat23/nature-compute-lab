"""Independent continuous-time resource/switch generator, not a cell model."""

import numpy as np


class ResourceSystem:
    def __init__(self, mechanism, seed=0):
        if mechanism not in ['none', 'direct', 'resource', 'mixed']:
            raise ValueError(mechanism)
        self.mechanism = mechanism
        rng = np.random.default_rng(seed)
        self.rate = rng.uniform(.8, 1.2, 4)
        self.threshold = rng.uniform(.25, .45)
        self.expression = rng.uniform(.7, 1., 2)

    def derivative(self, state, inputs):
        free, bound_a, bound_b, a, b = state.T
        ua, ub = inputs.T
        bind_a = self.rate[0]*(.1+.8*ua*(.2+.8*a))*free
        release_a = self.rate[1]*(.35+.2*b)*bound_a
        bind_b = self.rate[2]*(.1+.7*ub*(.3+.7*b))*free
        release_b = self.rate[3]*.4*bound_b
        pool = np.column_stack([-bind_a+release_a-bind_b+release_b,
                                bind_a-release_a, bind_b-release_b])
        resource = free/(self.threshold+free) if self.mechanism in ['resource', 'mixed'] else np.ones_like(free)
        direct = 1/(1+(a/self.threshold)**2) if self.mechanism in ['direct', 'mixed'] else np.ones_like(a)
        p = .06+np.column_stack([self.expression[0]*ua*resource,
                                self.expression[1]*ub*resource*direct])
        h = state[:, 3:]
        return np.column_stack([pool, (1-h)*p-.3*h])

    def trajectory(self, initial, inputs, interval=.2, reference_step=.01):
        count = round(interval/reference_step)
        if not np.isclose(count*reference_step, interval):
            raise ValueError('Incommensurate integration step')
        state = initial.copy()
        result = []
        for u in inputs.transpose(1, 0, 2):
            for _ in range(count):
                dt = reference_step
                k1 = self.derivative(state, u)
                k2 = self.derivative(state+dt*k1/2, u)
                k3 = self.derivative(state+dt*k2/2, u)
                k4 = self.derivative(state+dt*k3, u)
                state += dt*(k1+2*k2+2*k3+k4)/6
            if not np.isfinite(state).all() or (state < 0).any() or (state[:, 3:] > 1).any():
                raise FloatingPointError('Invalid reference trajectory')
            result.append(state.copy())
        return np.stack(result, axis=1)


def resource_dataset(system, seed, count, length):
    rng = np.random.default_rng(seed)
    pool = rng.dirichlet(np.ones(3)*2, count)*rng.uniform(.8, 1.2, (count, 1))
    initial = np.column_stack([pool, rng.uniform(.05, .95, (count, 2))])
    inputs = rng.uniform(.05, .95, (count, length, 2))
    inputs[:, 1::2] = inputs[:, ::2][:, :inputs[:, 1::2].shape[1]]
    return {'initial':initial, 'inputs':inputs, 'target':system.trajectory(initial, inputs)}
