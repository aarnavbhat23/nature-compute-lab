"""Independent continuous-time pilot generators; never call the learner's update."""

import numpy as np

FAMILIES = ('saturating', 'polynomial', 'forced_sinusoidal')


class System:
    def __init__(self, family, q=8, seed=0):
        if family not in FAMILIES:raise ValueError(family)
        self.family, self.q = family, q
        rng = np.random.default_rng(seed)
        self.parents = np.array([rng.choice(q, 3, replace=False) for _ in range(q)])
        self.strength = rng.uniform(.8, 1.2, (q, 2))
        self.phase = rng.uniform(-np.pi, np.pi, q)

    def derivative(self, x, u):
        a, b, c = (x[:, self.parents[:, i]] for i in range(3))
        if self.family == 'saturating':
            p = .08+.5*a/(.25+a)+.25*(b/(.4+b))*(c/(.3+c))+.15*u[:, :1]
            d = .12+.35*.3/(.3+c)+.12*u[:, 1:]
        elif self.family == 'polynomial':
            p = .08+.5*a*b+.15*u[:, :1]
            d = .10+.4*c*c+.12*u[:, 1:]
        else:
            p = .1+.35*(.5+.5*np.sin(2*np.pi*a+self.phase))+.15*u[:, :1]
            d = .12+.3*(.5+.5*np.cos(2*np.pi*b+self.phase))+.12*u[:, 1:]
        return (1-x)*p*self.strength[:, 0]-x*d*self.strength[:, 1]

    def trajectory(self, initial, inputs, interval=.2, reference_step=.01):
        substeps = round(interval/reference_step)
        if not np.isclose(substeps*reference_step, interval):raise ValueError('Incommensurate step')
        x = np.array(initial, dtype=float, copy=True)
        output = []
        for t in range(inputs.shape[1]):
            u = inputs[:, t]
            for _ in range(substeps):
                dt = reference_step
                k1 = self.derivative(x, u)
                k2 = self.derivative(x+dt*k1/2, u)
                k3 = self.derivative(x+dt*k2/2, u)
                k4 = self.derivative(x+dt*k3, u)
                x = x+dt*(k1+2*k2+2*k3+k4)/6
            if not np.isfinite(x).all() or not ((x >= 0) & (x <= 1)).all():
                raise FloatingPointError('Reference state escaped its declared support')
            output.append(x.copy())
        return np.stack(output, axis=1)


def dataset(system, seed, count, length, interval=.2):
    rng = np.random.default_rng(seed)
    initial = rng.uniform(.05, .95, (count, system.q))
    inputs = rng.uniform(.05, .95, (count, length, 2))
    # Known piecewise-constant interventions: levels held for two observations.
    inputs[:, 1::2] = inputs[:, ::2][:, :inputs[:, 1::2].shape[1]]
    if system.family == 'forced_sinusoidal':
        phase = rng.uniform(0, 2*np.pi, (count, 1, 2))
        inputs = .5+.45*np.sin(np.arange(length)[None, :, None]*interval*2+phase)
    target = system.trajectory(initial, inputs, interval)
    return {'initial':initial, 'inputs':inputs, 'target':target}
