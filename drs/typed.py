"""Coupled closed-pool, bounded-regulation, and abstract-memory pilot.

Both rate heads receive exactly the same states and use the same physical solver.
This implements a fixed typed model, not structural repair or experiment design.
"""

import numpy as np
from scipy.special import expit, logit
from components.exchange import exchange_step, exchange_vjp


class TypedDynamics:
    edges = np.array([[0, 1], [1, 0], [0, 2], [2, 0]])

    def __init__(self, head='saturating', seed=0):
        if head not in ['saturating', 'neural']:
            raise ValueError(head)
        self.head = head
        self.shapes = ([(12, 9), (12, 9), (12,)] if head == 'saturating'
                       else [(9, 10), (10,), (10, 12), (12,)])
        self.slices = []
        offset = 0
        for shape in self.shapes:
            size = int(np.prod(shape))
            self.slices.append(slice(offset, offset+size))
            offset += size
        rng = np.random.default_rng(seed)
        self.theta = np.zeros(offset+2)
        if head == 'saturating':
            self.theta[self.slices[0]] = rng.normal(0, .3, 108)
            self.theta[self.slices[1]] = logit((.3-.02)/1.98)
        else:
            self.theta[self.slices[0]] = rng.normal(0, .2, 90)
            self.theta[self.slices[2]] = rng.normal(0, .15, 120)
        self.theta[self.slices[-1]] = logit(.4/2)
        self.theta[-2:] = rng.normal(0, .2, 2)

    def rates(self, state, inputs):
        x = np.column_stack([state, inputs])
        parts = [self.theta[s].reshape(shape) for s, shape in zip(self.slices, self.shapes)]
        if self.head == 'saturating':
            w, raw_k, bias = parts
            fraction = expit(raw_k)
            k = .02+1.98*fraction
            denominator = k+x[:, None, :]
            features = x[:, None, :]/denominator
            score = np.einsum('brf,rf->br', features, w)+bias
            cache = (x, w, fraction, k, denominator, features)
        else:
            w1, b1, w2, b2 = parts
            hidden = np.tanh(x@w1+b1)
            score = hidden@w2+b2
            cache = (x, w1, hidden, w2)
        probability = expit(score)
        return .01+2*probability, (probability, cache)

    def rates_vjp(self, gradient, cache):
        probability, inner = cache
        g = gradient*2*probability*(1-probability)
        if self.head == 'saturating':
            x, w, fraction, k, denominator, features = inner
            wf = g[:, :, None]*w
            dx = np.sum(wf*k/denominator**2, axis=1)
            dw = np.einsum('br,brf->rf', g, features)
            dk = np.sum(-wf*x[:, None, :]/denominator**2, axis=0)*1.98*fraction*(1-fraction)
            pieces = [dw, dk, g.sum(axis=0)]
        else:
            x, w1, hidden, w2 = inner
            gh = (g@w2.T)*(1-hidden**2)
            dx = gh@w1.T
            pieces = [x.T@gh, gh.sum(axis=0), hidden.T@g, g.sum(axis=0)]
        parameter_gradient = np.zeros_like(self.theta)
        for sl, piece in zip(self.slices, pieces):
            parameter_gradient[sl] = piece.ravel()
        return dx[:, :7], parameter_gradient

    def step(self, state, inputs, delta):
        rates, rate_cache = self.rates(state, inputs)
        pools, exchange_cache = exchange_step(state[:, :3], rates[:, :4], self.edges, delta, True)
        p, d = rates[:, 4:8], rates[:, 8:12]
        denominator = 1+delta*(p+d)
        bounded = (state[:, 3:]+delta*p)/denominator
        return np.column_stack([pools, bounded]), (exchange_cache, bounded, denominator, rate_cache, delta)

    def step_vjp(self, gradient, cache):
        exchange_cache, bounded, denominator, rate_cache, delta = cache
        gx, gk = exchange_vjp(gradient[:, :3], exchange_cache)
        gh = gradient[:, 3:]
        gp = gh*delta*(1-bounded)/denominator
        gd = -gh*delta*bounded/denominator
        through_rates, parameters = self.rates_vjp(np.column_stack([gk, gp, gd]), rate_cache)
        direct = np.column_stack([gx, gh/denominator])
        return direct+through_rates, parameters

    def rollout(self, initial, inputs, delta=.1, substeps=2, tape=False):
        initial, inputs = np.asarray(initial), np.asarray(inputs)
        if initial.ndim != 2 or initial.shape[1] != 5:
            raise ValueError('Require three measured pools and two regulatory activities')
        if inputs.ndim != 3 or inputs.shape[0] != len(initial) or inputs.shape[2] != 2 or inputs.shape[1] < 1:
            raise ValueError('Require two known inputs at each interval')
        if not np.isfinite(delta) or delta <= 0 or substeps < 1 or int(substeps) != substeps:
            raise ValueError('Invalid integration step')
        if not all(np.isfinite(a).all() and (a >= 0).all() for a in [initial, inputs]):
            raise ValueError('Nonnegative finite measurements required')
        if (initial[:, 3:] > 1).any() or (inputs > 1).any():
            raise ValueError('Regulation and interventions must lie in [0,1]')
        memory = np.broadcast_to(expit(self.theta[-2:]), (len(initial), 2))
        state = np.column_stack([initial, memory])
        predictions, records = [], []
        for t in range(inputs.shape[1]):
            for sub in range(substeps):
                state, cache = self.step(state, inputs[:, t], delta)
                if not np.isfinite(state).all():
                    raise FloatingPointError('Non-finite coupled state')
                if tape:
                    records.append((cache, t if sub == substeps-1 else None))
            predictions.append(state[:, :5].copy())
        prediction = np.stack(predictions, axis=1)
        return (prediction, records) if tape else prediction

    def loss_gradient(self, initial, inputs, target, variance, delta=.1, substeps=2):
        prediction, records = self.rollout(initial, inputs, delta, substeps, True)
        residual = prediction-target
        loss = np.mean(residual**2/variance)
        scored_gradient = 2*residual/variance/residual.size
        adjoint = np.zeros((len(initial), 7))
        gradient = np.zeros_like(self.theta)
        for cache, t in reversed(records):
            if t is not None:
                adjoint[:, :5] += scored_gradient[:, t]
            adjoint, parameter_gradient = self.step_vjp(adjoint, cache)
            gradient += parameter_gradient
        memory = expit(self.theta[-2:])
        gradient[-2:] += adjoint[:, 5:].sum(axis=0)*memory*(1-memory)
        if not np.isfinite(loss) or not np.isfinite(gradient).all():
            raise FloatingPointError('Non-finite loss or gradient')
        return float(loss), gradient
