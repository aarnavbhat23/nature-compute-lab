"""Bounded state dynamics and exact reverse-mode derivatives, using NumPy only."""

import numpy as np
from scipy.special import expit, logit


class RateModel:
    """Shared integrator; subclasses learn nonnegative production/decay rates."""

    def initial_state(self, observed):
        hidden = expit(self.theta[-self.hidden:])
        return np.column_stack([observed, np.broadcast_to(hidden, (len(observed), self.hidden))])

    def rollout(self, initial, inputs, delta=.1, substeps=2, tape=False):
        initial, inputs = np.asarray(initial), np.asarray(inputs)
        if initial.ndim != 2 or initial.shape[1] != self.q:
            raise ValueError('Initial state must have shape (batch, observed coordinates)')
        if inputs.ndim != 3 or inputs.shape[0] != len(initial) or inputs.shape[2] != 2:
            raise ValueError('Inputs must have shape (batch, observation intervals, 2)')
        if delta <= 0 or substeps < 1 or int(substeps) != substeps:
            raise ValueError('Require positive delta and integer substeps')
        if not all(np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all() for a in [initial, inputs]):
            raise ValueError('States and inputs must be finite and in [0, 1]')
        h = self.initial_state(initial)
        predictions, records = [], []
        for t in range(inputs.shape[1]):
            for sub in range(substeps):
                p, d, cache = self.rates(h, inputs[:, t])
                denominator = 1 + delta*(p+d)
                h = (h+delta*p)/denominator
                if not np.isfinite(h).all():
                    raise FloatingPointError('Non-finite rollout; do not hide by clipping')
                if tape:
                    records.append((h, denominator, cache, t if sub == substeps-1 else None))
            predictions.append(h[:, :self.q].copy())
        predictions = np.stack(predictions, axis=1)
        return (predictions, records) if tape else predictions

    def loss_gradient(self, initial, inputs, target, variance, delta=.1, substeps=2):
        prediction, records = self.rollout(initial, inputs, delta, substeps, tape=True)
        residual = prediction-target
        loss = np.mean(residual**2/variance)
        output_gradient = 2*residual/variance/residual.size
        adjoint = np.zeros((len(initial), self.n))
        gradient = np.zeros_like(self.theta)
        for h, denominator, cache, scored_time in reversed(records):
            if scored_time is not None:
                adjoint[:, :self.q] += output_gradient[:, scored_time]
            gp = adjoint*delta*(1-h)/denominator
            gd = -adjoint*delta*h/denominator
            dh, dt = self.rate_vjp(gp, gd, cache)
            gradient += dt
            adjoint = adjoint/denominator + dh
        hidden = expit(self.theta[-self.hidden:])
        gradient[-self.hidden:] += adjoint[:, self.q:].sum(axis=0)*hidden*(1-hidden)
        if not np.isfinite(loss) or not np.isfinite(gradient).all():
            raise FloatingPointError('Non-finite objective or gradient')
        return float(loss), gradient


class Substrate(RateModel):
    """Fixed gate wiring; k, K, inhibition mixing, and hidden starts are learned."""

    def __init__(self, q=8, hidden=4, seed=0):
        self.q, self.hidden, self.n = q, hidden, q+hidden
        rng = np.random.default_rng(seed)
        layout = []
        for dest in range(self.n):
            layout.extend([(dest, 0, []), (dest, 1, [])])
            # All observed-to-compartment one-source couplings: no truth graph supplied.
            for source in range(q):
                layout.extend([(dest, 0, [source]), (dest, 1, [source])])
            layout.extend([(dest, 0, [self.n]), (dest, 1, [self.n+1])])
            # A hidden compartment can modulate one external input at each destination.
            layout.append((dest, dest % 2, [q+dest % hidden, self.n+dest % 2]))
        if self.n > 64 or len(layout) > 256:
            raise ValueError('Version-zero resource cap exceeded; this fixed pilot layout supports q=8')
        self.dest = np.array([r[0] for r in layout])
        self.decay = np.array([r[1] for r in layout], dtype=bool)
        self.sources = np.zeros((len(layout), 2), dtype=int)
        self.mask = np.zeros_like(self.sources, dtype=bool)
        for i, (_, _, sources) in enumerate(layout):
            self.sources[i, :len(sources)] = sources
            self.mask[i, :len(sources)] = True
        self.gates = len(layout)
        self.k_slice = slice(0, self.gates)
        self.k_indices = np.arange(self.gates)
        offset = self.gates
        self.K_indices = np.arange(offset, offset+self.mask.sum())
        offset += self.mask.sum()
        self.rho_indices = np.arange(offset, offset+self.mask.sum())
        offset += self.mask.sum()
        self.theta = np.zeros(offset+hidden)
        self.theta[self.k_slice] = logit(np.where(self.mask.any(axis=1), .03, .08)/2)
        self.theta[self.K_indices] = logit((.3-.02)/1.98)
        self.theta[self.rho_indices] = rng.normal(0, .5, self.mask.sum())
        self.theta[-hidden:] = rng.normal(0, .2, hidden)

    def rates(self, h, u):
        x = np.column_stack([h, u])
        sk = expit(self.theta[self.k_slice])
        sK, sr = expit(self.theta[self.K_indices]), expit(self.theta[self.rho_indices])
        k = 2*sk
        K, rho = np.ones_like(self.sources, dtype=float), np.zeros_like(self.sources, dtype=float)
        K[self.mask], rho[self.mask] = .02+1.98*sK, sr
        z = x[:, self.sources]
        factor = ((1-rho)*z+rho*K)/(K+z)
        factor[:, ~self.mask] = 1
        gate = k*factor.prod(axis=2)
        p, d = np.full_like(h, .01), np.full_like(h, .01)
        for i in range(self.n):
            p[:, i] += gate[:, (self.dest == i) & ~self.decay].sum(axis=1)
            d[:, i] += gate[:, (self.dest == i) & self.decay].sum(axis=1)
        return p, d, (z, K, rho, factor, sk, sK, sr)

    def rate_vjp(self, gp, gd, cache):
        z, K, rho, factor, sk, sK, sr = cache
        gate_gradient = np.where(self.decay, gd[:, self.dest], gp[:, self.dest])
        gradient = np.zeros_like(self.theta)
        gradient[self.k_slice] = (gate_gradient*factor.prod(axis=2)).sum(axis=0)*2*sk*(1-sk)
        factor_gradient = gate_gradient[:, :, None]*(2*sk)[None, :, None]*factor[:, :, ::-1]
        factor_gradient *= self.mask
        denom = (K+z)**2
        dz = factor_gradient*(1-2*rho)*K/denom
        dK = (factor_gradient*(2*rho-1)*z/denom).sum(axis=0)
        drho = (factor_gradient*(K-z)/(K+z)).sum(axis=0)
        gradient[self.K_indices] = dK[self.mask]*1.98*sK*(1-sK)
        gradient[self.rho_indices] = drho[self.mask]*sr*(1-sr)
        dh = np.zeros((len(gp), self.n))
        for slot in range(2):
            for source in range(self.n):
                dh[:, source] += dz[:, self.sources[:, slot] == source, slot].sum(axis=1)
        return dh, gradient


class NeuralRates(RateModel):
    """Capacity-matched fixed neural rate control with the same persistent state/integrator."""

    def __init__(self, q=8, hidden=4, seed=0, budget=736):
        self.q, self.hidden, self.n = q, hidden, q+hidden
        self.width = max(1, round((budget-2*self.n-hidden)/(3*self.n+3)))
        shapes = [(self.n+2, self.width), (self.width,), (self.width, 2*self.n), (2*self.n,)]
        self.shapes, self.slices = shapes, []
        offset = 0
        for shape in shapes:
            size = int(np.prod(shape)); self.slices.append(slice(offset, offset+size)); offset += size
        self.theta = np.zeros(offset+hidden)
        rng = np.random.default_rng(seed)
        self.theta[self.slices[0]] = rng.normal(0, .2, (self.n+2)*self.width)
        self.theta[self.slices[2]] = rng.normal(0, .1, self.width*2*self.n)
        self.theta[self.slices[3]] = logit(.3/2)
        self.theta[-hidden:] = rng.normal(0, .2, hidden)

    def unpack(self):
        return [self.theta[s].reshape(shape) for s, shape in zip(self.slices, self.shapes)]

    def rates(self, h, u):
        w1, b1, w2, b2 = self.unpack()
        x = np.column_stack([h, u])
        a = np.tanh(x@w1+b1)
        s = expit(a@w2+b2)
        rates = .01+2*s
        return rates[:, :self.n], rates[:, self.n:], (x, a, s, w1, w2)

    def rate_vjp(self, gp, gd, cache):
        x, a, s, w1, w2 = cache
        g = np.column_stack([gp, gd])*2*s*(1-s)
        ga = (g@w2.T)*(1-a*a)
        pieces = [x.T@ga, ga.sum(axis=0), a.T@g, g.sum(axis=0)]
        gradient = np.zeros_like(self.theta)
        for sl, piece in zip(self.slices, pieces):gradient[sl] = piece.ravel()
        return (ga@w1.T)[:, :self.n], gradient
