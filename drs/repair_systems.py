"""Small post-change generators for legal exchange-edge repair diagnostics."""

import numpy as np
from drs.resource_systems import ResourceSystem, resource_dataset


class ChangedResourceSystem(ResourceSystem):
    def __init__(self, seed, change):
        super().__init__('mixed', seed)
        if change not in ['none', 'parameter', 'edge_ab', 'edge_ba', 'both', 'noise']:
            raise ValueError(change)
        self.change = change

    def derivative(self, state, inputs):
        result = super().derivative(state, inputs)
        if self.change == 'parameter':
            flow = .6*state[:, 1]
            result[:, 1] -= flow;result[:, 0] += flow
        if self.change in ['edge_ab', 'both']:
            flow = .6*state[:, 1]*(.2+.8*state[:, 4])
            result[:, 1] -= flow;result[:, 2] += flow
        if self.change in ['edge_ba', 'both']:
            flow = .6*state[:, 2]*(.2+.8*state[:, 3])
            result[:, 2] -= flow;result[:, 1] += flow
        return result


def changed_dataset(system, seed, count, length, noise_sd=.01):
    data = resource_dataset(system, seed, count, length)
    if system.change == 'noise':
        # Observation noise only. Initial measurements stay exact in this pilot.
        data['target'] += np.random.default_rng(seed+7723).normal(0, noise_sd, data['target'].shape)
    return data
