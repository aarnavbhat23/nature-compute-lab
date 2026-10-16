"""Implicit exchange step: positivity and conserved total for a closed pool.

This is a standard backward-Euler transport construction, not a new numerical
method. Rates are frozen for the step; this is not a general reaction integrator.
"""

import numpy as np


def exchange_step(state,rates,edges,delta,return_cache=False):
    state=np.asarray(state,dtype=float);rates=np.asarray(rates,dtype=float)
    edges=np.asarray(edges)
    if state.ndim!=2 or rates.ndim!=2 or state.shape[0]!=rates.shape[0]:
        raise ValueError('Require batched states and rates with the same batch size')
    if edges.shape!=(rates.shape[1],2) or not np.issubdtype(edges.dtype,np.integer):
        raise ValueError('Each edge must have integer source and destination indices')
    if not np.isfinite(delta) or delta<=0:
        raise ValueError('Require a finite positive time step')
    if not all(np.isfinite(x).all() and (x>=0).all() for x in [state,rates]):
        raise ValueError('States and rates must be finite and nonnegative')
    n=state.shape[1]
    if ((edges<0)|(edges>=n)).any():raise ValueError('Edge outside the state space')
    matrix=np.broadcast_to(np.eye(n),(len(state),n,n)).copy()
    for e,(source,dest) in enumerate(edges):
        matrix[:,source,source]+=delta*rates[:,e]
        matrix[:,dest,source]-=delta*rates[:,e]
    next_state=np.linalg.solve(matrix,state[:,:,None])[:,:,0]
    if not np.isfinite(next_state).all():raise FloatingPointError('Non-finite exchange solution')
    return (next_state,(matrix,next_state,edges,delta)) if return_cache else next_state


def exchange_vjp(output_gradient,cache):
    matrix,next_state,edges,delta=cache
    adjoint=np.linalg.solve(matrix.transpose(0,2,1),np.asarray(output_gradient)[:,:,None])[:,:,0]
    rate_gradient=np.stack([delta*next_state[:,source]*(adjoint[:,dest]-adjoint[:,source])
                            for source,dest in edges],axis=1)
    return adjoint,rate_gradient
