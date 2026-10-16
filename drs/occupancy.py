"""Competing-state rate model: finite local occupancy, learned state affinities."""

import numpy as np
from scipy.special import expit, logit, softmax

from drs.models import RateModel, Substrate


class PolarizedSubstrate(Substrate):
    """Initialization diagnostic only: break the half-inhibition cancellation."""

    def __init__(self, q=8, hidden=4, seed=0):
        super().__init__(q,hidden,seed)
        rng=np.random.default_rng(seed+731)
        self.theta[self.rho_indices]=rng.choice([-2.5,2.5],len(self.rho_indices))


class OccupancyRates(RateModel):
    """Alternative local states compete with one inactive reference state.

    Real-valued affinity exponents are phenomenological, not identified binding
    stoichiometries or measured free energies. No global mass conservation follows.
    """

    def __init__(self, q=8, hidden=4, seed=0, states=4):
        self.q,self.hidden,self.n=q,hidden,q+hidden
        self.states=states
        self.shapes=[(self.n,states,self.n+2),(self.n,states),(self.n,states,2)]
        self.slices=[];offset=0
        for shape in self.shapes:
            size=int(np.prod(shape));self.slices.append(slice(offset,offset+size));offset+=size
        self.theta=np.zeros(offset+hidden)
        rng=np.random.default_rng(seed)
        self.theta[self.slices[0]]=rng.normal(0,.12,int(np.prod(self.shapes[0])))
        self.theta[self.slices[1]]=rng.normal(0,.1,int(np.prod(self.shapes[1])))
        self.theta[self.slices[2]]=logit(.45/2)+rng.normal(0,.12,int(np.prod(self.shapes[2])))
        self.theta[-hidden:]=rng.normal(0,.2,hidden)

    def unpack(self):
        return [self.theta[sl].reshape(shape) for sl,shape in zip(self.slices,self.shapes)]

    def rates(self,h,u):
        w,b,raw_capacity=self.unpack()
        x=np.column_stack([h,u]);logx=np.log(x+.05)
        scores=np.einsum('bs,nms->bnm',logx,w)+b
        probabilities=softmax(np.concatenate([np.zeros((*scores.shape[:2],1)),scores],axis=2),axis=2)
        occupied=probabilities[:,:,1:]
        fraction=expit(raw_capacity);capacity=2*fraction
        rates=.01+np.einsum('bnm,nmr->bnr',occupied,capacity)
        return rates[:,:,0],rates[:,:,1],(x,logx,occupied,capacity,fraction,w)

    def rate_vjp(self,gp,gd,cache):
        x,logx,pi,capacity,fraction,w=cache
        output_gradient=np.stack([gp,gd],axis=2)
        gc=np.einsum('bnm,bnr->nmr',pi,output_gradient)*2*fraction*(1-fraction)
        gpi=np.einsum('bnr,nmr->bnm',output_gradient,capacity)
        gs=pi*(gpi-np.sum(pi*gpi,axis=2,keepdims=True))
        gw=np.einsum('bnm,bs->nms',gs,logx)
        gb=gs.sum(axis=0)
        dx=np.einsum('bnm,nms->bs',gs,w)/(x+.05)
        gradient=np.zeros_like(self.theta)
        for sl,piece in zip(self.slices,[gw,gb,gc]):gradient[sl]=piece.ravel()
        return dx[:,:self.n],gradient
