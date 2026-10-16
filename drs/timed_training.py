"""Elapsed-time adaptation of the pilot optimizer; separate from fixed-step studies."""

import time
import numpy as np
from drs.training import nmse


def fit_timed(model,train,valid,variance,learning_rate,seconds,batch_size,seed):
    if seconds<=0:raise ValueError('Positive budget required')
    rng=np.random.default_rng(seed)
    m,v=np.zeros_like(model.theta),np.zeros_like(model.theta)
    best,weights,trace=float('inf'),model.theta.copy(),[]
    start=time.perf_counter();step=0;previous_horizon=2
    while time.perf_counter()-start<seconds or step==0:
        elapsed=time.perf_counter()-start
        horizon=2 if elapsed<seconds/3 else (4 if elapsed<2*seconds/3 else 8)
        previous_horizon=horizon;step+=1
        batch=rng.choice(len(train['initial']),batch_size,replace=False)
        loss,g=model.loss_gradient(train['initial'][batch],train['inputs'][batch,:horizon],train['target'][batch,:horizon],variance)
        norm=np.linalg.norm(g)
        if norm>5:g*=5/norm
        m=.9*m+.1*g;v=.999*v+.001*g*g
        model.theta-=learning_rate*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
        if step%20==0:
            error=nmse(valid['target'],model.rollout(valid['initial'],valid['inputs']),variance)
            trace.append({'step':step,'horizon':horizon,'seconds':time.perf_counter()-start,'validation_nmse':error})
            if error<best:best,weights=error,model.theta.copy()
    if not trace or trace[-1]['step']!=step:
        error=nmse(valid['target'],model.rollout(valid['initial'],valid['inputs']),variance)
        trace.append({'step':step,'horizon':previous_horizon,'seconds':time.perf_counter()-start,'validation_nmse':error})
        if error<best:best,weights=error,model.theta.copy()
    model.theta[:]=weights
    return {'validation_nmse':best,'optimizer_steps':step,'seconds':time.perf_counter()-start,'trace':trace}
