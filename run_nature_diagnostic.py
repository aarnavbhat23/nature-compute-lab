"""Fresh development comparison of initialization, training budget, and state competition."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import scipy

from drs.models import Substrate,NeuralRates
from drs.occupancy import OccupancyRates,PolarizedSubstrate
from drs.systems import System,dataset
from drs.training import fit,nmse,affine_fit,affine_rollout

ROOT=Path(__file__).resolve().parent


def make_model(name,seed):
    constructors={'original':Substrate,'polarized':PolarizedSubstrate,'occupancy':OccupancyRates}
    if name in constructors:return constructors[name](seed=seed)
    if name in ['neural_730','neural_808']:
        return NeuralRates(seed=seed,budget=int(name.split('_')[1]))
    raise ValueError(name)


def run():
    config=ROOT/'configs/nature_dynamics_diagnostic.json';cfg=json.loads(config.read_text())
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise RuntimeError('Commit this diagnostic before evaluation')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=ROOT/'results'/('nature_diagnostic_'+stamp);out.mkdir()
    files=[Path(__file__),*sorted((ROOT/'drs').glob('*.py')),config,ROOT/'research/repair-driven-development/NATURE_DIAGNOSTIC.md']
    provenance={'status':'running','started_utc':stamp,'config':cfg,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'machine':platform.platform()}
    (out/'protocol.json').write_text(json.dumps(provenance,indent=2)+'\n')
    cases=[]
    for family in cfg['families']:
        for seed in cfg['seeds']:
            start=time.perf_counter();system=System(family,q=8,seed=seed)
            train=dataset(system,seed*100+1,cfg['train_trajectories'],cfg['train_intervals'])
            valid=dataset(system,seed*100+2,cfg['validation_trajectories'],cfg['validation_intervals'])
            heldout=dataset(system,seed*100+3,cfg['heldout_trajectories'],cfg['heldout_intervals'])
            for name,data in [('train',train),('validation',valid),('heldout',heldout)]:
                np.savez_compressed(out/f'{family}_{seed}_{name}.npz',**data)
            variance=np.maximum(train['target'].var(axis=(0,1)),cfg['variance_floor'])
            affine,details=affine_fit(train,valid,variance)
            affine_pred=affine_rollout(affine,heldout['initial'],heldout['inputs'])
            case={'family':family,'seed':seed,'affine':{'nmse_48':nmse(heldout['target'],affine_pred,variance),**details},'models':{}}
            weights={'variance':variance,'affine':affine};predictions={'affine':affine_pred}
            for steps in cfg['optimizer_budgets']:
                conditions={}
                for name in cfg['models']:
                    trials=[];best=None
                    for lr in cfg['learning_rates']:
                        model=make_model(name,seed+10000)
                        trial=fit(model,train,valid,variance,lr,steps,cfg['batch_size'],seed+20000)
                        trial['learning_rate']=lr;trials.append(trial)
                        if best is None or trial['validation_nmse']<best[0]:best=(trial['validation_nmse'],lr,model.theta.copy())
                    model=make_model(name,seed+10000);model.theta[:]=best[2]
                    prediction=model.rollout(heldout['initial'],heldout['inputs'])
                    if not np.isfinite(prediction).all():raise FloatingPointError(name)
                    conditions[name]={'parameters':len(model.theta),'selected_learning_rate':best[1],
                        'validation_nmse':best[0],'heldout_nmse_12':nmse(heldout['target'][:,:12],prediction[:,:12],variance),
                        'heldout_nmse_48':nmse(heldout['target'],prediction,variance),
                        'training_seconds':sum(t['seconds'] for t in trials),'trials':trials,
                        'all_bounded':bool(((prediction>=0)&(prediction<=1)).all())}
                    key=f'{name}_{steps}';weights[key]=model.theta.copy();predictions[key]=prediction
                    print(f'{family} {seed} {steps} {name}: selected validation nMSE={best[0]:.6g}',flush=True)
                case['models'][str(steps)]=conditions
            case['seconds']=time.perf_counter()-start;cases.append(case)
            np.savez_compressed(out/f'{family}_{seed}_weights.npz',**weights)
            np.savez_compressed(out/f'{family}_{seed}_predictions.npz',**predictions)
            (out/'metrics.json').write_text(json.dumps(cases,indent=2)+'\n')
            print(f'Finished {family} {seed}',flush=True)
    provenance['status']='complete';provenance['finished_utc']=datetime.now(timezone.utc).isoformat()
    (out/'protocol.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Results: '+str(out),flush=True)


if __name__=='__main__':run()
