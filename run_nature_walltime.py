"""Implementation-specific equal-wall-time follow-up on fresh development systems."""

import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import platform
import subprocess

import numpy as np
from drs.systems import System,dataset
from drs.training import nmse,affine_fit,affine_rollout
from drs.timed_training import fit_timed
from run_nature_diagnostic import make_model

ROOT=Path(__file__).resolve().parent


def run():
    cfg_path=ROOT/'configs/nature_walltime.json';cfg=json.loads(cfg_path.read_text())
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise RuntimeError('Commit source first')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=ROOT/'results'/('nature_walltime_'+stamp);out.mkdir()
    files=[Path(__file__),ROOT/'run_nature_diagnostic.py',*sorted((ROOT/'drs').glob('*.py')),cfg_path,ROOT/'research/repair-driven-development/WALLTIME_PROTOCOL.md']
    p={'status':'running','config':cfg,'started_utc':stamp,'machine':platform.platform(),'python':platform.python_version(),'numpy':np.__version__,
       'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
    (out/'protocol.json').write_text(json.dumps(p,indent=2)+'\n');rows=[]
    for family in cfg['families']:
        for seed in cfg['seeds']:
            system=System(family,seed=seed)
            train=dataset(system,seed*100+1,32,8);valid=dataset(system,seed*100+2,8,12);held=dataset(system,seed*100+3,16,48)
            variance=np.maximum(train['target'].var(axis=(0,1)),1e-4)
            for name,data in [('train',train),('validation',valid),('heldout',held)]:np.savez_compressed(out/f'{family}_{seed}_{name}.npz',**data)
            weights={'variance':variance};predictions={};row={'family':family,'seed':seed,'models':{}}
            affine,details=affine_fit(train,valid,variance);ap=affine_rollout(affine,held['initial'],held['inputs'])
            row['affine']={'heldout_nmse_48':nmse(held['target'],ap,variance),**details}
            weights['affine']=affine;predictions['affine']=ap
            for name in cfg['models']:
                trials=[];best=None
                for lr in cfg['learning_rates']:
                    model=make_model(name,seed+10000)
                    trial=fit_timed(model,train,valid,variance,lr,cfg['seconds_per_trial'],cfg['batch_size'],seed+20000)
                    trial['learning_rate']=lr;trials.append(trial)
                    if best is None or trial['validation_nmse']<best[0]:best=(trial['validation_nmse'],lr,model.theta.copy())
                model=make_model(name,seed+10000);model.theta[:]=best[2]
                pred=model.rollout(held['initial'],held['inputs'])
                if not np.isfinite(pred).all():raise FloatingPointError(name)
                row['models'][name]={'parameters':len(model.theta),'selected_learning_rate':best[1],
                    'validation_nmse':best[0],'heldout_nmse_48':nmse(held['target'],pred,variance),
                    'seconds_all_trials':sum(t['seconds'] for t in trials),'trials':trials}
                weights[name]=model.theta.copy();predictions[name]=pred
                print(f'{family} {seed} {name}: selected validation nMSE={best[0]:.6g}',flush=True)
            rows.append(row);np.savez_compressed(out/f'{family}_{seed}_weights.npz',**weights)
            np.savez_compressed(out/f'{family}_{seed}_predictions.npz',**predictions)
            (out/'metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
    p['status']='complete';p['finished_utc']=datetime.now(timezone.utc).isoformat()
    (out/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
    print('Results: '+str(out),flush=True)


if __name__=='__main__':run()
