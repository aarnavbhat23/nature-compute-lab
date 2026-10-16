"""Paired checkpoint-selection diagnostic: short vs deployment-length validation."""

import hashlib,json,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from drs.systems import System,dataset
from drs.training import nmse
from run_nature_diagnostic import make_model

ROOT=Path(__file__).resolve().parent


def trial(model,train,valid,variance,lr,steps,seed):
    rng=np.random.default_rng(seed);m=np.zeros_like(model.theta);v=m.copy()
    best={h:(float('inf'),None,None) for h in [12,48]};trace=[]
    start=time.perf_counter()
    for step in range(1,steps+1):
        horizon=2 if step<=steps//3 else (4 if step<=2*steps//3 else 8)
        batch=rng.choice(32,8,replace=False)
        _,g=model.loss_gradient(train['initial'][batch],train['inputs'][batch,:horizon],train['target'][batch,:horizon],variance)
        norm=np.linalg.norm(g)
        if norm>5:g*=5/norm
        m=.9*m+.1*g;v=.999*v+.001*g*g
        model.theta-=lr*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
        if step%20==0:
            pred=model.rollout(valid['initial'],valid['inputs'])
            errors={h:nmse(valid['target'][:,:h],pred[:,:h],variance) for h in [12,48]}
            trace.append({'step':step,'validation_nmse':errors})
            for h,error in errors.items():
                if error<best[h][0]:best[h]=(error,model.theta.copy(),step)
    return best,{'learning_rate':lr,'seconds':time.perf_counter()-start,'trace':trace}


def run():
    cfgfile=ROOT/'configs/nature_horizon.json';cfg=json.loads(cfgfile.read_text())
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise RuntimeError('Commit source first')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out=ROOT/'results'/('nature_horizon_'+stamp);out.mkdir()
    files=[Path(__file__),ROOT/'run_nature_diagnostic.py',*sorted((ROOT/'drs').glob('*.py')),cfgfile,ROOT/'research/repair-driven-development/HORIZON_PROTOCOL.md']
    p={'status':'running','config':cfg,'started_utc':stamp,'numpy':np.__version__,
       'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
    (out/'protocol.json').write_text(json.dumps(p,indent=2)+'\n');rows=[]
    for family in cfg['families']:
        for seed in cfg['seeds']:
            system=System(family,seed=seed)
            train=dataset(system,seed*100+1,32,8);valid=dataset(system,seed*100+2,8,48);held=dataset(system,seed*100+3,16,48)
            variance=np.maximum(train['target'].var(axis=(0,1)),1e-4)
            for name,data in [('train',train),('validation',valid),('heldout',held)]:np.savez_compressed(out/f'{family}_{seed}_{name}.npz',**data)
            row={'family':family,'seed':seed,'models':{}};weights={'variance':variance};predictions={}
            for name in cfg['models']:
                chosen={h:None for h in [12,48]};trials=[]
                for lr in cfg['learning_rates']:
                    model=make_model(name,seed+10000)
                    best,details=trial(model,train,valid,variance,lr,cfg['optimizer_steps'],seed+20000);trials.append(details)
                    for h in chosen:
                        if chosen[h] is None or best[h][0]<chosen[h][0]:chosen[h]=(*best[h],lr)
                result={'parameters':len(model.theta),'trials':trials,'selectors':{}}
                for h,selection in chosen.items():
                    error,theta,step,lr=selection;model=make_model(name,seed+10000);model.theta[:]=theta
                    pred=model.rollout(held['initial'],held['inputs'])
                    result['selectors'][str(h)]={'validation_nmse':error,'checkpoint':step,'learning_rate':lr,
                        'heldout_nmse_48':nmse(held['target'],pred,variance)}
                    key=f'{name}_val{h}';weights[key]=theta;predictions[key]=pred
                row['models'][name]=result
                print(f'{family} {seed} {name}: validation selection complete',flush=True)
            rows.append(row);np.savez_compressed(out/f'{family}_{seed}_weights.npz',**weights)
            np.savez_compressed(out/f'{family}_{seed}_predictions.npz',**predictions)
            (out/'metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
    p['status']='complete';p['finished_utc']=datetime.now(timezone.utc).isoformat()
    (out/'protocol.json').write_text(json.dumps(p,indent=2)+'\n');print('Results: '+str(out),flush=True)


if __name__=='__main__':run()
