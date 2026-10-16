"""Run the frozen fixed-substrate development pilot; no structural repair."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import scipy
from scipy.integrate import solve_ivp

from drs.models import Substrate, NeuralRates
from drs.systems import System, dataset
from drs.training import fit, nmse, affine_fit, affine_rollout

ROOT=Path(__file__).resolve().parent


def source_provenance(config):
    files=[Path(__file__),*sorted((ROOT/'drs').glob('*.py')),config]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def ivp_reference(system, initial, inputs, interval=.2):
    """Different solver for teacher verification; one trajectory is enough for this check."""
    h=initial[:1].copy();out=[]
    for u in inputs[:1].transpose(1,0,2):
        sol=solve_ivp(lambda _,x:system.derivative(x[None],u)[0],(0,interval),h[0],rtol=1e-10,atol=1e-12)
        if not sol.success:raise RuntimeError(sol.message)
        h=sol.y[:,-1][None];out.append(h.copy())
    return np.stack(out,axis=1)


def model_reference(model, initial, inputs, interval=.2):
    h=model.initial_state(initial);out=[]
    for t in range(inputs.shape[1]):
        def rhs(_, state):
            state=state.reshape(h.shape)
            p,d,_=model.rates(state,inputs[:,t])
            return ((1-state)*p-state*d).ravel()
        sol=solve_ivp(rhs,(0,interval),h.ravel(),rtol=1e-9,atol=1e-11)
        if not sol.success:raise RuntimeError(sol.message)
        h=sol.y[:,-1].reshape(h.shape);out.append(h[:,:model.q].copy())
    return np.stack(out,axis=1)


def run(config_path):
    config_path=config_path.resolve();cfg=json.loads(config_path.read_text())
    if cfg['q']!=8 or cfg['hidden']!=4 or cfg['observation_interval']!=.2 or cfg['model_microstep']!=.1:
        raise ValueError('This first implementation freezes q=8, four hidden states, and the declared integrator')
    if cfg['reference_microstep']!=.01 or cfg['measurement_noise']!=0 or cfg['rate_penalty']!=0 or cfg['size_penalty']!=0:
        raise ValueError('Noise/penalties/reference-step extensions require a new implementation protocol')
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())
    if dirty:raise RuntimeError('Commit source and protocol before this pilot; do not run unrecorded code')
    timestamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=ROOT/'results'/('drs_fixed_'+timestamp);out.mkdir(parents=True)
    metadata={'status':'running','started_utc':timestamp,'source_commit':sha,'source_dirty':dirty,
              'source_sha256':source_provenance(config_path),'config':cfg,'python':platform.python_version(),
              'numpy':np.__version__,'scipy':scipy.__version__,'machine':platform.platform(),
              'notes':['Development pilot, not confirmatory.','No structural edits, repair, selector, or real biology.',
                       'CPU timings; no measured peak-memory instrumentation.','Fully observed, noiseless synthetic systems.']}
    (out/'protocol.json').write_text(json.dumps(metadata,indent=2)+'\n')
    cases=[]
    for family in cfg['families']:
        for seed in cfg['seeds']:
            started=time.perf_counter();case={'family':family,'seed':seed,'metrics':{},'models':{}}
            system=System(family,cfg['q'],seed)
            split={name:dataset(system,seed*100+offset,count,length) for name,offset,count,length in [
                ('train',1,cfg['train_trajectories'],cfg['train_intervals']),
                ('validation',2,cfg['validation_trajectories'],cfg['validation_intervals']),
                ('test',3,cfg['test_trajectories'],cfg['test_intervals'])]}
            train,valid,test=(split[n] for n in ['train','validation','test'])
            for name,data in split.items():
                np.savez_compressed(out/f'{family}_{seed}_{name}.npz',**data)
            variance=np.maximum(train['target'].var(axis=(0,1)),cfg['variance_floor'])
            reference=ivp_reference(system,test['initial'],test['inputs'][:,:12])
            reference_error=float(np.max(np.abs(reference-test['target'][:1,:12])))
            if reference_error>1e-7:raise RuntimeError('Teacher integration verification failed')
            case['reference_max_abs_error']=reference_error
            predictions={};weights={}
            for name,constructor in [('substrate',Substrate),('neural_rates',NeuralRates)]:
                trials=[];selected=None;total_time=0
                for lr in cfg['learning_rates']:
                    model=constructor(q=cfg['q'],hidden=cfg['hidden'],seed=seed+10000)
                    trial=fit(model,train,valid,variance,lr,cfg['optimizer_steps'],cfg['batch_size'],seed+20000)
                    trial['learning_rate']=lr;trials.append(trial);total_time+=trial['seconds']
                    if selected is None or trial['validation_nmse']<selected[0]:
                        selected=(trial['validation_nmse'],lr,model.theta.copy())
                    print(f'{family} seed={seed} {name} lr={lr}: validation nMSE {trial["validation_nmse"]:.5g}',flush=True)
                model=constructor(q=cfg['q'],hidden=cfg['hidden'],seed=seed+10000)
                model.theta[:]=selected[2]
                predictions[name+'_untrained']=model.__class__(q=cfg['q'],hidden=cfg['hidden'],seed=seed+10000).rollout(test['initial'],test['inputs'])
                inference_start=time.perf_counter()
                predictions[name]=model.rollout(test['initial'],test['inputs'])
                inference_seconds=time.perf_counter()-inference_start
                weights[name]=model.theta.copy()
                case['models'][name]={'parameters':len(model.theta),'hidden_states':model.hidden,
                    'selected_learning_rate':selected[1],'selected_validation_nmse':selected[0],
                    'all_trial_training_seconds':total_time,'test_rollout_seconds':inference_seconds,
                    'trials':trials}
                if name=='substrate':
                    case['models'][name]['gates']=model.gates
                    initial=test['initial'][:2];inputs=test['inputs'][:2,:12]
                    accurate=model_reference(model,initial,inputs)
                    convergence={}
                    for substeps in [2,4,8]:
                        delta=.2/substeps
                        pred=model.rollout(initial,inputs,delta=delta,substeps=substeps)
                        convergence[str(delta)]={'rmse_vs_ivp':float(np.sqrt(np.mean((pred-accurate)**2)))}
                    case['substrate_numerical_convergence']=convergence
                    long=model.rollout(initial,np.tile(test['inputs'][:2],(1,4,1)))
                    case['long_rollout']={'observation_intervals':long.shape[1],
                        'physical_duration':float(.2*long.shape[1]),'min':float(long.min()),'max':float(long.max()),
                        'all_finite':bool(np.isfinite(long).all()),'all_bounded':bool(((long>=0)&(long<=1)).all())}
            affine,affine_details=affine_fit(train,valid,variance)
            weights['affine']=affine
            predictions['affine']=affine_rollout(affine,test['initial'],test['inputs'])
            case['models']['affine']={**affine_details,'parameters':int(affine.size),
                'teacher_forced_training':True,'closed_loop_evaluation':True}
            predictions['persistence']=np.broadcast_to(test['initial'][:,None],test['target'].shape).copy()
            for name,pred in predictions.items():
                if not np.isfinite(pred).all():raise FloatingPointError('Non-finite held-out prediction: '+name)
                case['metrics'][name]={f'nmse_{h}':nmse(test['target'][:,:h],pred[:,:h],variance) for h in cfg['score_horizons']}
                case['metrics'][name]['mse_full']=float(np.mean((test['target']-pred)**2))
                case['metrics'][name]['bounded_fraction']=float(((pred>=0)&(pred<=1)).mean())
            np.savez_compressed(out/f'{family}_{seed}_fitted.npz',**weights,variance=variance)
            np.savez_compressed(out/f'{family}_{seed}_predictions.npz',**predictions)
            case['elapsed_seconds']=time.perf_counter()-started;cases.append(case)
            (out/'metrics.json').write_text(json.dumps(cases,indent=2)+'\n')
            print(f'Completed {family} seed={seed}: '+json.dumps(case['metrics']),flush=True)
    metadata['status']='complete';metadata['finished_utc']=datetime.now(timezone.utc).isoformat()
    (out/'protocol.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Results: '+str(out),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/fixed_substrate_pilot.json')
    run(parser.parse_args().config)
