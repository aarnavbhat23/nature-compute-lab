"""Nonlearned legal-edge selection vs fitting-budget-matched parameter adaptation."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import numpy as np
from drs.typed import TypedDynamics
from drs.resource_systems import ResourceSystem, resource_dataset
from drs.repair_systems import ChangedResourceSystem, changed_dataset
from drs.training import fit, nmse

ROOT = Path(__file__).resolve().parent
EDITS = {'parameter':[], 'ab':[[1,2]], 'ba':[[2,1]], 'both':[[1,2],[2,1]]}


def combine(first, second):
    return {key:np.concatenate([first[key], second[key]], axis=0) for key in first}


def run():
    cfgpath = ROOT/'configs/typed_repair_pilot.json'
    cfg = json.loads(cfgpath.read_text())
    if subprocess.check_output(['git','status','--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Commit protocol and sources first')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = ROOT/'results'/('typed_repair_'+stamp);out.mkdir()
    files = [Path(__file__), cfgpath, ROOT/'drs/typed.py', ROOT/'drs/resource_systems.py',
             ROOT/'drs/repair_systems.py', ROOT/'drs/training.py', ROOT/'components/exchange.py',
             ROOT/'tests/test_typed.py', ROOT/'research/repair-driven-development/TYPED_REPAIR_PROTOCOL.md']
    protocol = {'status':'running','config':cfg,'started_utc':stamp,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        'python':platform.python_version(),'numpy':np.__version__,'machine':platform.platform(),
        'openblas_num_threads':os.environ.get('OPENBLAS_NUM_THREADS')}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    base_records, rows = [], []
    for seed in cfg['seeds']:
        system = ResourceSystem('mixed',seed)
        train = resource_dataset(system,seed*100+1,32,8)
        valid = resource_dataset(system,seed*100+2,8,32)
        retained = resource_dataset(system,seed*100+7,8,32)
        for name,data in [('train',train),('validation',valid),('retained',retained)]:
            np.savez_compressed(out/f'base_{seed}_{name}.npz',**data)
        variance = np.maximum(train['target'].var(axis=(0,1)),1e-4)
        best, base_trials = None, []
        for lr in cfg['initial_learning_rates']:
            base = TypedDynamics('neural',seed+10000)
            details = fit(base,train,valid,variance,lr,cfg['initial_steps'],8,seed+20000)
            details['learning_rate'] = lr;base_trials.append(details)
            if best is None or details['validation_nmse']<best[0]:
                best = (details['validation_nmse'],base.theta.copy())
        base = TypedDynamics('neural',seed+10000);base.theta[:]=best[1]
        np.savez_compressed(out/f'base_{seed}_weights.npz',base=base.theta,variance=variance)
        base_records.append({'seed':seed,'validation_nmse':best[0],'trials':base_trials})
        (out/'initial_fits.json').write_text(json.dumps(base_records,indent=2)+'\n')
        replay = {key:value[:1] for key,value in train.items()}
        retention_valid = {key:value[:1] for key,value in valid.items()}
        for change in cfg['changes']:
            changed = ChangedResourceSystem(seed,change)
            datasets = {}
            for offset,split,count,length in [
                (4,'adaptation_fit',cfg['adaptation_fit_trajectories'],cfg['adaptation_fit_intervals']),
                (5,'adaptation_validation',cfg['adaptation_validation_trajectories'],cfg['adaptation_validation_intervals']),
                (6,'future',cfg['future_trajectories'],cfg['future_intervals'])]:
                data = changed_dataset(changed,seed*100+offset,count,length,cfg['noise_standard_deviation'])
                datasets[split] = data
                np.savez_compressed(out/f'{change}_{seed}_{split}.npz',**data)
            fitting = combine(datasets['adaptation_fit'],replay)
            selection = combine(datasets['adaptation_validation'],retention_valid)
            models = {'frozen':base};details = {}
            weights = {'variance':variance,'frozen':base.theta.copy()};predictions = {}
            shared_parameter_trials = {}
            for name,edges in EDITS.items():
                trials, chosen = [], None
                for lr in cfg['edit_learning_rates']:
                    model = base.add_exchange_edges(edges)
                    record = fit(model,fitting,selection,variance,lr,cfg['adaptation_steps'],4,seed+30000)
                    record['learning_rate'] = lr;trials.append(record)
                    if name == 'parameter':shared_parameter_trials[lr] = (record,model.theta.copy())
                    if chosen is None or record['validation_nmse']<chosen[0]:
                        chosen = (record['validation_nmse'],model.theta.copy(),lr)
                model = base.add_exchange_edges(edges);model.theta[:]=chosen[1]
                models[name] = model
                details[name] = {'validation_nmse':chosen[0],'selected_learning_rate':chosen[2],
                                 'trials':trials,'fitting_seconds':sum(t['seconds'] for t in trials)}
            trials, chosen = [], None
            for lr in cfg['parameter_learning_rates']:
                if lr in shared_parameter_trials:
                    record,theta = shared_parameter_trials[lr]
                else:
                    model = base.add_exchange_edges([])
                    record = fit(model,fitting,selection,variance,lr,cfg['adaptation_steps'],4,seed+30000)
                    record['learning_rate'] = lr;theta = model.theta.copy()
                trials.append(record)
                if chosen is None or record['validation_nmse']<chosen[0]:chosen=(record['validation_nmse'],theta.copy(),lr)
            model = base.add_exchange_edges([]);model.theta[:]=chosen[1]
            models['parameter_budget'] = model
            details['parameter_budget'] = {'validation_nmse':chosen[0],'selected_learning_rate':chosen[2],
                'trials':trials,'fitting_seconds':sum(t['seconds'] for t in trials)}
            details['frozen'] = {'validation_nmse':nmse(selection['target'],base.rollout(selection['initial'],selection['inputs']),variance),
                                 'fitting_seconds':0.,'trials':[]}
            selected = min(['frozen',*EDITS],key=lambda name:details[name]['validation_nmse'])
            parameter_selected = min(['frozen','parameter_budget'],key=lambda name:details[name]['validation_nmse'])
            row = {'seed':seed,'change':change,'selected_edit':selected,'selected_parameter':parameter_selected,
                   'edit_search_seconds':sum(details[name]['fitting_seconds'] for name in EDITS),
                   'parameter_search_seconds':details['parameter_budget']['fitting_seconds'],'models':{}}
            future = datasets['future']
            for name,model in models.items():
                pred = model.rollout(future['initial'],future['inputs'])
                keep = model.rollout(retained['initial'],retained['inputs'])
                mass = float(np.max(np.abs(pred[:,:,:3].sum(axis=2)-future['initial'][:,:3].sum(axis=1)[:,None])))
                if not np.isfinite(pred).all() or not (pred>=0).all() or not (pred[:,:,3:]<=1).all():
                    raise FloatingPointError(name)
                row['models'][name] = {**details[name],'parameters':len(model.theta),
                    'edges':model.edges.tolist(),'future_nmse':nmse(future['target'],pred,variance),
                    'retained_nmse':nmse(retained['target'],keep,variance),'pool_total_error':mass}
                weights[name] = model.theta.copy();predictions[name] = pred;predictions[name+'_retained'] = keep
            rows.append(row)
            np.savez_compressed(out/f'{change}_{seed}_weights.npz',**weights)
            np.savez_compressed(out/f'{change}_{seed}_predictions.npz',**predictions)
            (out/'metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(f'{seed} {change}: validation chose {selected}; parameter control {parameter_selected}',flush=True)
    protocol['status']='complete';protocol['finished_utc']=datetime.now(timezone.utc).isoformat()
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print('Results: '+str(out),flush=True)


if __name__ == '__main__':run()
