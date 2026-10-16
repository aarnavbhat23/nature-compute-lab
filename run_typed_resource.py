"""Freeze source, fit two equally constrained heads, preserve every pilot case."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import numpy as np
import scipy

from drs.typed import TypedDynamics
from drs.resource_systems import ResourceSystem, resource_dataset
from drs.training import fit, nmse

ROOT = Path(__file__).resolve().parent


def run():
    cfgpath = ROOT/'configs/typed_resource_pilot.json'
    cfg = json.loads(cfgpath.read_text())
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Commit source and protocol before evaluation')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = ROOT/'results'/('typed_resource_'+stamp)
    out.mkdir()
    sources = [Path(__file__), cfgpath, ROOT/'drs/typed.py', ROOT/'drs/resource_systems.py',
               ROOT/'drs/training.py', ROOT/'components/exchange.py', ROOT/'tests/test_typed.py',
               ROOT/'research/repair-driven-development/TYPED_RESOURCE_PROTOCOL.md']
    protocol = {'status':'running', 'started_utc':stamp, 'config':cfg,
        'source_commit':subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'python':platform.python_version(), 'numpy':np.__version__, 'scipy':scipy.__version__,
        'openblas_num_threads':os.environ.get('OPENBLAS_NUM_THREADS'), 'machine':platform.platform()}
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    rows = []
    for mechanism in cfg['mechanisms']:
        for seed in cfg['seeds']:
            system = ResourceSystem(mechanism, seed)
            datasets = {}
            for i, split in enumerate(['train', 'validation', 'heldout'], start=1):
                data = resource_dataset(system, seed*100+i, cfg[split+'_trajectories'], cfg[split+'_intervals'])
                datasets[split] = data
                np.savez_compressed(out/f'{mechanism}_{seed}_{split}.npz', **data)
            train, valid, held = (datasets[s] for s in ['train', 'validation', 'heldout'])
            variance = np.maximum(train['target'].var(axis=(0, 1)), cfg['variance_floor'])
            weights, forecasts = {'variance':variance}, {}
            row = {'mechanism':mechanism, 'seed':seed, 'heads':{}}
            for head in cfg['heads']:
                trials, best = [], None
                initialized = TypedDynamics(head, seed+10000)
                initial_error = nmse(held['target'], initialized.rollout(held['initial'], held['inputs']), variance)
                for lr in cfg['learning_rates']:
                    model = TypedDynamics(head, seed+10000)
                    details = fit(model, train, valid, variance, lr, cfg['optimizer_steps'], cfg['batch_size'], seed+20000)
                    details['learning_rate'] = lr
                    trials.append(details)
                    if best is None or details['validation_nmse'] < best[0]:
                        best = (details['validation_nmse'], lr, model.theta.copy())
                model = TypedDynamics(head, seed+10000)
                model.theta[:] = best[2]
                prediction = model.rollout(held['initial'], held['inputs'])
                repeats = int(np.ceil(cfg['long_rollout_intervals']/held['inputs'].shape[1]))
                long_inputs = np.tile(held['inputs'], (1, repeats, 1))[:, :cfg['long_rollout_intervals']]
                long = model.rollout(held['initial'], long_inputs)
                mass_error = float(np.max(np.abs(long[:, :, :3].sum(axis=2)-held['initial'][:, :3].sum(axis=1)[:, None])))
                row['heads'][head] = {'parameters':len(model.theta), 'validation_nmse':best[0],
                    'selected_learning_rate':best[1], 'heldout_nmse':nmse(held['target'], prediction, variance),
                    'heldout_pool_nmse':nmse(held['target'][:, :, :3], prediction[:, :, :3], variance[:3]),
                    'heldout_regulatory_nmse':nmse(held['target'][:, :, 3:], prediction[:, :, 3:], variance[3:]),
                    'initialized_heldout_nmse':initial_error, 'training_seconds':sum(t['seconds'] for t in trials),
                    'long_pool_total_error':mass_error, 'long_all_nonnegative':bool((long >= 0).all()),
                    'long_regulation_bounded':bool((long[:, :, 3:] <= 1).all()), 'trials':trials}
                weights[head] = model.theta.copy();forecasts[head] = prediction
                print(f'{mechanism} {seed} {head}: validation={best[0]:.6g}', flush=True)
            rows.append(row)
            np.savez_compressed(out/f'{mechanism}_{seed}_weights.npz', **weights)
            np.savez_compressed(out/f'{mechanism}_{seed}_predictions.npz', **forecasts)
            (out/'metrics.json').write_text(json.dumps(rows, indent=2)+'\n')
    protocol['status'] = 'complete'
    protocol['finished_utc'] = datetime.now(timezone.utc).isoformat()
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    print('Results: '+str(out), flush=True)


if __name__ == '__main__':run()
