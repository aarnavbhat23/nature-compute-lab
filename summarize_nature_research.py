"""Verify saved development artifacts and regenerate the research tables/figure."""

import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from drs.training import affine_rollout, nmse
from run_nature_diagnostic import make_model

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'research/repair-driven-development'
RUNS = {
    'updates': 'nature_diagnostic_20261016T015441102245Z',
    'walltime': 'nature_walltime_20261016T020053486333Z',
    'horizon': 'nature_horizon_20261016T020716259962Z',
}
FAMILIES = ['saturating', 'polynomial', 'forced_sinusoidal']
verified = []
tables = []
all_rows = {}
fits = 0


def check_close(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)


for study, folder in RUNS.items():
    path = ROOT/'results'/folder
    protocol = json.loads((path/'protocol.json').read_text())
    assert protocol['status'] == 'complete'
    for source, digest in protocol['source_sha256'].items():
        original = subprocess.check_output(['git', 'show', f"{protocol['source_commit']}:{source}"], cwd=ROOT)
        assert hashlib.sha256(original).hexdigest() == digest, source
    rows = json.loads((path/'metrics.json').read_text())
    assert len(rows) == 9
    all_rows[study] = rows
    forecasts = 0
    for row in rows:
        stem = f"{row['family']}_{row['seed']}"
        data = {s: dict(np.load(path/f'{stem}_{s}.npz')) for s in ['train', 'validation', 'heldout']}
        initial = np.concatenate([data[s]['initial'] for s in data])
        assert len(np.unique(initial, axis=0)) == len(initial)
        weights = dict(np.load(path/f'{stem}_weights.npz'))
        predictions = dict(np.load(path/f'{stem}_predictions.npz'))
        variance = np.maximum(data['train']['target'].var(axis=(0, 1)), 1e-4)
        check_close(weights['variance'], variance)
        for key, prediction in predictions.items():
            if key == 'affine':
                result = affine_rollout(weights[key], data['heldout']['initial'], data['heldout']['inputs'])
                metric = row['affine'].get('nmse_48', row['affine'].get('heldout_nmse_48'))
            else:
                if study == 'updates':
                    name, steps = key.rsplit('_', 1)
                    details = row['models'][steps][name]
                    selected_validation = details['validation_nmse']
                elif study == 'horizon':
                    name, horizon = key.rsplit('_val', 1)
                    details = row['models'][name]['selectors'][horizon]
                    selected_validation = details['validation_nmse']
                else:
                    name = key
                    details = row['models'][name]
                    selected_validation = details['validation_nmse']
                model = make_model(name, row['seed']+10000)
                model.theta[:] = weights[key]
                result = model.rollout(data['heldout']['initial'], data['heldout']['inputs'])
                valid_prediction = model.rollout(data['validation']['initial'], data['validation']['inputs'])
                h = int(horizon) if study == 'horizon' else valid_prediction.shape[1]
                check_close(selected_validation, nmse(data['validation']['target'][:, :h], valid_prediction[:, :h], variance))
                trials = row['models'][name]['trials'] if study == 'horizon' else details['trials']
                candidates = [t['validation_nmse'][horizon] if study == 'horizon' else t['validation_nmse']
                              for trial in trials for t in trial['trace']]
                check_close(selected_validation, min(candidates))
                metric = details['heldout_nmse_48']
            check_close(result, prediction)
            check_close(metric, nmse(data['heldout']['target'], result, variance))
            forecasts += 1
    if study == 'updates':
        fits += 9*5*2*3
        conditions = [(str(s), n) for s in [240, 960] for n in protocol['config']['models']]
        get = lambda row, c: row['models'][c[0]][c[1]]['heldout_nmse_48']
        label = lambda c: f'{c[1]} / {c[0]} updates'
    elif study == 'walltime':
        fits += 9*4*3
        conditions = protocol['config']['models']
        get = lambda row, c: row['models'][c]['heldout_nmse_48']
        label = lambda c: c
    else:
        fits += 9*4*3
        conditions = [(n, str(h)) for n in protocol['config']['models'] for h in [12, 48]]
        get = lambda row, c: row['models'][c[0]]['selectors'][c[1]]['heldout_nmse_48']
        label = lambda c: f'{c[0]} / validation {c[1]}'
    tables += [f'### {study}', '', '| Model / condition | Saturating | Polynomial | Forced sinusoidal |', '|---|---:|---:|---:|']
    for condition in conditions:
        cells = []
        for family in FAMILIES:
            values = [get(row, condition) for row in rows if row['family'] == family]
            cells.append(f'{np.mean(values):.6g} ± {np.std(values, ddof=1):.3g}')
        tables.append('| '+label(condition)+' | '+' | '.join(cells)+' |')
    tables.append('')
    verified.append({'study':study, 'path':folder, 'source_commit':protocol['source_commit'],
                     'cases':len(rows), 'forecasts_reloaded':forecasts,
                     'checks':'source hashes, disjoint initial states, variance, saved forecasts, metrics, validation selection'})

assert fits == 486
improved = sum(m['selectors']['48']['heldout_nmse_48'] < m['selectors']['12']['heldout_nmse_48']
               for row in all_rows['horizon'] for m in row['models'].values())
(OUT/'RESULT_TABLES.md').write_text('# Development result tables\n\n'
    'Held-out 48-interval normalized mean squared error; lower is better. Each cell is the mean ± sample standard deviation across three generated systems within that family, not a confidence interval. '
    'These familiar families were used during development; do not pool them into a universal ranking.\n\n'
    +'\n'.join(tables)+f'\nLong validation selection improved {improved}/36 individual model/system outcomes, and all 12 model/family means.\n')
(OUT/'artifact_verification.json').write_text(json.dumps({'training_trials':fits,
    'paired_horizon_individual_improvements':improved,'studies':verified},indent=2)+'\n')

fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), sharey=False)
colors = ['#176b65', '#bb6b22', '#4976ae', '#81509a']
for ax, family in zip(axes, FAMILIES):
    for name, color in zip(['original','occupancy','neural_730','neural_808'], colors):
        rows = [r for r in all_rows['horizon'] if r['family']==family]
        points = np.array([[r['models'][name]['selectors'][str(h)]['heldout_nmse_48'] for h in [12,48]] for r in rows])
        ax.plot([12,48],points.mean(axis=0), marker='o', color=color, label=name)
    ax.set(yscale='log',xticks=[12,48], xlabel='Validation intervals', title=family.replace('_',' '))
    ax.grid(alpha=.2)
axes[0].set_ylabel('Mean held-out 48-interval nMSE (log scale)')
axes[-1].legend(fontsize=8)
fig.suptitle('Matching validation to forecast duration helps all family means')
fig.text(.5,.01,'Paired checkpoint selection on identical fitted candidates; 3 systems per family. Longer validation uses 4× the labels.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.06,1,.93])
fig.savefig(OUT/'validation_horizon.png', dpi=180)
print(json.dumps({'fits':fits,'forecasts_verified':sum(v['forecasts_reloaded'] for v in verified),'paired_improvements':improved}))
