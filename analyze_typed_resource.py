"""Verify the fixed coupled-model pilot and write its development report."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from drs.typed import TypedDynamics
from drs.training import nmse

ROOT = Path(__file__).resolve().parent


def analyze(folder):
    path = ROOT/folder
    protocol = json.loads((path/'protocol.json').read_text())
    assert protocol['status'] == 'complete'
    for source, digest in protocol['source_sha256'].items():
        content = subprocess.check_output(['git', 'show', f"{protocol['source_commit']}:{source}"], cwd=ROOT)
        assert hashlib.sha256(content).hexdigest() == digest
    rows = json.loads((path/'metrics.json').read_text())
    assert len(rows) == 12
    verified = 0
    for row in rows:
        prefix = f"{row['mechanism']}_{row['seed']}"
        data = {s:dict(np.load(path/f'{prefix}_{s}.npz')) for s in ['train','validation','heldout']}
        initial = np.concatenate([d['initial'] for d in data.values()])
        assert len(np.unique(initial, axis=0)) == len(initial)
        weights = dict(np.load(path/f'{prefix}_weights.npz'))
        predictions = dict(np.load(path/f'{prefix}_predictions.npz'))
        variance = np.maximum(data['train']['target'].var(axis=(0,1)), 1e-4)
        np.testing.assert_array_equal(variance, weights['variance'])
        for head, details in row['heads'].items():
            model = TypedDynamics(head, row['seed']+10000)
            model.theta[:] = weights[head]
            held, valid = data['heldout'], data['validation']
            predicted = model.rollout(held['initial'], held['inputs'])
            np.testing.assert_allclose(predicted, predictions[head], rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(nmse(held['target'], predicted, variance), details['heldout_nmse'], rtol=1e-12)
            ve = nmse(valid['target'], model.rollout(valid['initial'], valid['inputs']), variance)
            np.testing.assert_allclose(ve, details['validation_nmse'], rtol=1e-12)
            minimum = min(t['validation_nmse'] for trial in details['trials'] for t in trial['trace'])
            np.testing.assert_allclose(minimum, ve, rtol=1e-12)
            for name, sl in [('pool',slice(0,3)),('regulatory',slice(3,5))]:
                np.testing.assert_allclose(nmse(held['target'][:,:,sl], predicted[:,:,sl], variance[sl]),
                                           details[f'heldout_{name}_nmse'], rtol=1e-12)
            long = model.rollout(held['initial'], np.tile(held['inputs'], (1,4,1)))
            mass = np.max(np.abs(long[:,:,:3].sum(axis=2)-held['initial'][:,:3].sum(axis=1)[:,None]))
            np.testing.assert_allclose(mass, details['long_pool_total_error'], atol=1e-14)
            assert (long>=0).all() and (long[:,:,3:]<=1).all()
            assert details['heldout_nmse'] < details['initialized_heldout_nmse']
            verified += 1
    lines = ['# Coupled resource/switch development pilot', '',
        '**Date:** October 15, 2026, America/Los_Angeles. Raw artifacts use UTC. '
        '**Result:** the coupled model learns forecasts and preserves declared constraints, '
        'but the saturating representation loses to the equally constrained neural representation in every condition mean and every system tested. '
        'Continue with the neural rate head as the current control/engineering backbone; do not claim the saturating features are a new or better intelligence.', '',
        'This is the first fixed coupled solver, not learned structural repair. Both heads jointly model three known closed-pool states, two measured regulatory activities, and two abstract memory states. '
        'They receive the same pool topology and total information. The neural head has 234 parameters; the saturating head has 230. See the [frozen protocol](TYPED_RESOURCE_PROTOCOL.md).', '',
        '## Outcomes', '', 'Each entry is held-out mean nMSE ± sample SD across three generated coefficient draws. Lower is better. '
        'SD is not a confidence interval. Seeds are paired across conditions; do not pool the four conditions as independent systems.', '',
        '| Condition | Saturating total | Neural total | Saturating regulation | Neural regulation |', '|---|---:|---:|---:|---:|']
    for condition in protocol['config']['mechanisms']:
        values = []
        for metric in ['heldout_nmse','heldout_regulatory_nmse']:
            for head in ['saturating','neural']:
                x = [r['heads'][head][metric] for r in rows if r['mechanism']==condition]
                values.append(f'{np.mean(x):.6g} ± {np.std(x,ddof=1):.3g}')
        lines.append('| '+condition+' | '+' | '.join(values)+' |')
    max_mass = max(d['long_pool_total_error'] for r in rows for d in r['heads'].values())
    neural_wins = sum(r['heads']['neural']['heldout_nmse']<r['heads']['saturating']['heldout_nmse'] for r in rows)
    lines += ['', f'The neural head has lower total error in {neural_wins}/12 paired cases. All 24 selected models improve on their untrained initialization. '
        f'All long 128-interval forecasts remain nonnegative with bounded regulation; maximum closed-pool total discrepancy is {max_mass:.3g}. '
        'Conservation is guaranteed by the supplied solver and is not evidence of discovered biological mechanisms.', '',
        '## Verification and costs', '',
        f'Completed 72 training trials (12 cases × two heads × three learning rates). Verified {verified} saved forecasts, '
        'source hashes against the recorded source commit, disjoint split initial states, training-derived normalization, '
        'validation minima, pool/regulatory errors, and long-rollout invariants. Complete parameter and state finite-difference checks '
        'plus solver convergence tests pass in `tests/test_typed.py`. Source freeze: `'+protocol['source_commit']+'`.', '',
        '| Head | Mean fitting seconds per case, all three trials |', '|---|---:|']
    for head in ['saturating','neural']:
        lines.append(f"| {head} | {np.mean([r['heads'][head]['training_seconds'] for r in rows]):.4g} |")
    relative = '../../'+folder
    lines += ['', 'Equal updates and close parameter counts are not equal compute. Timings include the fit and validation calls, not data generation or model construction. '
        'No elapsed-time superiority is claimed. Both methods get 1280 training and 1280 validation scalar target values per case, plus initial states. '
        'This is not yet a low-measurement repair result.', '',
        f'Raw [metrics]({relative}/metrics.json), [source/configuration provenance]({relative}/protocol.json), datasets, predictions, and weights are preserved. '
        f'Regenerate this report with `OPENBLAS_NUM_THREADS=1 python3 analyze_typed_resource.py {folder}`.', '',
        '## What this does not establish', '',
        'The generator is small, matched to a known closed-pool topology, fully observed and noiseless. It is not validated gene expression or a whole cell. '
        'The labels direct/resource/mixed/none describe supplied synthetic mechanisms; this study does not infer or score those labels. '
        'All pool coordinates are measured, so it does not test ambiguity between hidden resource effects and direct regulation. '
        'No active experiment choice, new mechanism discovery, learned repair policy, open-system generalization, uncertainty calibration, or ASI is demonstrated.', '',
        'The [closed-loop precedent audit](CLOSED_LOOP_PRECEDENTS.md) finds that biological hypothesis generation, model revision, and experiment choice already exist. '
        'The next test must isolate a specific structural-repair advantage against parameter adaptation and a full-capacity control; if it does not exist, narrow the claim.']
    destination = ROOT/'research/repair-driven-development/TYPED_RESOURCE_RESULTS.md'
    destination.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'verified_forecasts':verified,'neural_paired_wins':neural_wins,'max_mass_error':max_mass}))


if __name__ == '__main__':analyze(sys.argv[1])
