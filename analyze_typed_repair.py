"""Reload legal-edit pilot artifacts and report the full negative/positive record."""

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
    assert protocol['status']=='complete'
    for source,digest in protocol['source_sha256'].items():
        content = subprocess.check_output(['git','show',f"{protocol['source_commit']}:{source}"],cwd=ROOT)
        assert hashlib.sha256(content).hexdigest()==digest
    rows = json.loads((path/'metrics.json').read_text())
    base_records = json.loads((path/'initial_fits.json').read_text())
    assert len(rows)==18 and len(base_records)==3
    verified,actual_fits = 0,9
    for base in base_records:
        seed = base['seed']
        weights = dict(np.load(path/f'base_{seed}_weights.npz'))
        valid = dict(np.load(path/f'base_{seed}_validation.npz'))
        train = dict(np.load(path/f'base_{seed}_train.npz'))
        model = TypedDynamics('neural');model.theta[:]=weights['base']
        np.testing.assert_array_equal(weights['variance'],np.maximum(train['target'].var(axis=(0,1)),1e-4))
        score = nmse(valid['target'],model.rollout(valid['initial'],valid['inputs']),weights['variance'])
        np.testing.assert_allclose(score,base['validation_nmse'],rtol=1e-12)
        np.testing.assert_allclose(score,min(t['validation_nmse'] for trial in base['trials'] for t in trial['trace']),rtol=1e-12)
    for row in rows:
        seed,change = row['seed'],row['change'];prefix=f'{change}_{seed}'
        data = {s:dict(np.load(path/f'{prefix}_{s}.npz')) for s in ['adaptation_fit','adaptation_validation','future']}
        old_valid = dict(np.load(path/f'base_{seed}_validation.npz'))
        old_train = dict(np.load(path/f'base_{seed}_train.npz'))
        retained = dict(np.load(path/f'base_{seed}_retained.npz'))
        selection = {key:np.concatenate([data['adaptation_validation'][key],old_valid[key][:1]]) for key in old_valid}
        initial = np.concatenate([d['initial'] for d in [*data.values(),retained,old_train,old_valid]])
        assert len(np.unique(initial,axis=0))==len(initial)
        weights = dict(np.load(path/f'{prefix}_weights.npz'))
        saved = dict(np.load(path/f'{prefix}_predictions.npz'))
        for name,details in row['models'].items():
            model = TypedDynamics('neural',extra_edges=details['edges'][4:]);model.theta[:]=weights[name]
            np.testing.assert_array_equal(model.edges,details['edges'])
            assert len(model.theta)==details['parameters']
            for split,suffix,metric in [(data['future'],'','future_nmse'),(retained,'_retained','retained_nmse')]:
                prediction = model.rollout(split['initial'],split['inputs'])
                np.testing.assert_allclose(prediction,saved[name+suffix],rtol=1e-12,atol=1e-14)
                np.testing.assert_allclose(nmse(split['target'],prediction,weights['variance']),details[metric],rtol=1e-12)
                assert (prediction>=0).all() and (prediction[:,:,3:]<=1).all()
                if suffix=='':
                    mass = np.max(np.abs(prediction[:,:,:3].sum(axis=2)-split['initial'][:,:3].sum(axis=1)[:,None]))
                    np.testing.assert_allclose(mass,details['pool_total_error'],atol=1e-14)
                verified += 1
            score = nmse(selection['target'],model.rollout(selection['initial'],selection['inputs']),weights['variance'])
            np.testing.assert_allclose(score,details['validation_nmse'],rtol=1e-12)
            if details['trials']:
                np.testing.assert_allclose(score,min(t['validation_nmse'] for trial in details['trials'] for t in trial['trace']),rtol=1e-12)
        candidates = ['frozen','parameter','ab','ba','both']
        assert row['selected_edit']==min(candidates,key=lambda n:row['models'][n]['validation_nmse'])
        assert row['selected_parameter']==min(['frozen','parameter_budget'],key=lambda n:row['models'][n]['validation_nmse'])
        for trial in row['models']['parameter']['trials']:
            shared = [t for t in row['models']['parameter_budget']['trials'] if t['learning_rate']==trial['learning_rate']]
            assert shared==[trial]
        actual_fits += sum(len(row['models'][n]['trials']) for n in ['parameter','ab','ba','both','parameter_budget'])-2
    assert actual_fits==261
    def key(row,method):
        return row['selected_edit'] if method=='edit' else row['selected_parameter'] if method=='parameter' else 'both' if method=='full' else 'frozen'
    wins = sum(r['models'][key(r,'edit')]['future_nmse']<r['models'][key(r,'parameter')]['future_nmse'] for r in rows)
    ties = sum(r['models'][key(r,'edit')]['future_nmse']==r['models'][key(r,'parameter')]['future_nmse'] for r in rows)
    breaches = {m:[f"{r['change']}/{r['seed']}" for r in rows if r['models'][key(r,m)]['retained_nmse']>1.05*r['models']['frozen']['retained_nmse']] for m in ['edit','parameter']}
    lines = ['# Legal exchange-edit development pilot', '',
        '**Date:** October 15, 2026, America/Los_Angeles. Raw artifacts use UTC. '
        '**Decision:** do not advance to training a repair selector on the current formulation. '
        'Nonlearned edit search shows no broad advantage over fitting-budget-matched parameter adaptation, and retention fails badly on several changed systems. '
        'Preserve these outcomes and address context/retention semantics before scaling.', '',
        '## Study and information', '',
        'Three new base systems are each tested under six changes: 18 paired change cases, only three independent coefficient draws. '
        'Nine initial fits plus 252 distinct adaptation fits yield **261 actual training trials**. Each main procedure is charged eight 300-update tuning trials per case; '
        'two trials are shared exactly between the procedures. Extra edges increase parameters from 234 to at most 256. '
        'All methods receive the same known closed-pool graph information, 120 new fit scalar targets, 320 new validation scalar targets, '
        '25 new initial-state values, and the declared stored old data. See the [frozen protocol](TYPED_REPAIR_PROTOCOL.md).', '',
        'Selection uses new-condition validation and retained validation in a 2:1 trajectory ratio, permitting a frozen/no-edit choice. '
        'An edit candidate can add A→B, B→A, or both; this deliberately matches some generator changes. '
        'It is nonlearned model selection over four known topologies, not novel mechanism generation, a learned selector, or active experiment choice.', '',
        '## Future forecast error', '',
        'Mean nMSE ± sample SD over three base draws, lower better. SD is not a confidence interval. '
        '“Edit selected” and “Parameter selected” include the frozen option. Full graph receives only two tuning trials, so it is not budget-matched.', '',
        '| Change | Frozen | Edit selected | Parameter selected | Full graph |', '|---|---:|---:|---:|---:|']
    for change in protocol['config']['changes']:
        cells = []
        for method in ['frozen','edit','parameter','full']:
            values = [r['models'][key(r,method)]['future_nmse'] for r in rows if r['change']==change]
            cells.append(f'{np.mean(values):.6g} ± {np.std(values,ddof=1):.3g}')
        lines.append('| '+change+' | '+' | '.join(cells)+' |')
    lines += ['', f'Edit search wins {wins}/18 individual future-error comparisons against the parameter procedure, ties {ties}/18, and loses {18-wins-ties}/18. '
        'Its mean error is slightly lower for A→B additions and no-change cases, but worse for the parameter and B→A changes; the both-edge and noise cases provide no broad edit benefit. '
        'Three seeds and favorable edit priors do not support a superiority claim.', '',
        '## Retention and false changes', '',
        '| Change | Frozen retained error | Edit-selected retained error | Parameter-selected retained error |', '|---|---:|---:|---:|']
    for change in protocol['config']['changes']:
        values = [np.mean([r['models'][key(r,m)]['retained_nmse'] for r in rows if r['change']==change]) for m in ['frozen','edit','parameter']]
        lines.append('| '+change+' | '+' | '.join(f'{v:.6g}' for v in values)+' |')
    false_growth = sum(r['change'] in ['none','noise'] and r['selected_edit'] in ['ab','ba','both'] for r in rows)
    lines += ['', f'Edit selection exceeds 5% retained-error deterioration in **{len(breaches["edit"])}/18** cases; parameter selection does so in **{len(breaches["parameter"])}/18**. '
        'The current objective does not protect old behavior. Breach cases: '+', '.join(breaches['edit'])+'.', '',
        f'No-change/noise false structural growth: {false_growth}/6. All three noise cases select frozen. '
        'Existing-edge parameter changes select extra edges in all three cases, demonstrating that selected graph structure is not a recovered causal mechanism. '
        'All three both-edge cases select frozen under the declared objective. Every choice and its errors remains in the raw record.', '',
        '## Costs and verification', '',
        f'Mean charged search time: edit {np.mean([r["edit_search_seconds"] for r in rows]):.4g} seconds; '
        f'parameter {np.mean([r["parameter_search_seconds"] for r in rows]):.4g} seconds per case. '
        'Model construction, data generation, and initial fitting are outside those adaptation times. Equal trials/updates do not imply equal wall time.', '',
        f'Verified {verified} saved forecast arrays (future and retained), all selected validation scores/minima, graph choices, reused trials, '
        'parameter counts, source hashes, and disjoint within-case initial states. '
        f'Maximum recorded future pool-total discrepancy: {max(d["pool_total_error"] for r in rows for d in r["models"].values()):.3g}. '
        'The constraints are supplied and guaranteed; they do not demonstrate correct biology. Source freeze: `'+protocol['source_commit']+'`.', '',
        f'Regenerate with `OPENBLAS_NUM_THREADS=1 python3 analyze_typed_repair.py {folder}`. '
        f'Raw [metrics](../../{folder}/metrics.json), [initial fitting traces](../../{folder}/initial_fits.json), '
        f'[source/configuration](../../{folder}/protocol.json), datasets, weights, and forecasts are preserved.', '',
        '## A design limitation exposed by this test', '',
        'The same initial state and intervention can have different correct continuations before and after a hidden mechanism change. '
        'The current model initializes abstract memory to the same learned vector for every trajectory; it receives no observed calibration history or regime context at forecast initialization. '
        'A single deterministic transition cannot represent two different derivatives at the same complete input. '
        'Replaying old labels while overwriting one global dynamics model can therefore impose incompatible tasks. '
        'This does not excuse the forecast losses, prove that every loss comes from ambiguity, or establish that edits never help. '
        'It limits what the retention experiment can establish and what a more realistic architecture must specify.', '',
        'The next protocol should distinguish knowledge retention from using one dynamics law in two regimes. '
        'A straightforward baseline stores the old model and conditions forecasts on an explicit experimental-context record. '
        'A harder model infers context from the same actually revealed calibration observations available to all methods. '
        'Neither gets the true mechanism label or future outcomes. Query context, memory, model storage, and inference cost must be counted. '
        'Context-conditioned learning and model banks are existing ideas; this correction is not a new intelligence claim.', '',
        'Before learning a repair policy, test whether structural edits add value against this context-aware parameter/control setup. '
        'Do not retune the present 18 cases or call a corrected protocol a fresh confirmatory result on them. '
        'No ASI, calibrated uncertainty, biological validation, or novel architecture advantage is demonstrated.']
    (ROOT/'research/repair-driven-development/TYPED_REPAIR_RESULTS.md').write_text('\n'.join(lines)+'\n')
    summary = {'actual_training_trials':actual_fits,'forecast_arrays_verified':verified,
               'edit_wins':wins,'edit_ties':ties,'retention_breaches':breaches}
    (path/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':analyze(sys.argv[1])
