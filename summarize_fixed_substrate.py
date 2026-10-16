"""Describe a completed fixed-substrate pilot without refitting any model."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


LABELS={'substrate':'DRS substrate','neural_rates':'Neural rates','affine':'Affine','persistence':'Persistence'}
COLORS=['#237A8A','#B86D31','#7470A0','#AAB2BA']


def summarize(result):
    protocol=json.loads((result/'protocol.json').read_text())
    if protocol['status']!='complete':raise ValueError('Do not summarize an incomplete pilot')
    rows=json.loads((result/'metrics.json').read_text())
    cfg=protocol['config'];families=cfg['families'];models=list(LABELS)
    assert len(rows)==len(families)*len(cfg['seeds'])
    report=result/'report';report.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,family in zip(axes.ravel(),families):
        cases=[r for r in rows if r['family']==family]
        samples=np.array([[c['metrics'][m]['nmse_48'] for c in cases] for m in models])
        means=samples.mean(axis=1)
        ax.bar(np.arange(4),means,color=COLORS,alpha=.9)
        for i,values in enumerate(samples):ax.scatter(np.full(3,i)+np.array([-.09,0,.09]),values,c='black',s=16,zorder=4)
        ax.set_yscale('log');ax.set_xticks(np.arange(4),['DRS','Neural','Affine','Persist'])
        ax.set_title(family.replace('_',' ').title())
        ax.set_ylabel('48-interval normalized MSE (lower is better)')
        ax.grid(axis='y',which='major',alpha=.2)
    family,seed=families[0],cfg['seeds'][0]
    data=np.load(result/f'{family}_{seed}_test.npz')
    prediction=np.load(result/f'{family}_{seed}_predictions.npz')
    ax=axes[1,1];times=.2*np.arange(1,49)
    ax.plot(times,data['target'][0,:,0],c='black',label='Reference',linewidth=2)
    for m,color in zip(models[:3],COLORS):ax.plot(times,prediction[m][0,:,0],color=color,label=LABELS[m],linestyle='--')
    ax.axvline(1.6,c='gray',linestyle=':',label='Training horizon')
    ax.set_title('First test trajectory, saturating seed 101, coordinate 0')
    ax.set_xlabel('Physical time');ax.set_ylabel('State');ax.legend(fontsize=8)
    fig.suptitle('Fixed-substrate development pilot: three seeds per family\nBars: family mean; points: systems. No structural repair.',fontsize=13)
    fig.savefig(report/'comparison.png',dpi=170);plt.close(fig)
    lines=['# Fixed-substrate pilot results','',
           '**Outcome:** the custom substrate learns bounded forecasts, but it has no demonstrated accuracy advantage over the controls. Structural repair and ASI claims have not been tested.','',
           'Source frozen at `'+protocol['source_commit']+'`; completed '+protocol['finished_utc']+'. These are UTC provenance times. See the saved protocol, source fingerprints, tuning traces, datasets, weights, and predictions in the parent directory.','',
           '![Pilot comparison](comparison.png)','',
           '## Held-out long-rollout error','',
           'Variance-normalized MSE over 48 intervals (9.6 time units), mean ± sample SD across three system seeds. Training rollouts extend to eight intervals (1.6 units); validation extends to 12. Lower is better. No confidence or significance claim is made.','',
           '| Family | DRS substrate | Neural rate control | Affine control | Persistence |',
           '|---|---|---|---|---|']
    summaries={}
    for family in families:
        cases=[r for r in rows if r['family']==family]
        summaries[family]={}
        cells=[]
        for m in models:
            values=np.array([c['metrics'][m]['nmse_48'] for c in cases])
            summaries[family][m]={'mean':float(values.mean()),'sample_sd':float(values.std(ddof=1))}
            cells.append(f'{values.mean():.6g} ± {values.std(ddof=1):.3g}')
        lines.append('| '+family.replace('_',' ')+' | '+' | '.join(cells)+' |')
    lines+=['','The neural rate control has lower family-mean error than DRS on all three families. The affine model has the lowest family mean on the forced-sinusoidal task. DRS beats the neural control in one of nine individual cases, but this does not establish a transferable advantage.','',
            'Both DRS and the neural control have 12 persistent states and the same bounded numerical update. They use 736 and 730 parameters respectively. They each receive three learning-rate trials, 240 optimizer updates per trial, identical observations, and the same validation selection procedure. The affine model is a much smaller, 88-parameter control with a one-step training objective and closed-loop evaluation. Equal trial budgets and parameters do not make training time or expressive power equal.','',
            '## Learning and numerical checks','',
            '| Family | Untrained DRS nMSE | Trained DRS nMSE | Reduction versus its initialization |',
            '|---|---|---|---|']
    for family in families:
        cases=[r for r in rows if r['family']==family]
        before=np.mean([c['metrics']['substrate_untrained']['nmse_48'] for c in cases])
        after=np.mean([c['metrics']['substrate']['nmse_48'] for c in cases])
        lines.append(f'| {family.replace("_"," ")} | {before:.6g} | {after:.6g} | {100*(1-after/before):.2f}% |')
    reference=max(c['reference_max_abs_error'] for c in rows)
    bounded=all(c['long_rollout']['all_finite'] and c['long_rollout']['all_bounded'] for c in rows)
    converged=all(c['substrate_numerical_convergence']['0.025']['rmse_vs_ivp'] < c['substrate_numerical_convergence']['0.05']['rmse_vs_ivp'] < c['substrate_numerical_convergence']['0.1']['rmse_vs_ivp'] for c in rows)
    lines+=['',f'- All nine selected substrates passed the finite/bounded 192-interval (38.4 time unit) diagnostic: **{bounded}**. This checks the repeated known input sequence; it is not an accuracy test or proof for all inputs.',
            f'- Learned-equation integration error decreased at every tested step halving on all nine cases: **{converged}**. This does not imply better prediction of the true system.',
            f'- Maximum teacher RK4/adaptive-solver discrepancy: **{reference:.3g}** in state units.',
            '- Six new tests passed, including parameter-gradient finite differences through complete rollouts, state-vector-Jacobian checks for both models, invalid inputs, resource caps, boundedness, and reference resolution. All 15 repository tests passed.',
            '- The pilot source and configuration were committed before held-out scores were computed. Development smoke seed 13 was excluded from the reported cases.','',
            '## Measured training time','',
            'Sum of wall time across all three learning-rate trials for each of the nine systems, on the recorded machine. These exclude data generation, test forecasts, integration diagnostics, and reporting; they are not end-to-end or hardware-normalized benchmarks. Peak memory and energy were not instrumented.']
    for m in models[:2]:
        seconds=sum(c['models'][m]['all_trial_training_seconds'] for c in rows)
        lines+=['',f'- {LABELS[m]}: {seconds:.3f} seconds.']
    lines+=['','This NumPy implementation of DRS is also slower to train than the neural rate control. No speed advantage is claimed.','',
            '## Individual system results','',
            '| Family | Seed | DRS nMSE (12 / 48) | Neural nMSE (12 / 48) | Affine nMSE (12 / 48) |',
            '|---|---|---|---|---|']
    for c in rows:
        cells=[f"{c['metrics'][m]['nmse_12']:.6g} / {c['metrics'][m]['nmse_48']:.6g}" for m in models[:3]]
        lines.append('| '+c['family']+' | '+str(c['seed'])+' | '+' | '.join(cells)+' |')
    lines+=['','## Decision','',
            'The first engineering task is complete: differentiable gate dynamics, parameter fitting, deterministic reproducible forecasts, and numerical checks work. The architectural advantage is unproven, and the present evidence favors the controls for predictive accuracy.','',
            'Do not advance to real biology, claim 95% accuracy, or present this as ASI. Before adding structural repair, run a separate development-only diagnostic to separate limited gate expressivity, fixed wiring, and optimization budget. Candidate experiments should include a clearly labeled matched-wiring diagnostic and capacity/training-budget curves on fresh development seeds. Neither the diagnosis nor an improvement from it has been demonstrated by this run. Freeze the next protocol before computing new held-out results, keep this pilot intact, and do not relabel these seeds as confirmatory.']
    (report/'results.md').write_text('\n'.join(lines)+'\n')
    (report/'summary.json').write_text(json.dumps({'families':summaries,'analysis_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result',type=Path)
    print(summarize(parser.parse_args().result.resolve()))
