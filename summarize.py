"""Build a descriptive report from the newest run of each distinct experiment."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PANELS = [
    ('01_development', 'Development: long cascades', ['grown_program_mse', 'shared_linear_mse', 'fixed_full_dictionary_mse'], ['Grown', 'Linear', 'Full dictionary']),
    ('02_catalysts', 'Catalysts: held-out ingredients', ['catalyst_mse', 'linear_mse', 'full_chemistry_mse'], ['Selected rules', 'Linear', 'Full library']),
    ('03_symbiosis', 'Symbiosis: joint extremes', ['partnership_mse', 'independent_modules_mse', 'fixed_interactions_mse'], ['Partnerships', 'Independent', 'Full interactions']),
    ('04_repair', 'Repair: changed interaction', ['sparse_repair_mse', 'frozen_mse', 'full_refit_mse', 'regularized_full_patch_mse'], ['Sparse patch', 'Frozen', 'Full refit', 'Full patch']),
    ('05_immune_memory', 'Immune memory: changing regimes', ['repertoire_mse', 'cumulative_mse', 'forgetting_mse'], ['Repertoire', 'Cumulative', 'Forgetting']),
    ('06_stable_regimes', 'Stable regimes: closed-loop switches', ['hysteresis_error_rate', 'memoryless_error_rate', 'autoregressive_error_rate', 'transition_table_error_rate'], ['Hysteresis', 'Memoryless', 'AR rule', 'State table']),
    ('07_clocks', 'Clocks: trajectory error', ['asynchronous_mse', 'fitted_dense_mse', 'uniform_clock_mse'], ['Async', 'Dense', 'Uniform clock']),
]

VERDICTS = [
    'Transfers across depth, but no reliable advantage over the fixed nonlinear dictionary.',
    'Sparse rule selection helps on a supplied, correctly specified chemistry; rule recovery is imperfect.',
    'Two useful joins are retained on a composition-friendly synthetic task; stronger tuning remains needed.',
    'Sparse repair helps when the generating change is sparse and already in the dictionary.',
    'Negative against forgetting regression, both overall and immediately after returns.',
    'The matched hysteresis prior works; a generic state transition table is already competitive.',
    'Fewer updates than dense evaluation, but no error advantage over a similar-budget uniform clock.',
]


def collect(root, preview_clock=None):
    records = []
    for stem, *_ in PANELS:
        candidates = sorted((root/'results').glob(stem+'_*/metrics.json'))
        if stem == '07_clocks' and preview_clock:
            path = Path(preview_clock)
        elif candidates:
            path = candidates[-1]
        else:
            raise FileNotFoundError(f'No results for {stem}')
        record = json.loads(path.read_text())
        if record['experiment'] != stem or [r['seed'] for r in record['rows']] != [7, 17, 29, 43, 71]:
            raise ValueError(f'Unexpected experiment or seed set in {path}')
        # Repeated verifications are not pooled into extra independent seeds.
        records.append((path, record))
    return records


def build(root, output, preview_clock=None):
    output.mkdir(parents=True, exist_ok=True)
    records = collect(root, preview_clock)
    fig, axes = plt.subplots(4, 2, figsize=(12, 14), constrained_layout=True)
    for ax, panel, (_, record) in zip(axes.flat, PANELS, records):
        stem, title, keys, labels = panel
        values = [record['summary'][k]['mean'] for k in keys]
        colors = ['#237b83'] + ['#a6b3bd']*(len(keys)-1)
        ax.scatter(np.arange(len(keys)), values, color=colors, marker='_', s=950, linewidths=4, zorder=2)
        # Individual seed values expose variation without implying confidence intervals.
        for i, key in enumerate(keys):
            ax.scatter(np.full(5, i)+np.linspace(-.12, .12, 5), [r['metrics'][key] for r in record['rows']], color='#27323b', s=14, zorder=3)
        ax.set_xticks(np.arange(len(keys)), labels, rotation=15)
        ax.set_yscale('log')
        ax.set_ylim(min(r['metrics'][k] for r in record['rows'] for k in keys)*.6,
                    max(r['metrics'][k] for r in record['rows'] for k in keys)*1.8)
        ax.grid(axis='y', alpha=.15)
        ax.set_ylabel('Error rate' if stem == '06_stable_regimes' else 'MSE')
        ax.set_title(title, loc='left', fontsize=11)
        ax.spines[['top', 'right']].set_visible(False)
        for i, val in enumerate(values):
            ax.text(i, val*1.15, f'{val:.2g}', ha='center', fontsize=8)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(0, .9, 'Read each panel separately.\nDifferent tasks and units.\nDots: five seeds, not confidence intervals.\n\nSynthetic mechanism probes.\nNo biological or ASI validation.\nSeveral controls are equally good or better.', va='top', fontsize=12)
    fig.suptitle('Seven ideas, seven controlled probes', fontsize=19)
    fig.savefig(output/'comparison.png', dpi=180)
    plt.close(fig)
    text = ['# Seven nature-inspired mechanism probes', '',
            f'Generated {datetime.now(timezone.utc).isoformat()}.', '',
            '**Status:** '+('Local preview; experiments 06 and 07 await scheduled repository checkpoints.' if preview_clock else 'Initial staged study complete.'), '',
            'Evaluate claims against the code and saved measurements.', '',
            'These are seven small algorithmic abstractions, not seven finished general machine learning architectures. All data are synthetic. No novelty, biology, or ASI claim is validated by these results.', '',
            'Each task has its own scale and controls. The table is descriptive, not a cross-task leaderboard. Repeated verification runs are not additional independent seeds.', '',
            '![Within-task results](comparison.png)', '',
            '| Probe | Prototype error, mean ± sample SD | Strong same-task comparison | Interpretation |',
            '|---|---|---|---|']
    for panel, (_, record), verdict in zip(PANELS, records, VERDICTS):
        stem, title, keys, labels = panel
        primary = record['summary'][keys[0]]
        best_key = min(keys[1:], key=lambda k: record['summary'][k]['mean'])
        best = record['summary'][best_key]
        label = labels[keys.index(best_key)]
        text.append(f"| {title} | {primary['mean']:.5g} ± {primary['std']:.3g} | {label}: {best['mean']:.5g} ± {best['std']:.3g} | {verdict} |")
    clocks = records[-1][1]['summary']
    text += ['', '## What deserves follow-up', '',
             'Repair and compact composition are the clearest positive mechanism demonstrations in this first pass, but both benefit from supplied operation families. They need fresh task families and tuned controls before guiding an architecture decision.', '',
             'The immune repertoire and asynchronous allocation both fail against meaningful controls. Those results are retained rather than tuned away after evaluation.', '',
             f"For clocks, mean per-seed dense/asynchronous update ratio is {clocks['dense_to_async_update_ratio']['mean']:.3g}. This is an event-count ratio, not a measured wall-time speedup; the similar-budget uniform clock is more accurate on average.", '',
             '## Limits', '',
             '- Several generators exactly match the supplied hypothesis class.',
             '- Five seeds are an exploratory check, not a powered significance study.',
             '- Parameter counts and tuning budgets are not fully equalized.',
             '- Sparse algebraic programs, threshold fitting, and simple online rules are intentionally small proxies for the brainstormed ideas.',
             '- Real molecular data, learned primitives, and broader transfer remain untested.', '',
             '## Result provenance', '']
    for path, record in records:
        text.append(f"- `{record['experiment']}`: `{path.relative_to(root).as_posix() if path.is_relative_to(root) else path.name}`; source `{record.get('source_commit', record.get('source_sha256', 'unknown'))}`.")
    (output/'comparison.md').write_text('\n'.join(text)+'\n')
    print(output/'comparison.md')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--output', type=Path)
    p.add_argument('--preview-clock', type=Path)
    args = p.parse_args()
    build(args.root, args.output or args.root/'report', args.preview_clock)
