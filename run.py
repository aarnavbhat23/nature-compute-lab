"""Run one completed experiment and preserve raw measurements in a fresh directory."""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parent


def revision():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def run_experiment(path, seeds):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    started = datetime.now(timezone.utc)
    source = revision()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
    rows = []
    for seed in seeds:
        tick = time.perf_counter()
        row = module.run(seed)
        rows.append({'seed': seed, 'elapsed_seconds': time.perf_counter() - tick, **row})
    metrics = rows[0]['metrics']
    summary = {}
    for key in metrics:
        values = np.asarray([r['metrics'][key] for r in rows], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f'Non-finite metric: {key}')
        summary[key] = {'mean': float(values.mean()), 'std': float(values.std(ddof=1)) if len(values) > 1 else 0.0}
    record = {
        'experiment': path.stem, 'started_utc': started.isoformat(),
        'finished_utc': datetime.now(timezone.utc).isoformat(),
        'source_commit': source, 'source_tree_dirty': dirty,
        'python': platform.python_version(), 'numpy': np.__version__,
        'platform': platform.platform(), 'rows': rows, 'summary': summary,
        'scope': 'Synthetic proof of mechanism; no biological or ASI validation.',
    }
    target = ROOT / 'results' / f'{path.stem}_{started:%Y%m%dT%H%M%S%fZ}'
    target.mkdir(parents=True)
    (target / 'metrics.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(path.stem, json.dumps(summary, indent=2))
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True, help='all, a numbered prefix, or full filename stem')
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 17, 29, 43, 71])
    args = parser.parse_args()
    paths = sorted((ROOT / 'experiments').glob('*.py'))
    selected = paths if args.experiment == 'all' else [p for p in paths if p.stem.startswith(args.experiment)]
    if not selected:
        parser.error('No matching completed experiment')
    for path in selected:
        run_experiment(path, args.seeds)
