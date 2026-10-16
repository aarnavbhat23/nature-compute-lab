# Nature compute lab

Seven small experiments about learning from development, catalysis, symbiosis, repair, immune memory, stable regimes, and biological clocks.

The ideas came from a brainstorming session. This repository asks which parts can be made executable and tested. These are synthetic proof-of-mechanism studies, not validated biological models, new-model-family claims, or evidence of ASI.

## Experiments

| Experiment | Question | Planned control |
|---|---|---|
| 01 Development | Can a small learned local program construct a solver for longer cascades? | Shared linear transfer; fixed polynomial transfer |
| 02 Catalysts | Can conditional reaction rules be selected from observations? | Linear and full-library regression |
| 03 Symbiosis | Do useful compositions emerge when modules can exchange intermediate outputs? | Independent modules; fixed interaction dictionary |
| 04 Repair | Can a small structural patch adapt a model after one mechanism changes? | Frozen model; full refit; regularized full patch |
| 05 Immune memory | Does retaining competing local explanations help when a regime returns? | Cumulative and forgetting online regression |
| 06 Stable regimes | Can learned hysteresis preserve the relevant input history? | Memoryless threshold; autoregressive regression |
| 07 Clocks | Can learned timescales support cheaper asynchronous updates? | Same fitted rates with dense updates |

Each has a different task. Compare a prototype to controls within its experiment; do not rank all seven using their unrelated error values. Candidate operators, permitted side information, data splits, and limitations belong beside the code.

## Run

Python 3.11 or newer. Tested dependencies are pinned in `requirements.txt`.

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
python3 run.py --experiment all
```

The runner writes seed-level measurements, an aggregate, timings, the source revision, and timestamps into a new directory under `results/`. Results are added as the experiments are implemented and run.

## Research record

See [PROTOCOL.md](PROTOCOL.md) for the initial evaluation plan. Code and results will be committed as separate experiments are completed, with actual commit times. Negative results stay in the record. No biological sequences, private data, or live experiments are used in this first pass.

The implementation and documentation are intended to make every reported number reproducible and every limitation visible.
