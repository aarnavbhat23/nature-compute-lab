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

The subsequent architecture research is documented in [repair-driven development](research/repair-driven-development/README.md). Start with [latest progress](research/repair-driven-development/RESEARCH_PROGRESS.md), the [architecture decision](research/repair-driven-development/ARCHITECTURE_DECISION.md), and [specific closed-loop precedents](research/repair-driven-development/CLOSED_LOOP_PRECEDENTS.md). The earlier [486-trial report](research/repair-driven-development/RESEARCH_RESULTS.md) is followed by 333 trials of a limited coupled typed solver and nonlearned legal repair. Current prototypes show no broad advantage over conventional controls. Learned repair selection and active experiments remain to be built; context/retention semantics must be resolved first.

Developed with Codex assistance. The implementation and documentation are intended to make every reported number reproducible and every limitation visible.

The latest direction is a [survey of nature-derived learning primitives](research/nature-learning-survey/README.md), following the request to investigate mechanisms beyond regeneration. It screens 14 natural systems, records existing computational work, and prioritizes compartment-confined conformational memory for learning-rule discovery. The proposed learner is not yet specified or validated; the [mathematical audit](research/nature-learning-survey/SEQUESTRATION_LEARNING.md) explains why a simple sequestration gate would not be a new architecture.
