# First task: learn a fixed dynamical substrate

**Written October 15, 2026, before the nine-system pilot was run.** A separate polynomial development smoke case, seed 13, was used to check that both learned models reduce training error in 60 optimizer steps. It is excluded from pilot reporting. No confirmatory evaluation or structural repair is implemented in this task.

## Implemented subset

- A persistent substrate with eight observed coordinates, four learned hidden initial values, and 252 fixed-wiring gates.
- Trainable gate strengths, saturation constants, and activation/inhibition mixtures: 736 parameters total.
- Zero-, one-, and two-source gates and the bounded synchronous update in SPEC.md.
- Exact reverse-mode derivatives in NumPy, checked against finite differences through complete rollouts. SciPy supplies numerical utility functions and the independent solver check; no neural-training framework is required.
- Adam parameter learning with a 2/4/8-observation rollout curriculum. All forecasts are closed-loop from the initial measured state and the known intervention schedule.

No gates or compartments grow, disappear, or change wiring. There is no repair selector, online adaptation, missing-observation inference, uncertainty estimator, or biological sequence generator.

## Frozen pilot protocol

Configuration: `configs/fixed_substrate_pilot.json`.

Three independent generator implementations use continuous-time production/decay dynamics: saturating regulation, polynomial interactions, and sinusoidal rate functions with known periodic forcing. The last family is not claimed to demonstrate autonomous biological oscillation. All coordinates have known unit-interval support. There is no measurement noise in this first task. The generators do not call the learner's update or expose their parent indices to it.

For each family use system seeds 101, 211, 307. Parameters drawn with the same seed are paired across family functional forms; do not treat the resulting nine cases as nine independent replications of a single task. Each system has 32 training trajectories (8 intervals), 8 validation trajectories (12 intervals), and 16 held-out test trajectories (48 intervals). Separate deterministic random streams initialize each partition. The observation interval is 0.2, so the supervised training horizon is 1.6 time units and the full test horizon is 9.6. This is a development pilot; confirmatory families and damage episodes remain untouched.

Reference trajectories use RK4 with step 0.01. Check a held-out trajectory against an independently implemented adaptive SciPy solver with tight tolerances. A disagreement above 1e-7 fails the run. Synthetic initial states and interventions are supplied exactly; predicting from these inputs does not identify real biological mechanisms.

## Controls and selection

1. Persistence: hold the initial measured state fixed.
2. Affine transition regression: fit on training transitions, select ridge penalty from 1e-6, 0.001, 0.1 using validation nMSE, then roll out without test teacher forcing. Its 88 parameters are deliberately a simpler control, not a capacity-matched architecture.
3. Neural rate model: a conventional one-hidden-layer tanh network generates production/decay rates. It has the same 12 persistent states, learned hidden initial values, nonnegative rates, synchronous integrator, and input information as DRS. Its 730 parameters are within 1% of DRS's 736.
4. Untrained initializations of DRS and the neural control, reported as learning diagnostics.

Both learned models use the same three learning rates (0.003, 0.01, 0.03), 240 Adam updates per trial, batch size 8, and validation checks every 20 updates. Select both checkpoint and learning rate on the 12-interval validation forecast. Then evaluate held-out test trajectories. Models receive identical training labels; the affine control uses one-step fitting whereas the learned models optimize multistep rollouts. This is an explicit objective difference, not extra data.

The gate graph includes all observed-to-compartment single-source couplings, two constant gates per compartment, production from the first external input, decay from the second, and one hidden/input interaction per destination. This wiring is human-specified and frozen; it is not structural discovery. The saturating family is favorably related to the model's functional grammar. Polynomial and sinusoidal families probe functional mismatch within the same broad bounded-dynamics class.

Rate/size penalties are zero in this task; there is no claim of learned sparsity. Numerical microstep 0.1, two updates per observation, is fixed before the pilot. No optimizer, training budget, or layout is selected using test scores. Resource matching here means similar parameter counts and identical trial/update/label budgets, not identical wall time or expressive power. Report wall time; do not call it a hardware-normalized benchmark. Peak memory and energy are not instrumented.

## Measurements

Estimate coordinate variances from training target values with floor 1e-4. Report variance-normalized forecast MSE on the first 12 test intervals and the full 48, as well as raw full-horizon MSE. These windows share trajectories, so they are not independent tests.

For each selected substrate, compare fixed-parameter forecasts at microsteps 0.1, 0.05, and 0.025 with an adaptive ODE reference over the first 12 intervals of two test trajectories. This is numerical integration convergence of the learned equations, not better learned predictions. Also run 192 observation intervals (38.4 time units) to check finiteness and boundedness. Do not train or select models based on either diagnostic.

Report all three seeds separately and family means/sample standard deviations. No significance claim or universal accuracy percentage is appropriate for three development seeds. The first engineering goal is executable, differentiably trainable, bounded dynamics with reproducible forecasts. A win over persistence is insufficient evidence for the architecture; the neural and affine controls remain central to the interpretation.

## Reproduce and provenance

From the repository root:

```sh
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
python3 run_fixed_substrate.py
```

The runner requires committed, clean source before creating a new timestamped results directory. It saves the source commit and SHA256 fingerprints, complete configuration, versions, numerical checks, every tuning trace, selected weights, split datasets, and predictions. A partial run remains marked `running`; only a completed run receives `complete`. The runner never changes source or pushes Git history.

## Results

**Post-run addition, October 15, 2026:** [the completed pilot report](../../results/drs_fixed_20261016T011815110991Z/report/results.md) records all nine cases, including negative comparisons. The substrate learned useful forecasts and passed numerical checks, but had higher family-mean forecast error than the neural control on all three families. This does not test structural repair. The seven earlier mechanism experiments and their remaining scheduled checkpoints are unchanged.
