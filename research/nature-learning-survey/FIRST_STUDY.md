# First study: discover and falsify the learning rule

Status: prospective design. The learning rule, datasets, seeds, and numerical budgets are not yet frozen. No experiments in this document have been run. A final protocol must be committed before confirmatory test outcomes are inspected.

## Task 0: the required next deliverable

Produce a complete specification for one candidate: input encoding, available state, persistent memory, inference observable, learning signal, update events, initialization, reset/retention semantics, capacity bound, and computational cost. Derive the simplest limiting cases. Compare them with a bounded gate, complex memory, an associative chemical rule, and a bounded model bank.

**Go:** identify a specific operational difference and a task on which that difference makes a falsifiable prediction.

**Stop or relabel:** the proposed rule is only a reparameterization of a known learner, depends on an unspecified intelligent controller, or provides no mechanism for learning a relationship that was not prewired. Do not enlarge the simulator to hide this failure.

## First predictive capability

Predict how a benign dynamical system responds to a stimulus after different exposure histories. Begin with abstract reaction/regulatory dynamics; no DNA synthesis, sequence design, or live experiments are needed.

The simulator should include:

- Histories with the same present input and different, inferable internal contexts.
- Changed cue–outcome relationships, their later recurrence, and genuinely new combinations.
- Transient adaptation that should disappear and persistent change that should remain.
- Interactions requiring more than one cue, so a single habituating threshold is insufficient.
- Partial/noisy observations as a later condition, after fully observed behavior is understood.

The history must contain enough information to infer the context. Hiding the distinguishing evidence while asking the learner to select the correct hidden regime would repeat the [previous retention problem](../repair-driven-development/CONTEXT_AND_RETENTION.md). Context labels may be supplied only to a clearly marked oracle upper bound.

## Separate three claims

| Claim | Evidence needed | Common false positive |
|---|---|---|
| The substrate has memory | Same input produces a history-dependent response | A persistent state with no useful relationship to outcomes |
| The substrate learns | Experience improves later predictions; withholding/shuffling the learning signal removes the improvement | Hand-coded cue-to-output mapping or a trained readout doing all the work |
| The new mechanism helps | Its distinctive operation improves a preregistered metric against fair controls | More state, more labels, a favorable generator, extra tuning, or an oracle context |

## Required controls and ablations

1. Memoryless predictor and a linear history model, to establish that history is necessary.
2. A small recurrent predictor and a finite-state/switching dynamical model, both receiving identical histories.
3. A bounded stored-model library with observable-history-based selection. Match total persistent storage, including unused models.
4. A scalar sequestration gate and a multiscale-memory control, to test whether simpler known memory explains the effect.
5. A chemical associative-learning control matched to the selected task, when a faithful implementation is feasible.
6. Candidate ablations: no persistence, no confinement, fixed assembly membership, randomized memory transfer, and equal-capacity independent compartments.

Conventional controls are necessary to evaluate originality and value. They need not be components of the candidate. External code must be inspected for license and reproduced before its numbers are used; the source ledger does not imply reproduction.

## Measurements

- **Forecast quality:** error normalized by training-set scale, integrated over a fixed horizon; report each task family separately. Avoid normalization by near-zero target variance without a declared floor.
- **Learning efficiency:** error as a function of observed examples and outcomes, not just training iterations.
- **Retention and recovery:** old-context error after a change and observations needed to recover when it returns. Compare to the pre-change model on the same held-out cases.
- **Compositional transfer:** performance on withheld combinations and independent generator families, not only new random trajectories of a familiar rule.
- **Capacity:** useful retained relationships versus total stored numbers/bits, assembly types, and compartments.
- **Compute:** end-to-end wall time, solver steps, development search budget, and peak memory. Passive molecular dynamics in a hypothetical device do not imply cheap simulation on a CPU.
- **Uncertainty, when implemented:** proper scoring rules and interval coverage/width under declared noise and distribution shifts. An ensemble is a comparator, not evidence that the core primitive is new.

Report paired uncertainty across independent systems. Trajectories from one coefficient draw are not independent replications. A 95% confidence interval is a reporting convention, not the earlier 95% accuracy target.

## Avoiding an experiment designed to favor the candidate

Use development tasks from at least two distinct generating mechanisms. Reserve a third family and new coefficient draws for confirmation. Give all controls the same observations, labels, prediction horizons, and context evidence. Match both parameter/state budgets and measured runtime in separate comparisons; one matching scheme does not imply the other.

Before the confirmatory run, freeze the exact update law, encoding, data-generation code, splits, seeds, tuning allocations, metrics, and smallest useful effect. Keep all runs and negative results. If development failure motivates a new rule, version it and use a fresh confirmatory evaluation.

Include shuffled cue–outcome histories and irrelevant cues. A substrate that produces attractive stable structures but cannot distinguish these controls has not established predictive learning.

## Progression after the first result

1. **Mathematical candidate:** complete the rule and audit reductions.
2. **Minimal learning demonstration:** learn an unseen mapping with a fixed readout and a bounded substrate.
3. **Independent synthetic generalization:** beat the relevant controls on a declared tradeoff, not necessarily raw error alone.
4. **Public biological data:** identify a reusable dataset with adequate exposure history, metadata, license, and held-out conditions. The Whi3 papers motivate the mechanism; their figures are not automatically a ready training benchmark.
5. **Scaling study:** vary components, assembly complexity, tasks, and data; report gains together with memory/runtime and failure rates.

A laptop-scale simulation is a reasonable starting hypothesis, but runtime cannot be estimated responsibly before choosing the rule and solver. No GPU budget or wet-lab commitment is justified yet. The first decision should be whether the proposed learning operation survives Task 0.
