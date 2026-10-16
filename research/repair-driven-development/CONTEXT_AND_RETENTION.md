# Context must be part of a testable repair architecture

**Date:** October 15, 2026. **Status:** design correction motivated by the completed [legal-edit pilot](TYPED_REPAIR_RESULTS.md); not an evaluated new capability or a frozen confirmatory protocol.

## The missing information

After a hidden mechanism change, the same measured initial state and known intervention can have a different correct continuation. The current pilot starts each forecast with the same learned memory vector. It cannot infer a changed mechanism from identical complete inputs before seeing a distinguishing outcome. This is a limit of the available information, not a reason to introduce arbitrary randomness.

An elementary counterexample uses a closed two-state pool with transfer rate k. Starting at x=(1,0), the exact state at time t is `(exp(-k*t), 1-exp(-k*t))`. Two regimes with different k have the same initial state and input but different targets a and b. For any common deterministic prediction p,

    0.5*(||p-a||² + ||p-b||²)
      = ||p-(a+b)/2||² + ||a-b||²/4.

The irreducible average squared Euclidean error without a distinguishing input is at least `||a-b||²/4`. For a mean over coordinates divide by the number of coordinates. This mathematical example does not quantify how much of the pilot's measured error comes from ambiguity; fitting and selection can fail independently as well.

“Retain the old model's knowledge” must therefore be distinguished from “make one context-free transition function match old and changed dynamics simultaneously.” Replay and a retention penalty cannot supply a missing regime signal.

## Corrected architecture requirement

Every forecast needs an explicit statement of what context is available:

- **Recorded context:** the experiment belongs to a known batch or environment record, without revealing the hidden mechanism identity. All methods get that record. A model bank retaining old and adapted snapshots is a necessary simple baseline. Its old-context retention is largely guaranteed by storage, not learned intelligence.
- **Inferred context:** the model receives an actually observed calibration history from the queried system and infers a contextual state or distribution over models. All methods get the same scalar observations, time points, and intervention metadata. No ground-truth edge labels or future outcomes are available.
- **Insufficient context:** preserve multiple explanations and report unresolved predictive uncertainty. A conditional-mean model cannot claim a calibrated predictive distribution merely by retaining several models.

Context-adaptive dynamics is already studied, including [CoDA, ICML 2022](https://proceedings.mlr.press/v162/kirchmeyer22a.html). Context conditioning, model banks, and keeping versioned explanations are controls and design requirements, not an established novel contribution here.

## Next experiment to specify

The next narrow question is whether **structural repair adds value after context is handled fairly**. Do not train a cross-system edit selector on the current retention setup.

1. Keep the completed 18 cases as development evidence. Use new coefficient draws for a corrected protocol; publish code, budgets, and selectors before evaluation.
2. Start with recorded context. Compare stored old + parameter-adapted models against stored old + structurally adapted models. Give all procedures the same calibration/validation data, trial budgets, initialization, context records, replay policy, and fixed invariants.
3. Charge full parameter storage, number of retained models, fitting, selection, and query-time computation. Do not call snapshot retention free or evidence of learned memory.
4. Report future error, post-change measurements, uncertainty limitations, and retention per context. Include no change and noise-only changes; a repair need not be correct merely because the forecast improves.
5. Compare a fixed full-capacity graph, a context-adaptive control, and sparse correction with the same known candidate grammar. Include changes outside that grammar. A matched edge dictionary remains privileged prior knowledge.
6. Only if structural adaptation offers a meaningful cost/error benefit, test whether a policy trained on different systems can choose edits more efficiently than the same candidates with nonlearned/random selection.
7. Treat active experiment choice as a separate later intervention-budget study, with existing model-discrimination approaches as controls. Do not infer its benefit from passive-fit outcomes.

This does not prescribe an unbounded expansion of features. The immediate prerequisite is a precise context interface and a straightforward stored-model control. If parameter adaptation plus context meets the need, retain that simpler result. If it does not, identify a concrete residual failure before adding machinery.

## Novelty and scale

The [closed-loop precedent audit](CLOSED_LOOP_PRECEDENTS.md) shows that biological model revision, experiment selection, signal/mass distinctions, and edit prioritization already exist. A possible contribution must be a distinct algorithm or measured transfer/cost capability beyond that work. The current studies have not established it. Scaling computation without resolving these definitions would scale an ambiguous experiment, not demonstrate ASI.
