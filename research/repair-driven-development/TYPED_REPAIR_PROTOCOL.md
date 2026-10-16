# Nonlearned legal-edit vs parameter-adaptation pilot

**Date:** October 15, 2026. **Status:** prospective development diagnostic, committed before evaluation. Motivated by the [coupled-model pilot](TYPED_RESOURCE_RESULTS.md), where the neural rate head won every paired case. This experiment uses that head. It tests a small legal edit search, not a learned edit policy or active experiment selection.

## Question and favorable assumptions

Can adding an exchange interaction after a change improve future prediction over tuning existing rates? The known physical type is one-for-one exchange within the same closed pool; allowed edits are exactly A→B, B→A, both, or neither. This is a deliberately favorable grammar: some truth changes use those same permitted edges. A positive result would not establish open-ended mechanism discovery. The original star-graph model is flexible and can compensate through other paths, so an edge-label match is not required for good forecasts and does not prove causal identification.

## Fixed implementation

`TypedDynamics.add_exchange_edges` validates the grammar before fitting and copies existing rate functions, feature transformations, and memory starts. New rate outputs start near 0.01001 because the current head has a 0.01 rate floor; expanded rollouts are therefore not exactly identical before adaptation. New edges conserve the same pool total. Memory does not receive pool mass. The base neural model has 234 parameters; one edge gives 245 and both give 256. The largest increase is under 10%.

## Systems and splits

Three fresh base coefficient draws, seeds 1601,1709,1801, use the mixed direct/resource generator. Each is initially fitted from 32 trajectories × 8 intervals and selected on eight validation trajectories × 32 intervals. Initial learning rates are 0.003,0.01,0.03 with 600 updates each: nine initial trials. Use existing training curriculum and variance floor 0.0001, calculated from initial training targets only.

Each base system has six changes: none; a rate increase along an existing release edge; a new A→B transfer; new B→A; both transfers; or no dynamics change plus Gaussian target-observation noise, SD 0.01. Initial measurements remain exact even in the noise condition. True noisy measurements can fall outside physical support; do not clip them or pretend the model outputs measurement uncertainty. Same initial states and intervention schedules are paired across change conditions. The 18 cases contain only three independent base coefficient draws; do not pool them into 18 independent systems for confidence intervals.

Post-change fitting uses three new trajectories × 8 intervals (**120 new scalar targets**) plus one stored initial-training replay trajectory. Selection uses two independent post-change trajectories × 32 intervals (**320 new scalar targets**) plus the first stored initial-validation trajectory. Thus selection minimizes 2/3 new-condition nMSE + 1/3 retained-validation nMSE. It uses **440 new scalar target values**, plus 25 scalar initial values across the five new fit/selection trajectories. Prior fit/validation/replay labels are shared equally. No active queries are permitted.

Future evaluation uses 16 new post-change trajectories × 32 intervals. Retention evaluation uses eight independent original-system trajectories × 32 intervals. Their outcomes never enter fitting or selection. Simulator equations and true change identities are not learner inputs. This pilot fully observes all five physical/regulatory coordinates without noise except the declared target-noise condition; missing-state biology remains separate.

## Methods and budgets

- **Frozen:** unchanged pretrained model, no adaptation.
- **Nonlearned edit selection:** fit the four topologies (existing graph, A→B, B→A, both) at two rates 0.003,0.01. Select the lowest combined validation score, also allowing the frozen model. At most four fitted graph candidates, eight tuning trials.
- **Parameter-only control:** existing graph with eight learning rates 0.0003,0.001,0.003,0.006,0.01,0.02,0.03,0.06. Select by the same combined score, also allowing frozen. This has eight tuning trials, equal to the whole edit search.
- **Full-graph control:** the both-edges candidate, fitted at the same two rates, reported separately. This is a smaller tuning budget than the two main procedures; disclose it rather than representing it as an equal-cost comparison.

Each adaptation trial has 300 Adam updates, batch size four (all three new fitting trajectories and one replay trajectory), curriculum 2/4/8 intervals, norm-five clipping, and validation every 20 updates. All candidates start from the same pretrained model, with the explicit small new-rate initialization above. The two parameter trials shared between the edit search and parameter-only control are executed once and reused exactly, not charged twice as new actual fits. There are 14 distinct adaptation fits per case, 252 total, plus nine pretraining fits: **261 actual training trials**. Each main procedure is charged its eight trials for its own cost comparison. Equal trials/updates are not equal measured wall time or expressive power; report actual time.

## Outcomes and decision

Report all case-level future/retained nMSE, condition means ± sample SD over three base draws, selected topology, parameters, all learning-rate/checkpoint traces, source fingerprints, and measured search time. Preserve weights, predictions, and raw splits. No held-out retuning. Report false growth on no-change and noise-only cases, recognizing that selection can react to baseline approximation error even without a true mechanism change. A selected edge is a model-choice result, not verified true biology.

Correct derivatives and invariants must continue to hold after expansion. Both post-change and retained behavior matter. The current selection score uses retained measurements but does not guarantee a ≤5% retention deterioration; explicitly report any breaches rather than claiming protection.

If legal edit selection fails to improve the tradeoff against budget-matched parameter updates, do not train a selector simply to make the architecture more elaborate. If it helps only on the matched edge changes, call it specialization. Even a broad pilot win would need more independent systems, withheld mechanisms, a random/nonlearned selector comparison, missing/noisy observations, and real data before a learned repair or biological claim.

This protocol is narrower than the prospective confirmatory plan in EVALUATION.md and does not satisfy its progression gate. It is an internal development freeze, not external preregistration. Run `OPENBLAS_NUM_THREADS=1 python3 run_typed_repair.py` from a clean committed checkout. Existing scheduled experiment publications remain separate.
