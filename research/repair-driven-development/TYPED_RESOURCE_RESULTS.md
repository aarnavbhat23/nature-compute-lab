# Coupled resource/switch development pilot

**Date:** October 15, 2026, America/Los_Angeles. Raw artifacts use UTC. **Result:** the coupled model learns forecasts and preserves declared constraints, but the saturating representation loses to the equally constrained neural representation in every condition mean and every system tested. Continue with the neural rate head as the current control/engineering backbone; do not claim the saturating features are a new or better intelligence.

This is the first fixed coupled solver, not learned structural repair. Both heads jointly model three known closed-pool states, two measured regulatory activities, and two abstract memory states. They receive the same pool topology and total information. The neural head has 234 parameters; the saturating head has 230. See the [frozen protocol](TYPED_RESOURCE_PROTOCOL.md).

## Outcomes

Each entry is held-out mean nMSE ± sample SD across three generated coefficient draws. Lower is better. SD is not a confidence interval. Seeds are paired across conditions; do not pool the four conditions as independent systems.

| Condition | Saturating total | Neural total | Saturating regulation | Neural regulation |
|---|---:|---:|---:|---:|
| none | 0.00699638 ± 0.00139 | 0.00138348 ± 0.000246 | 0.00701177 ± 0.00315 | 0.00111581 ± 0.000129 |
| direct | 0.00658454 ± 0.00251 | 0.00153182 ± 0.000287 | 0.00464985 ± 0.00239 | 0.00132084 ± 0.000264 |
| resource | 0.00611917 ± 0.000333 | 0.00240207 ± 0.000226 | 0.00390318 ± 0.0019 | 0.00250423 ± 0.000369 |
| mixed | 0.00605657 ± 0.00201 | 0.00167028 ± 0.0002 | 0.00526929 ± 0.00336 | 0.00159452 ± 0.00019 |

The neural head has lower total error in 12/12 paired cases. All 24 selected models improve on their untrained initialization. All long 128-interval forecasts remain nonnegative with bounded regulation; maximum closed-pool total discrepancy is 4.22e-15. Conservation is guaranteed by the supplied solver and is not evidence of discovered biological mechanisms.

## Verification and costs

Completed 72 training trials (12 cases × two heads × three learning rates). Verified 24 saved forecasts, source hashes against the recorded source commit, disjoint split initial states, training-derived normalization, validation minima, pool/regulatory errors, and long-rollout invariants. Complete parameter and state finite-difference checks plus solver convergence tests pass in `tests/test_typed.py`. Source freeze: `6877e7f33c4b51ff6592b9a6283b5c23817192eb`.

| Head | Mean fitting seconds per case, all three trials |
|---|---:|
| saturating | 1.685 |
| neural | 1.593 |

Equal updates and close parameter counts are not equal compute. Timings include the fit and validation calls, not data generation or model construction. No elapsed-time superiority is claimed. Both methods get 1280 training and 1280 validation scalar target values per case, plus initial states. This is not yet a low-measurement repair result.

Raw [metrics](../../results/typed_resource_20261016T023212260119Z/metrics.json), [source/configuration provenance](../../results/typed_resource_20261016T023212260119Z/protocol.json), datasets, predictions, and weights are preserved. Regenerate this report with `OPENBLAS_NUM_THREADS=1 python3 analyze_typed_resource.py results/typed_resource_20261016T023212260119Z`.

## What this does not establish

The generator is small, matched to a known closed-pool topology, fully observed and noiseless. It is not validated gene expression or a whole cell. The labels direct/resource/mixed/none describe supplied synthetic mechanisms; this study does not infer or score those labels. All pool coordinates are measured, so it does not test ambiguity between hidden resource effects and direct regulation. No active experiment choice, new mechanism discovery, learned repair policy, open-system generalization, uncertainty calibration, or ASI is demonstrated.

The [closed-loop precedent audit](CLOSED_LOOP_PRECEDENTS.md) finds that biological hypothesis generation, model revision, and experiment choice already exist. The next test must isolate a specific structural-repair advantage against parameter adaptation and a full-capacity control; if it does not exist, narrow the claim.
