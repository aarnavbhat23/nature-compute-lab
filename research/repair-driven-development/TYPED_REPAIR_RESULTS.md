# Legal exchange-edit development pilot

**Date:** October 15, 2026, America/Los_Angeles. Raw artifacts use UTC. **Decision:** do not advance to training a repair selector on the current formulation. Nonlearned edit search shows no broad advantage over fitting-budget-matched parameter adaptation, and retention fails badly on several changed systems. Preserve these outcomes and address context/retention semantics before scaling.

## Study and information

Three new base systems are each tested under six changes: 18 paired change cases, only three independent coefficient draws. Nine initial fits plus 252 distinct adaptation fits yield **261 actual training trials**. Each main procedure is charged eight 300-update tuning trials per case; two trials are shared exactly between the procedures. Extra edges increase parameters from 234 to at most 256. All methods receive the same known closed-pool graph information, 120 new fit scalar targets, 320 new validation scalar targets, 25 new initial-state values, and the declared stored old data. See the [frozen protocol](TYPED_REPAIR_PROTOCOL.md).

Selection uses new-condition validation and retained validation in a 2:1 trajectory ratio, permitting a frozen/no-edit choice. An edit candidate can add A→B, B→A, or both; this deliberately matches some generator changes. It is nonlearned model selection over four known topologies, not novel mechanism generation, a learned selector, or active experiment choice.

## Future forecast error

Mean nMSE ± sample SD over three base draws, lower better. SD is not a confidence interval. “Edit selected” and “Parameter selected” include the frozen option. Full graph receives only two tuning trials, so it is not budget-matched.

| Change | Frozen | Edit selected | Parameter selected | Full graph |
|---|---:|---:|---:|---:|
| none | 0.00233403 ± 0.000507 | 0.0024211 ± 0.000438 | 0.00247055 ± 0.000387 | 0.00345344 ± 0.00202 |
| parameter | 0.57483 ± 0.245 | 0.126543 ± 0.0329 | 0.102976 ± 0.0155 | 0.125136 ± 0.0344 |
| edge_ab | 0.450683 ± 0.262 | 0.0890398 ± 0.0491 | 0.0934327 ± 0.0348 | 0.0893163 ± 0.0484 |
| edge_ba | 0.390158 ± 0.0261 | 0.22036 ± 0.142 | 0.0806699 ± 0.0158 | 0.0954819 ± 0.046 |
| both | 0.0639541 ± 0.0404 | 0.0639541 ± 0.0404 | 0.0594793 ± 0.0432 | 0.0462069 ± 0.0185 |
| noise | 0.00847879 ± 0.00122 | 0.00847879 ± 0.00122 | 0.00847879 ± 0.00122 | 0.0186817 ± 0.00651 |

Edit search wins 3/18 individual future-error comparisons against the parameter procedure, ties 6/18, and loses 9/18. Its mean error is slightly lower for A→B additions and no-change cases, but worse for the parameter and B→A changes; the both-edge and noise cases provide no broad edit benefit. Three seeds and favorable edit priors do not support a superiority claim.

## Retention and false changes

| Change | Frozen retained error | Edit-selected retained error | Parameter-selected retained error |
|---|---:|---:|---:|
| none | 0.0021934 | 0.00215581 | 0.0022098 |
| parameter | 0.0021934 | 0.430702 | 0.467186 |
| edge_ab | 0.0021934 | 0.428628 | 0.368044 |
| edge_ba | 0.0021934 | 0.130853 | 0.142403 |
| both | 0.0021934 | 0.0021934 | 0.0209361 |
| noise | 0.0021934 | 0.0021934 | 0.0021934 |

Edit selection exceeds 5% retained-error deterioration in **8/18** cases; parameter selection does so in **11/18**. The current objective does not protect old behavior. Breach cases: parameter/1601, edge_ab/1601, edge_ba/1601, parameter/1709, edge_ab/1709, edge_ba/1709, parameter/1801, edge_ab/1801.

No-change/noise false structural growth: 0/6. All three noise cases select frozen. Existing-edge parameter changes select extra edges in all three cases, demonstrating that selected graph structure is not a recovered causal mechanism. All three both-edge cases select frozen under the declared objective. Every choice and its errors remains in the raw record.

## Costs and verification

Mean charged search time: edit 2.07 seconds; parameter 1.87 seconds per case. Model construction, data generation, and initial fitting are outside those adaptation times. Equal trials/updates do not imply equal wall time.

Verified 216 saved forecast arrays (future and retained), all selected validation scores/minima, graph choices, reused trials, parameter counts, source hashes, and disjoint within-case initial states. Maximum recorded future pool-total discrepancy: 2.44e-15. The constraints are supplied and guaranteed; they do not demonstrate correct biology. Source freeze: `2dd116ed7693e956ada2ec80034536a4e6618b5d`.

Regenerate with `OPENBLAS_NUM_THREADS=1 python3 analyze_typed_repair.py results/typed_repair_20261016T023951695435Z`. Raw [metrics](../../results/typed_repair_20261016T023951695435Z/metrics.json), [initial fitting traces](../../results/typed_repair_20261016T023951695435Z/initial_fits.json), [source/configuration](../../results/typed_repair_20261016T023951695435Z/protocol.json), datasets, weights, and forecasts are preserved.

## A design limitation exposed by this test

The same initial state and intervention can have different correct continuations before and after a hidden mechanism change. The current model initializes abstract memory to the same learned vector for every trajectory; it receives no observed calibration history or regime context at forecast initialization. A single deterministic transition cannot represent two different derivatives at the same complete input. Replaying old labels while overwriting one global dynamics model can therefore impose incompatible tasks. This does not excuse the forecast losses, prove that every loss comes from ambiguity, or establish that edits never help. It limits what the retention experiment can establish and what a more realistic architecture must specify.

The next protocol should distinguish knowledge retention from using one dynamics law in two regimes. A straightforward baseline stores the old model and conditions forecasts on an explicit experimental-context record. A harder model infers context from the same actually revealed calibration observations available to all methods. Neither gets the true mechanism label or future outcomes. Query context, memory, model storage, and inference cost must be counted. Context-conditioned learning and model banks are existing ideas; this correction is not a new intelligence claim.

Before learning a repair policy, test whether structural edits add value against this context-aware parameter/control setup. Do not retune the present 18 cases or call a corrected protocol a fresh confirmatory result on them. No ASI, calibrated uncertainty, biological validation, or novel architecture advantage is demonstrated.
