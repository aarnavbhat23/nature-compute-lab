# Research progress: coupled dynamics and legal repair

**Date:** October 15, 2026, America/Los_Angeles. This follow-up responds to the request to investigate whether the specific self-revising biological model already exists and start building/testing it. It adds two completed studies and **333 actual training trials**. Git records the actual source/result commit times; raw artifacts use UTC.

## What changed

| Work | Outcome | Evidence |
|---|---|---|
| Specific closed-loop literature audit | The broad concept exists. Six additional primary precedents include biological hypothesis generation, experiment selection, quantitative signaling/metabolic integration, and automated revision. The earlier audit covered 13 other methods. | [CLOSED_LOOP_PRECEDENTS.md](CLOSED_LOOP_PRECEDENTS.md) |
| Fixed coupled typed model | Joint physical/regulatory/memory state, differentiable nonnegative rates, valid closed-pool update, complete cross-state gradients. The similarly sized constrained neural head beats the saturating head in all 12 paired cases. | [72-trial protocol](TYPED_RESOURCE_PROTOCOL.md), [results](TYPED_RESOURCE_RESULTS.md), source freeze `6877e7f` |
| Nonlearned structural repair | Four legal graph candidates vs an eight-trial parameter-only control. Edits win 3/18 future-error comparisons, tie six, lose nine; no broad advantage. Selected-edit retention exceeds 5% deterioration in 8/18 cases. | [261-trial protocol](TYPED_REPAIR_PROTOCOL.md), [results](TYPED_REPAIR_RESULTS.md), source freeze `2dd116e` |
| Design diagnosis | Old and changed dynamics can require different predictions from the same initial state/input. A fixed shared memory initialization does not resolve that ambiguity. Retention requires a specified context interface and an appropriate stored-model control. | [CONTEXT_AND_RETENTION.md](CONTEXT_AND_RETENTION.md) |

## Current implementation status

- Implemented: fixed bounded and competing-state predictors; a limited closed-pool coupled solver; two rate-head controls; legal one-for-one exchange-edge additions with warm starts; nonlearned validation-based model selection; explicit replay and retention evaluation.
- Verified: 29 repository tests, including all 464 default-head parameter derivatives by finite differences, complete coupled step derivatives, expanded-model derivative samples, physical invariants, and numerical convergence. The new studies' **240 saved forecast arrays** reload and reproduce; source fingerprints and validation selection verify.
- Not implemented: a policy that learns repair decisions across systems, context inference from calibration history, active experiment choice, general biochemical stoichiometry/open-system operators, a calibrated uncertainty model, real biological prediction, protein/sequence design, or ASI.

The known pool total and exchange topology are provided to both controls. They are not discovered. These are small synthetic development tasks, fully observed apart from abstract memory and the declared noise condition. The repair study has only three independent base coefficient draws across 18 change cases; it is not an 18-system confirmatory study.

## Decision

Use the stronger neural rate head for present engineering controls. Do not advertise a superior nature-derived rate architecture, train a repair selector on the current setup, or extrapolate these results to ASI. The nature-inspired representation has value as an explicit, testable scientific model; its additional learning advantage remains unproven.

The next justified task is the context/retention interface and a stored-model baseline, followed by an independent structural-versus-parameter adaptation test under that corrected interface. Only a demonstrated structural benefit justifies learning the edit policy. Context-conditioned models and robot scientists already exist; novelty remains a separate requirement.

The earlier [architecture decision](ARCHITECTURE_DECISION.md) records the initial proposal. This follow-up supersedes its unimplemented-coupled-solver status and its preliminary rate-head recommendation. It preserves the prior negative results and the existing scheduled seven-experiment publications.
