# Evaluation plan before implementation

**Status:** prospective local protocol, October 15, 2026; not an external preregistration. This was written before DRS implementation. A fixed-substrate development pilot is now implemented under [FIRST_TASK.md](FIRST_TASK.md); none of the structural-repair or confirmatory claims below have been tested. Amendments must be dated and distinguish pilot changes from confirmatory decisions.

## Hypotheses

H1: A bounded substrate with structural edits recovers after mechanism changes better than parameter-only adaptation under comparable observation and resource budgets.

H2: A selector trained on other damage episodes improves the quality/cost tradeoff over random and nonlearned edit selection with the same proposals and fitted-candidate budget.

H3: Any benefit extends beyond sparse changes and generators exactly expressible by the supplied gate grammar.

These claims are independent. A positive H1 result does not establish H2 or H3, architectural novelty, real biological validity, or ASI.

## Pilot and confirmatory separation

Use three inexpensive pilot task families with development-only seeds to debug, tune, and estimate variance. Never use the seven original probes as confirmatory evidence for DRS. Their results motivated this proposal.

Before confirmatory evaluation, freeze a release commit, dependency versions, generator code, task-family assignments, resource caps, candidate-generation rule, hyperparameter search spaces, selector training, failure handling, and analysis script. Commit the protocol and record its hash before opening confirmatory outcomes. This is an internal freeze, not proof of independence from the research team.

Generate a separately recorded confirmatory seed manifest after the implementation freeze. Use 20 independent generated systems per confirmatory family, with multiple trajectories per system. The system, not individual time points, is the independent statistical unit. Keep the same systems, noise realizations, observation masks, and intervention schedules paired across models. A result from three families does not support claims about all biology.

## Systems and changes

Use q = 8 and 16 observed coordinates with unit-interval measurement support fixed by the benchmark before data generation, and a documented mixture of sparse and dense dependencies. Specify any transform from an unbounded internal generator state in advance. Estimate loss variance normalizers from training data only. Start fully observed; partial observation is a separately reported extension, not an unimplemented hidden advantage.

| Role | Generator family | Purpose |
|---|---|---|
| Pilot / meta-training | Smooth saturating production-decay systems | Train and debug the specified substrate; disclose the favorable match. |
| Pilot / meta-training | Polynomial interactions with damping | Expose different functional forms and dense effects. |
| Pilot / meta-training | Oscillatory systems with known external forcing | Test multistep prediction and recovery. |
| Confirmatory transfer | Rational regulation with competitive interactions withheld from meta-training | Function-family transfer beyond the simple product gate. |
| Confirmatory transfer | Systems with explicit delays and a held-out delay range | Test whether added hidden state helps, rather than just extra static coefficients. |
| Confirmatory transfer | Thresholded/hysteretic hybrid systems withheld from meta-training | Test regime changes and structural mismatch. |

Define each generator independently of DRS source code. Mathematical expressibility differs from a guaranteed exact-match dictionary; disclose any overlap. Check reference trajectories with tighter numerical tolerances and a second integration method where appropriate. The truth generator's numerical error must be smaller than measured model differences.

Every family includes these change conditions, reported separately:

1. One sparse coupling or rate change.
2. A dense, coordinated change affecting many interactions.
3. An added delay or unobserved dynamic influence.
4. An intervention combination held out during training.
5. An observation-noise or missingness change without a mechanism change.
6. No change at all.

The last two penalize needless structural growth. Ambiguous systems that observationally resemble one another are an identifiability test: accurate forecasts cannot establish that the hidden mechanism was uniquely recovered. Ground-truth structure is diagnostic only and is never provided to the repair learner.

Test two evaluation modes separately: forecasting after adapting to a changed system, and predicting a new intervention without adaptation labels from that intervention. Do not call the former zero-shot intervention prediction.

## Observation and adaptation budget

Each system has disjoint initial-fit, initial-validation, adaptation-fit, adaptation-validation, retained-task, and future-test trajectories. Avoid overlapping windows or trajectories split across partitions. Meta-training and confirmatory systems never share generator parameter draws.

Development budgets: 24, 96, and 384 scalar post-change measurement values. Split each budget equally between adaptation fit and adaptation validation. A measurement of eight coordinates consumes eight scalar values, not one label. The default primary comparison uses 96 scalar values; the other budgets estimate a sample-efficiency curve. Explicitly count measured coordinates, observation times, and experiments as separate resources.

Fit and validation trajectories are separate from future-test trajectories. All controls receive identical observations and intervention descriptions. Partial-observation masks, if introduced later, must be fixed and shared. An adaptation method cannot ask for additional measurements unless an active-measurement extension is declared for all applicable controls.

For retention, allow an equal, fixed training-era replay buffer to all adapting methods. Select its size on development data and freeze it. Report memory and retention with and without replay as an ablation.

## Required controls

- Frozen initial predictor: no adaptation; a lower-cost reference.
- Original sparse dictionary repair: carry over the old favorable assumption transparently.
- Fixed DRS graph with parameter-only adaptation: isolates structural change.
- Full-sized fixed DRS graph: tests whether spare capacity provides the same benefit as growth.
- DRS with random candidate ranking and the same four-candidate fitting budget.
- DRS with a nonlearned residual/sensitivity ranking and the same budget.
- Flexible, fixed nonlinear transition model with multistep rollout and tuned full-model adaptation.
- A flexible fixed dynamics model that allows latent state, with tuned adaptation: necessary for a fair delay comparison.
- An oracle edit location, given only for diagnostics and labeled as privileged; never use it as a deployable competitor.

Tune each control using equal development search budgets and the same measurements. Freeze a short, explicit implementation choice for the two flexible controls after pilot validation; names such as "strong baseline" are not sufficient. Do not deliberately cripple conventional controls because the goal is a nature-inspired design.

For the primary comparison, use the control selected by mean validation adaptation error across meta-development episodes, excluding frozen and privileged oracles. The comparator is selected before confirmatory test outcomes. Report all controls and an additional per-condition best-control envelope for context; do not switch the primary comparator retrospectively.

Model sizes should be within 10% of the declared parameter budget where possible. Also produce measured error-versus-wall-time and error-versus-label curves, since equal parameters, equal search, equal wall time, and equal expressive power are different constraints. The four-candidate limit applies to all structural selection controls. Count failed proposals and all candidate training.

## Primary metric

For coordinate j, estimate a positive scale v_j from training-era target variance, with a development-fixed floor. For a scored trajectory window W, define

    nMSE(W) = mean over observed scored (t,j) of
              (prediction[t,j] - target[t,j])^2 / v_j.

Compute this on independent future-test trajectories after the allowed adaptation episode. For each system, average nMSE over three development-fixed physical-time horizons; score equally weighted horizons. Pair DRS and controls on the same system. Integrator microsteps are not scored as extra observations.

For a family, relative improvement is

    I = 1 - [mean system-level nMSE_DRS] /
            [mean system-level nMSE_primary_control].

Report numerator and denominator, each system, and each change condition. A development-fixed small denominator floor is used only for numerical reporting and is explicitly flagged; near-zero baseline error is not an opportunity for a misleading relative improvement claim.

Bootstrap paired independent systems within each family, retaining all trajectories and horizons belonging to a system, for 95% intervals. The primary progression rule uses at least 10% improvement with an interval above zero in at least two of three confirmatory families. Treat this as an engineering gate, not a multiple-testing-adjusted declaration of a scientific discovery. For inferential claims, use predeclared family-wise correction or simultaneous intervals and report both corrected and uncorrected results.

## Secondary outcomes and guardrails

- Adaptation nMSE across the three scalar-label budgets; area under that budget/error curve with its exact weights disclosed.
- Retained-task nMSE. Progression requires no more than 5% relative deterioration versus the model immediately before the adaptation episode; use an absolute tolerance fixed on development data when the pre-adaptation error is near zero.
- No-change false repair rate and unnecessary added compartments/gates.
- Long-rollout boundedness, numerical failures, and integration-step convergence. Bounded outputs alone are not evidence of a useful forecast.
- Actual trainable parameters, active gates, optimizer operations, elapsed time, peak memory, replay storage, and candidate-search cost.
- Effects on sparse, dense, delayed, noisy, and hybrid conditions separately.
- Learned-selector improvement over random and nonlearned selectors, with paired system-level intervals. If it fails to improve this comparison, drop H2 regardless of any win over a frozen model.

No calibration or uncertainty-coverage claim is permitted for the conditional-mean version. If a probabilistic observation model is added, evaluate proper scoring rules and calibration separately, with noise training and calibration splits specified in a dated amendment.

## Decision and failure reporting

Advance to a biological dataset only if the progression gate is met under declared caps, retained-task degradation is within tolerance, and the structural advantage survives the fixed/full-sized controls. If selector learning is ineffective but structural edits help, continue only with the narrower structural claim and say so.

All failures count. Do not discard seeds, tune on test data, silently extend budgets, replace a failed generator family, or describe numerical clipping as successful recovery. A failure on dense changes may establish a specialization; it does not prove the model is useless. A win on matched sparse systems does not establish a general architecture advantage.

The real-data phase requires its own protocol for dataset licensing, donor/batch splits, intervention overlap, missing covariates, measurement noise, leakage, biological interpretation, and validation limits. It is not preapproved as a sequence-design system or a replacement for experiments.
