# First coupled typed-model pilot

**Date:** October 15, 2026. **Status:** prospective development protocol, frozen in Git before running the pilot. This is the first implementation gate in ARCHITECTURE_DECISION.md, not the full repair/active-experiment architecture or a confirmatory result.

## Question

Can a learned fixed model jointly advance a known closed resource pool, bounded regulatory activities, and abstract memory with correct complete derivatives and useful forecasts? Does a simple saturating rate representation add value against a closely sized conventional rate network given exactly the same constraints?

## Mathematical model

State is three nonnegative resource-pool fractions x, two measured bounded activities h, and two unmeasured bounded computational states z. Inputs are two known dimensionless intervention levels. Fractions are normalized to a fixed reference amount; their trajectory-specific sum can be 0.8–1.2. Hidden states have no physical mass. The fixed pool topology is free ↔ bound-A and free ↔ bound-B, supplied equally to both learners; no topology is discovered.

From the previous joint state and input, compute 12 nonnegative rates: four pool exchange rates, four production rates, and four decay rates. Exchange uses the standard implicit component; activities and memory use `(h + dt*p)/(1 + dt*(p+d))`. Compute all rates before either update. The full reverse pass includes rates' dependence on both pools and bounded states, including cross-type feedback and memory initialization. No clipping.

Two interchangeable heads receive the same nine inputs and emit the same 12 rates in (0.01,2.01):

- **Saturating, 230 parameters:** per-output saturating features `x/(K+x)`, learned signed weights and K in (0.02,2), output sigmoid, and learned memory initialization. This is a conventional phenomenological Hill-like parameterization, not literal reaction identification or new mathematics.
- **Neural, 234 parameters:** tanh network with width 10 and output sigmoid, plus learned memory initialization. It uses exactly the same typed solver and prior knowledge. Parameter counts differ by less than 2%.

## Independent generator and information

`drs/resource_systems.py` defines a separate continuous-time five-variable system. RK4 with 0.01 steps supplies trajectories, with finer-step agreement checked. Pool binding depends on inputs and activities; regulatory production may depend on direct inhibition, free resource, both, or neither. Four named conditions (`direct`, `resource`, `mixed`, `none`) designate those regulatory influences; they are not episodes of structural change. Pool dynamics occur in every condition. These equations are deliberately small and biologically inspired, not a validated gene-expression system.

All five measured states are fully observed without noise, including the resource pool. Total amount and the star exchange topology are valid known priors given to both heads. True coefficients and mechanism labels are not learner inputs. Measuring all pools makes this an easier initial forecasting task; it does not test distinguishing observationally ambiguous mechanisms or missing-state inference. Neither head calls the generator.

## Frozen study

Use seeds 1301,1409,1511 in each of four conditions: 12 generated cases. The same seed pairs coefficient draws across conditions, so conditions are not independent replications to pool into a confidence interval. Each has separate initial states and interventions for 32 train trajectories × 8 intervals, 8 validation × 32 intervals, and 16 held-out × 32 intervals. Interval 0.2; interventions held for two observations. Every scalar output label counts: 1280 training and 1280 validation values per case, plus five initial values per trajectory, supplied equally. This is not a measurement-limited adaptation experiment.

For each head, fit all three learning rates 0.003,0.01,0.03 with Adam for 600 updates, batch size eight, gradient clipping at norm five, and the existing 2/4/8-interval training curriculum. Select checkpoint every 20 updates and learning rate by full 32-interval validation nMSE. Two heads × three rates × 12 cases = 72 training trials. No seed removal or retuning after held-out inspection. Source must be committed with clean Git status; raw records include source hashes, all tuning traces, datasets, selected weights, and predictions. Run with `OPENBLAS_NUM_THREADS=1`.

## Measurements and interpretation

Primary: mean held-out normalized squared error across the five measured coordinates and all 32 intervals, dividing by training variance with floor 0.0001. Report pool and regulatory errors separately, each case, condition means, and sample SD (not confidence intervals). Report actual fitting times; equal updates are not equal compute. Untrained error is a diagnostic, not a competitive baseline.

Numerical checks: finite-difference every parameter for both heads, complete step state derivative, conservation/positivity/boundedness, convergence to an independent adaptive ODE solver, and generator step agreement. Long 128-interval model rollouts repeat each held-out trajectory's input sequence four times and report invariants without a forecast-accuracy claim.

Progression requires correct derivatives/invariants and learned forecasts improving over initialization. A saturating head losing to the constrained neural head means its specific representation has no demonstrated advantage here. In either event, no learned repair advantage follows. Next gate is a separately specified nonlearned edit-vs-parameter adaptation experiment, only after examining this pilot. The full learned edit policy, active measurement choice, retention, open-system operators, uncertainty calibration, mechanism-family transfer, and real biological validity remain untested.

## Prior-art consequence

The [closed-loop novelty follow-up](CLOSED_LOOP_PRECEDENTS.md) finds automated biological hypothesis generation, model revision, and experiment selection already represented in Robot Scientist, LGEM+, Genesis, and biochemical model-discrimination research. This pilot is a transparent engineering prerequisite. A future contribution must demonstrate a distinct repair algorithm/capability beyond those precedents; it is not justified by calling the architecture nature-inspired.
