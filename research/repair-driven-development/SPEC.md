# Developmental Repair Substrate: version-zero specification

**Proposed design; fixed-substrate dynamics and parameter learning now have a first implementation. Structural edits and selector learning are not implemented; architectural novelty is unverified.** See [FIRST_TASK.md](FIRST_TASK.md) for the exact implemented subset. The rules below specify the broader target.

## 1. Inputs, state, and outputs

An episode supplies an initial measured state, a measurement mask, a known intervention schedule, observation times, and any allowed adaptation measurements. No mechanism labels, change location, generating equations, or unobserved test outcomes are supplied to the learner.

For the first fully observed experiment, q = 8 or 16 measured coordinates have known [0, 1] support specified by the synthetic benchmark before data generation. Intervention inputs have declared nonnegative ranges, mapped to [0, 1] using those ranges. This avoids silently clipping out-of-range test values to training extrema. The substrate has n compartments, with q output coordinates and n-q hidden coordinates. Its state is h in [0, 1]^n. Output is the first q coordinates. A bounded synthetic output representation does not solve normalization or extrapolation for arbitrary real measurements. Hidden initial values are learned constants, not encodings of future outcomes. A partial-observation extension must separately specify how initial hidden and unmeasured output states are inferred from past measurements.

Start with n = q+4. Hard limits: n <= 64 and at most 256 reaction-like gates. Each gate reads at most two sources from the current state or normalized, known intervention inputs. Arbitrary-length text tokens are not the core representation. An eventual DNA or protein encoder is a separate interface problem.

## 2. One gate

For a nonnegative source z and positive K, define

    A(z; K) = z / (K + z)
    I(z; K) = K / (K + z)
    F(z; K, rho) = (1-rho) A(z; K) + rho I(z; K)

where 0 <= rho <= 1. A gate r computes

    g_r(h,u) = k_r * product over its sources j of F(z_j; K_rj, rho_rj).

Here k_r >= 0, and z_j is either a compartment or an intervention coordinate. Gate destinations and source indices are discrete structure; k, K, and rho are trainable parameters. Use fixed positive lower bounds for K and bounded positive k to prevent division by zero and uncontrolled rate inflation. Zero-, one-, and two-source gates are permitted; the empty product is 1.

Each gate contributes to either a production sum p_i or a decay sum d_i at one destination i:

    p_i = epsilon_p + sum of production gates arriving at i
    d_i = epsilon_d + sum of decay gates arriving at i

with fixed small positive epsilon_p and epsilon_d. Signs are represented by production versus decay, not negative concentrations. Gate wiring and these two destination types are the version-zero computational grammar. This remains a human-imposed prior; it is more flexible than a dictionary containing the exact generating equations, but not prior-free discovery.

## 3. Fast dynamics

For an integration interval Delta > 0, freeze p_i and d_i at the start of the interval and update synchronously:

    h_i(next) = [h_i + Delta * p_i] / [1 + Delta * (p_i + d_i)].

This is a positivity-preserving numerical rule motivated by

    dh_i/dt = (1-h_i) p_i - h_i d_i.

It is not claimed to be the exact ODE solution. If 0 <= h_i <= 1 and rates are nonnegative, the next state is also in [0, 1]: the numerator is nonnegative, and denominator minus numerator is 1-h_i+Delta*d_i >= 0. Boundedness does not guarantee accurate integration, stable gradients, correct oscillations, or correct biology. Validate convergence as Delta is reduced and report integration cost.

Use one global Delta in version zero, chosen on development data. Multiple microsteps can occur between measurements. Do not silently align numerical update cost to the ground-truth solver's cost. Test observation times are not training labels.

## 4. Initial parameter learning

For a fixed graph, optimize trainable gate parameters and hidden initial constants using gradients through a finite rollout and a masked squared forecast loss. Positive parameters can be represented with bounded logistic transforms. Optimization is a tool here; using gradients does not imply a Transformer or require token prediction.

Training objective:

    L = masked normalized forecast error
        + lambda_rate * summed gate rates
        + lambda_size * gate count.

Gate count is constant during a fixed-graph fit and affects comparisons between candidate structures, not its continuous gradient. Estimate loss variance normalizers from training data only; the unit-interval measurement support is fixed by the benchmark, not inferred from test targets. Learn noise scales on development data if later adding distributional outputs; version zero predicts conditional means and does not claim calibrated uncertainty.

Train first with observed initial states and progressively longer rollouts. Record the curriculum and compare controls with the same sequence lengths. Integration clipping must not conceal NaNs or invalid input normalization; fail those runs visibly.

## 5. Slow repair

Repair requires new measurements. The model cannot discover its own predictive error from an unobserved outcome. At a change episode, split a predeclared adaptation budget into fit, candidate-selection validation, and future evaluation data. Every adapting control receives the same fit and validation observations.

Allowed edits:

- **Retune:** change a bounded subset of existing gate parameters.
- **Add interaction:** add one gate between existing sources and a destination.
- **Add compartment:** add one hidden state, its learned initial constant, and at most four incident gates. This adds computational state and potentially history dependence.
- **Prune:** remove one gate, or remove an isolated hidden compartment after removing its incident gates.
- **No edit:** retain the current model.

A new hidden compartment is not a claim of discovering a new biological species. Hidden units are generally non-identifiable without additional measurements or constraints.

Candidate edits are generated from adaptation-fit residuals, activity summaries, and sensitivity estimates. Freeze the proposal generator before confirmatory evaluation. No test outcomes or true damaged-site identifiers can guide the proposals. Proposals may use automatic differentiation to estimate sensitivities; count its time and memory.

Version-zero selection procedure:

    predict before obtaining the next labeled measurement
    update residual summaries after that measurement is revealed
    generate a fixed-budget edit candidate pool using fit observations
    rank candidates using the selector
    fit at most four candidates, each for a predeclared update budget
    score candidates on adaptation-validation observations and fixed retention probes
    include the no-edit candidate
    accept the best eligible edit only if it exceeds a development-chosen margin
    freeze structure and parameters for future held-out forecasts

Only one selection round is allowed per adaptation episode in the first benchmark. This limits repeated selection on the same validation measurements. All methods may perform their allowed fit/validation selection before future scoring. Later sequential adaptation needs an explicit prequential protocol and must never update from a target before scoring its prediction.

## 6. Learning the selector

Do not start by claiming a learned repair rule. First measure whether the structural candidates are useful with a nonlearned selector.

For the learned-selector stage, each edit a has a fixed-length feature vector:

    f(a) = [fit residual energy near destination,
            absolute residual/activity correlation for proposed sources,
            forecast sensitivity near proposed destination,
            local parameter-gradient norm,
            destination degree, added states, added gates,
            estimated fit operations, action-type indicators].

For hidden destinations, there is no measured hidden-state residual. Define its error credit as the Jacobian projection of observed-output residuals onto that state along the fitted rollout. This is model-based sensitivity, not evidence of the true causal change location. For a proposed new compartment, use the proposed incident gates and their provisional initialization when computing sensitivity; include this cost.

Standardize features using meta-training episodes only. A small linear ranker scores s_phi(a) = phi^T f(a). Its training target is the negative held-out validation loss plus declared retention and resource penalties after fitting the candidate. Candidate outcomes come only from meta-training damage episodes. Fit phi using regularized pairwise logistic ranking or squared utility regression; choose between these on development episodes, then freeze the choice.

This deliberately simple selector makes its contribution inspectable. It is not claimed to be a novel general learner. Meta-training search cost must be reported separately from adaptation cost. At confirmatory test time phi is frozen: it ranks proposals, and each selected candidate is fit using the allowed adaptation observations. The learning claim is that a repair-selection strategy transfers, not that it receives privileged supervision at test time.

### What is shared across systems

Version zero fits substrate parameters and structure separately to each training-era system. The selector parameters phi, the gate grammar, and the proposal-generation procedure are shared across systems. It does not yet learn a single pretrained biological world model or an entity-aligned representation shared across unrelated assays. Test transfer refers to the repair strategy across new systems and change families. A shared initialization or cross-system substrate pretraining would be a separately tested extension.

## 7. Acceptance, retention, and resource accounting

Choose an edit using

    J(a) = normalized adaptation-validation error
           + lambda_keep * retained-task error
           + lambda_cells * active compartment count
           + lambda_gates * active gate count
           + lambda_compute * measured candidate-fitting cost.

All coefficients and the acceptance margin are selected on development episodes. Report the separate components, not just J. Retention probes come from permitted training-era replay data; every control may use the same buffer. Count that storage. Replay is an explicit resource and does not imply the failed immune-repertoire mechanism has been validated.

Use hard caps in addition to penalties: 64 compartments, 256 gates, four fitted candidates per episode, a declared number of optimizer steps, and a fixed replay buffer size. Report actual parameter counts, gate evaluations, wall time, peak memory, labels, and discarded-candidate cost. An edit-count ratio is not a speedup.

## 8. What is and is not the claimed contribution

The proposed computational substrate is the bounded gate system. The proposed learning procedure couples its parameter fitting to budgeted structural edits. Their value and the selector's value are separate hypotheses.

The implementation is mathematically a learned dynamical system with structural adaptation. It can be represented using existing mathematical machinery. We cannot promise that nobody has built an equivalent formulation; this proposal makes no such claim. The first meaningful contribution would be reproducible evidence that the specified structural adaptation improves recovery and transfer under the protocol.
