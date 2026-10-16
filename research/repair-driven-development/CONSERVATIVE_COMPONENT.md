# A physical constraint that can be guaranteed

This component implements standard implicit transport mathematics. It is not claimed as a novel integrator or a trained architecture. It is separate from the gate and competing-state forecasting experiments.

**Follow-up:** it is now connected to jointly learned rates in the [coupled typed pilot](TYPED_RESOURCE_RESULTS.md), with complete state-dependent derivatives, and used in the [legal-edit pilot](TYPED_REPAIR_RESULTS.md). Those studies do not show a broad learning advantage. The component's restriction to known closed, one-for-one exchange still applies.

## Closed exchange pool

Let x_i >= 0 be amounts in compatible units. An edge e transfers one unit of pool content from source s(e) to destination d(e) at nonnegative first-order rate k_e. Freeze rates during a step and form a column-conservative generator Q:

    Q[source,source] -= k_e
    Q[destination,source] += k_e
    dx/dt = Q x
    x_next = (I - Delta Q)^(-1) x.

The columns of Q sum to zero. Therefore 1^T x_next = 1^T x. Since Q has nonnegative off-diagonal entries, the backward-Euler matrix is a nonsingular M-matrix for finite nonnegative rates and positive Delta; its inverse is nonnegative. The step preserves nonnegativity without clipping. This is a numerical invariant, not a learned discovery. Finite-precision errors still need tolerances.

For an output derivative g, solve a = (I-Delta Q)^(-T) g. Then

    dL/dx = a
    dL/dk_e = Delta * x_next[source] * (a[destination] - a[source]).

If rates depend on the previous state through another module, its chain-rule contribution must also be included. The component returns the direct state derivative and the rate derivative separately; it does not silently assume state-dependent rates are constants in the full gradient.

## What this establishes

`components/exchange.py` is differentiable and can be connected to a learned nonnegative-rate module. Its tests check conservation and positivity at varied rates/time steps, finite-difference derivatives, invalid inputs, and convergence toward the exact matrix exponential for constant rates.

This component only represents closed, first-order, one-for-one exchange. It does not model arbitrary A+B-to-C reactions, synthesis, degradation, growing cells, changing volumes, measured mass units with unequal conversion factors, thermodynamic equilibrium, or physical energy expenditure. A gene expression system is not automatically a closed pool. Unknown biological constraints must not be invented to gain a benchmark advantage.

In an eventual typed architecture, genuinely conserved measured pools use such a block, abstract memory uses bounded computational dynamics, and known boundary feeds/removal are modeled explicitly. A structural edit inside the exchange block may add an exchange edge while preserving the same pool total. A new hidden computational state is not a new molecule and must not receive fictitious conserved mass.

## Next model-level test

Before using this block to claim improved learning, train a complete rate learner and compare it with a conventional model given the same conservation information, including a properly constrained generic control. Evaluate closed and intentionally open systems separately. A zero conservation violation is guaranteed here and must not be represented as evidence that the model identified correct rates, predicts well, or is a new AI architecture.
