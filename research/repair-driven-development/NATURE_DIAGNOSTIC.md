# Nature dynamics: prospective diagnostic round

Written October 15, 2026, after the first fixed-substrate pilot and before the fresh diagnostic outcomes. This is adaptive development, not independent confirmation. The earlier pilot stays intact.

## Questions and predictions

1. **Initialization:** at rho=0.5, the original factor ((1-rho)z+rho K)/(K+z) equals 0.5 at every z. Its input derivative is zero. Nearby initializations may therefore learn weak input sensitivities. Compare the original initialization to polarized rho with logits ±2.5, leaving layout, parameter count, loss, and data unchanged. An improvement would implicate initialization; no improvement would limit this explanation.
2. **Optimization budget:** increase both models' optimizer updates fourfold, with identical per-trial schedules and selection rules. If the gap disappears, the short budget was a sufficient explanation on these cases. If not, representational or search issues remain plausible, but are not identified conclusively.
3. **Competing local states:** instead of independently summing many weak activator/inhibitor gates, allocate each compartment's finite activity across four alternative states and one inactive reference state. Test whether a context-dependent, capacity-normalized rate law helps at a matched neural-control parameter budget. A forecast win would motivate this representation; it would not validate a physical binding model or novel AI invention.

The original layout has only one hidden/input two-source gate per destination, so it omits direct observed/observed two-source products. This is a plausible restriction. This round does not independently manipulate only that wiring, so do not claim to establish its causal contribution.

## Competing-state equations

Let x concatenate current compartment values and the two known intervention inputs. For compartment i and local state m:

    score_im = b_im + sum_j W_imj log(x_j + 0.05)
    pi_im = exp(score_im) / [1 + sum_l exp(score_il)]
    pi_i,inactive = 1 / [1 + sum_l exp(score_il)]
    p_i = 0.01 + sum_m pi_im * 2 sigmoid(v_im,production)
    d_i = 0.01 + sum_m pi_im * 2 sigmoid(v_im,decay)

Use numerically stable normalization. Occupancies sum to one including the inactive state. Each production/decay rate lies in [0.01,2.01]. Use the original bounded state update and exact reverse-mode gradients. Four active states and twelve compartments produce 820 trainable parameters, versus 808 for the matched neural control; the original gate substrate has 736 versus its 730-parameter neural control.

This takes inspiration from molecular state competition and finite capacities. The real-valued W values are phenomenological learned affinities, not literal binding stoichiometries or measured free energies. No global mass conservation, thermodynamic consistency, detailed balance, physical energy expenditure, or cell identity follows. Normalization alone is not a new architecture contribution. The experiment tests the specified representation, including its log input map; it does not isolate competition from every other representational difference.

## Frozen experiment

Configuration is `configs/nature_dynamics_diagnostic.json`. Three new system seeds (401,503,601), three previously seen family types, five learned models, two optimizer budgets (240 and 960), and three learning rates give 270 fits. These are new generated systems within familiar families, not new-family transfer or real biology. Both budgets restart from identical model initializations and use the same 2/4/8 rollout curriculum apportioned over their own budget. A checkpoint is selected every 20 updates, so the long-budget condition also has more validation opportunities; the budget comparison includes that effect.

Use 32 training, eight validation, and sixteen held-out trajectories with separate seeds. The same initial states, inputs, targets, loss variance, batch random stream, learning-rate set, optimizer settings, and observation horizons are shared across models within a case. Score the full 48-interval held-out forecast only after model/checkpoint selection on the twelve-interval validation forecast. No adaptation labels, structural repair, or change episodes are introduced.

Report all systems, all trials, normalized errors, trained parameter counts, elapsed times, and predicted arrays. Compare each substrate with its capacity-matched neural model and report the original affine control as context. Do not choose the winner from held-out errors and call it a confirmed architecture. Selection for the next study must use validation performance, with held-out results used to audit that choice.

Before executing, commit source, configuration, and this protocol. Record source fingerprints. Do not change these models or budgets after observing this round's held-out results. Any subsequent change requires a dated protocol and fresh system seeds. No significance claim from three development systems per family.
