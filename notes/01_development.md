# A local developmental program

The executable abstraction is a compact local instruction that repeatedly creates a new computational stage. Its transfer function is learned from noisy transitions in short cascades. Greedy residual-guided growth proposes terms, and separate validation cascades choose program size.

Test cascades are four times the training depth. Seed amplitudes have the same distribution; this tests depth transfer, not amplitude extrapolation. The true nonlinearity is in the supplied grammar, an explicit favorable prior.

Controls share the local-transfer assumption: a linear transfer and a regularized full dictionary. Their rollouts have the same depth and inputs. The full dictionary is a serious control, not an intentionally size-limited global network.

This is program growth/repetition, not a learned cellular morphogenesis process. It does not learn morphology, connectivity, or new primitive operations. A useful next test is transfer to branching structures and a generator absent from the supplied grammar.

Initial five-seed run: mean rollout MSE was 3.70e-5 for the grown program, 1.60e-2 for shared linear transfer, and 4.45e-5 for the fixed full dictionary. The nonlinear grammar matters here; these measurements do not demonstrate a reliable advantage over the full dictionary. The selected program used 5 of 6 available terms on average, so compression is modest.
