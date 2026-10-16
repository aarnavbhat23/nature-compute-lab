# Paired validation-horizon diagnostic

Written October 15, 2026, after the equal-wall-time follow-up. That follow-up shows that twelve-interval validation ranking can disagree with forty-eight-interval held-out ranking. It does not by itself establish why. This paired study holds fitted candidate trajectories fixed and changes only how the checkpoint/learning rate is selected.

Use new system seeds 1009,1103,1201 for the same three familiar families. Training: 32 trajectories, eight intervals; validation: eight trajectories, forty-eight intervals; independent held-out evaluation: sixteen trajectories, forty-eight intervals. Preserve the existing observation interval, noise-free/fully observed inputs, variance floor, integrator, parameter counts, three learning rates, batch size, Adam, and 2/4/8 curriculum. Fit original, occupancy, neural_730, and neural_808 for 960 updates per learning rate (108 fits total).

Save/evaluate a candidate every twenty updates. Two selectors choose from the *same* fitted candidate set: one minimizes error over the first twelve validation intervals; the other over all forty-eight. Each then forecasts the independent held-out trajectories. The long selector receives four times as many validation target values. This is a validation-information effect, not free data or a change to the model's architecture/training trajectory. Do not present one selector as receiving the same label budget as the other.

Report paired changes per system and family, every model, both selectors, all candidate validation scores, chosen checkpoints, weights, and forecasts. No test-based best selector or method is retrospectively called confirmed. A longer selector may help, hurt, or make little difference. A negative result limits the horizon-mismatch explanation.

This study is adaptive development after previous outcomes, not unseen-family transfer, an ASI benchmark, a real-data experiment, or a new-architecture validation. Any next architecture decision must acknowledge that these three functional families have guided development. Freeze source/configuration/protocol before computing held-out results; preserve all earlier rounds.
