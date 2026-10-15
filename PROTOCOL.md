# Initial protocol

Written before the prototype results were inspected. This is an exploratory protocol, not an external preregistration.

## Common rules

- Seeds: 7, 17, 29, 43, 71. Each seed creates fresh training and test problems.
- Use train/validation/test partitions with no test-based hyperparameter choice.
- Report each seed, the mean and sample standard deviation. Five seeds are an initial sensitivity check, not a powered significance claim.
- No seed is discarded because its result is inconvenient.
- Report prediction error and resource costs where meaningful. No universal 95% target.
- Candidate programs and feature dictionaries are human-specified priors. Selecting from them is not discovering arbitrary mathematics.
- Controls get the same observations unless a row explicitly identifies an oracle/reference calculation.
- Time measurements are single-machine wall time, without a hardware-normalized speed claim.

## Scope by experiment

1. Development: fit a local scalar transfer from short simulated cascades; evaluate rollout over longer cascades. Track selected grammar terms and the growth trace. This abstracts development as repeated local program construction, not embryonic morphology.
2. Catalysts: recover a sparse combination of conditional reaction features. Separate train and test input regions; compare sparse selection to the entire dictionary. The generating functions are available in the dictionary, an explicit favorable prior.
3. Symbiosis: learn intermediate module readouts, propose pairwise compositions, retain partners using validation data, and freeze before test. Test interaction combinations withheld by input region. No claim of autonomous organism-level evolution.
4. Repair: learn an initial dynamical map, change one coefficient, provide a small labeled adaptation set, and score independent post-change data. All adapting controls see exactly that set. The original dictionary includes the changed mechanism.
5. Immune memory: retain and reuse local explanations in a streaming regime sequence. Predict before each label arrives. Never score a prediction after training on its own target. Compare to cumulative and forgetting regressions. Report recovery after switches and the cost of retained memory.
6. Stable regimes: learn on/off hysteresis thresholds from sequences. Score closed-loop sequences, with no teacher-forced previous test outputs. The hysteresis model is structurally matched to the generator; that advantage is disclosed.
7. Clocks: infer rates from observed relaxation pairs, then choose asynchronous update intervals on validation trajectories. Test different rates and durations through process-specific fitted rates. Compare with the same inferred kinetics evaluated densely. Count update events and measure state approximation error. This does not learn arbitrary reaction kinetics.

## What a result means

A positive result demonstrates that one executable abstraction can help on the particular synthetic task. It does not establish a novel general AI architecture, biological causality, clinical utility, or ASI. An equally good simpler control is an important null result.

## Deferred work

Real biological data, a shared cross-task benchmark, statistical power analysis, equalized tuning budgets, independent implementation review, learned primitive discovery, and transfer outside the supplied task families.
