# Store history in switching regimes

The learner estimates two hysteresis thresholds from noisy sequences. Training error shortlists threshold pairs and independent validation sequences choose one. Test predictions run closed-loop, retaining the model's own state. No previous test target is supplied.

Controls are a memoryless threshold, an autoregressive linear predictor, and a learned input-bin/previous-state transition table. The latter is a stronger stateful comparison. All see the same input and labeled training sequences. Test sequences are independent and twice as long.

The generator is itself hysteretic, so this is a favorable, explicitly matched inductive prior. Success would demonstrate learned switch memory in this family, not a general phase-based machine learning architecture. Oscillatory, multistable, partially observed, and non-hysteretic processes remain untested.

Local five-seed preview before the scheduled checkpoint: mean error rate was 0.00242 for learned hysteresis, 0.11365 for the memoryless threshold, 0.40716 for the simple autoregressive rule, and 0.00771 for the transition table. The stateful table is already competitive. This favorable matched-family result is not evidence that a new phase-based model family has been established.
