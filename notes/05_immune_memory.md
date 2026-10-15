# Keep a repertoire of explanations

A repertoire retains frozen local response laws. Recent labeled prediction errors determine which law is recalled. Sustained mismatch can create a new explanation fit to a recent window; successful old explanations remain available. This abstracts diversity, selection, and memory, without implementing actual immune cells or a full clonal-selection process.

The stream visits three hidden regimes, including returns. Regime identities are available only to the scorer. Every method first predicts and then receives that example's label. Cumulative and exponentially forgetting online regressions have the same warmup and observe every same subsequent label. No method sees the current target before predicting it.

Window sizes and thresholds are fixed, not selected from test-stream results. The task favors recurring linear-in-features rules; selection may lag at switches, and keeping old rules consumes extra memory. Compare whole-stream performance as well as the first 20 examples after a previously seen regime returns. Those return-window scores use known simulator boundaries only for evaluation.
