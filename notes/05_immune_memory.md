# Keep a repertoire of explanations

A repertoire retains frozen local response laws. Recent labeled prediction errors determine which law is recalled. Sustained mismatch can create a new explanation fit to a recent window; successful old explanations remain available. This abstracts diversity, selection, and memory, without implementing actual immune cells or a full clonal-selection process.

The stream visits three hidden regimes, including returns. Regime identities are available only to the scorer. Every method first predicts and then receives that example's label. Cumulative and exponentially forgetting online regressions have the same warmup and observe every same subsequent label. No method sees the current target before predicting it.

Window sizes and thresholds are fixed, not selected from test-stream results. The task favors recurring linear-in-features rules; selection may lag at switches, and keeping old rules consumes extra memory. Compare whole-stream performance as well as the first 20 examples after a previously seen regime returns. Those return-window scores use known simulator boundaries only for evaluation.

Initial five-seed result is negative against the strongest control. Whole-stream MSE: repertoire 0.0649, cumulative regression 0.2720, forgetting regression 0.0474. Return-window MSE: repertoire 0.5371 versus forgetting 0.4237. Retaining explanations does not compensate for this selector's delay. The repertoire retained 5.8 rules on average for only three true regimes. These results stay unchanged; a different selector would require a separately labeled follow-up and fresh evaluation streams.
