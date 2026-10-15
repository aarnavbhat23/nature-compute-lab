# Conditional rules as catalysts

A catalyst is an executable feature with explicit participants and an activation law. Nonnegative rates combine production and consumption channels. Residual-guided proposals add channels; held-out validation ingredients choose the retained set. The learned object is a list of reactions and their rates, rather than a dense hidden-layer network.

Training and validation cover `a+b < 1.1`; test ingredients cover `a+b > 1.5`. The true rules are present in the supplied library. Test errors use noiseless generator expectations; noisy observed targets are used for fitting. The full-library nonnegative model gets exactly the same training data.

The rule F1 is a synthetic diagnostic relative to a planted dictionary representation. It does not establish biochemical causality or unique mechanistic identification: correlated rules can make similar predictions. This prototype selects known rules; it does not invent new reaction primitives or implement physical chemistry.

Initial five-seed run: mean test MSE 3.97e-5 for selected catalysts, 1.58e-2 for the linear model, and 7.51e-5 for the full chemistry. Planted-rule F1 averaged 0.897, despite strong prediction. This illustrates that good outputs do not imply exact recovery of the generating explanation. The apparent improvement over the full dictionary needs testing with tuned regularization and a misspecified library.
