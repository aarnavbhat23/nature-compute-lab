# Repair a damaged predictor

The initial model is fit on one simulated response map. A previously absent interaction becomes active. The repair learner proposes a sparse residual correction and selects patch size on independent adaptation validation data. It retains the original predictor and adds the patch.

All adapting methods get 12 fit labels and 12 validation labels. Both full-model controls tune regularization on that same validation set. The full patch retains the same old-model prior as sparse repair, avoiding an unfair comparison only to a model trained from scratch. The frozen control receives no updates.

Independent test examples come from the changed system and are evaluated only after adaptation. The changed interaction is already a candidate in the human-specified dictionary. This is a sparse parametric patch, not discovery of an unrestricted new mechanism, autonomous biological healing, or a general self-repairing computer.
