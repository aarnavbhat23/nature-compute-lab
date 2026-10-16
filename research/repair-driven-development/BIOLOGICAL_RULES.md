# Biological rules worth turning into computation

Research round: October 15, 2026. This is a primary-biological-literature audit and a set of engineering inferences, not a search of closest AI precedents. Novelty relative to existing AI architectures remains unverified, in accordance with the user's earlier request not to center brainstorming on those precedents.

The design goal is to turn a biological rule into an explicit invariant, state variable, transition, learning action, and falsifiable prediction. Nature is not one algorithm; mechanisms that help one task can be harmful or irrelevant on another.

## 1. Alternative molecular states and allostery

**Source:** Monod, Wyman and Changeux, *On the Nature of Allosteric Transitions: A Plausible Model*, J. Mol. Biol. (1965), [DOI](https://doi.org/10.1016/S0022-2836(65)80285-6), [author-paper scan](https://math.nyu.edu/~peskin/modsim_lecture_notes/monod_wyman_changeaux_allosteric.pdf). Inspected the model assumptions and state/saturation equations, pages 2–5 of the scan.

**Biological result/model:** alternative conformational states have differing ligand affinities; ligand binding shifts state occupancy and can alter regulatory responses. This is a specified physical model with assumptions, not a universal account of regulation.

**Engineering inference:** let alternative computational states compete through normalized occupancies, then generate rates from their occupancy-weighted activities. This provides context-sensitive combinations and a finite local capacity. The current learned log-affinity candidate is only phenomenologically inspired by the model. Its unrestricted real exponents are not molecular stoichiometries or measured binding energies.

**Test:** compare forecasting and transfer with a capacity-matched neural rate control and the original independent-gate substrate. Check normalization and rate bounds separately from accuracy. A softmax expression does not by itself establish architectural novelty.

## 2. Finite cellular resources

**Source:** Scott et al., *Interdependence of cell growth and gene expression: origins and consequences*, Science (2010), [DOI](https://doi.org/10.1126/science.1192588), [primary-paper abstract](https://pubmed.ncbi.nlm.nih.gov/21097934/). Abstract inspected; not a full reanalysis of the experiments.

**Biological finding:** physiological constraints on resource allocation link protein synthesis, growth, and expression burden. These relationships are conditional empirical descriptions of bacterial physiology.

**Engineering inference:** computational capacity should be explicitly budgeted. Extra modules, repeated candidate fitting, and longer internal simulations should be charged, and interacting modules may compete for a shared capacity. A normalization constant is an allocation constraint, not measured cellular energy or carbon accounting.

**Test:** perturb a shared capacity and measure whether the model predicts coupled effects in independently regulated components. Compare an uncoupled model and an equally constrained generic control. Do not claim faster or more efficient learning from fewer counted gates alone.

## 3. Resource-mediated interactions can masquerade as regulation

**Source:** Borkowski et al., *Cell-free prediction of protein expression costs for growing cells*, Nature Communications (2018), [paper](https://www.nature.com/articles/s41467-018-03970-x). Inspected the article's description of capacity measurements and expression burden; no experimental instructions or sequences are reproduced.

**Biological finding:** a construct can change another expression readout by competing for cellular resources. Direct regulation is not the only explanation of observed coupling.

**Engineering inference:** include shared environmental/resource variables when the data support them, and distinguish a direct interaction from a hidden common constraint. An edge inferred from correlated outputs is not a discovered causal biochemical interaction.

**Test:** use synthetic systems in which direct-coupling and shared-resource explanations fit the same observational data. Score predictions on discriminating interventions; permit an explicit unresolved-identifiability result when observations cannot distinguish them.

## 4. Interconnection changes component behavior

**Source:** Del Vecchio, Ninfa and Sontag, *Modular cell biology: retroactivity and insulation*, Molecular Systems Biology (2008), [DOI/paper](https://doi.org/10.1038/msb4100204). Abstract and model framing inspected.

**Biological/modeling principle:** a connected downstream system can load an upstream module, changing its behavior. Independent components need not retain their isolated input/output response after composition.

**Engineering inference:** compositional modules should expose both forward effects and loading/resource effects. A computational module is not inherently reusable merely because its isolated forecast is accurate.

**Test:** learn two modules separately, connect them under different loads, and evaluate joint trajectories. Compare a one-way composition with an explicitly bidirectional or shared-resource formulation. This is a proposed new benchmark, not a result from the current pilot.

## 5. Feedback makes regulation possible under specified conditions

**Source:** Aoki et al., *A universal biomolecular integral feedback controller for robust perfect adaptation*, Nature (2019), [paper](https://www.nature.com/articles/s41586-019-1321-1). Abstract and extended-data descriptions inspected; full subscription text was not accessible through the publisher preview.

**Biological result:** a biomolecular integral-feedback controller was engineered and its adaptation properties demonstrated. Guarantees belong to the controller assumptions and regulated quantities; imperfections such as dilution and saturation affect performance.

**Engineering inference:** distinguish fast prediction state from slower feedback-driven parameter or structure changes. A residual accumulator can summarize persistent error, but needs an actually measured outcome or another defensible supervision source. Internal consistency alone cannot tell a predictor whether an unobserved forecast is correct.

**Test:** compare adaptation after real feedback with frozen prediction and with parameter-only adaptation. Test noise-only shifts and absent feedback to detect unnecessary editing or imagined error signals.

## 6. Robustness can depend on network structure

**Source:** Araujo and Liotta, *The topological requirements for robust perfect adaptation in networks of any size*, Nature Communications (2018), [paper](https://www.nature.com/articles/s41467-018-04151-6). Abstract and introduction inspected.

**Biological/mathematical result:** under the study's modeling conditions, adaptation-capable networks admit a structured modular decomposition. This does not imply that every biological task or model should use the same motifs.

**Engineering inference:** useful properties can be enforced by allowable structural edits rather than encouraged solely with a soft error penalty. For example, edits to a known closed exchange pool should not create or destroy its total mass.

**Test:** verify the chosen invariant after every allowed edit, and compare with a generic model supplied the same constraint. An invariant guaranteed by construction is a valuable property, but not evidence of superior learning by itself.

## 7. Error rejection consumes resources

**Source:** Hopfield, *Kinetic proofreading: a new mechanism for reducing errors in biosynthetic processes requiring high specificity*, PNAS (1974), [DOI](https://doi.org/10.1073/pnas.71.10.4135), [primary-paper abstract](https://pubmed.ncbi.nlm.nih.gov/4530290/). Abstract inspected.

**Biological/modeling principle:** an actively driven kinetic pathway can improve discrimination. Biological accuracy is tied to specific mechanisms and expenditure, not arbitrary randomness that creates knowledge.

**Engineering inference:** a structural edit may need staged screening before consolidation. Charge the extra simulations, measurements, and fitting. Do not describe repeated self-evaluation as an independent source of evidence.

**Test:** compare one-stage versus staged candidate screening under equal total computation and measurement budgets. Defer this feature until the simpler repair loop works; the paper does not demonstrate that it improves a forecasting architecture.

## 8. Regeneration depends on context

**Source:** Owlarn et al., *Generic wound signals initiate regeneration in missing-tissue contexts*, Nature Communications (2017), [paper](https://www.nature.com/articles/s41467-017-02338-x). Abstract available through search; publisher full-page retrieval failed.

**Biological finding:** wound signals can have different regenerative consequences depending on tissue context and missing structures.

**Engineering inference:** permit no-edit decisions. A residual may reflect observation noise, unmeasured input, or an inadequate representation; it should not automatically trigger growth. This analogy does not supply a validated repair algorithm.

**Test:** evaluate no-change, noise-only, sparse-change, and dense-change episodes separately. Count false repairs and retained-task deterioration.

## Design synthesis and priorities

| Rule | Computational consequence | Current status |
|---|---|---|
| Alternative-state competition | Normalized local state occupancies and bounded rate capacity | Implemented; fresh development comparison frozen before evaluation |
| Resource limits | Explicit parameter, fitting, simulation, and observation budgets | Implemented in pilot budgets; no cellular energy model |
| Context-dependent interconnection | Shared resource/context variables and loading-aware ports | Proposed; not evaluated |
| Feedback | Error-guided correction after actual measurements | Parameter fitting implemented; structural repair unimplemented |
| Structural invariants | Conservative exchange blocks for genuinely closed pools | Separate mathematical/component study; not a general reaction learner |
| Costly rejection | Candidate screening with measured total cost | Deferred |
| Contextual regeneration | Growth is conditional; no-edit is a valid action | Proposed repair evaluation |

The strongest architectural research direction is a typed dynamical substrate: represent quantities according to their physical or computational role, learn local reaction/occupancy laws, preserve only justified invariants, and allow feedback to propose budgeted structural changes. This is a research recommendation, not an experimentally validated complete architecture. Forecasting evidence must decide which implemented rate representation survives. Do not attach every biological analogy to one model simply to make it sound more biological.

## Independent review option

No Claude Science tool or connector is exposed in this session's tool catalog, and no Claude review has occurred. The user offered it as an optional independent check. A review prompt can be prepared for the point where a formal or biological claim needs external scrutiny; no consensus should be claimed from an unperformed review. The present source studies and experiments can proceed without that dependency.
