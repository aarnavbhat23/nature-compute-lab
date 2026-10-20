# Nature as a source of learning primitives

## Scope and interpretation

This survey responds to the request for a successor-level learning idea: start with the operating rules of a natural system, then investigate whether those rules support a new computational learner. It screens **14 mechanism families**, follows the strongest leads into primary work, and records concrete overlaps instead of declaring a mechanism unexplored from a sparse keyword search. It is a targeted research audit, not an exhaustive systematic review or a patent search.

Three questions must remain separate:

1. **Does the natural system store information or adapt?** A persistent physical state can satisfy this without learning a general predictor.
2. **Has someone already turned the mechanism into computation or learning?** Using machine learning to analyze the natural system is a different direction of influence.
3. **What additional algorithmic claim could we test?** A gap in application coverage is not automatically a new learning principle.

The judgments below are our research inferences. The numbered references identify what the papers actually support; details and access levels are in [SOURCES.md](SOURCES.md). Candidate gaps are proposals, not statements that the entire field has neglected them.

## Fourteen systems screened

| Natural system | Candidate computational primitive | What already exists | Specific question still worth investigating | Decision for this project |
|---|---|---|---|---|
| **1. Yeast conformational memory and selective inheritance** | Experience changes functional availability by assembling components; barriers control propagation of persistent memory | Whi3 cellular memory, membrane-associated confinement, and context-dependent condensate interactions [S01–S03]. Sequestration classifiers and bistable genetic memory already exist [S04–S05]. | Can assembly membership and its selective propagation form a trainable, compositional predictive memory, with no neural controller and bounded storage? | **First discovery priority.** Good biological grounding; artificial learning rule and originality remain unresolved. |
| **2. Multicomponent condensation** | Collective phase selection computes a response; composition controls accessible states | Boltzmann-liquid learning [S06]; surface classification with hidden species and reprogrammable reservoir composition [S07] | Can a finite shared pool support online formation of useful compartments under conflicting histories, beyond existing energy-based learning and gating? | Useful comparator. Generic phase-based classification is already occupied territory. |
| **3. Molecular self-assembly and nucleation** | Collective assembly paths recognize inputs; kinetic accessibility influences which structure forms | Associative assembly theory and experimental pattern recognition [S08–S09]; programmed nonreciprocal shape changes [S10] | Can local assembly/disassembly events acquire predictive temporal rules and reuse subassemblies on unseen combinations, without a separately trained encoder? | **Second discovery priority.** Strong evidence for computation, substantial novelty risk. |
| **4. Cytoskeletal mechanochemistry** | Strain-sensitive binding and active forces remodel a mechanical substrate | Explicit learning through mechanosensitivity and motors [S11]; temporal contrastive learning [S12]; physical elastic-network learning [S13] | Can turnover and changing connectivity yield a demonstrably different learning operator rather than implement contrastive learning? | Do not present a motor/spring learner as our new family. Excellent reference mechanism. |
| **5. Slime-mould transport and habituation** | Flow modifies conductance; structure and chemical state retain history | Adaptive path-finding model [S14], habituation experiment [S15], tube-diameter memory study [S16]; reservoir-style anticipation [S17] | Can local transport adaptation learn arbitrary input-to-future mappings with explicit credit assignment? | Not a fresh broad architecture. Flow optimization is not general predictive learning. |
| **6. Dormancy and microbial seed banks** | Preserve states during inactivity and reactivate them under suitable conditions | Quantitative dormancy/seed-bank models [S18]; proposed electrochemical pulse integration [S19], with an experimental measurement challenge [S20] | Can selective reactivation improve retention per stored byte beyond a bounded model bank or multiscale memory? | Potential component, weak standalone claim to a new learner. Do not treat the disputed spore readout as settled. |
| **7. Ecological communities** | Distributed populations turn interactions and environmental history into a dynamical response | Ecological reservoir computation [S21] | Can the community itself acquire transferable predictive rules instead of relying on a fitted output map? | Natural computation already demonstrated; a new substrate alone fails our criterion. |
| **8. Fungal mycelium** | Growth and electrical dynamics provide a distributed state with memory | A published mycelial-growth framework aimed at reservoir computing [S22] | Can a well-specified intrinsic plasticity rule learn under component turnover and imperfect observation? | Interesting physical substrate; current project evidence does not justify a new model-family claim. |
| **9. Active nematic defects** | Defect creation, motion and interaction carry information through topology | Active-defect logic and nematic universal gates [S23–S24]; local updates for nonequilibrium control [S25] | Can experience modify a bounded set of defect pathways so the substrate learns unseen temporal relations, rather than executes programmed gates? | **Third, high-risk reserve.** Distinct physical representation; learning and implementation costs are substantial. |
| **10. Chromatin organization** | Accessibility and spatial organization determine which operations can interact | A chromatin/condensate-inspired DNA-computer architecture proposal [S26]; loop-extrusion-inspired optimization [S27] | Can accessibility changes provide new predictive credit assignment, rather than masks, variable grouping or memory gates? | Do not claim that chromatin-inspired architecture is untouched. Distinguish proposal from demonstrated general learning. |
| **11. Selective membranes and compartments** | Local rules and restricted exchange determine computation | Membrane/P systems with rule creation [S28] and a processor architecture [S29] | Can membrane formation itself be learned from experience with an advantage beyond existing rule systems and modular models? | Long-established computing paradigm; boundary learning is a narrower hypothesis. |
| **12. Kinetic proofreading** | Repeated, energy-consuming checks suppress premature commitment | Quantitative speed/error/dissipation theory [S30]; explicit ML analogy and classifier defense [S31] | Can a learned checking policy improve calibrated prediction per observation and computation budget? | Useful reliability mechanism; not a new broad learning family by itself. |
| **13. Driven matter and mechanical hysteresis** | Training history changes accessible physical states and responses | Low-rattling theory with experiments [S32]; cyclic-training-induced physical memory [S33] | When does physical adaptation improve held-out prediction rather than simply encode the forcing range? | Foundational research lead; adaptation alone does not establish a learner. |
| **14. Ciliary coordination** | Hydrodynamic interactions coordinate distributed oscillatory units | Reinforcement-learning-assisted coordination study and reduced physical model [S34] | Can native interactions acquire an input/output rule without an external RL controller? | Preliminary screen only. Coordination and optimization are not enough to infer a new learning algorithm. |

## Important nearby work that cuts across the table

**Chemistry can already learn without literal neurons.** Autonomous CRN-ensemble learning of generative distributions is described in [S35]. Chemical associative-learning agents are studied in [S36]. A 2026 preprint reports in-context learning in reaction networks without explicit attention [S37]. These are different projects, not one proof of general intelligence. Their existence rules out broad claims such as “the first chemical learner,” “the first non-neural learner,” or “the first attention-free chemical contextual inference.”

**Training is not the same as inference.** In the examined version of [S07], differentiable optimization adjusts interactions and reservoir parameters. In [S37], chemical dynamics implement contextual inference, but an outer supervised training process sets parameters. Those works matter even if our goal is to replace the outer learning mechanism too. “No attention at inference” does not establish “no conventional optimization during training.”

**Multiple memory timescales are not new.** Complex synaptic memory models already use coupled processes with different timescales [S38]. A protein-inspired variable with a slow decay constant cannot be our entire contribution. Chemical Hebbian learners and experimental DNA-network training further narrow the novelty space [S39–S40].

## Why prioritize conformational memory?

The choice follows the user's emphasis on a different way of learning. It does not follow from a measured performance ranking. The Whi3 line provides three experimentally motivated ingredients in one biological setting: experience-dependent assembly, persistence, and restricted inheritance [S01–S03]. We can ask whether those ingredients imply a useful *update operation*, rather than merely replacing a neuron's activation function.

The intended computational difference would be that experience changes the available repertoire of operations by redistributing components into persistent assemblies. Information would be stored partly in assembly membership and location. Predictive computation would then depend on the remaining functional repertoire and its interaction with new observations. Neither a chemical concentration nor a slow state is sufficient for originality; the joint update and reuse rules would have to contribute something specific.

There is also a serious weakness: the biological observations demonstrate particular forms of cellular memory and decision adjustment. They do **not** supply a general credit-assignment rule. Inventing such a rule is the central work, not a detail to hide inside a trainable controller. The [mathematical audit](SEQUESTRATION_LEARNING.md) makes this missing step explicit.

We found no verified artificial learner matching the complete proposed specification in this targeted search. Searches for Whi3 and “mnemon” were often dominated by biological studies, ML used to analyze those studies, or unrelated LLM-memory products. This gives only weak negative evidence. We therefore checked broader chemical learning, sequestration classifiers, complex synapses, and condensate computation. These are the substantive competitors, regardless of their names.

## Why keep the other two directions?

**Temporal molecular assembly** has a clearer path to a computational simulator and direct evidence for high-dimensional pattern recognition [S08–S10]. The research question is about acquiring and composing temporal rules within the assembly dynamics. A fixed assembly library trained by an external optimizer would reproduce an existing direction. A trace-based reinforcement rule could also reduce to known Hebbian or reinforcement learning. Both reductions must be tested before calling the model new.

**Active-defect learning** offers a more unusual state representation: information resides in configurations and allowed interactions of defects. Existing work already implements logic [S23–S24], and recent control learning narrows the gap [S25]. A novel defect-mediated learner would need an experience-dependent modification rule and reproducible generalization. Numerical fluid simulation may be far more expensive than the task it solves. This is a reserve, not the first implementation investment.

## Ranking without false precision

| Criterion | Conformational memory | Temporal assembly | Active defects |
|---|---|---|---|
| Grounding of the motivating natural mechanism | Strong for specific yeast memory phenomena | Strong for self-assembly; computation also demonstrated experimentally | Strong physics; logic demonstrated in model systems |
| Existing nearby computational work | Substantial; exact proposed learning combination unverified | Very substantial | Substantial for logic/control; general learner unresolved |
| Main unsolved piece | Credit assignment, useful assembly formation, selective transfer | Acquiring compositional temporal rules intrinsically | Learning rule and efficient representation/readout |
| Practical first environment | Small deterministic abstract reaction/compartment simulator | Coarse-grained assembly simulator | Reduced defect dynamics before full hydrodynamics |
| Largest novelty trap | Adaptive gates or complex synapses with renamed variables | Hopfield memory, temporal associations or RL on assembly paths | Programmed automaton or physical reservoir |
| Priority under this user's objective | First research bet | Second | High-risk reserve |

These are qualitative judgments, not probabilities of novelty or success. None has demonstrated an advantage on our biological forecasting tasks.

## What “new” should mean here

A software model necessarily uses existing mathematics and numerical tools. Being expressible as a state-space system, or simulatable by a sufficiently large neural network, does not make it algorithmically identical to that neural network. Conversely, changing physical vocabulary does not make an existing update rule novel.

The defensible claim would concern a specific representation/update/inference combination, an explanatory result about that combination, and evidence that the difference matters. We should be able to state it without mentioning the natural organism. For example: “A bounded population of reusable operators can acquire context-specific temporal predictions by changing higher-order persistent assembly membership through a local rule.” That is a **target claim**, not a result of this survey.

## Connection to biology and the ASI ambition

The first biological capability should be predicting how prior exposure changes a system's response to a later benign stimulus. This directly tests history and context, which are central to the proposed primitive. Sequence design, organism-wide simulation, and universal biological forecasting would be much later capabilities requiring different data and validation.

Unpredictability should be represented and measured, not added for its own sake. A deterministic learner can model some useful dynamics; uncertainty in observations, missing variables, and inherently random outcomes still requires calibrated predictive distributions or intervals. Exact prediction of every microscopic outcome is not a credible success criterion.

The sequence of evidence must be: distinct rule → demonstrated learning → unseen-system generalization → real-data validation → scaling evidence. Biological superhuman performance in one task would still not establish ASI. Nothing in the reviewed literature or our results provides an assured path from these mechanisms to general superintelligence.
