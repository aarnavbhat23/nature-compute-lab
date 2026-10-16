# Has the self-revising biological world-model idea been done?

**Research date:** October 15, 2026. This follow-up checks the specific claim in the conversation: a biological model that predicts interventions, revises mechanisms after contradictory evidence, and chooses experiments to distinguish explanations. It extends the [earlier 13-method architecture audit](NOVELTY_AUDIT.md).

## Answer

**The broad concept has already been done.** Automated biological hypothesis generation, biological model revision, and experiment selection have explicit primary precedents. We must not describe this loop as a previously unknown type of intelligence. The exact proposed learned repair policy over a typed quantitative dynamics program remains unspecified and unverified as a novel contribution. Not finding an exact implementation match would not prove novelty.

The appropriate response is to make a narrower research contribution testable. We can build our own implementation and compare algorithms, while attributing the scientific loop and constituent methods correctly. ASI does not follow from this architecture, even if its next experiment succeeds.

## Stronger precedents found in this follow-up

| Primary source | What it covers | Consequence for our proposal |
|---|---|---|
| [King et al., Robot Scientist, Nature 2004](https://www.nature.com/articles/nature02236), DOI 10.1038/nature02236 | A robotic biological system generates hypotheses, selects and executes experiments, interprets observations, rejects inconsistent hypotheses, and repeats. It uses a logical pathway model for yeast gene function. | The overall hypothesis → experiment → evidence → revised explanation loop is established. Different model representations could still be useful, but do not make the loop new. |
| [Mélykúti et al., BMC Systems Biology 2010](https://link.springer.com/article/10.1186/1752-0509-4-38), DOI 10.1186/1752-0509-4-38 | Designs perturbations to separate competing biochemical dynamical models that fit previous data. It treats initial conditions, input profiles, and parameter changes as possible experimental choices. | Our suggested experiment to distinguish direct regulation from shared-resource coupling is an instance of a known model-discrimination problem. Disagreement-based intervention choice alone is not a novelty claim. |
| [Gower et al., LGEM+, Discovery Science 2023](https://doi.org/10.1007/978-3-031-45275-8_42), [author preprint](https://arxiv.org/abs/2306.06065), [institutional full paper](https://research.chalmers.se/publication/538006/file/538006_Fulltext.pdf) | Represents compartmentalized metabolism in first-order logic, abducts candidate model improvements, and evaluates hypotheses using gene-essentiality prediction and flux-balance constraints. The paper describes intended subsequent experimental testing, not completion of every proposed test. | Semantic biological models with automated theory repair already exist. Quantitative learned transient dynamics and transferred edit decisions are possible differences to evaluate, not proof of priority. |
| [Tiukova et al., Genesis, 2024 preprint](https://arxiv.org/abs/2408.10689), [full author text](https://arxiv.org/html/2408.10689v1) | Describes a systems-biology robot-scientist project combining model improvement, experimental infrastructure, structured knowledge, revision records, and LGEM+. Some capacities are project goals rather than completed demonstrated performance. | Even the system-level integration of revision, experiments, and biological models is a close precedent. We must compare the specific algorithm, information, task, and costs rather than claim the assembled vision is first. |
| [Kronström et al., RIMBO, 2023](https://link.springer.com/chapter/10.1007/978-3-031-45275-8_35) | An ontology to describe and explain biological-model changes for revision databases. | An auditable record of scientific edits is valuable engineering but is not itself a new learning architecture. |

These findings substantially strengthen the overlap already identified for constrained reaction learning, context adaptation, sparse model correction, and expandable networks. The cited systems are not interchangeable with our prototype; equally, differences in application or terminology cannot establish that our version is novel.

## What could still be a useful research question?

Ask whether a **repair policy learned across systems** can choose valid structural modifications to a **quantitative, typed dynamical model**, using fewer new scalar measurements or less candidate-search computation than equally informed parameter adaptation, conventional model selection, sparse repair, and active model discrimination.

Specify before claiming a contribution:

1. An explicit program representation and edit grammar, including assumptions that make each physical constraint valid.
2. The policy inputs, training distribution, supervision, legal actions, search budget, and stop/no-edit rule.
3. The role of unresolved explanations: multiple fitting mechanisms must remain possible when data cannot distinguish them.
4. A common experimental-information budget, including initial conditions, measured pools, known interventions, validation labels, and any active queries.
5. Tests on withheld mechanisms, retention, false structural changes, noise-only shifts, and real measurements.

A nonlearned search over models would be a baseline implementation of known ideas. A transferred policy that improves the error/measurement/cost tradeoff could be a contribution if the novelty audit and direct comparisons survive further scrutiny. A combination of known ingredients is not automatically a new type of AI.

## Immediate engineering step

Proceed with the [fixed coupled typed-model pilot](TYPED_RESOURCE_PROTOCOL.md): a closed resource pool, bounded regulation, and abstract memory, fitted jointly with complete gradients. Both rate representations receive the same valid conservation/topology information. This prerequisite can fail and is not claimed as the full repair model. Only afterward specify the edit-vs-parameter adaptation experiment.

## Search and access record

Representative queries:

- `robot scientist Adam 2009 functional genomics hypotheses experiments Science King`
- `automated model revision systems biology robot scientist mechanistic models experiments`
- `active learning biochemical network structure discovery intervention model discrimination original paper`
- `"LGEM+" model revision metabolic models abductive learning`
- `"Genesis" "model revision" "learning" biology 2025 2026`
- `"Discriminating between rival biochemical network models" authors 2010`

Inspected official primary abstracts, the 2010 full publisher article's background/problem formulation, the institutional LGEM+ paper and its method descriptions, Genesis's author full text, and the RIMBO publisher page. Robot Scientist's accessible publisher abstract was inspected; no subscription-only experiments were independently reanalyzed. Existing published performance is not reproduced here. Search result snippets from secondary explainers and unverified generated-looking pages were not used as evidence. This is a focused research audit, not an exhaustive patent or priority search.
