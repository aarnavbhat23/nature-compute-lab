# Focused architecture novelty audit

**Date:** October 15, 2026. **Authorization:** the user explicitly requested both biology-led research and a focused closest-method comparison during this round. Earlier brainstorming deliberately omitted this comparison. This is a scoped literature audit, not an exhaustive priority, patent, or freedom-to-operate search.

**Follow-up:** [CLOSED_LOOP_PRECEDENTS.md](CLOSED_LOOP_PRECEDENTS.md) identifies five additional primary precedents for biological hypothesis generation, model revision, experiment selection, and revision records. The broad self-revising scientific loop already exists; our exact learned typed repair procedure remains a research hypothesis.

## Finding

The current components do **not** establish a previously unknown category of AI. Learned reaction dynamics, fractional-occupancy regulation, regeneration, capacity expansion, constraint-preserving computation, and adaptation to changed systems all have substantial precedents.

We can build an original research implementation and potentially make a publishable contribution. Its contribution must be a precisely different mechanism, guarantee, learning procedure, evaluation, or demonstrated capability. Calling compartments “cells” or a correction “regeneration” is insufficient. None of the current experiments establish ASI or general biological intelligence.

## Closest methods and the consequences for our claims

| Primary work | Relevant overlap | Difference that remains to investigate | Claim we must not make |
|---|---|---|---|
| [Ji & Deng, Chemical Reaction Neural Network, 2021](https://pubs.acs.org/doi/10.1021/acs.jpca.0c09316) / [author preprint](https://arxiv.org/abs/2002.09062) | Learns reaction pathways/kinetics from concentration trajectories using physically structured rate laws. | Our rate parameterization is different and our intended task includes post-change structural repair. That joint behavior is unimplemented. | “Reaction-based learning from biological dynamics is new.” |
| [Döppel & Votsmeier, atom-conserving CRNN, 2024](https://doi.org/10.1016/j.proci.2024.105507) | Enforces atom conservation while learning reaction mechanisms through a dedicated balance layer. | Our exchange step is a narrower known transport construction; investigate whether a learned edit policy across heterogeneous module types adds anything. | “Conservation-preserving learned reaction discovery is new.” |
| [Mjolsness, cooperative quasi-equilibrium transcription models, 2007](https://pubmed.ncbi.nlm.nih.gov/17636856/) | Derives regulation models from statistical mechanics and identifies conditions for artificial-neural-network approximations. | Our normalized log-affinity rates are a particular phenomenological choice. No distinctive capability has been established. | “Thermodynamic state occupancy is an entirely separate, unprecedented alternative to neural computation.” |
| [Mordvintsev et al., Growing Neural Cellular Automata, 2020](https://distill.pub/2020/growing-ca/) | Learns local state updates capable of growth, persistence, and regeneration. | Their morphogenesis setting differs from forecasting a changed measured system under a limited new-observation budget. | “A learned model that grows or regenerates has never been built.” |
| [Rackauckas et al., Universal Differential Equations, 2020 preprint](https://arxiv.org/abs/2001.04385) | Combines known scientific dynamics and learnable components, including constraints, delays, and stochastic terms. | The proposed typed edit grammar and cross-system repair policy need independent justification beyond assembling a hybrid model. | “Combining biological equations with learned unknown dynamics is new.” |
| [Hoedt et al., MC-LSTM, ICML 2021](https://proceedings.mlr.press/v139/hoedt21a.html) | Uses hard conservation constraints in learned memory dynamics. | Different update equations and intended scientific semantics do not alone establish a better or novel capability. | “Conserved quantities in a learned computational architecture are new.” |
| [Papadopoulos & Guichard, MaCE, 2025](https://arxiv.org/abs/2507.12306) / [author-hosted paper](https://infoscience.epfl.ch/bitstreams/3b3aac9a-43f7-4012-9d76-132ba8a3788c/download) | Adds mass-conserving local dynamics to cellular automata, including neural variants. | Our typed physical pools are not the same as conserved abstract lattice activation; that semantic difference requires a task-level benefit. | “Local nature-inspired cellular computation plus conservation is new.” |
| [Yoon et al., Dynamically Expandable Networks, ICLR 2018](https://arxiv.org/abs/1708.01547) | Selective retraining, network expansion, and mechanisms to limit drift across tasks. | We propose reaction-graph edits subject to physical-type constraints and explicit measurement budgets. | “Learning when to grow or split computational units is new.” |
| [Kirchmeyer et al., CoDA, ICML 2022](https://proceedings.mlr.press/v162/kirchmeyer22a.html) | Learns context-conditioned dynamics for sample-efficient adaptation across systems. | Our proposed structural rather than solely contextual adaptation needs a measured advantage under equal resources. | “Learning to adapt dynamical models to new systems with few observations is new.” |
| [Chen, Chen & Wu, CEBoosting, 2023](https://arxiv.org/abs/2304.07863) / [author code](https://github.com/AIMS-Madison/CEBoosting) | Detects regime switches and rectifies sparse dynamical models from sequential data. | We must test what learned primitives, constraint-preserving edits, or transferred repair decisions add beyond an existing library-based switch/repair method. | “Detecting changed mechanisms and sparsely repairing a dynamics model is new.” |
| [Mouli, Alam & Ribeiro, MetaPhysiCa, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f0e5cde3850e7dd0db125c0ebae16680-Abstract-Conference.html) | Meta-learning causal structure for OOD dynamical forecasting. | Our proposed edit semantics and observation/retention tradeoffs require a direct method-level comparison. | “Meta-learning transferable mechanisms for dynamical prediction is new.” |
| [Srivastava, HyperODE, August 1, 2026 preprint](https://arxiv.org/abs/2608.00852) | Uses structured compartmental systems for cross-structure trajectory prediction/inference. | A supplied model-structure surrogate differs from learning which structural edit is needed after new observations. This distinction does not prove priority. | “A model that generalizes across different compartmental structures is new.” |
| [Escrig Molina & Probst, Neural Petri Flow, October 6, 2026 preprint](https://arxiv.org/abs/2610.08750) | Hard-wires local firing semantics, conservation, and enabling while learning rate laws/readouts. | Their reaction-prediction/valence setting is distinct from our proposed time-series repair policy, but the invariant-by-construction idea strongly overlaps. | “Typed local reaction dynamics with hard semantic guarantees are new.” |

These are original papers, proceedings, author preprints, and author code. The 2026 preprints' existence and claims were checked at official arXiv records; their reported performance is not independently verified here and is not used as an experimental result for our system.

## Algebraic overlap in our competing-state model

Our score has the form

    s_m = b_m + sum_j W_mj log(x_j + epsilon).

Consequently

    exp(s_m) = exp(b_m) * product_j (x_j + epsilon)^W_mj.

After normalization, this is a ratio of generalized positive monomials, related to fractional-occupancy/statistical-mechanical models. It is not evidence that we discovered new mathematics. Unrestricted exponents also prevent interpreting every learned coefficient as literal binding stoichiometry. The useful empirical question is what this particular rate representation buys under the stated task and budget.

## The narrower research lead

Investigate **measurement-limited structural repair of a typed dynamical model**:

- The model distinguishes measured conserved pools, bounded regulatory variables, abstract memory, and explicit environmental ports.
- Every proposed edit carries a checkable type/invariant condition; hidden computational state does not invent physical mass.
- A policy trained across systems chooses between parameter correction, a new interaction, an additional latent context variable, observation recalibration, or no edit.
- Decisions use actually revealed observations and are evaluated for future forecasting, new-measurement cost, retention, and total candidate-search cost.
- Structurally different explanations that fit available data remain explicitly unresolved; a permitted discriminating measurement can be proposed in a later active-measurement extension.

**Status:** a concrete research hypothesis with substantial neighboring work. This search did not establish that the exact joint formulation is new. Its individual pieces have precedents, and a combination is not automatically a contribution. A reviewer could reasonably regard the first implementation as a structured dynamics learner plus model selection unless we show a distinct algorithmic or empirical result.

## What a defensible contribution would require

1. Specify the edit-policy training algorithm and invariant-preserving edit grammar completely.
2. Compare with constrained reaction learners, context adaptation, sparse switch discovery, and generic expansion/retention controls, giving them equal observation and tuning budgets.
3. Hold out mechanisms and system families; do not count more development seeds from the same three generators as broader biological transfer.
4. Demonstrate that learned edit selection beats nonlearned and random selection from the same candidates.
5. Demonstrate that invariants improve the relevant prediction/adaptation tradeoff against a conventional model given the same constraint knowledge; a mathematical guarantee alone is not an accuracy win.
6. Publish failures, costs, and ambiguity. Real biological validation remains separate.

If these tests fail, retain the simpler model or narrower engineering contribution. Do not keep adding biological labels until the implementation sounds unprecedented.

## Access and verification limits

MWC equations were inspected in the primary-paper scan; the biology audit separately records abstract-only sources. For closest methods, the audit inspected official abstracts/primary-page extracts and selected mathematical descriptions. Some publisher/OpenReview full-text fetches failed; author preprints and indexed primary-paper text were used where available. This is enough to reject broad first-of-its-kind claims, not enough to certify the novelty of the exact complete method. We have not reproduced these methods' published results.

## Search log: representative exact queries

- `chemical reaction neural network autonomous discovery reaction pathways interpretable Ji Deng 2021`
- `Growing Neural Cellular Automata 2020 Mordvintsev Randazzo Niklasson regeneration`
- `learning reaction diffusion equations neural cellular automata differentiable morphogenesis`
- `neural networks thermodynamic models state occupancy partition function gene regulation learning`
- `"neural cellular automata" "conservation"`
- `"neural" "reaction" "network" "online" "structure" learning`
- `"mass conserving" "neural" "differential"`
- `"constraint preserving" "structural" "learning" neural network`
- `"chemical reaction neural network" "atom" "conserving"`
- `"chemical reaction neural network" "continual"`
- `"online" "reaction network" "discovery" "learning"`
- `"dynamically expandable" "network" "continual learning" 2018`
- `CoDA Context-Informed Dynamics Adaptation 2022 paper`
- `MetaPhysiCa improving generalization machine learning dynamical systems 2024`
- `universal differential equations scientific machine learning Rackauckas 2020`
- `"online" "sparse identification" "dynamical" "change"`
- `"Lifelong Learning with Dynamically Expandable Networks" arxiv`
- `"MetaPhysiCa" ICLR 2024 arxiv`
- `"MaCE: General Mass Conserving Dynamics for CAs"`
- `"CEBoosting" "Causation Entropy" arxiv`

A search returning no exact phrase match would not establish an unstudied idea. This audit found multiple close matches and narrows the contribution accordingly.
