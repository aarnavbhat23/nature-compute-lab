# Optional independent Claude Science review

**Status:** prompt prepared; not submitted. No Claude response, agreement, or independent validation is claimed. This session exposes no Claude Science connector. The review is optional and is not needed to reproduce the current experiments.

The following can be pasted into Claude Science with the public repository/branch. Only public research material is referenced.

---

Act as a skeptical quantitative biology and dynamical-systems reviewer. Review this public research branch and its draft PR:

https://github.com/aarnavbhat23/nature-compute-lab/pull/1

Read the current branch versions of:

- `research/repair-driven-development/BIOLOGICAL_RULES.md`
- `research/repair-driven-development/NATURE_DIAGNOSTIC.md`
- `research/repair-driven-development/WALLTIME_PROTOCOL.md`
- `research/repair-driven-development/CONSERVATIVE_COMPONENT.md`
- the latest architecture decision and results reports linked from the research README
- `drs/models.py`, `drs/occupancy.py`, and `components/exchange.py`

This work does not claim validated novelty, ASI, physical free-energy identification, or real-data biological validity. It includes synthetic development experiments, not confirmatory results. Please do not infer biological identity for learned latent states. Do not treat a second AI's agreement as experimental evidence.

Answer these questions:

1. Which biological principles are represented faithfully, which are only loose analogies, and which assumptions would fail in open, growing, partially observed biological systems?
2. Check the bounded update, competing-state normalization/capacity, and the conservative exchange step. Verify the state/rate reverse-mode derivatives and state clearly where state-dependent rates require an additional chain-rule term.
3. Is the distinction between conserved measured amounts, bounded regulatory activity, and abstract memory defensible? What measurements or prior knowledge are required to assign these types without inventing constraints?
4. Do the fixed-update and fixed-wall-time comparisons actually support the interpretation in the results report? Look for hidden hyperparameter selection, incompatible resource accounting, data leakage, unidentifiability, and conclusions that exceed three seeds per family.
5. Propose one small experiment that could most efficiently falsify the architecture's central research claim. Specify the competing explanations and an outcome that would make you reject the idea.
6. Give your strongest disagreement with the proposed next step. Identify an equally plausible simpler model or explanation.
7. Check the specific overlap and remaining distinctions in `NOVELTY_AUDIT.md`. The user has now authorized a focused closest-method comparison. Identify any claimed distinction that is already covered by a cited method; provide a verifiable primary reference instead of asserting novelty or relying on another AI summary.

Use primary biological or mathematical sources where necessary. Separate source-supported facts, mathematical deductions, and speculative suggestions. Do not invent references or imply that this architecture is unprecedented. A focused closest-method novelty survey is now authorized; distinguish an unverified joint formulation from a demonstrated novel contribution.

Return a concise review with: fatal issues, repairable issues, claims supported by current evidence, claims unsupported, and the single proposed falsification experiment.

---

## How to use a returned review

Record Claude's exact model/product identity, date, prompt, and returned text. Check cited sources independently. Create a disagreement ledger with each claim, evidence on each side, and a proposed resolving test. Do not silently replace the architecture with a suggested alternative or report consensus without resolving substantive disagreements. Review feedback must not be used to reinterpret old held-out data as a fresh confirmatory test.
