# Searching for a different learning primitive

Research date: October 15, 2026, America/Los_Angeles. Prepared with Codex assistance. This is a literature audit and research proposal, not a completed new architecture or a performance result.

## Decision

The objective is to discover a new way for a computer to learn biological behavior. A neural network with biological terminology, a different physical implementation of a known learner, or a simulator with a trained readout would not meet that objective on its own.

**Prioritize compartment-confined conformational memory as a learning-rule research direction.** The motivating natural system is protein assembly and selective memory inheritance in single yeast cells. The working hypothesis is that experience could modify a computational system by changing which components are available, which assemblies persist, and where those assemblies can propagate. Call this *learning by sequestration* for now; the phrase is descriptive, not a novelty claim or a product name.

This is our highest-priority **discovery bet under an originality-first objective**, not an evidence-based claim that it is the best-performing architecture. The biological mechanism is documented. A scalable, general learning algorithm derived from it is not established by this survey. We have not found a verified end-to-end implementation of the exact proposed combination in the sources examined, but the search does not establish that none exists.

The two alternatives worth retaining are **learning temporal pathways through molecular assembly** and **learning with the topology of active-matter defects**. Both have substantial prior art and unresolved learning problems. Generic condensate classifiers, mechanical learning, ecological reservoirs, and slime-mould optimization already exist.

## Read the research

- [Survey of 14 natural systems and computational mechanisms](SURVEY.md): existing work, candidate gaps, and why the shortlist was chosen.
- [Learning by sequestration: hypothesis and mathematical audit](SEQUESTRATION_LEARNING.md): a concrete substrate, its limitations, and what a genuinely different learning rule would need to do.
- [First study and decision gates](FIRST_STUDY.md): tests for novelty, learning, predictive value, and scalability. No new experimental results are claimed.
- [Source ledger](SOURCES.md): primary papers, publication status, access limitations, and author software.
- [Search record](SEARCH_LOG.md): scope, representative executed searches, exclusions, and remaining coverage gaps.

## What this changes in Nature Compute Lab

This research opens a separate direction after the [repair studies](../repair-driven-development/RESEARCH_PROGRESS.md). Those results remain evidence against broad superiority of our existing prototypes. Their next engineering step would still be resolving context and retention, but that is no longer the default answer to the user's request for a fundamentally different learner.

No earlier results, scheduled checkpoints, repository visibility, or publication dates were changed. This update contains research documents only. No new learner was trained, and none of the external projects was reproduced during this survey.

## The standard for success

Before calling anything a new model family, we need an explicit representation, an experience-dependent update rule, an inference procedure, and a measurable difference from its nearest existing algorithms. A useful biological prediction result is another requirement. Neither a new natural metaphor nor a synthetic demonstration establishes ASI. There is currently no evidence that any shortlisted mechanism will scale to ASI.
