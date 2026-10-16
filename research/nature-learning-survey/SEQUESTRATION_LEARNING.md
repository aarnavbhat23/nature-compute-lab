# Learning by sequestration: a candidate, not yet an architecture

## The proposed learning primitive

Start with a finite collection of computational components that can participate in several operations. Experience can place some components into persistent assemblies. Assembly changes their availability or role. A compartment can retain an assembly while allowing other components to move, so a memory need not spread everywhere.

The question is whether a useful learner can arise from **experience-dependent assembly, functional sequestration, and selective propagation**, with knowledge represented in the resulting organization. This is a software abstraction of a natural mechanism. It does not require growing organisms or implementing molecular hardware.

The biological motivation is Whi3-associated cellular memory and membrane confinement [S01–S03 in the source ledger](SOURCES.md). The proposed general learner goes beyond those experiments. In particular, flexible erasure, arbitrary prediction errors, reusable computational operators, and generalization across tasks are engineering hypotheses, not documented functions of Whi3.

## What would change compared with a conventional neural implementation?

| Property | Common neural implementation | Intended candidate mechanism |
|---|---|---|
| Basic object | An activation unit connected through parameters | A reusable functional component, free or incorporated into an assembly |
| Persistent knowledge | Learned parameters, with possible recurrent/external state | Assembly composition, compartment membership, and selective mobility |
| Inference | Propagation through a specified network | Competition and exchange among available components; responses depend on prior organization |
| Learning | A rule changes model parameters | A local rule changes the repertoire and distribution of available operations |
| Memory transfer | Copy parameters/state or route to a stored module | Transfer selected free components/assemblies while retaining others |

This table is a design intention. Neural models can also have adaptive structure, complex synapses, and persistent states. The candidate must be distinguished at the level of its actual update rule, not by this comparison alone.

## A minimal substrate we can specify now

Let there be components `i`, compartments `q`, and assembly types `a`.

- `c[q,i] >= 0`: free amount of component `i` in compartment `q`.
- `b[q,a] >= 0`: amount of assembly `a` in compartment `q`.
- `nu[i,a]`: amount of component `i` consumed by one unit of assembly `a`.
- `P[q,r,i] >= 0`: permeability for component `i` between compartments `q` and `r`.
- `u(t)`: observed input and intervention history, without hidden ground-truth regime labels.

For a closed substrate the accounting identity is

$$
M_i=\sum_q c_{qi}+\sum_{q,a}\nu_{ia}b_{qa}.
$$

Assembly and disassembly must preserve each `M_i`. If we later allow synthesis or removal, their fluxes must appear explicitly; “growth” cannot create free computational capacity without cost.

An illustrative deterministic substrate is

$$
\dot b_{qa}=v^+_{qa}(c_q,b_q,u)-v^-_{qa}(c_q,b_q,u),
$$
$$
\dot c_{qi}=\sum_r J_{r\to q,i}
-\sum_a\nu_{ia}(v^+_{qa}-v^-_{qa}),
\qquad
J_{r\to q,i}=P_{rq,i}(c_{ri}-c_{qi}),
$$

with symmetric permeability `P[r,q,i] = P[q,r,i]` in this simplest passive-exchange example. Compartment volumes are taken equal; unequal volumes require amount/concentration conversion. Assembly transport would require a separate conservative flux. Rates must vanish appropriately at depleted reactants to preserve nonnegativity. These equations define a substrate, **not a new learning rule**.

Prediction would be a fixed, explicitly specified observable of this substrate: for example, a fraction of output components activated over a declared interval. We must count the cost of any encoding or readout. A powerful neural encoder or trained readout cannot silently perform the task while the proposed substrate provides decoration.

The candidate's strongest possible distinction would require learning **which higher-order assemblies exist and where they persist**, using reusable components, rather than adjusting a scalar gate on each member of a pre-specified model bank. Changing assembly membership must use conservative dissociation/reassembly events; changing `nu` in place would silently violate the accounting identity above.

## An immediate reduction that disqualifies the simplest version

Suppose a component of total amount `T` merely switches between free and sequestered states, with sequestered amount `m`:

$$
\dot m=k_+ f(u)(T-m)-k_-m.
$$

Define `p=m/T`. Then

$$
\dot p=k_+ f(u)(1-p)-k_-p.
$$

The active fraction is `1-p`. This is an ordinary bounded dynamical gate. Calling `m` a protein aggregate does not change the algorithm. Adding self-reinforcement can introduce bistability, but sequestration-based bistable memory is already known [S05]. Giving it fast and slow internal variables also overlaps complex memory models [S38].

Therefore, **we should not implement this scalar gate and announce a new architecture**. It can serve as an inexpensive control. This analytic check is a reason to refine the hypothesis before training a large model.

## The missing learning rule

To turn the substrate into a learner, the following questions need explicit answers:

1. **What writes a memory?** What locally available event initiates or modifies an assembly? Repetition alone can cause habituation without learning predictive relationships.
2. **What assigns credit?** When an outcome arrives later, how does a local assembly receive information that distinguishes a useful earlier event from a coincidental one?
3. **What is reusable?** How can a component or subassembly learned in one history help a new combination without copying a complete task-specific model?
4. **What allows correction?** How does the substrate recover from an incorrect persistent memory without erasing unrelated memories?
5. **What determines confinement and transfer?** Where does compartment identity come from, and how does the system infer context from actual observations?
6. **What scales?** Does the number of assembly types grow exponentially with input combinations, and does useful capacity exceed the cost of storing separate rules?

Writing `update = U(error, state)` would conceal the hardest part. At this stage there is no justified general `U`, no convergence result, and no measured predictive advantage.

## Constraints on the rule-discovery stage

We will examine local reaction-style updates generated by measured co-occurrence, dwell time, resource occupancy, and a declared outcome signal. Their natural plausibility is a hypothesis to assess. We should not select a familiar learning algorithm first and then assign biological names to its variables.

The initial candidate should use one declared set of update laws across tasks. It may have parameters, but their role and selection must be explicit. A fixed global parameter search on development tasks is different from online learning inside the substrate, and its computational cost must be reported. Any conventional optimizer used to discover the rule must also be disclosed; it does not by itself make the resulting rule novel or non-novel.

We should allow ordinary arithmetic, numerical integration, and deterministic dynamics. Excluding these would prevent implementation without helping originality. We should also permit conventional models as controls. A comparison does not make them the proposed architecture.

## Closest collisions to resolve

| Existing direction | What it already covers | What must remain different for this proposal to be worth pursuing |
|---|---|---|
| Sequestration classifiers [S04] | Molecular titration, classifier behavior, shared resource effects | Intrinsic experience-dependent organization and retention, rather than a chemically realized perceptron |
| Bistable sequestration memory [S05] | Persistent molecular state from sequestration and positive feedback | Acquisition and composition of new predictive relationships, beyond a switch |
| Boltzmann liquids / surface condensates [S06–S07] | Learning or designing collective classification and phase-dependent outputs | A distinct update rule for persistent assemblies and their propagation; finite resources alone are insufficient |
| Autonomous chemical learning [S35–S36, S39–S40] | Generative or associative learning through chemical networks | A specific new operation or theorem, not merely autonomous chemistry |
| Chemical in-context learning [S37] | Contextual inference without explicit attention | A distinct acquisition mechanism and evidence beyond the demonstrated contextual task |
| Complex synaptic memory [S38] | Retention through coupled internal processes and timescales | Higher-order reusable assembly structure whose contribution survives a matched memory control |
| Conformational synaptic hardware [S41] | Conformational changes controlling transport and memory | A different learning operation, rather than a new implementation of synaptic state |
| Context-conditioned model banks | Store separate rules and select/reuse them | Transfer of reusable components that cannot be explained by extra storage or an oracle context label |

An explicit equivalence to a known update would downgrade the work to an implementation or application contribution. That could still be useful science, but it would not meet the present objective.

An author-listed 2026 preprint on genetic-regulatory plasticity during cellular learning [S42] is also a required biological follow-up. Its full text was not accessible in this audit, so the older yeast studies should not be treated as a complete account of the current mechanism.

## What we can responsibly claim today

We identified a natural system and a narrowly specified algorithm-discovery problem. We also showed why its simplest abstraction fails the novelty requirement. The stronger candidate is not yet a complete artificial learner. The next deliverable is an explicit rule and a comparison of its operations with the closest methods, followed by the [first study](FIRST_STUDY.md). No ASI or benchmark claim follows from this document.
