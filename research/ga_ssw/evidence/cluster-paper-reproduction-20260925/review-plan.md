# Model-cluster reproduction contract — 2026-09-25

Question: can the existing independent SSW kernel find published LJ/Morse global minima with useful cost, without the confounding effect of an ML potential? Audience: project research decisions. Unit of evidence: a paper-specific case and disclosed protocol, not an algorithm name. Include original SSW2013, LS2024/SI, original basin hopping1997 and Cambridge reference structures. Exclude unsourced benchmark recipes and default ASE cutoff LJ as equivalents to the full pair potential. These model cases supplement real systems and do not satisfy random-C60 acceptance.

Extract potential/scales, initial ensemble, local/outer stops, random repeats, target definition, first-hit costs and omitted settings. Distinguish reported values, interpretations, local choices and contradictions. No new default or algorithm mechanism follows from source inspection.

Priorities: LJ55 as positive control, LJ38 as difficult alternate-funnel case, short-range Morse as a distinct landscape. LJ75 is a later expensive escalation, not the first pilot. Keep random-to-GM separate from known-SLM-to-GM. Use held-out reference coordinates only for post-hoc qualification. Preserve failed/censored runs and all force calls. Before search, certify reference energies/forces with the exact untruncated potential on a CPU node. Before scaling, freeze an explicit missing-data policy and trial budget.

Source audit is complete only with source ledger and explicit reproduction gaps; search success requires an independently qualified target, not an energy decrease or completed job.
