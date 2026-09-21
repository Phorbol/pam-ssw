# Fixed-cell production lifecycle: boundary resume

User-approved priority: complete the fixed-cell SSW/LS/GA lifecycle before
integrating constrained/surface variants. This work changes persistence, not
the potential, proposals, acceptance or adaptive parameter rules.

## Boundary and invariant

First support completed outer SSW attempts, after true landing, MC selection
and LS response update. At this boundary the next attempt already creates a
new Gaussian history and local optimizer. Saving their previous internal
states is unnecessary. Save current and best structures, initial certificate,
records/minima, LS frozen potential and response state, random-generator type
and state, configuration/policies, next outer index and cumulative requests.
The supplied surface/calculator remains caller-owned and is not serialized.
Matching calculator/model/settings and environment are required; persistence
of electronic SCF state is outside this contract.

Resume must skip initial quenching and LS initialization, retain attempted-step
numbering and rejected landings, and add new requests to the prior total without
rewriting a new surface's counter. An error termination is not a pause.
No optional checkpoint should change the old default path's evaluations.

GA requires its own controller state: archive, observations, stages/failures,
walk results, RNG and total budget, plus the next generation/cycle. Prefer
completed generation/cycle boundaries first, where pending offspring do not
need regeneration. A saved population alone is not a reproducible restart.

## Implementation and acceptance

1. Implement optional SSW boundary capture and trusted-local save/load; expose
   resume through ordinary SSW and both LS settings variants.
2. Compare continuous runs with split + disk reload on real EMT clusters and
   fixed-cell material, checking requests, coordinates, MC decisions and LS
   state. Include mismatched settings and terminal failure checks. Extend to
   a bounded molecular case once deterministic-backend tests pass.
3. Add GA boundary state without re-running initialization, partition or
   proposals. Verify continuous/split population lineage, RNG and stage costs.
4. Keep physical minima validation separate: restart identity does not prove
   search efficiency or global-minimum discovery.

Parallel scope: Luna implements SSW/LS; root audits integration and completed
real campaigns; a separate read-only review identifies GA closure state.
No VC/RC expansion, GPU jobs, default tuning, or new search heuristics.
