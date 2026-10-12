# Proposal: explicit variable-cell mode in the research LASP–ASE bridge

Status: proposed, not implemented or approved. Independent Python SSW defaults
and its calculators are unaffected. This is the next prerequisite for a useful
same-potential native VC baseline, after the completed TiO2 component panel.

## Problem and evidence

`research/ga_ssw/lasp_external_ase.py::run_lasp` currently enforces a fixed cell
and returns E/F. The socket client writes no stress. Simply removing that cell
check would be wrong: the native external reader tolerates absent tensor data,
which can leave a zero tensor. Static input/read contracts are documented in
[native-stress-backend-boundary.md](native-stress-backend-boundary.md).

The private Cu4/EMT probe at branch experiment/native-external-stress-20261012,
source1065e56 and CPU1757580, uses three genuinely changing-cell requests.
Explicit `external.dedcell F` plus nine values of -ASE tensile-positive full
stress gives matching legacy stress and class-consumer stress within 7.11e-8
 eV/Angstrom^3, far below our .001 physical stress gate. This resolves the
bounded transport question; it does not qualify MACE or a full cell gradient.
[Raw evidence and limits](../../../native-external-stress-20261012/research/ga_ssw/evidence/native-external-stress-20261012/decision.md).

## Recommended minimal contract

Add a keyword-only `cell_mode="fixed"` to the existing research `run_lasp`
entry point, with explicit `"variable"` as the only new mode. Keep the fixed
mode's request validation, E/F response and existing callers unchanged.

Variable mode has the following declared scope and behavior:

- Fully periodic, finite nonsingular 3D cells, fixed atom identities/order,
  no ASE constraints; initially qualify the zero-external-pressure protocol.
- Materialize the current request cell and positions together in the existing
  Atoms template. Obtain E/F and `get_stress(voigt=False)` on that same Atoms.
  Unsupported or failed stress is an error, never silently zero-filled.
- The response adds the nine full-tensor entries; the existing client writes
  them only when explicitly supplied. For the qualified native branch, wire
  tensor is `-ASE stress` in eV/Angstrom^3, serialized in Fortran order.
- The caller still owns prepared LASP input and bounded process supervision.
  Variable mode requires the explicit `external.dedcell F` input contract;
  input/parsed-key evidence is retained with the qualification run. It must
  not silently accept a dedcell=T experiment or infer mode from properties.
- Preserve request charging, actual-call accounting and failure cleanup.
  Extra stress work remains in the same request and must be counted; a
  response failure cannot leave stale external.ene or allow further paid calls.
- Logs explicitly identify mode and include the request cell and E/F/stress.
  No new checkpoint, population state, native-memory hook or generic execution
  framework. Production calls do not depend on GDB; GDB is a qualification tool.

This is a research interface extension, not a promise that every ASE calculator
supports variable cells. Models without stress remain usable in fixed mode.
The inspected uploaded ELF is the qualified native version; other binaries
need their own interface check. No nonperiodic unwrapping, slab cell constraints,
external-field support, or arbitrary ASE constraint support is added here.

## Alternatives and tradeoff

1. Continue one-off private probes: no shared interface change, but native TiO2
   comparisons would duplicate the E/F/stress service and its failure semantics.
2. Recommended explicit mode: reuse the working service and keep old semantics;
   modest added contract/validation burden, no core SSW changes.
3. Separate generalized bridge framework: broader architecture and migration
   without evidence that it is needed; not recommended.

The decision is interface semantics, not permission for another resource grant.
Per AGENTS.md section8, important public/interface architecture is discussed
before implementation. The old approved fixed-cell contract did not include
stress transmission or changing cells. Approving this proposal does not mean
the mode is already qualified or that it improves optimization.

## Minimal acceptance and next decision

First, reuse existing E/F bridge checks and add only variable-cell protocol,
missing/nonfinite stress, and counted failure-path regressions. Then qualify
OMAT-small on the two saved TiO2 structures with the frozen periodic convention,
one bounded finite-strain check per required mode and native tensor observation.
Record signs, units, cell representation, actual requests and failure outcomes.
No precision chase below the existing physical stress criterion. Pre-register
the exact finite-strain protocol and bounded cost before submission.

Only after those checks pass should a new, bounded same-OMAT native-VC versus
Python search protocol be prepared. Failure retains raw evidence and blocks
that comparison; no default optimizer/search-policy change follows from this
bridge proposal. The finished phase87/139 component panel is not rerun to tune
its atomic schedule, width, height, rotation budget or acceptance thresholds.
