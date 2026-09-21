# Independent periodic TYPE1 three-stage controller

2026-09-10. `periodic_ga_reference.py` now executes supplied initial quick walks,
region-selected TYPE1 offspring plus quick walks, and ranked-region fine walks.
The default walker is independent joint VC. A callback with the same result
contract can select block VC; it is called for the entire configured chain,
so block's internal scheduling is retained. Every input/offspring gets exactly
its walker's one initial quench, with no extra cluster or fixed-cell prequench.

Three supplied reference structures are descriptorized before PES requests and
held fixed for routing. The periodic descriptor, explicit bond lengths/range and
six similarity weights produce the three projections used by the existing
legacy-inspired partition routine. Ranking uses the empirical energy/variance
formula on eV-valued objective (E+pV); no funnel connectivity or kinetic claim.
The descriptor does not decide structural identity. Caller must supply a matcher;
optional `pymatgen_identity(ltol,stol,angle_tol)` uses scale=False,
primitive_cell=True, attempt_supercell=True with explicit tolerance parameters.
No default tolerance or phase-identity claim is invented.

All certified walker minima are observed, including MC-rejected landings and
initializations, with raw walker records retaining acceptance and failed-stage
state. Observation walk/minimum indices trace back to these records. Archive
identities are separate: matching observations remain in the full observation
list; the lower-objective representative is retained for later routing.
Offspring keep per-atom parent archive lineage and operator details. Full E/F/
stress arrays and residual certificates are stored; there are no hidden fresh
oracle checks. Failures, missing routing/identity, and every paid walk cost remain
visible. Surface-owned budget denial/exhausted flags produce censored output.

The source proposal requires >=2 regions, >2 parents and nonzero energy span.
An unsatisfied domain produces explicit no_proposal records, not a mutation
fallback. Partition and proposal draw/cut/batch budgets are caller settings.
Incomplete proposal batches and collision counts remain in the proposal result.
No implicit random initial structure generation, Java full-schedule parity,
TYPE2/4 operators, or automatic matcher tolerance qualification is claimed.

`tests/standalone/test_periodic_ga_reference.py`: **3 passed**. A bounded real
Cu4/EMT run exercises quick initialization, actual TYPE1 proposal/offspring walks,
and a genuine joint-cell fine step, with matched EFS accounting and unchanged
inputs. Its intentionally exact representation matcher is a wiring fixture;
it can retain nearly identical physical minima and is NOT a coverage metric.
Separate tests show independent duplicate archive handling/source-domain rejection
and retention of paid initial cost under a one-call budget. No MACE or campaign
was launched. This completes the periodic pipeline mechanics, not scientific
GA search advantage, crystal identity validation, or complete GA type coverage.
