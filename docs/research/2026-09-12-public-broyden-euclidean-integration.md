# Public Euclidean Broyden integration boundary

`rotation_solver='broyden-euclidean'` is an explicit opt-in numerical comparison backend. It reuses the recovered raw Broyden state and direct endpoint HVP certificate, with Euclidean contractions and fixed `FACT=0.05` from the native `CBD.fact` level-0 initializer. The existing Ritz default, dimer option, height rules, stopping policy, and LS/GA configuration flow remain unchanged.

The public entry uses the same direction solver signature as `paper_dimer_direction`. Ordinary Broyden requires one endpoint HVP; staged mode retains the dimer presweep and permits Broyden as the main stage with the same shared budget accounting. Its `projected_symmetry_error` is zero because this solver does not sample a Ritz projection matrix. Its residual stopping status is a direct endpoint certificate and is not native CBD `ftol` parity.

The implementation is copied into standalone private state/history modules and a public low-level wrapper. Research originals remain frozen for evidence comparison and archived runner compatibility. `atomic_climb` rejects this solver before any PES request because that separate path has no Broyden dispatch. LS and GA need no new knobs: their existing `SSWConfig` objects carry the explicit choice through initial and offspring walks.

Validation covers nonzero-center analytic force certificates, ordinary and staged budget-one configuration, and staged public dispatch. Scientific efficacy and default promotion are outside this integration.


Minimal use with an existing validated fixed-cell configuration:

```python
from dataclasses import replace
from pamssw.standalone import run_ssw

broyden_config = replace(config, rotation_solver="broyden-euclidean")
result = run_ssw(atoms, surface, steps=steps, config=broyden_config, rng=rng)
```

Pass the same `broyden_config` to the LS wrapper or GA's `ssw_config` to use
this main solver. A separately supplied GA offspring config must also select
it if both stages should use Broyden. Existing default configs continue to
select Ritz. The Euclidean metric is an independent physical choice, not the
recovered native block-sum form; the latter remains a research-only comparison.

Public SSW replay is complete for Cu13/EMT, fixed-cell Cu31/EMT, and
bicyclobutane/GFN2-xTB, each with seeds 11 and 29. All six paid evaluation
ledgers are byte-identical to the corresponding research-adapter runs.
The total is 6,879 search requests plus 17 independent final checks; all
saved structures pass the specified force, energy-agreement and fixed-cell
checks. Evidence: `research/ga_ssw/evidence/public-broyden-ssw-20260912/root-public-replay-audit.json`.
This establishes public-entry equivalence, not improved search efficiency.
The standalone test suite reports 470 passed and 1 skipped.
Public Broyden LS integration is complete: six SSW/paper-LS/native-LS arms
on trans-butadiene/GFN2-xTB used 36,000 search requests plus 42 fresh checks.
All saved records pass the specified force/energy/cell checks; all arms reached
the request cap without backend failures, and none improved the initial best
energy. Paper and native response magnitudes remain different.

The public GA replay is complete as a bounded integration check: two seeds
used 24,000 search requests plus 16 fresh archive checks, all force/energy
qualified. Each produced four actual offspring SSW runs and exhausted its
budget during generation-short exploration, before fine exploration. Each
archive includes one fragmented structure; force qualification is not intact
molecular validity. These runs do not establish Broyden superiority, full
fine-stage completion or a global minimum. Audits are in the respective
`public-broyden-ls-20260912` and `public-broyden-ga-20260912` evidence folders
as `root-final-audit.json`.
