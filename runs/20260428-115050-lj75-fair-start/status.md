Current phase: verified diagnostic implementation.

Completed:
- Patched fair-start trace accounting so SSW, ASE-BH, and ASE-GA all start from the same seed-0 quenched LJ cluster at step 0.
- Patched ASE-GA population construction so the first initial population member is the same random-cluster seed used by SSW/BH; remaining members still use ASE StartGenerator.
- Reproduced the LJ75 SSW isolated-atom artifact in the exported minima and quantified it with nearest-neighbor geometry diagnostics.
- Added generic SSW proposal trust controls:
  - bounded proposal L-BFGS-B coordinate trust radius,
  - cumulative per-atom walk displacement clip relative to the seed node,
  - optional non-periodic finite-cluster fragmentation guard.
- Kept the fragmentation guard disabled by default in SSWConfig; the LJ benchmark enables it explicitly because disconnected fragments are invalid for finite LJ-cluster comparisons.
- Added diagnostics for walk displacement clips and fragment rejections.
- Added unit/integration tests for trust-radius bounds, walk clipping, fragment rejection, and fair-start traces.

Verification:
- `pytest -q`: 68 passed in 4.48 s.
- Fair-start spot check: SSW, ASE-BH, and ASE-GA LJ13 seed0 all start at best_energy -40.24277387226225, gap 4.084027127737755 at step 0.
- LJ75 seed0 budget60 after guard and deterministic ASE-GA seeding: SSW best_energy -381.49794547522214, gap 15.99438552477784; ASE-GA gap 19.924216191825963; ASE-BH gap 24.104957666394967.

Known limitations:
- LJ75 result is still far from the Cambridge Cluster Database global minimum -397.492331 and should not be presented as literature-grade global optimization.
- Proposal relaxations are still frequently unconverged on LJ75, so the next production step should tune SSW walk/proposal relaxation convergence and trust policy before larger benchmark claims.
