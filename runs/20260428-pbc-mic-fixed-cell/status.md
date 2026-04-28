# Fixed-Cell PBC/MIC Implementation Status

Date: 2026-04-28

## Scope

This round implements fixed-cell PBC geometry handling for bulk and slab SSW paths. It does not implement variable-cell moves or cell relaxation.

## Implemented

- Added shared MIC helpers in `pamssw/pbc.py`:
  - `mic_displacement`
  - `wrap_positions`
  - `mic_distance_matrix`
- Reused the shared MIC displacement in periodic fingerprints.
- Made archive RMSD use MIC when both states have the same cell and PBC flags.
- Wrapped periodic coordinates after Cartesian SSW displacements.
- Made walk displacement clipping use MIC distances.
- Made `previous_direction` use MIC displacement, so crossing a periodic boundary does not create a false long direction.
- Projected periodic translation gauge modes for fixed-cell PBC systems:
  - bulk projects x/y/z translations;
  - slab projects only periodic in-plane translations.
- Left coordinate trust-radius bounds unconstrained along periodic axes and bounded along non-periodic axes.
- Enabled dynamic non-neighbor bond candidates for slabs with MIC distances.
- Kept dynamic non-neighbor bond candidates disabled for fully periodic bulk, because the current bond-formation heuristic is a cluster/slab proposal, not a bulk defect/exchange move.
- Enabled nearest-neighbor fragment diagnostics for slab-like systems via MIC distances when `fragment_guard_factor` is configured; fully periodic bulk remains excluded.

## Verification

```bash
pytest -q tests/unit/test_pbc.py tests/unit/test_archive.py tests/unit/test_coordinates.py tests/unit/test_relax.py tests/unit/test_rigid_modes.py tests/unit/test_walker_policy.py
```

Result: `52 passed in 5.15s`

```bash
pytest -q
```

Result: `90 passed in 5.16s`

## Remaining Before Production Slab/Bulk Claims

- Run a controlled slab smoke test with an ASE calculator first, then MACE/AgxOy.
- Add exported-minima validation that every periodic minimum preserves `cell` and `pbc`.
- For slab adsorption, define adsorbate/surface fragmentation semantics instead of relying only on global nearest-neighbor spread.
- For bulk, add domain-appropriate periodic proposal diagnostics; current code is fixed-cell position search, not defect-generation or compositional global optimization.
