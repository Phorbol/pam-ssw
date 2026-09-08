# Certified Minima and Cell Quench Implementation Plan

> **For agentic workers:** use superpowers:executing-plans for the connected search-contract changes and an isolated worker/reviewer for the independent cell relaxer. Track completed tasks here.

**Goal:** prevent false minima and support opt-in cell-relaxed true quenching.
**Architecture:** preserve atomic biased propagation; enforce certified archive boundaries; add a stress-aware ASE cell relaxer.
**Tech Stack:** Python, numpy/scipy, ASE, pytest.
**Spec:** ../specs/2026-09-08-certified-cell-quench-design.md

## Global Constraints

- Base fb02469; all source changes stay in this worktree.
- No new GPU/HPC runs, no remote branch deletion or push during implementation.
- Fixed atoms in cell mode have fixed fractional semantics.
- Cell pressure is GPa; stress residual tolerance is eV/Angstrom^3.
- Cell search is opt-in; posterior cell search is explicitly unsupported.

## Task 1: Certified archive boundary

Files: pamssw/relax.py, result.py, walker.py, __init__.py,
exploration/ssw_worker.py; tests/integration/test_certified_minima.py.

- [x] Regression: a double-well initial state with ase-fire/maxiter=1 must raise
  QuenchConvergenceError with nonzero evaluation_counts and no archive result.
- [x] Regression: a certified double-well starter followed by an under-relaxed
  proposal must leave one certified entry and increment uncertified rejections.
- [x] Run these tests red, add the gate and typed exception, then run green.
- [x] Verify posterior worker preserves INVALID and the exact purpose ledger.

Interface: has_minimum_convergence_certificate(result, fmax, stress_tol=None),
QuenchConvergenceError(message, relaxation=..., evaluation_counts=...).
RelaxResult adds optional stress_norm, potential_energy, volume fields.

## Task 2: Geometry-first matching

Files: pamssw/archive.py; tests/unit/test_archive.py.

- [x] Regression: add(s, -10, None); add(s, -10.002, None) returns one entry.
- [x] Regression: identical coordinates in different periodic cells remain distinct.
- [x] Preserve representative and record energy discrepancy telemetry; find_match
  remains read-only. Separate cell tolerance from Cartesian RMSD tolerance.
- [x] Run archive and exploration-controller regressions.

## Task 3: Cell relaxer and search integration

Files: pamssw/cell_relax.py, config.py, walker.py, exploration/ssw_worker.py,
README.md; tests/unit/test_cell_relax.py, tests/integration/test_cell_quench.py.

- [x] Write physical tests first: E=0.5*k*(V-V0)^2 relaxes toward V0;
  pressure shifts optimum to V0-p/k; independently check forces and stress.
- [x] CellRelaxer(calculator, optimizer='ase-lbfgs', mode='shape',
  pressure_gpa=0, stress_tol=1e-3).relax(state, fmax, maxiter,
  trajectory_callback=None, trajectory_stride=1) returns RelaxResult.
- [x] Verify stress unavailable, invalid cell, slab mask, fixed fractional atoms,
  true force-budget errors, and nonzero-deformation gradient consistency.
- [x] Wire primary/fallback selection and certificate checks into true quench.
  Preserve fixed-cell proposal path and expose truthful E/H/V/stress metadata.
- [x] Reject posterior cell mode before any calculator evaluation.
- [x] Run periodic LJ smoke, affected suites, and git diff --check.

## Task 4: Review and delivery

- [x] Independent code review of the resulting diff; fix numerical/contract issues.
- [x] Record tests and scientific limitations in docs/research.
- [x] Commit reviewed changes locally and retain the worktree for inspection.
- [x] Produce an exact archive/cleanup manifest; do not delete unique refs or data.
