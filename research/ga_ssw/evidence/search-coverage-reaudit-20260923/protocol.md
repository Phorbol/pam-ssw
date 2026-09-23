# Zero-PES fixed-cell trajectory coverage re-audit

## Question and scope

For each archived periodic SSW/Basin-Hopping arm, distinguish the number of stored minimum frames from the number of geometrically distinct periodic structures under a declared matching rule, and measure how often later stored frames return to the arm's initial quenched structure. These are retrospective analyses of existing runs, not new independent searches, not estimates of basin counts, and not PES or energetic qualification.

This bounded first pass includes the 12 periodic arms only: brookite48 (SSW/BH, seeds 11/29) and the two archived AlOH cases aloh2/aloh3 (SSW/BH, seeds 11/29). The four nonperiodic Cu55 arms are excluded from this pass; no conclusion about them follows. Existing source minima, summaries, initial geometries, and logs are read-only.

## Inputs and denominator

Each arm contributes its existing `minima.extxyz`; its existing `initial.extxyz` is the reference quenched start. The archived runner writes the first converged quench to this `initial.extxyz` and stores all converged minima, including rejected SSW landings, in `minima.extxyz`. Report every readable stored frame as the denominator, plus the archived run status and any missing/unreadable frames. Do not filter frames by energy or SSW acceptance. Confirm that each arm's first minimum matches its initial reference; if not, mark that arm's initial-return statistic unresolved rather than silently substituting another reference.

## Periodic structure comparison

Convert ASE frames to pymatgen `Structure` objects preserving species, cell, and fractional coordinates. Use `pymatgen.core.structure_matcher.StructureMatcher` with `primitive_cell=False`, `scale=False`, `attempt_supercell=False`, and `ElementComparator()`. This permits origin, symmetry, lattice-basis, and atom-permutation matching while retaining the supplied cell size and scale and requiring exact element identity. Require equal composition and periodic boundary flags before matching.

Run two predeclared tolerance levels:

- Tight: `ltol=0.05`, `stol=0.10`, `angle_tol=2.0` degrees.
- Broad: `ltol=0.20`, `stol=0.30`, `angle_tol=5.0` degrees (pymatgen's standard default tolerances).

`ltol` is fractional lattice-length tolerance; `stol` is the fraction of average free length per atom; `angle_tol` is degrees. Report group counts and initial-return proportions separately at both levels. No threshold is selected after seeing outcomes. If the first archived minimum does not match the declared initial reference at a tolerance, mark that tolerance's return statistic unresolved.

Form deterministic representative groups in archived frame order: assign a frame to the first existing group representative it matches; otherwise start a new group. Record representative frame indices and group sizes. This is a compact, reproducible coverage summary, but order-sensitive representative clustering can split or merge borderline cases; it is not a transitive equivalence relation or a rigorous basin identity. Do not label groups as basins.

## Return-to-initial statistic

At each tolerance, compare every stored minimum to that arm's `initial.extxyz` reference with the same matcher. Report (i) all matching frames / all stored frames and (ii) subsequent matching frames / subsequent stored frames after excluding the first archived minimum, which is the initial member and otherwise guarantees a trivial hit. Also report how many representative groups contain any matching frame. A geometric return is not a dynamical return probability.

## Execution and outputs

The analysis reads the archived periodic minima and references only; it makes no calculator/model calls. Run one single-task, five-minute CPU-only Slurm job on `CPU-MISC`, QOS `rush-cpu`, account `sjtu-caoxiaoming`, using `/home/gengjianrui/.conda/envs/mace_env/bin/python`. Do not request explicit CPU or memory resources. Preserve original data. Write only the analysis script, job script/logs, protocol, and result files under `research/ga_ssw/evidence/search-coverage-reaudit-20260923/`.

Report per arm: stored frame count, group counts at both tolerance levels, initial-match counts and denominators (including and excluding the first frame), run status, and any invalid/incomplete inputs. Summarize sensitivity without treating small arm counts or reused inputs as independent evidence. The re-audit can change a geometric coverage interpretation and SSW/LS prioritization; it cannot establish new search performance or scientific validity.

Pymatgen matching options and tolerance definitions follow the official `StructureMatcher` documentation: https://pymatgen.org/pymatgen.core.html#pymatgen.core.structure_matcher.StructureMatcher
