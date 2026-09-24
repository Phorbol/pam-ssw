# GA active-walk real-system recovery qualification

Purpose: verify the approved active-walk checkpoint implementation preserves the actual ASE search across independent Python processes. This is execution/reproducibility qualification, not GA search-quality validation or a parameter study.

First gate: existing GA/SSW pause baseline (CPU1483690), focused regression for all four phases, independent diff review. Do not execute this protocol against an unfinished interface.

## EMT phase

Reuse the three Cu13 inputs and descriptor/optimizer settings from population-comparison-20260923/cu13-seed3.json. Keep the existing raw structures, Cu-Cu descriptor length2.259876, six existing weights, neighbor_range2, TYPE0 operators and explicit seed3. To test a pause inside a walk, set quick/generation/fine/offspring walks to2 steps, one generation,4 candidates,1 cycle. These shortened lengths are an engineering fixture, not fitted scientific defaults. Preserve original NG14/width0.2/fmax0.01/Safe-total settings.

Run one uninterrupted default reference, one uninterrupted opt-in reference, and one paused/resumed execution for each of quick, generation_short, fine and offspring_ssw. Each paused result is saved to trusted pickle, then loaded in a fresh Python process with fresh surface and RNG; RNG input deliberately differs on resume, since the checkpoint must restore it. Pause after the first completed SSW outer step in the selected phase. If the phase is not reached, record missing coverage and diagnose rather than call it a pass.

Compare canonical JSON representations of full GA result data (archive, observations, lineage, stages, walks, failure records, cumulative requests) and final main RNG, excluding only callback-management checkpoint objects. Check full cost equals actual prefix plus suffix surface requests, no repeated archive entries or proposals, same budget and actual model. Independently re-evaluate final archive geometries with fresh EMT and report force/energy; equality of serialization alone is not force qualification.

Maximum120000 search requests across six complete-equivalent executions (each per-run20000, shared between prefix/suffix), plus at most1000 fresh EMT evaluations. One CPU allocation at most30min. No GPU. Stop on a protocol or implementation failure, retain the failing phase and outputs; make one focused correction before rerunning the failed check. Do not silently relax equality or extend total budget.

## MH1 phase (prepare only until EMT gate passes)

Use the already authorized MH-1/omol C60 model and stored nonperiodic development inputs. Prepare a small quick/fine cross-process pause with a declared total cap before submission. No long global optimization, no new potential, no bitwise GPU-trajectory guarantee. Separate state serialization correctness and physical force qualifications from numerical trajectory sensitivity. Exact resources and stop criteria must be recorded after the EMT result establishes how many steps are sufficient to exercise the mechanism.

## Focused repair after first execution

CPU1483789 consumed55500 search calls. Full trajectories and final RNG agree, but four resumed arms each duplicate one observation (40 versus39); acceptance failed. Preserve runs/. Repair object-identity-based GA landing collection and callback error propagation, then run the unchanged scientific fixture as runs-v2. Expected additional55500, combined111000 below original120000 search ceiling. New series identifier is administrative; no equality/force threshold or physical configuration change. No MH1 execution before this gate passes.
