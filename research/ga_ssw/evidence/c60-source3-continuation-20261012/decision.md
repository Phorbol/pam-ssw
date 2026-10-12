# Decision: C60 source-3 SSW versus paper-LS continuation

**Date:** 2026-10-12  
**Evidence:** `run-v1/summary.md`, the four per-arm `summary.json` and `cumulative-readout.json` files, `run-v1/paired-cumulative-readout.json`, the child request ledgers, and the immutable parent records under `c60-local-defect-qualification/.../prepared-20261007-a`.

## Result

The saved source-3 trajectories resumed without replaying parent work and all four reached the planned cumulative 100 outer attempts. Parent and child costs close exactly, the child request ledgers have no errors, and the parent artifacts remained unchanged. LS continuation state also advanced as expected: each LS child began from a byte-identical parent checkpoint (including RNG and native-MC state) and ended with response step 100; the SSW arms remained LS-free. The source/model/configuration stayed tied to the frozen parent snapshot.

No arm found a candidate meeting the frozen reference-energy window. Each arm produced force-qualified cage-like structures, and the one child fresh check per arm passed the numerical E/F repeat check, but all four fresh structures remained about 1.842 eV above the reference and none matched Ih at the three required graph cutoffs. The manual three-dimensional cage review was not performed. Therefore the physical target was not reached.

The paired best-connected-energy result has opposite signs across the two repeated seeds, with differences around 0.00005 eV. This supplies no consistent evidence that paper-LS improves the fixed-horizon SSW outcome. It does not test whether the algorithm would benefit from a longer horizon: all trajectories reached 100, but that finite horizon and this target model leave broader explanations unresolved. In particular, this is not evidence for or against a barrier mechanism.

## Interpretation boundary

This is one published defect input, two already-used search seeds, one frozen historical implementation, and MACE-MH-1 (`omol`, float64 CUDA). The C60 paper's G-NN surface and paper search statistics were not reproduced; the result is not PBE. It is not a random-start success-rate estimate, a general LS validation, or a physical confirmation of a carbon-cage basin.

The readout's connected graph-class counts summarize cutoff graph isomorphism only. They are not counts of distinct geometric minima or basins and must not be used as basin coverage. Fresh numerical qualification, graph-cage candidacy, strict Ih graph identity, target energy, and manual 3D review are separate gates.

## Disposition

Retain run-v1 as a completed fixed-horizon paired continuation with a negative target-discovery outcome and no consistent LS advantage in endpoint energy. Do not promote the LS behavior or claim a barrier explanation from this case. Any future broader claim requires new independent initial structures/seeds and physical validation on the intended reference surface; this run should remain an historical MH-1 comparison rather than be retuned against its outcome.

## Independent readout check

A separate read-only review recomputed graph isomorphism classes from the saved
checkpoints: seed791 LS/SSW=73/79 and seed792 LS/SSW=76/70. These counts include
the initial graph; the older parent analysis excluded it, so the two versions
must not be subtracted directly. Every best structure matches its initial
source-3 graph at 1.70 Å. All 404 initial/landing observations fail the energy
target, independently of the stricter Ih identity gate. Saved child landing
coordinates agree with checkpoint positions within extxyz rounding (5e-9 Å).
No additional PES evaluations were used for this review.
