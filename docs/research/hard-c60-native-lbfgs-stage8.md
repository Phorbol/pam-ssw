# Original LBFGS instructions reach the force threshold on the frozen C60 LS failure

Evidence: `research/ga_ssw/evidence/hard-c60-native-stage8/`; runner `research/ga_ssw/probe_native_lbfgs_c60_stage8.py`. This is one deliberately selected failed local subproblem, not an independent benchmark population or a rerun of LS-SSW. The earlier search failure remains unchanged.

The original stage8 start and entire frozen objective are recovered from the exact offline replay:90 frozen LS pairs and8 Gaussian terms plus GFN2-xTB. Original input-coordinate error is0. Safe-total's423 physical E/F/400 accepted-step result is reused without recomputation. New budget was frozen before PES: at most423 optimizer E/F plus1 independent modified-endpoint check,400 accepted steps,600 seconds, one CPU thread. Actual new cost **315 E/F,235.38 seconds**.

| Local kernel | Optimization E/F | Accepted steps | Modified max force eV/Å | Modified energy eV | Outcome |
|---|---:|---:|---:|---:|---|
| Safe-total archived |423|400|.0113545810|−3477.832151868307|Step limit, fails .01 threshold|
| Original ELF LBFGS |314|309|.0097104135 (fresh)|−3477.825698664686|Passes .01 threshold;1 additional fresh E/F|

Unicorn executes original LBFGS→MCSRCH→MCSTEP instructions (ELF SHA256 bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704). Counters: LBFGS314, MCSRCH622, MCSTEP4. Host memcpy adaptation is the same prior31-Cu harness. C60 requires an additional1 MiB scratch mapping for N180/history400; this changes allocation capacity only. At each accepted-state hook, machine-code x equals supplied evaluation coordinates exactly (maximum error0 over309 accepted states).

Numerical settings reuse the previously frozen kernel protocol: history400, initialized GTOL900, STPMIN1e-4, maxstep .5, FTOL1e-4, exact ELF-derived EPS/XTOL, gradient scale1. GTOL900 is the inspected ELF initialization/kernel path value, not a claim that all full-program runs retain it. No BFGSDRIVER .05 force scaling, LASP main, protection bypass, native Gaussian arithmetic, or native LS scheduler is executed. The supplied independent objective is conservative E/F. This is an original-instruction local-kernel test, not full LASP trajectory parity and not SciPy/ASE replacement of those instructions.

A newly constructed GFN2 calculator independently recomputes the last accepted geometry. Modified energy error is0 and maximum component force difference3.912e-7 eV/Å; fresh maximum force remains below .01. The certificate concerns the frozen modified surface, not a true-PES carbon minimum. Saved raw endpoint energy is−3486.735856308325 eV. No final bias-free quench was performed.

The native endpoint's modified energy is **.00645320 eV higher** than Safe-total's endpoint, while the aligned same-atom RMSD between endpoints is .002439 Å. Passing the max-force criterion sooner therefore does not mean obtaining a lower energy or proving a better minimum. The result does demonstrate that, at the same starting objective and request ceiling, local-optimizer path differences can determine whether this particular climbing stage clears its stopping threshold. Safe-total is not numerically identical to the native optimizer. It does not establish why the trajectories differ, general algorithm superiority, full LS success, or justification for changing defaults based on this selected failure alone.

Prior costs remain separate: hard-C60 comparison1806 E/F; qualification39 E/F; earlier Ih test1292 E/F. This diagnostic adds315, with no retries or parameter changes. Further analysis can use these saved trajectories without new PES evaluations.
