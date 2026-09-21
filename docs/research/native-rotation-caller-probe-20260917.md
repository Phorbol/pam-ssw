# Post-`cbd_rotation` caller probe

The bounded probe `research/ga_ssw/probe_native_rotation_caller.py` starts at
`0x5c40cf`, immediately after the first `unbiasedrot -> cbd_rotation` call.
It executes the real `for_cpstr` comparisons and stops at the subsequent
`biasedrot`, `set_status`, or return branch. Allocator calls are same-shape
hooks; the distinct `work1`, `fa`, `tf0`, `n0` and saved anchor descriptors are
initialized with the real `(3,N)` layout. The tested converged path confirms
the executed `work1 -> fa` copy value-for-value. No LASP main, rotation body,
calculator, or PES is entered.

For the tested `CBD_PreRot` matrix, with curvature values below, equal to, and
above the literal `-1e-6` and both `LCONVERGE` values, all eight branch traces
completed. `LCONVERGE=false` returns through the caller tail. With
`LCONVERGE=true`, curvature `<= -1e-6` follows the non-biased status path,
whereas curvature `> -1e-6` reaches `CBD_biasedRot`. The transition writes
`CBD_biasedRot`; `CBD_UnbiasedRot` is the intermediate mode written only on
the negative-curvature side before its later status path.

Evidence: `research/ga_ssw/evidence/native-cbd-reentry-guard-20260917/rotation-caller.json`.
