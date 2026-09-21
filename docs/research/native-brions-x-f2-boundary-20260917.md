# Native `rotate_dimer` BRIONS/f2 boundary

The call at `rotate_dimer` `0x6e6390–0x6e639a` passes `rsi = r14`, where
`r14` is the saved original fourth register argument (`r8`) and is the
endpoint coordinate vector. The post-call reconstruction at
`0x6e6405–0x6e6415` computes `(r14-r15)/dr` into the candidate direction
buffer, confirming that BRIONS receives the endpoint `x`, not the direction
`n`.

Before the call, the force workspace is scaled in place by `FACT1` at
`0x6e602b–0x6e60b3`. The BRIONS call receives that workspace as its force
argument (`rdx = [rbp+0x10]` at `0x6e6392`). BRIONS stores its own module
copies/history and computes the proposal `X`; the inspected caller path has no
writeback from BRIONS into the caller force workspace. The reported quantity
at `0x6e7b78–0x6e7b84` is therefore

```text
sqrt(sum(f2_scaled**2)) / FACT1 * 10
```

where `f2_scaled` is the same FACT1-scaled tangent-force workspace supplied to
BRIONS. It is not an HVP residual. Algebraically, if `f2_raw` is the force
workspace before scaling, the reported value is
`10 * ||FACT1*f2_raw|| / FACT1 = 10*||f2_raw||` for positive FACT1; the
workspace itself remains scaled while BRIONS runs.

The recovered Broyden state can consequently consume the endpoint `x` and the
scaled force response, but the native caller must retain the separate
`FACT1`, perform scaling before `BroydenState.step`, and compute native
reported-force termination from the pre-update scaled workspace. This note is
an assembly boundary audit; it does not change the solver.

Root subsequently executed the original `rotate_dimer` prefix through the
BRIONS entry: `probe_native_rotation_response.py`, N=1/2/5, dr=.001/.03,
FACT1=.05/.032, rotnum=2, fixed cell without constraints. All 12 cases confirm
`x=r0+dr*n` and `f2_scaled=FACT1*(f_endpoint-f_center+dr*curvature*n)`;
maximum response error is 1.39e-17. Only equivalent memcpy is hooked, with no
force-arithmetic replacement. This closes the prefix algebra boundary for
those explicit conditions, not the complete rotation or constraint branches.
The corresponding tolerance conversion and real-input interface check are in
[CBD stage and tolerance audit](2026-09-17-cbd-stage-and-tolerance.md).
