# Native `localatomgroup_mode` recovery

The archived LASP ELF has SHA256
`bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`.
Static disassembly identifies `newssw_basics_mp_localatomgroup_mode_` at
`0x6e4c00`, returning at `0x6e4e4c` (the implementation ends at `0x6e4e48`).
The caller at `0x5d8255` passes `rdi=&natoms`, `rsi=coords`, `rdx` (not read by
this helper), `rcx=freedom_mask`, `r8=output`, `r9=&pair_descriptor`, and the
first stack argument `&group_mask`. The pair descriptor contains two 1-based
atom indices; the group mask is a raw integer `N` vector, not an array
descriptor.

For each atom `k` whose `group_mask[k] == 1`, the recovered geometry is

```text
output[k] = (coords[k] - coords[pair[0]]) × (coords[k] - coords[pair[1]])
```

where the vectors are the contiguous `3N` Cartesian doubles used by the
native routine. Endpoint rows are therefore zero by construction. The helper
then multiplies all `3N` output entries by the integer Cartesian freedom mask.
The raw ELF routine does not clear the output buffer for unselected rows; its
caller zeroes that buffer before entering the helper. The Python helper returns
a fresh zeroed array, which gives the caller-visible contract without retaining
stale values.
There is no RNG, neighbor-list call, species lookup, normalization, or PES
evaluation in this function. Pair/group selection remains a caller concern.

The bounded Unicorn probe `research/ga_ssw/probe_native_local_group.py` ran four
explicit geometries/masks against the original function and matched the pure
reference with maximum absolute error `0.0` in every case. The probe used no
main program, protection, PES, or GPU path. The corresponding isolated Python
helper is `pamssw/standalone/native_local_group.py`; it intentionally requires
the selected pair and group mask explicitly and is not wired into a driver.

## Root integration verification and physical interpretation

The root independently executed the actual generator, this helper and
setconstraints together in `probe_native_group_geometry.py`, with ASE ethane,
methanol and benzene geometries. Each used a specified axis and either a
one-side group or all atoms. All six cases agreed with the production Python
helper plus the Euclidean rigid projector, maximum error 1.11e-15.
The earlier `probe_native_group_projection.py` separately isolated actual
projection on water, methane and benzene (six cases, projection error at most
1.34e-15). These are geometry/interface checks, not PES or search experiments.

Both integrated probes initialize a valid same-N translation-mode cache;
the original instructions build and orthogonalize rotational modes. RNG is
fixed and memory-runtime operations are replaced by equivalent operations.
The final geometry probe does **not** substitute local generation or rigid
projection. Native axis/group selection and cold-cache initialization remain
outside its claim. Artifacts are `group-geometry.json` and
`group-projection.json` in
`research/ga_ssw/evidence/native-cbd-reentry-guard-20260917/`.

Algebraically, writing a and b for axis-atom positions,

    (x-a) cross (x-b) = (b-a) cross (x-a).

Thus the selected atoms receive a common infinitesimal rotation about the
axis. Distances between selected atoms are preserved to first order because
delta_x dot ((b-a) cross delta_x) = 0. This does not preserve finite-step
rigidity under a Cartesian displacement, nor does it establish RC-SSW.
Selecting all atoms creates an overall rotation which rigid cleanup removes;
the integrated probes explicitly cover that case. Group boundaries can still
deform bonds to unselected off-axis atoms. No connectivity or chemical
validity claim follows from the cross product alone.

The Python helper is a bounded port, with no new heuristic or tunable parameter.
Its returned raw vector has length-squared units before the caller normalizes
it. It accepts nonperiodic atoms and explicit freedom masks; automatic group
selection and driver integration are not implemented here.

Root verification commands (from the research checkout):

```sh
env PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/pam-ssw-unicorn-probe:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/gengjianrui/.conda/envs/mace_env/bin/python research/ga_ssw/probe_native_group_geometry.py
env PYTHONNOUSERSITE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/gengjianrui/.conda/envs/mace_env/bin/python -m pytest tests/standalone/test_native_local_group.py -q
```

Results: 6/6 differential cases; 5 tests passed. Tests include nonzero masked
outputs, geometric covariance and first-order distance preservation.
