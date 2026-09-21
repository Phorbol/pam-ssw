# Native LJ stress producer: arithmetic closure

2026-09-11. Uploaded ELF SHA256
`bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`.
This is instruction evidence, not a native end-to-end search or an efficacy test.

## Recovered chain

`cal_pes` passes `module_str_mp_strt_ + 0x110` as the stress output to
`ljperipot_` at call `0x49a996`. The latter saves its fifth argument in `r15`.
At `0x5081ed` it calls `volcel_loc_` (`0x504000`) on the cell. That helper
evaluates the three-by-three determinant and applies an absolute-value mask.
Its result is saved at `[rbp-0x190]`.

For the inspected pair loop, let `r=|d|` and let `s(r)` denote the cutoff
value passed back by `tanh4_`. Literal constants give

```
A = 150830.0777733409
B = 388.36848195153647
u(r) = 2 (A/r^12 - B/r^6) s(r)
u'(r) = 2 [(6B/r^7 - 12A/r^13)s(r)
           + (A/r^12 - B/r^6)s'(r)]
```

`0x509512–0x509712` adds `u` to the energy accumulator, opposite pair
vectors `±d u'/r` to the two force arrays, and `d d^T u'/r` to the stored
triangle of the stress accumulator. Here the sign assigned to a particular
atom additionally depends on the upstream definition of `d`; the stress
outer product is independent of that orientation.

After MPI reduction, `0x509b2f–0x509b8a` divides every stress component by
`abs(det(cell))` and copies the off-diagonal entries to make a symmetric tensor.
Thus the inspected backend block has the energy-strain derivative sign,
`sigma = (1/V) sum d d^T u'/r`, not the negative pressure-tensor sign.
No claim about pair counting, input units, cutoff implementation, or all other
backends follows from this isolated block.

## Independent original-instruction check

Run from the research worktree:

```
env PYTHONPATH=/tmp/pam-ssw-unicorn-probe:. \
 /home/gengjianrui/.conda/envs/mace_env/bin/python \
 research/ga_ssw/probe_native_lj_stress.py --output NEW_RESULT.json
```

Evidence: `research/ga_ssw/evidence/native-stress-producer-review/lj-pair-oracle.json`.
Six cases span `r=2.5,3.5,5.0`, a nonorthogonal cell with either determinant
sign, and controlled cutoff inputs `(s,s')=(1,0)` or
`(exp(-0.17r),-0.17 exp(-0.17r))`. The exponential is a test input, not a claim
about native tanh4. Original instruction blocks, including the volume helper,
are executed without native initialization, neighbor generation or PES calls.
Maximum discrepancies: energy 0, force `3.55e-15`, stress `1.39e-17`;
opposite-force sum is exactly zero.

## Remaining boundary and design decision

Ordinary `run_ssw` copies its input stress directly into CSSW storage; see
`native-vc-gap-review.md`. However `cal_pes` combines backends through
`pesfact` and has a date-dependent postprocessing term; see the new section
in `native-expiry-review.md`. The raw LJ output and final CSSW input must not
be equated without qualifying those intermediate operations. NPT's separate
`-V*stress` transfer is not the ordinary VC-SSW path.

The evidence supports keeping the independent ASE energy/force/stress
convention and its consistent `E+pV` derivatives. It does not justify a new
search component, a changed strain metric, or a performance claim. No
production defaults changed.
