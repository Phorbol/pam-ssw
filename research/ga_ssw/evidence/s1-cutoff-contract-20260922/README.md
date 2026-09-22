# S1 cutoff arithmetic verification: prepared, execution blocked

## 2026-09-22 16:48 follow-up

The same five-minute CPU shape now passes `sbatch --test-only` under
`sjtu-caoxiaoming`; `private-gengjianrui` still returns `AssocGrpBilling`.
Actual group-account submission 1447817 executed and failed at import with
`ModuleNotFoundError: No module named 'unicorn'`. Its Slurm log is retained.
Earlier probes used a temporary Unicorn 2.1.4 installation which is no longer
present. Restore that pinned dependency in the research tools directory and
set the probe's PYTHONPATH explicitly; do not change the production package
dependencies. A subsequent run remains necessary for numerical validation.

Purpose: discriminate the recovered `tanh(1-r/c)^3` cutoff and its radial
derivative from a mistaken reading of adjacent ELF constants. This is an
implementation check, not evidence for Q-mode search efficacy.

Root independently read the uploaded ELF using:

```
objdump -s --start-address=0x4a75af0 --stop-address=0x4a75b28 lasp
objdump -d --start-address=0xa24bfe --stop-address=0xa24c88 lasp
```

`0xa24c47` loads the double at `0x4a75af8`, whose little-endian bytes
`00000000000008c0` encode -3. The argument uses xmm3 loaded from
`0x4a75b1c`, bytes `0000803f`, encoding float 1. The adjacent 0.5 and
10000 constants are not these operands. Native arithmetic is
`t=tanhf(1-r/c)`, value `t^3`, derivative `-3*t^2*(1-t^2)/c`.
The filter at `0xa248c6–0xa248d9` additionally requires `c-r` to exceed
the float guard stored at `0x4a75b20`.

The proposed check executes two bounded arithmetic blocks in
`research/ga_ssw/probe_native_s1_cutoff.py`, with explicit registers and
memory. The tanhf call itself is not executed: a float-rounded Python tanh
is supplied between the blocks. Ten radius/cutoff pairs check the original
instructions against the recovered arithmetic. No native main, optimizer,
license/expiry path, calculator, or PES evaluation is involved.

Budget: one CPU task, five minutes maximum, zero E/F requests. Existing
`mace_env` and Unicorn dependencies are reused, without installation.

Both submission attempts on 2026-09-22 were rejected before allocation:

1. `sbatch --wait --parsable check.sbatch` with explicit `sjtu-caoxiaoming`:
   `AssocGrpBilling`.
2. Same script with `--account=private-gengjianrui`: `AssocGrpBilling`.

There is no job ID or numerical result. The root script has only syntax
and source review; execution and assertions remain unverified. Do not
infer native full-gradient, periodic, descriptor-library, or Q-controller
parity from the static formula or the existence of this probe.

## Related parallel-work verification provenance

Two delegated agents ran preliminary pytest commands on `login-01` despite
the task instruction reserving numerical checks for root-submitted CPU jobs:
the GA contract/population subset reported 12 passes; the first LS rejection
test reported one pass. These are recorded as a process deviation, not the
final verification of the reviewed code. The later strengthened LS assertions,
S1 primitive/tests, original-instruction probe and permutation audit have not
been executed. Root stopped further agent execution and did not submit retries
after the two account rejections. Only syntax and `git diff --check` are
accepted checks for this final source snapshot.
