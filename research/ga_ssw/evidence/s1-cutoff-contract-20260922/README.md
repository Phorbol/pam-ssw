# S1 and related contract checks: execution history and results

## Completed follow-up checks

CPU1447868 exited zero: `tests-1447868.txt` records four passing checks
(three S1 radial/gradient/guard tests and the strengthened Cu13/EMT LS
rejection regression). The explicit LS surface caps evaluation requests at
500; the test output does not report an actual total, so none is inferred.
The water15 permutation audit in the same job made zero PES calls.

The arbitrary atom reversal changes ordered OHH topology. CPU1447885 then
tested six legal TYPE3 permutations using `legal-permutation-manifest.json`:
three saved geometries with whole molecules reversed, then the equivalent
hydrogens exchanged within every OHH group. All preserve ordered species and
the same contiguous group layout. All six raw projection drifts exceed the
unchanged 0.0001 threshold; all six full-fingerprint-order drifts are zero.
This is representation consistency, not six independent GA search successes.
See [minimal design and results](../../../../docs/research/2026-09-22-ga-row-order-proposal.md).

Both jobs used one CPU task, `CPU-MISC`, `rush-cpu`, `sjtu-caoxiaoming`, with
a five-minute ceiling. Job1447868 is reproduced by `validate.sbatch`;
job1447885 ran the same analysis module with the legal manifest and its own
output path. Source base was `1ec2fe3`, plus the archived validation script
and explicit-permutation extension included in the subsequent result commit.
The earlier rejected submissions and import failure below remain recorded.

## 2026-09-22 16:48 follow-up

After restoring the isolated pinned dependency, CPU job 1447836 exited zero.
`result-1447836.json` contains ten passing original-instruction arithmetic
cases and zero PES calls. The checks confirm the constants 1 and -3 and the
cutoff guard `9.999999747378752e-06`. This closes this arithmetic probe only;
the supplied-tanh and full-Q limitations below still apply.

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
