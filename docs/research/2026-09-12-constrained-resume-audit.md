# Constrained checkpoint replay audit

Offline audit of `research/ga_ssw/evidence/constrained-resume-multicase-20260912`.
No PES was run in this audit and the completed directory was not rerun.

The exact execution command recorded by the parent orchestration was:

```text
PYTHONNOUSERSITE=1 PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/gengjianrui/.conda/envs/mace_env/bin/python research/ga_ssw/run_constrained_resume_multicase.py --output research/ga_ssw/evidence/constrained-resume-multicase-20260912 --execute
```

The runner copies `pamssw` to `source/pamssw` before starting its child and
asserts `Path(pamssw.__file__).is_relative_to(output/source)`. The output
contains that source snapshot and `runner.py`; no separate stdout import-path
log was saved. Therefore the artifact supports the runner's snapshot-import
assertion, but this audit does not claim an independently logged runtime
`pamssw.__file__` value.

## Results

| case | continuous paid | split paid | result requests | exact paid sequence | fresh checks |
|---|---:|---:|---:|---|---:|
| Cu EMT | 443 | 70 + 373 | 443 | true | 6/6 |
| Al EMT | 401 | 111 + 290 | 401 | **false** | 6/6 |

The six fresh checks per case are three continuous minima plus three resumed
minima, hence twelve fresh checks overall. All recorded validation booleans
for both cases are true: three minima, record indices `[0,1]`, request
accounting, final geometry comparison, RNG state comparison, and fresh active
force/energy/fixed-cell checks.

## Al exactness correction

The first Al paid-sequence difference is at zero-based paid index `123` (the 124th request), at identical
geometry. Energy differs as below; the largest force-component difference at
that same request is `8.881784197001252e-15` eV/Angstrom:

```text
continuous  = 2.951520535575970
split       = 2.951520535575973
 difference = 2.6645352591003757e-15 eV
```

Across all 401 paired paid requests, offline maxima are:

| quantity | maximum absolute difference | index |
|---|---:|---:|
| energy | `1.2550743999639735e-08` eV | 379 |
| force component | `5.158514170822137e-07` eV/Angstrom | 395 |
| position component | `1.7321083056742737e-07` Angstrom | 395 |

The mismatch is therefore a small floating-point/trajectory divergence that
starts at machine precision and grows during the later relaxation. It is not
safe to call the Al ledger exactly identical. The artifact does not establish
which internal operation causes the drift, and does not justify treating it as
an algorithmic or physical difference.

The checkpoint replay is thus numerically close and passes the configured
fresh certificates, while exact paid-ledger identity is demonstrated for Cu
only. This is a persistence/accounting result, not evidence of solver quality.

Total cost: 1688 search E/F requests plus 12 independent fresh E/F checks,
1700 combined. Both Al runs reject the second landing; equality of the final
selected current therefore does not imply equality of every landing. The root
independent full suite passed 507 tests with one native-oracle opt-in skip
(`/tmp/pam-root-constrained-checkpoint-final-20260912.log`).
