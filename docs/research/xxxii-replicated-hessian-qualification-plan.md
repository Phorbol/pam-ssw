# XXXII replicated endpoint Hessian qualification

This is a numerical qualification of the two full precision endpoint
structures saved by `xxxii-replicated-endpoint-quench/result.json`. It is a
separate research artifact and does not alter the SSW walker or any production
calculator.

The frozen representation is the audited XXXII model with repetitions
`(1,1,2)`. Each endpoint uses a fresh center E/F/stress check, followed by a
fresh persistent replicated engine for the Hessian columns. Coordinates use
`SymmetricLogStrainChart(strain_length=5, pressure=0)`, exactly the six cell
coordinates plus the 3N atomic coordinates. The three uniform atomic
translations are removed with the SVD construction in
`research/ga_ssw/qualify_material_gate.py`; for N=172 this leaves 519
coordinates.

For each `h` in `1e-4` and `5e-5`, every projected column is

```text
P [ g(q + h u) - g(q - h u) ] / (2 h)
```

where `g` is the chart gradient of `E+pV` and `P` is the translation-free SVD
basis transpose. Thus each endpoint costs 519*2*2 = 2076 E/F evaluations,
plus one independent center check: 2077 per endpoint and 4154 in total. The
per-endpoint guard is 2200 API requests and 90 seconds. Calls are charged
before evaluation and partial matrices are saved as `.npy` after every
column; progress JSON remains compact and no full matrix is embedded in JSON.

The intended outputs are the raw matrices, their skew Frobenius norms, and
eigenvalue arrays/minimum/maximum summaries for each step. Cross-step
differences and operator norms are reported as numerical perturbation
diagnostics. They are not a positive-stability claim: the backend has a known
ERFC non-conservative floor, and this calculation is neither a DFT Hessian,
phonon calculation, nor an all-wave-vector stability test. Any failed or
timed-out column remains visible in the partial matrix and charged ledger.

The prepared runner is
`research/ga_ssw/qualify_xxxii_replicated_hessian.py`. Run `--prepare` only to
create a new frozen package. Real execution is intentionally withheld until
the parent reviews the frozen source and plan.
