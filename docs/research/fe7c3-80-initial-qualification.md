# Fe7C3 80-atom initial qualification

This is an input/backend qualification, not an LS-SSW search.  The corrected
run is archived under
`research/ga_ssw/evidence/fe7c3-80-input-qualification/`.

## Execution and provenance

The first attempt, job `1252364`, charged zero evaluations and failed before
PES construction with `KeyError('symbols')` while reading the source input.
Its failure record is retained under `failed-input-job1252364/` and is not a
scientific result.  The corrected attempt, job `1252408`, completed on a Tesla
V100 in 39 s according to the allocation/runtime record, with 990 charged
E/F/stress requests.  The saved runtime identifies Python 3.12.11, ASE 3.26.0,
NumPy 2.0.2, SciPy 1.16.0, PyTorch 2.8.0+cu128, MACE-torch 0.3.16, one torch
thread and the model
`/home/gengjianrui/.cache/mace/mace-omat-0-small.model`.
Model SHA-256 is
`0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5`.

The supplied public computational structure is a 20-atom Fe14C6 cell from
the CatalystHub URL recorded in `source-metadata.json`, replicated `(2,2,1)`
to 80 atoms (Fe56C24).  The distributor attributes it to Materials Project,
but the original MP identifier is not independently verified.  It matches the
Fe7C3 formula and hexagonal class only; it is not the LS paper's supplied
Fe7C3 initial geometry.  No atom substitution or AFLOW prototype substitution
was performed.

## Measured qualification

The initial 20-atom cell is fully periodic with volume
`181.36252713059918 Å^3`; its 80-atom replicated cell has volume
`725.4501085223967 Å^3` before quench.  The initial replicated physical
evaluation has energy `-170.14040094970304 eV` per 20-atom cell, maximum force
`0.3690892331 eV/Å`, and maximum absolute stress
`0.07088427194 eV/Å^3`, so it is deliberately not treated as a stationary
starting minimum.

The 80-atom variable-cell quench converged under the registered
`fmax=1e-4 eV/Å`, `stress_tol=1e-5 eV/Å^3`, `maxiter=500`, `max_step=0.2`
settings.  It used 14 optimizer requests.  The independent fresh endpoint
certificate gives `fmax=3.2799307260e-6 eV/Å` and maximum stress
`7.6637213783e-8 eV/Å^3`, with zero recorded energy error against the saved
endpoint and force/stress discrepancies below `1.2e-14` and `1.7e-16` in the
recorded units.

The final cell volume is `686.1044372817767 Å^3`, with singular values
`[16.55404995, 9.55748525, 4.33652975] Å`; the endpoint remains periodic and
has composition Fe56C24.  These are endpoint identity facts at the level of
atom ordering/composition and replicated cell, not a proof of a particular
Fe7C3 phase or chemical stability.

Two representation checks were also recorded: unimodular rebasing gives energy
error `0`, force error `5.0515e-15 eV/Å`, and stress error
`3.9552e-16 eV/Å^3`; exact `(2,2,1)` replication gives per-atom energy error
`-3.5527e-15 eV`, force error `2.8550e-14 eV/Å`, and stress error
`9.5757e-16 eV/Å^3`.

The 243-mode endpoint finite-difference Hessian has minimum eigenvalues
`1.2939794013` (`h=1e-4 Å`) and `1.2939793983` (`h=5e-5 Å`), with no negative
eigenvalues in either saved spectrum.  This is a positive finite-difference
curvature result for this MACE finite-cell endpoint; it is not a DFT phonon,
magnetic stability, or global-minimum claim.

## Fe–C lookup boundary

The separate frozen-ELF lookup audit reports raw `bondeneval_` Fe–C value
`13.779999732971191` and raw `bondlenval_` value
`1.9199999570846558 Å`; C–C is `3.4468400478363037` and
`1.5399999618530273 Å`.  These are release lookup returns, not independently
published Fe–C chemical constants.  In native initialization the energy value
enters the normalized table through

```text
B_ab = D_ab * f_ab * scale * float64(float32(N)*0.02)
      / (N_b * D_CC)
```

and later pair/atom factors; the raw `13.78` must therefore not be reported as
the final Fe–C LS amplitude or as a standalone physical bond energy.  The
qualification above did not use LS and does not validate these lookup values
on the MACE Fe7C3 endpoint.

Artifacts: `qualification/result.json`, `qualification/equivalence.json`,
`qualification/eigenvalues-1e-04.npy`, `qualification/eigenvalues-5e-05.npy`,
`qualification/quench-endpoint.json`, `qualification/runtime.json`, and the
preserved failed-attempt record.
