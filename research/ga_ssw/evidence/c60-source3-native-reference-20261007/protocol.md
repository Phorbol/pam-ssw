# Published C60 #3: native periodic SSW reference

## Question and scope

This is a bounded native-LASP reference on the newly qualified published
non-Ih C60 #3 cage. It asks whether the native SSW stack, started from that
single fixed geometry, reaches a qualified low-energy Ih cage within the
available MH-1 request budget. Compare the outcome with the already completed
Python SSW and paper-form LS source3 study. This is a whole-stack reference,
not a single-component ablation: native LASP uses a periodic 50 Å storage
cell, native random-number and Monte Carlo paths, and its own local optimizer;
the Python arms use nonperiodic geometry and a different implementation.
Seed numbers therefore identify native LASP trajectories only and do not
make the methods' internal trajectories identical.

The two seeds are repeated searches from one qualified source structure, not
two independent structures. The exact input is
`c60-ls-source3-qualification-20261007/qualification-1664282/final-candidate.extxyz`,
SHA256 `dcc81d45c4e7193b306a0c1957b49c19e44751231fa95fef75201183280f0758`.
The run uses its qualified native ARC serialization and the qualified Ih
reference carried forward by source3 input qualification. No redraw,
relaxation, rotation, confinement, or replacement is allowed.

The input is member `c60/c60-iso-3_opt.xyz` of the coordinate supplement to
Liu, Jin and Liu, *Mapping structure-property relationships in fullerene
systems: a computational study from C20 to C60*, npj Computational Materials
10,227 (2024), DOI10.1038/s41524-024-01410-7. It is an additional published
local-defect case; it is not asserted to be the LS2024 paper's exact initial
structure or its G-NN potential. The existing source3 eligibility archive
retains the source ZIP member and accompanying CSV identity.

## Frozen model and native settings

Use cached MACE-MH-1, model SHA256
`a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47`,
`omol` head, float64 CUDA, cueq/oeq disabled. Native LASP binary SHA256
`bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`.
Use the qualified external ASE socket/callback contract and the frozen
geometry gate. The source3 qualification passed the periodic input callback
and the 50 Å model-image separation check; every search callback is still
checked against the model cutoff before it is paid.

Each input fixes `SSW.NG=12`, `SSW.ds_atom=0.6 Å`, `SSW.Temp=150 K`,
`SSW.ftol=0.03/sqrt(3)=0.0173205080756888 eV/Å`,
`SSW.MaxOptstep=1000`, `SSW.SSWsteps=1001`, `SSW.internal_LJ=F`,
`SSW.globalcompress=0.0001`, and `SSW.vapor_cri=1.7`. `ranseed` is set to
26100791 or 26100792. Explicit seed parsing and `NG=12` were observed in the
native `allkeys.log` probe. The larger step/outer limits are the intended
native search settings; the earlier interface qualification used one step
and was only a callback test.

## Cost boundary and records

Each seed receives at most 16,000 paid E/F requests, including failed MACE
calls, or 900 seconds of supervised LASP time, whichever comes first. The
20-minute single-V100 allocation leaves up to five minutes for Python/MACE
initialization, final ledger flush, and scheduler cleanup. A denied request
does not enter the calculator and is logged with `error_kind=request_cap`.
Malformed requests and geometry-gate failures are logged separately and do
not consume a paid slot; once the geometry gate fails, the case remains
failed. The socket server checks its paid counter before every Calculator
entry, independent of LASP's callback client; failed calculator attempts
consume a slot before entry. There is no resume, retry, cap extension, or
budget transfer between seeds.

The append-only `request.jsonl` records callback coordinates, geometry
diagnostics, paid index, energy, forces, failure class, and elapsed request
time. `actual_calculate_calls` counts ASE Calculator entries independently
of paid requests. Preserve native `allstr.arc`, `allfor.arc`, `all.arc`,
`best.arc`, `SSWtraj`, `lasp.out`, process status, and runtime provenance.
Scheduler completion, process completion, callback/request counts, numerical
qualification, physical cage status, and any scientific conclusion remain
separate. Missing, denied, failed, timed-out, or truncated cases stay in the
two-seed denominator.

## Readout and interpretation

The offline readout maps each printed `Min found` energy to the cumulative
integer E/F count at that point and accepts it only when a ledger paid request
matches its energy within `1e-4 eV` and maximum force-component difference
within `5e-4 eV/Å`. The initial ordinal-zero result is separate from search
minima. Report the best connected, force-qualified minimum and first Ih
target at common paid-request prefixes 5,000, 10,000, and 15,000 only where
the run actually reached each horizon; never extrapolate past a censored
run. A target must match the frozen Ih graph at cutoffs 1.64, 1.70, and
1.80 Å, satisfy the chosen energy window and force criterion, then pass a
separate fresh qualification and 3D review before being called a qualified
scientific target.

The ordinary source3 quench remained a non-Ih cage at 1.84216 eV above the
qualified Ih reference, so the search addresses local repair rather than
repeating an already successful relaxation. Any native advantage is limited
to these two seeds, this one source, this model, and the measured paid-cost
horizons. It does not establish a global success rate, reproduce the paper's
G-NN performance, establish equivalence to Python SSW, or isolate periodic
boundary conditions, RNG, Monte Carlo, or optimizer effects.

No fresh check is run by the search itself. The afterany offline readout and
dependent qualification select at most three frames per run (initial, best,
and first target) and allow at most three paid calls per run,
counting failed calls. Those calls do not enlarge the search budget. The
qualification reports its selected frame set and cost; a positive target
still requires manual 3D review. Stop after this readout and qualification.
Do not retune, extend, or add source structures based on intermediate
outcomes.
