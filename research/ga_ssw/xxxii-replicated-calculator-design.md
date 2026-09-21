# XXXII explicit replicated calculator

This research adapter addresses the primitive restricted-cell half-box image
branch found in the XXXII line-search diagnostic.  It is an exact periodic
representation diagnostic: it repeats the audited 172-atom engine internally,
divides total energy by the explicit replica count, and averages corresponding
image forces.  It does not change force-field parameters, Ewald settings,
topology, or SSW behavior.

`XXXIIReplicatedCalculator` requires `data_path`, `input_path`,
`model_manifest`, `reference_atoms`, and a tuple `repetitions=(nx, ny, nz)`.
The tuple is deliberately mandatory and must contain positive integers.  The
public ASE input is the original ordered 172-atom structure with a changing
positive full-periodic cell; the adapter explicitly constructs the internal
`atoms.repeat(repetitions)` representation for each evaluation.  Constructor
validation reuses the audited adapter's source hashes, element/type/charge
constants, and table-0 Ewald contract (`1e-12`, `gewald=0.47570069`).

Before any engine evaluation, each explicit original-topology graph pair at
bond distance 1--3 is transformed to the replicated restricted Prism frame.
If any corresponding lifted pair has a component at or beyond half of the
restricted replicated box length, `ReplicatedDomainError` is raised.  This
keeps the caller's unwrapped coordinates intact and refuses to hide an image
ambiguity by changing coordinates or applying an extra minimum-image rule.

Each API attempt increments `api_calls`; only an actual internal `run 0`
increments `engine_calls`, `requests`, and `atoms_evaluated` (the expanded
atom count).  `replicas` records the fixed replica count and
`last_force_image_max_difference` records corresponding-image force spread.
The LAMMPS engine is persistent and released by `close()` or the context
manager.  Returned stress uses the existing Prism tensor conversion and is
not divided by the number of replicas, while force blocks are reshaped to
`(replicas,172,3)` and averaged.

This module is research-only and has no production walker integration or PES
run in this change.  Numerical parity across 1x1x2, 1x1x3, and 2x2x2 remains
an experiment to be performed by the parent runner with its separately
registered budget.
