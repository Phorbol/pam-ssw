# K4 central-HVP GPU batch micro-gate

## Question

R1-B did not authorize deleting all-candidate HVP. Can the eight force
requests from four central finite-difference directions at one starter be
evaluated in MACE graph batches without changing the force/HVP semantics, and
with at least 1.3x wall-time speedup?

This gate changes execution only. It cannot reduce force evaluations and does
not test terminal search quality.

## Frozen workload

- Systems: C60 and fixed-bottom PdO.
- Inputs and SHA256 are inherited from the frozen production runner.
- Four deterministic unit-Gaussian directions per system; fixed PdO atoms are
  exactly zeroed.
- Central finite-difference epsilon: `1e-3 Å`.
- Eight geometries per workload.
- Modes: serial, batch 2, batch 4, batch 8.
- Five timed repetitions per mode and system, with rotated mode order.
- One untimed warm-up workload per mode and system.
- Every evaluated geometry counts as one `direction_oracle` FE, including
  warm-up. Total cap: 384 FE.

The batch path uses the same loaded MACE model, z-table, cutoff, heads,
float32 calculator configuration, PBC/cell, and model forward options as the
serial ASE calculator. C60 and PdO are never mixed in one graph batch.

## Equivalence gate

Relative to serial results for the identical geometries:

- max absolute energy error <= 0.005 eV;
- max absolute force-component error <= 0.0005 eV/Å;
- relative HVP-vector norm error <= 0.005;
- absolute directional-curvature error <= 0.1 eV/Å²;
- FE ledger closes exactly and `unattributed = 0`.

## Speed gate

For a batch size to survive:

- the equivalence gate passes in both C60 and PdO;
- median end-to-end workload speedup is at least 1.3x in both systems.

If no common batch size survives, close ForceService. If one survives, only a
minimal run-local service integration gate may open; no batch optimizer,
multi-GPU path, action-policy change, or production default is authorized.
