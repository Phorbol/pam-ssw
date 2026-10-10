#!/usr/bin/env python3
"""Offline fixed-W linear-response estimate from archived C60 #2 Hessian; no PES."""
from pathlib import Path
import json
import sys
import numpy as np
from ase.io import read

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from pamssw.standalone.softening import FrozenBondSoftening

EVIDENCE = ROOT / "research/ga_ssw/evidence/c60-local-defect-20260925"
NPZ = EVIDENCE / "curvature/isomer-2.npz"
GEOM = EVIDENCE / "qualification/isomer-2/final.extxyz"
OUT = Path(__file__).resolve().parent / "result.json"

z = np.load(NPZ, allow_pickle=False)
atoms = read(GEOM)
x = atoms.positions.copy()
X = np.asarray(z["positions"])
Q = np.asarray(z["internal_basis"])
Hraw = np.asarray(z["raw_hessian"])
assert len(atoms) == 60 and np.all(atoms.numbers == 6)
assert x.shape == X.shape == (60, 3) and Q.shape == (180, 174) and Hraw.shape == (180, 180)
assert np.max(np.abs(x-X)) < 1e-12, "Hessian geometry does not match qualified extxyz atom order"
assert np.max(np.abs(Q.T @ Q - np.eye(174))) < 2e-12
centered = X - X.mean(axis=0)
rigid = np.column_stack([np.tile(v, (60, 1)).ravel() for v in np.eye(3)] +
                        [np.cross(v, centered).ravel() for v in np.eye(3)])
assert np.max(np.abs(rigid.T @ Q)) < 2e-12
# Same symmetrization convention as the curvature runner.
H = Q.T @ ((Hraw + Hraw.T) * 0.5) @ Q
w = np.linalg.eigvalsh(H)
assert w.min() > 0

# C60 paper-form frozen LS: A_ij = fraction * 3.61 eV; xi=.2;
# freeze unordered C-C pairs at d <= 1.64 A and use each current d as r0.
# This is the recorded project cutoff convention and exact FrozenBondSoftening form.
cutoff, xi, bond_energy = 1.64, 0.2, 3.61
D = X[:, None, :] - X[None, :, :]
dist = np.linalg.norm(D, axis=2)
pairs = [(i, j) for i in range(60) for j in range(i+1, 60) if dist[i,j] <= cutoff]
if not pairs:
    raise RuntimeError("no frozen LS pairs")
soft = FrozenBondSoftening.from_atoms(
    atoms, bond_energies={(6, 6): bond_energy},
    bond_lengths={(6, 6): cutoff}, initial_fraction=0.03, xi=xi)
assert list(soft.pairs) == pairs
soft_energy, soft_forces = soft.evaluate(atoms)

def ls_gradient(fraction):
    grad = np.zeros_like(X)
    A = fraction * bond_energy
    for i, j in pairs:
        vec = X[j] - X[i]
        r = np.linalg.norm(vec)
        # d/dr A exp[-(r-r0)/(xi*r0)] evaluated at r=r0 is -A/(xi*r0).
        pair_grad = -A / (xi*r) * (vec/r)
        grad[i] -= pair_grad
        grad[j] += pair_grad
    return grad.ravel()

# Pure expansion ray after removing rigid-body components, normalized in Q coordinates.
qexp = Q.T @ centered.ravel()
qexp /= np.linalg.norm(qexp)
# d is the unnormalized internal uniform-dilation direction in Q coordinates.
d = Q.T @ centered.ravel()
results = {}
for fraction, label in ((0.03, "fraction_0.03"), (0.48, "fraction_0.48")):
    grad_cart = ls_gradient(fraction)
    g = Q.T @ grad_cart
    if fraction == 0.03:
        analytic_force = -grad_cart.reshape(60, 3)
        force_diff = float(np.max(np.abs(soft_forces - analytic_force)))
        if force_diff > 1e-12:
            raise RuntimeError(f"FrozenBondSoftening force mismatch: {force_diff}")
    y = np.linalg.solve(H, g)
    chi = float(g @ y)
    dilation_fraction = float((d @ g)**2 / ((d @ H @ d) * chi))
    displacement = -y
    projection = float(qexp @ displacement)
    results[label] = {
        "fraction": fraction,
        "number_of_frozen_pairs": len(pairs),
        "chi_eV": chi,
        "physical_V_quadratic_deformation_cost_a1_eV": 0.5 * chi,
        "modified_F_equals_V_plus_aW_local_energy_change_a1_eV": -0.5 * chi,
        "physical_V_deformation_cost_per_atom_a1_eV_per_atom": 0.5 * chi / 60,
        "modified_F_local_change_per_atom_a1_eV_per_atom": -0.5 * chi / 60,
        "H_metric_expansion_fraction_(dTg)^2_over_(dTHd_chi)": dilation_fraction,
        "predicted_internal_displacement_norm_a1_A": float(np.linalg.norm(displacement)),
        "predicted_expansion_projection_a1_A": projection,
        "predicted_expansion_projection_over_internal_radius_a1": projection / np.linalg.norm(centered),
        "predicted_expansion_fraction_of_displacement_a1": projection / np.linalg.norm(displacement),
        "g_norm_eV_per_A": float(np.linalg.norm(g)),
    }
# fraction .48 is 16 * .03, so same base W has a=16. Energies scale a^2;
# displacements and signed expansion projection scale a in linear response.
base = results["fraction_0.03"]
results["fraction_0.48_as_a16_of_fraction_0.03"] = {
    "a": 16,
    "physical_V_quadratic_deformation_cost_eV": 0.5 * 16**2 * base["chi_eV"],
    "modified_F_equals_V_plus_aW_local_energy_change_eV": -0.5 * 16**2 * base["chi_eV"],
    "physical_V_deformation_cost_per_atom_eV_per_atom": 0.5 * 16**2 * base["chi_eV"] / 60,
    "modified_F_local_change_per_atom_eV_per_atom": -0.5 * 16**2 * base["chi_eV"] / 60,
    "H_metric_expansion_fraction_(dTg)^2_over_(dTHd_chi)": base["H_metric_expansion_fraction_(dTg)^2_over_(dTHd_chi)"],
    "predicted_internal_displacement_norm_A": 16 * base["predicted_internal_displacement_norm_a1_A"],
    "predicted_expansion_projection_A": 16 * base["predicted_expansion_projection_a1_A"],
    "predicted_expansion_projection_over_internal_radius": 16 * base["predicted_expansion_projection_over_internal_radius_a1"],
}
result = {
    "scope": "offline fixed-W harmonic response from archived geometry/Hessian; no Calculator, PES, or minimization",
    "inputs": {"hessian_npz": str(NPZ), "geometry": str(GEOM), "curvature_source": str(EVIDENCE / "curvature/runner.py"),
               "configuration_source": str(ROOT / "research/ga_ssw/evidence/c60-source3-paper-ls-transfer-20261007/protocol.md")},
    "consistency": {"max_abs_geometry_difference_A": float(np.max(np.abs(x-X))),
                    "basis_orthogonality_max_abs": float(np.max(np.abs(Q.T@Q-np.eye(174)))),
                    "rigid_basis_overlap_max_abs": float(np.max(np.abs(rigid.T@Q))),
                    "projected_H_eigenvalue_min_eV_A2": float(w.min()),
                    "projected_H_eigenvalue_max_eV_A2": float(w.max()),
                    "raw_H_antisymmetric_spectral_norm_eV_A2": float(np.linalg.norm((Hraw-Hraw.T)/2, ord=2)),
                    "cutoff_A": cutoff, "xi": xi, "pair_strength_at_0.03_eV": .03*bond_energy,
                    "pair_strength_at_0.48_eV": .48*bond_energy, "frozen_pair_count": len(pairs),
                    "FrozenBondSoftening_pair_list_matches_independent_cutoff": list(soft.pairs) == pairs,
                    "FrozenBondSoftening_analytic_energy_at_reference_eV": soft_energy,
                    "analytic_gradient_vs_minus_evaluator_force_max_abs_eV_A": force_diff,
                    "atom_order_matches_extxyz": True, "units": "x=A; H=eV/A^2; gradW=eV/A; chi=eV"},
    "response": results,
    "interpretation_limits": [
        "Hessian comes from a single central force-difference stencil h=0.01 A, not a dual-step stability certificate.",
        "This is a fixed-frozen-pair, first-order response at one C60 isomer, not a relaxed finite-amplitude result.",
        "The quadratic +a^2 chi/2 is the physical V deformation cost; the modified F=V+aW local energy change is -a^2 chi/2. It is not a prediction that physical V decreases.",
        "At a=16 the predicted internal displacement and physical V deformation cost are large; use only as a warning that weak-response expansion is unlikely to be controlled, not as a quantitative prediction.",
        "No cross-reference geometry or Hessian was mixed; source geometry equals stored Hessian positions in original atom order."
    ]}
OUT.write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2))
