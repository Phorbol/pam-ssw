"""Recompute same-point pair/shape susceptibilities from archived E/F-free data.

This script reads only archived geometries and physical Hessians. It invokes
analytic frozen-bias energy/force methods, never an ASE calculator, and writes
calibration.json next to this script.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import null_space
from ase.io import read

from pamssw.standalone.cluster_frame import ClusterFrame
from pamssw.standalone.native_ls import HC_BOND_ENERGIES, HC_BOND_LENGTHS
from pamssw.standalone.shape_softening import FrozenShapeBondSoftening
from pamssw.standalone.softening import FrozenBondSoftening


ROOT = Path(__file__).resolve().parents[5]
EVIDENCE = ROOT / "research/ga_ssw/evidence/ls-theory-20261010"
OUTPUT = Path(__file__).with_name("calibration.json")
BASE_FRACTION = 0.03
PROTOCOL_PATH = "../shape-linear-response-protocol.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _c4_case(name: str, endpoint: str):
    folder = EVIDENCE / "channel-v1/run-1728173"
    return {
        "name": name,
        "input": folder / f"{endpoint}.extxyz",
        "campaign_input": (EVIDENCE / "shape-single-escape-v1/source/research/ga_ssw/"
                           f"evidence/ls-theory-20261010/channel-v1/run-1728173/{endpoint}.extxyz"),
        "hessian": folder / f"{endpoint}-hessian.npz",
        "hessian_result": EVIDENCE / "channel-v1/run-1728173/result.json",
        "hessian_key": "H_h0005",
        "dual_hessian_key": "H_h001",
        "chart_input": folder / "input.extxyz",
        "bond_energies": HC_BOND_ENERGIES,
        "bond_lengths": {key: value + 0.1 for key, value in HC_BOND_LENGTHS.items()},
        "expected_pairs": 9,
    }


def cases():
    c60 = EVIDENCE / "c60-finite-response-v1/run-1729152"
    c60_protocol_copy = (EVIDENCE / "shape-single-escape-v1/source/research/ga_ssw/"
                         "evidence/ls-theory-20261010/c60-finite-response-v1/run-1729152/minimum.extxyz")
    return [
        {
            "name": "c60-minimum",
            "input": c60 / "minimum.extxyz",
            "campaign_input": c60_protocol_copy,
            "hessian": c60 / "baseline.npz",
            "hessian_result": c60 / "result.json",
            "hessian_key": "H",
            "dual_hessian_key": None,
            "chart_input": c60 / "minimum.extxyz",
            "bond_energies": {(6, 6): 3.61},
            "bond_lengths": {(6, 6): 1.64},
            "expected_pairs": 90,
        },
        _c4_case("c4-gauche", "plus-0.050"),
        _c4_case("c4-trans", "minus-0.050"),
    ]


def _hessian_qualification(case, arrays, result):
    if case["name"].startswith("c60"):
        baseline = result["baseline"]
        h = arrays[case["hessian_key"]]
        eigenvalues = np.linalg.eigvalsh((h + h.T) / 2)
        spread = float(baseline["stencil_spread"])
        margin = float(eigenvalues[0] / spread)
        return {
            "qualification_status": "completed; resolved strict minimum",
            "physical_minimum_fmax_eV_A": baseline["fmax"],
            "physical_hessian_stencils_A": [0.001, 0.0005],
            "saved_hessian_stencil_A": 0.0005,
            "stencil_spread_eV_A2": spread,
            "antisymmetric_norms_eV_A2": baseline["antisymmetric_norms"],
            "lambda_min_eV_A2": float(eigenvalues[0]),
            "lambda_min_over_stencil_spread": margin,
            "acceptance": "lambda_min > 5 * dual-stencil spread",
            "positive_resolved": bool(margin > 5),
            "chart": "minimum-centered affine Eckart internal basis saved as Q",
        }

    endpoint = "plus-0.050" if case["name"].startswith("c4-gauche") else "minus-0.050"
    phase = result["phases"][endpoint]
    hphase = result["phases"][endpoint + "_hessian"]
    eigenvalues = arrays["eigenvalues"]
    spread = float(hphase["stencil_spread_eV_A2"])
    margin = float(eigenvalues[0] / spread)
    return {
        "qualification_status": "force-qualified stable minimum; dual-stencil Hessian qualified",
        "physical_minimum_fmax_eV_A": phase["fmax_eV_A"],
        "physical_hessian_stencils_A": hphase["steps_A"],
        "saved_hessian_stencil_A": 0.0005,
        "stencil_spread_eV_A2": spread,
        "antisymmetric_norms_eV_A2": hphase["antisymmetric_norms_eV_A2"],
        "lambda_min_eV_A2": float(eigenvalues[0]),
        "lambda_min_over_stencil_spread": margin,
        "acceptance": hphase["criterion"],
        "positive_resolved": bool(hphase["positive_resolved_hessian"]),
        "chart": "fixed affine Eckart internal chart based at channel-v1 input.extxyz",
    }


def calibrate(case):
    atoms = read(case["input"])
    atoms.calc = None
    if not case["campaign_input"].is_file():
        raise FileNotFoundError(case["campaign_input"])
    if not np.array_equal(read(case["campaign_input"]).positions, atoms.positions):
        raise ValueError(f"campaign geometry differs from Hessian geometry: {case['name']}")
    geometry_hash = sha256(case["input"])
    campaign_hash = sha256(case["campaign_input"])
    if geometry_hash != campaign_hash:
        raise ValueError(f"campaign geometry bytes differ from Hessian geometry: {case['name']}")

    hessian_result = json.loads(case["hessian_result"].read_text())
    arrays = np.load(case["hessian"])
    hessian = arrays[case["hessian_key"]]
    q = arrays["Q"]
    if q.shape != (3 * len(atoms), hessian.shape[0]):
        raise ValueError(f"Hessian/chart dimension mismatch: {case['name']}")
    qualification = _hessian_qualification(case, arrays, hessian_result)
    if not qualification["positive_resolved"]:
        raise ValueError(f"physical Hessian is not a resolved strict minimum: {case['name']}")

    pair = FrozenBondSoftening.from_atoms(
        atoms, bond_energies=case["bond_energies"], bond_lengths=case["bond_lengths"],
        initial_fraction=BASE_FRACTION, xi=0.2,
    )
    shape = FrozenShapeBondSoftening.from_atoms(
        atoms, bond_energies=case["bond_energies"], bond_lengths=case["bond_lengths"],
        initial_fraction=BASE_FRACTION, xi=0.2,
    )
    if len(pair.pairs) != case["expected_pairs"] or shape.pairs != pair.pairs:
        raise ValueError(f"pair selection mismatch: {case['name']}")
    pair_gradient = -pair.evaluate(atoms)[1].reshape(-1)
    shape_gradient = -shape.evaluate(atoms)[1].reshape(-1)
    pair_q = q.T @ pair_gradient
    shape_q = q.T @ shape_gradient
    chi_pair = float(pair_q @ np.linalg.solve(hessian, pair_q))
    chi_shape = float(shape_q @ np.linalg.solve(hessian, shape_q))
    if not (np.isfinite(chi_pair) and np.isfinite(chi_shape) and chi_pair > 0 and chi_shape > 0):
        raise ValueError(f"nonpositive/nonfinite susceptibility: {case['name']}")

    chart_record = {"basis_input": str(case["chart_input"].relative_to(ROOT)),
                    "basis_input_sha256": sha256(case["chart_input"])}
    if case["chart_input"] != case["input"]:
        reference = read(case["chart_input"])
        chart0 = ClusterFrame(reference).basis
        chart1 = ClusterFrame(atoms).basis
        current_internal = null_space(chart1.T)
        overlap = np.linalg.svd(chart0.T @ chart1, compute_uv=False)
        internal_overlap = np.linalg.svd(current_internal.T @ q, compute_uv=False)
        chart_record.update({
            "kind": "fixed affine Eckart section",
            "endpoint_rigid_subspace_overlap_singular_values": overlap.tolist(),
            "minimum_overlap_singular_value": float(overlap[-1]),
            "transverse_to_endpoint_rigid_modes": bool(overlap[-1] > 1e-10),
            "current_internal_basis_overlap_singular_values": internal_overlap.tolist(),
            "current_internal_basis_overlap_min_singular_value": float(internal_overlap[-1]),
            "saved_Q_transverse_to_current_rigid_modes": bool(internal_overlap[-1] > 1e-10),
        })
        if (not chart_record["transverse_to_endpoint_rigid_modes"] or
                not chart_record["saved_Q_transverse_to_current_rigid_modes"]):
            raise ValueError(f"saved chart/internal subspace mismatch: {case['name']}")
    else:
        chart_record.update({"kind": "minimum-centered affine Eckart section",
                             "transverse_to_endpoint_rigid_modes": True})

    factor = float(np.sqrt(chi_pair / chi_shape))
    dual_response = None
    if case["dual_hessian_key"] is not None:
        dual_hessian = arrays[case["dual_hessian_key"]]
        chi_pair_dual = float(pair_q @ np.linalg.solve(dual_hessian, pair_q))
        chi_shape_dual = float(shape_q @ np.linalg.solve(dual_hessian, shape_q))
        dual_response = {
            "stencil_A": 0.001,
            "chi_pair_eV": chi_pair_dual,
            "chi_shape_eV": chi_shape_dual,
            "amplitude_ratio_shape_over_pair": float(np.sqrt(chi_pair_dual / chi_shape_dual)),
        }
    return {
        "input_sha256": geometry_hash,
        "campaign_input_sha256": campaign_hash,
        "campaign_geometry_byte_identical": True,
        "initial_fraction": BASE_FRACTION * factor,
        "base_pair_fraction": BASE_FRACTION,
        "chi_pair_eV": chi_pair,
        "chi_shape_eV": chi_shape,
        "amplitude_ratio_shape_over_pair": factor,
        "dual_stencil_response_check": dual_response,
        "pair_count": len(pair.pairs),
        "radius0_A": shape.radius0,
        "bias_parameters": {
            "initial_fraction_pair": BASE_FRACTION,
            "xi": 0.2,
            "bond_energies_eV": {str(key): value for key, value in case["bond_energies"].items()},
            "bond_lengths_A": {str(key): value for key, value in case["bond_lengths"].items()},
        },
        "hessian": {
            "path": str(case["hessian"].relative_to(ROOT)),
            "sha256": sha256(case["hessian"]),
            "matrix_key": case["hessian_key"],
            "dual_matrix_key": case["dual_hessian_key"],
            "dimension": int(hessian.shape[0]),
            **qualification,
            **chart_record,
        },
        "calibration_source": str(case["hessian_result"].relative_to(ROOT)),
        "calibration_details": {
            "input_path": str(case["input"].relative_to(ROOT)),
            "hessian_result_path": str(case["hessian_result"].relative_to(ROOT)),
            "bias_definition": "FrozenBondSoftening and FrozenShapeBondSoftening; same pair list, bond tables, xi=0.2, base fraction 0.03",
            "equation": "chi = (Q^T grad W)^T H_Q^{-1} (Q^T grad W); shape uses its analytic pulled-back gradient at the same geometry",
        },
    }


def build_manifest():
    entries = {case["name"]: calibrate(case) for case in cases()}
    return {
        "status": "offline_same_point_linear_response_calibration",
        "scope": "No Calculator/PES/optimizer calls; susceptibility from archived physical Hessians and analytic bias gradients only.",
        "protocol_path": PROTOCOL_PATH,
        "calibration_script": str(Path(__file__).resolve().relative_to(ROOT)),
        "calibration_script_sha256": sha256(Path(__file__).resolve()),
        "base_initial_fraction": BASE_FRACTION,
        "starts": entries,
        "interpretation_limit": "The rescaled fraction equalizes only the O(a^2) physical-minimum relaxation energy response. It does not equalize response vectors, transition-state/barrier response, finite prequench behavior, or SSW outcomes.",
    }


if __name__ == "__main__":
    OUTPUT.write_text(json.dumps(build_manifest(), indent=2) + "\n")
    print(OUTPUT)
