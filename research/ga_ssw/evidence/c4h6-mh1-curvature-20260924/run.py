#!/home/gengjianrui/.conda/envs/mace_env/bin/python
"""One-pass analytic MH-1 Hessian check of three saved C4H6 pilot frames."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import torch
from ase import Atoms
from mace.calculators import MACECalculator


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PILOT = REPO / "research/ga_ssw/evidence/c4h6-mh1-lifecycle-20260924"
MODEL = Path("/home/gengjianrui/.cache/mace/mace-mh-1.model")
MODEL_SHA256 = "a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47"
EXPECTED_CORE_TREE = "46f0049ed02c773621ac614093386220227e3d3a"
CASES = (
    ("ssw", 0),
    ("paper_ls", 1),
    ("native_ls", 3),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_case(arm: str, minimum_index: int) -> tuple[Atoms, dict, dict, dict]:
    result_path = PILOT / arm / "result.json"
    fresh_path = PILOT / arm / "fresh-requests.jsonl"
    result = json.loads(result_path.read_text())
    fresh_lines = fresh_path.read_text().splitlines()
    if minimum_index >= len(fresh_lines):
        raise IndexError(f"missing fresh request for minimum index {minimum_index}")
    fresh = json.loads(fresh_lines[minimum_index])
    minimum = result["minima"][minimum_index]
    if fresh.get("kind") != "search" or fresh.get("request") != minimum_index + 1:
        raise ValueError(
            f"fresh request/index mismatch: arm={arm}, minimum={minimum_index}, "
            f"request={fresh.get('request')}, kind={fresh.get('kind')}"
        )
    for field in ("numbers", "positions", "cell", "pbc"):
        if fresh["atoms"].get(field) != minimum["atoms"].get(field):
            raise ValueError(
                f"fresh request and saved minimum disagree in {field}: "
                f"arm={arm}, minimum={minimum_index}"
            )
    frame_fields_match = ["numbers", "positions", "cell", "pbc"]
    atoms = Atoms(**fresh["atoms"])
    if len(atoms) != 10 or atoms.pbc.any():
        raise ValueError(f"unexpected input system for {arm} minimum {minimum_index}")
    provenance = {
        "pilot_result": str(result_path),
        "pilot_fresh_request": str(fresh_path),
        "pilot_minimum_index": minimum_index,
        "pilot_fresh_request_index_1based": fresh["request"],
        "fresh_minimum_frame_fields_verified_equal": frame_fields_match,
        "pilot_result_sha256": sha256_file(result_path),
        "pilot_fresh_requests_sha256": sha256_file(fresh_path),
    }
    reference = {
        "energy_eV": minimum["energy"],
        "max_force_eV_per_A": minimum["max_force"],
        "fresh_check_energy_eV": fresh["energy"],
        "fresh_check_max_force_eV_per_A": fresh["fmax"],
    }
    return atoms, provenance, reference, minimum


def hessian_matrix(raw: np.ndarray, natoms: int) -> np.ndarray:
    expected_shape = (3 * natoms, natoms, 3)
    if raw.shape != expected_shape:
        raise ValueError(f"expected raw Hessian shape {expected_shape}, got {raw.shape}")
    matrix = raw.reshape(3 * natoms, 3 * natoms)
    if not np.isfinite(matrix).all():
        raise ValueError("raw Hessian contains non-finite entries")
    return matrix


def rigid_basis(atoms: Atoms) -> np.ndarray:
    relative = atoms.positions - atoms.positions.mean(axis=0)
    rotational = np.column_stack(
        [np.cross(axis, relative).ravel() for axis in np.eye(3)]
    )
    rotation_vectors, singular_values, _ = np.linalg.svd(rotational, full_matrices=False)
    tolerance = np.finfo(float).eps * max(rotational.shape) * singular_values[0]
    if np.count_nonzero(singular_values > tolerance) != 3:
        raise ValueError("rigid-body projection requires a nonlinear molecule")
    translation = np.column_stack([
        np.broadcast_to(axis, relative.shape).ravel() / np.sqrt(len(atoms))
        for axis in np.eye(3)
    ])
    return np.column_stack((translation, rotation_vectors))


def analyze_case(arm: str, minimum_index: int, model_sha256: str) -> dict:
    name = f"{arm}-min{minimum_index:02d}"
    output = HERE / name
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    row: dict = {
        "case": name,
        "arm": arm,
        "minimum_index": minimum_index,
        "status": "started",
        "pilot_core_tree_reference": EXPECTED_CORE_TREE,
        "project_core_imports": [],
    }
    dump_json(output / "result.json", row)
    try:
        atoms, provenance, reference, _minimum = load_case(arm, minimum_index)
        row.update(provenance=provenance, reference=reference)
        dump_json(output / "result.json", row)
        dump_json(output / "input.json", {
            "provenance": provenance,
            "reference": reference,
            "numbers": atoms.numbers.tolist(),
            "positions_A": atoms.positions.tolist(),
            "cell_A": atoms.cell.array.tolist(),
            "pbc": atoms.pbc.tolist(),
        })

        calculator_kwargs = {
            "model_paths": str(MODEL),
            "head": "omol",
            "device": "cpu",
            "default_dtype": "float64",
            "enable_cueq": False,
            "enable_oeq": False,
        }
        row["calculator_requested_config"] = calculator_kwargs
        dump_json(output / "result.json", row)
        calculator_setup_started = time.perf_counter()
        calc = MACECalculator(**calculator_kwargs)
        row["calculator_setup_seconds"] = time.perf_counter() - calculator_setup_started
        atoms.calc = calc
        row["calculator"] = {
            **calculator_kwargs,
            "model_sha256": model_sha256,
            "energy_units_to_eV": float(calc.energy_units_to_eV),
            "length_units_to_A": float(calc.length_units_to_A),
            "hessian_units": "eV/Angstrom^2 (raw model output; conversion factors are 1)",
            "mace_torch_version": importlib.metadata.version("mace-torch"),
            "torch_version": torch.__version__,
            "ase_version": importlib.metadata.version("ase"),
            "model_count": int(calc.num_models),
            "model_type": calc.model_type,
            "active_head": calc.head,
            "available_heads": list(calc.available_heads),
            "hessian_api_kwargs": {
                "compute_hessian": True,
                "compute_stress": False,
                "training": bool(calc.use_compile),
            },
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "omp_num_threads": os.environ["OMP_NUM_THREADS"],
            "mkl_num_threads": os.environ["MKL_NUM_THREADS"],
            "openblas_num_threads": os.environ["OPENBLAS_NUM_THREADS"],
        }
        if row["calculator"]["model_sha256"] != MODEL_SHA256:
            raise ValueError("MH-1 model SHA256 differs from completed pilot provenance")
        if (calc.energy_units_to_eV != 1.0 or calc.length_units_to_A != 1.0
                or calc.num_models != 1 or calc.model_type != "MACE"
                or calc.head != "omol"):
            raise ValueError("calculator units/model contract differs from the planned Hessian units")
        dump_json(output / "result.json", row)

        row["fresh_energy_force_requests"] = 1
        started_ef = time.perf_counter()
        try:
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=float)
        finally:
            ef_seconds = time.perf_counter() - started_ef
            row["fresh_energy_force_seconds"] = ef_seconds
            dump_json(output / "result.json", row)
        if not np.isfinite(energy) or not np.isfinite(forces).all():
            raise ValueError("fresh energy/forces contain non-finite values")
        fmax = float(np.linalg.norm(forces, axis=1).max())
        row.update(
            fresh_energy_eV=energy,
            fresh_max_force_eV_per_A=fmax,
            energy_delta_from_pilot_eV=energy - float(reference["energy_eV"]),
            max_force_delta_from_pilot_eV_per_A=(
                fmax - float(reference["max_force_eV_per_A"])
            ),
            fresh_energy_force_seconds=ef_seconds,
        )
        dump_json(output / "result.json", row)

        row["fresh_hessian_requests"] = 1
        started_hessian = time.perf_counter()
        try:
            raw_hessian = np.asarray(calc.get_hessian(atoms))
        finally:
            hessian_seconds = time.perf_counter() - started_hessian
            row["fresh_hessian_seconds"] = hessian_seconds
            dump_json(output / "result.json", row)
        np.save(output / "raw_hessian.npy", raw_hessian)
        dump_json(output / "result.json", row)
        matrix = hessian_matrix(raw_hessian, len(atoms))
        rigid = rigid_basis(atoms)
        if rigid.shape != (3 * len(atoms), 6):
            raise ValueError(f"expected six rigid modes, got basis shape {rigid.shape}")
        complete_basis = np.linalg.qr(rigid, mode="complete")[0]
        external_basis = complete_basis[:, :6]
        internal_basis = complete_basis[:, 6:]
        symmetric = (matrix + matrix.T) / 2.0
        skew = (matrix - matrix.T) / 2.0
        internal_hessian = internal_basis.T @ symmetric @ internal_basis
        full_eigenvalues = np.linalg.eigvalsh(symmetric)
        internal_eigenvalues, internal_eigenvectors = np.linalg.eigh(internal_hessian)
        lowest_cartesian_mode = internal_basis @ internal_eigenvectors[:, 0]
        row.update(
            status="completed",
            raw_hessian_shape=list(raw_hessian.shape),
            hessian_matrix_shape=list(matrix.shape),
            hessian_all_finite=bool(np.isfinite(matrix).all()),
            hessian_max_abs_antisymmetric_eV_per_A2=float(np.max(np.abs(matrix - matrix.T))),
            hessian_antisymmetric_frobenius_eV_per_A2=float(np.linalg.norm(skew, ord="fro")),
            cartesian_eigenvalues_eV_per_A2=full_eigenvalues.tolist(),
            internal_eigenvalues_eV_per_A2=internal_eigenvalues.tolist(),
            negative_internal_eigenvalues_eV_per_A2=(
                internal_eigenvalues[internal_eigenvalues < 0.0].tolist()
            ),
            lowest_internal_eigenvalue_eV_per_A2=float(internal_eigenvalues[0]),
            lowest_internal_mode_cartesian=lowest_cartesian_mode.reshape((-1, 3)).tolist(),
            rigid_basis_shape=list(external_basis.shape),
            internal_basis_shape=list(internal_basis.shape),
            fresh_hessian_seconds=hessian_seconds,
            interpretation=(
                "Curvature at the saved geometry only. At nonzero force, a negative mode "
                "does not establish a first-order saddle or exclude a nearby minimum. "
                "No near-zero cutoff or minimum certification is applied."
            ),
        )
        np.savez(
            output / "hessian.npz",
            raw_hessian=raw_hessian,
            hessian_matrix=matrix,
            symmetrized_hessian=symmetric,
            rigid_basis=external_basis,
            internal_basis=internal_basis,
            projected_internal_hessian=internal_hessian,
            cartesian_eigenvalues=full_eigenvalues,
            internal_eigenvalues=internal_eigenvalues,
            lowest_internal_mode_cartesian=lowest_cartesian_mode,
        )
    except Exception as error:
        row.update(status="failed", error=repr(error), traceback=traceback.format_exc())
    row["total_case_seconds"] = time.perf_counter() - started
    dump_json(output / "result.json", row)
    return row


def main() -> None:
    if not MODEL.is_file():
        raise FileNotFoundError(MODEL)
    if not PILOT.is_dir():
        raise FileNotFoundError(PILOT)
    result_path = HERE / "results.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")
    torch.set_num_threads(1)
    started = time.perf_counter()
    model_hash_started = time.perf_counter()
    actual_model_sha256 = sha256_file(MODEL)
    model_hash_seconds = time.perf_counter() - model_hash_started
    if actual_model_sha256 != MODEL_SHA256:
        raise ValueError("MH-1 model SHA256 differs from completed pilot provenance")
    torch.set_num_interop_threads(1)
    rows = []
    for arm, minimum_index in CASES:
        rows.append(analyze_case(arm, minimum_index, actual_model_sha256))
        dump_json(result_path, {
            "status": "running",
            "model": str(MODEL),
            "model_sha256_expected": MODEL_SHA256,
            "model_sha256_actual": actual_model_sha256,
            "model_hash_seconds": model_hash_seconds,
            "head": "omol",
            "device": "cpu",
            "dtype": "float64",
            "pilot_core_tree_reference": EXPECTED_CORE_TREE,
            "cases": rows,
            "elapsed_seconds": time.perf_counter() - started,
        })
    summary = {
        "status": "completed" if all(row["status"] == "completed" for row in rows)
        else "completed_with_frame_failures",
        "model": str(MODEL),
        "model_sha256_expected": MODEL_SHA256,
        "model_sha256_actual": actual_model_sha256,
        "model_hash_seconds": model_hash_seconds,
        "head": "omol",
        "device": "cpu",
        "dtype": "float64",
        "pilot_core_tree_reference": EXPECTED_CORE_TREE,
        "cases": rows,
        "elapsed_seconds": time.perf_counter() - started,
        "scope": "three saved pilot frames; no relaxation, tuning, xTB, or automatic expansion",
    }
    dump_json(result_path, summary)
    print(json.dumps({
        "status": summary["status"],
        "cases": [
            {"case": row["case"], "status": row["status"],
             "fresh_energy_force_seconds": row.get("fresh_energy_force_seconds"),
             "fresh_hessian_seconds": row.get("fresh_hessian_seconds"),
             "negative_internal_eigenvalues_eV_per_A2": row.get(
                 "negative_internal_eigenvalues_eV_per_A2"
             )}
            for row in rows
        ],
        "elapsed_seconds": summary["elapsed_seconds"],
    }, indent=2))
    if summary["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
