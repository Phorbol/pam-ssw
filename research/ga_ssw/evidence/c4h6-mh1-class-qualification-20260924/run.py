#!/home/gengjianrui/.conda/envs/mace_env/bin/python
"""Fresh E/F and analytic MH-1 Hessians for fixed C4H6 class representatives."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import os
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
COVERAGE = REPO / "research/ga_ssw/evidence/c4h6-mh1-coverage-20260924"
CURVATURE_HELPER = REPO / "research/ga_ssw/evidence/c4h6-mh1-curvature-20260924/run.py"
MODEL = Path("/home/gengjianrui/.cache/mace/mace-mh-1.model")
EXPECTED_COUNTS = {
    "native_ls-seed61": 10,
    "native_ls-seed67": 8,
    "paper_ls-seed61": 6,
    "paper_ls-seed67": 8,
    "ssw-seed61": 5,
    "ssw-seed67": 5,
}
EXISTING_FMAX_LIMIT_EV_PER_A = 0.03
EXISTING_ENERGY_ERROR_LIMIT_EV = 1e-6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def case_name(frame: dict) -> str:
    return (
        f"{frame['arm_seed']}-class{frame['global_graph_class_id']:03d}"
        f"-min{frame['minimum_index']:03d}"
    )


def load_curvature_helpers():
    spec = importlib.util.spec_from_file_location("completed_c4h6_curvature_runner", CURVATURE_HELPER)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load curvature helper: {CURVATURE_HELPER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.rigid_basis, module.hessian_matrix


def result_path_for(frame: dict) -> Path:
    return (HERE / frame["source_json"]).resolve()


def output_path_for(frame: dict) -> Path:
    return HERE / case_name(frame)


def initial_row(frame: dict, manifest_sha256: str) -> dict:
    return {
        "case": case_name(frame),
        "status": "started",
        "frame": frame,
        "manifest_sha256": manifest_sha256,
    }


def write_failed_frame(frame: dict, manifest_sha256: str, error: Exception, details: str) -> dict:
    output = output_path_for(frame)
    output.mkdir(exist_ok=False)
    row = initial_row(frame, manifest_sha256)
    row.update(status="failed", error=repr(error), traceback=details)
    dump_json(output / "result.json", row)
    return row


def analyze_frame(
    frame: dict,
    minimum: dict,
    source_json: Path,
    source_sha256: str,
    manifest_sha256: str,
    calculator: MACECalculator,
    model_config: dict,
    rigid_basis_fn,
    hessian_matrix_fn,
) -> dict:
    output = output_path_for(frame)
    output.mkdir(exist_ok=False)
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    row = initial_row(frame, manifest_sha256)
    row.update(source_json=str(source_json), source_json_sha256=source_sha256, model=model_config)
    dump_json(output / "result.json", row)
    try:
        if minimum["energy"] != frame["energy_eV"]:
            raise ValueError("manifest energy differs from source result minimum")
        atom_data = minimum["atoms"]
        atoms = Atoms(**atom_data)
        numbers = atoms.numbers.tolist()
        if len(atoms) != 10 or numbers.count(6) != 4 or numbers.count(1) != 6:
            raise ValueError("source frame is not the expected C4H6 composition")
        if atoms.pbc.any():
            raise ValueError("source frame unexpectedly has periodic boundaries")
        input_record = {
            "frame": frame,
            "source_json": str(source_json),
            "source_json_sha256": source_sha256,
            "source_minimum_energy_eV": minimum["energy"],
            "source_minimum_max_force_eV_per_A": minimum["max_force"],
            "source_minimum_converged": minimum["converged"],
            "numbers": numbers,
            "positions_A": atoms.positions.tolist(),
            "cell_A": atoms.cell.array.tolist(),
            "pbc": atoms.pbc.tolist(),
        }
        dump_json(output / "input.json", input_record)
        row.update(
            input_record=input_record,
            source_minimum_energy_eV=minimum["energy"],
            source_minimum_max_force_eV_per_A=minimum["max_force"],
        )
        dump_json(output / "result.json", row)

        calculator.reset()
        atoms.calc = calculator
        row["fresh_energy_force_requests"] = 1
        ef_wall_start = time.perf_counter()
        ef_cpu_start = time.process_time()
        try:
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=float)
        finally:
            row["fresh_energy_force_wall_seconds"] = time.perf_counter() - ef_wall_start
            row["fresh_energy_force_cpu_seconds"] = time.process_time() - ef_cpu_start
            dump_json(output / "result.json", row)
        if not np.isfinite(energy) or not np.isfinite(forces).all():
            raise ValueError("fresh energy/forces contain non-finite values")
        fmax = float(np.linalg.norm(forces, axis=1).max())
        row.update(
            fresh_energy_eV=energy,
            fresh_max_force_eV_per_A=fmax,
            energy_error_from_source_eV=energy - float(minimum["energy"]),
            max_force_error_from_source_eV_per_A=fmax - float(minimum["max_force"]),
            fresh_fmax_nonzero=(fmax != 0.0),
            stationarity_claim="not established by this curvature check",
        )
        row["fresh_numerical_qualification"] = {
            "criteria_source": "existing coverage protocol: fmax <= 0.03 eV/A and abs(energy_error) <= 1e-6 eV",
            "fmax_limit_eV_per_A": EXISTING_FMAX_LIMIT_EV_PER_A,
            "energy_error_limit_eV": EXISTING_ENERGY_ERROR_LIMIT_EV,
            "fmax_pass": fmax <= EXISTING_FMAX_LIMIT_EV_PER_A,
            "energy_error_pass": abs(row["energy_error_from_source_eV"]) <= EXISTING_ENERGY_ERROR_LIMIT_EV,
        }
        row["fresh_numerically_qualified"] = bool(
            row["fresh_numerical_qualification"]["fmax_pass"]
            and row["fresh_numerical_qualification"]["energy_error_pass"]
        )
        dump_json(output / "result.json", row)

        row["analytic_hessian_requests"] = 1
        hessian_wall_start = time.perf_counter()
        hessian_cpu_start = time.process_time()
        try:
            raw_hessian = np.asarray(calculator.get_hessian(atoms))
        finally:
            row["hessian_wall_seconds"] = time.perf_counter() - hessian_wall_start
            row["hessian_cpu_seconds"] = time.process_time() - hessian_cpu_start
            dump_json(output / "result.json", row)
        np.save(output / "raw_hessian.npy", raw_hessian)
        row["raw_hessian_shape"] = list(raw_hessian.shape)
        row["raw_hessian_all_finite"] = bool(np.isfinite(raw_hessian).all())
        dump_json(output / "result.json", row)

        matrix = hessian_matrix_fn(raw_hessian, len(atoms))
        rigid = np.asarray(rigid_basis_fn(atoms), dtype=float)
        if rigid.shape != (3 * len(atoms), 6):
            raise ValueError(f"expected six rigid modes, got basis shape {rigid.shape}")
        complete_basis = np.linalg.qr(rigid, mode="complete")[0]
        external_basis = complete_basis[:, :6]
        internal_basis = complete_basis[:, 6:]
        antisymmetric_error = matrix - matrix.T
        symmetric = (matrix + matrix.T) / 2.0
        projected = internal_basis.T @ symmetric @ internal_basis
        eigenvalues, eigenvectors = np.linalg.eigh(projected)
        if not np.isfinite(eigenvalues).all():
            raise ValueError("projected internal spectrum contains non-finite values")
        if np.any(eigenvalues < 0.0):
            category = "negative_curvature"
        elif np.any(eigenvalues == 0.0):
            category = "exact_zero_sign_ambiguous"
        elif np.all(eigenvalues > 0.0):
            category = "strictly_positive_spectrum"
        else:
            raise ValueError("could not classify finite internal spectrum by exact sign")
        lowest_mode = internal_basis @ eigenvectors[:, 0]
        row.update(
            status="completed",
            raw_hessian_shape=list(raw_hessian.shape),
            hessian_matrix_shape=list(matrix.shape),
            hessian_all_finite=bool(np.isfinite(matrix).all()),
            hessian_max_abs_antisymmetric_eV_per_A2=float(np.max(np.abs(antisymmetric_error))),
            hessian_antisymmetric_frobenius_eV_per_A2=float(np.linalg.norm(antisymmetric_error, ord="fro")),
            internal_eigenvalues_eV_per_A2=eigenvalues.tolist(),
            negative_internal_eigenvalues_eV_per_A2=eigenvalues[eigenvalues < 0.0].tolist(),
            lowest_internal_eigenvalue_eV_per_A2=float(eigenvalues[0]),
            lowest_internal_mode_cartesian=lowest_mode.reshape((-1, 3)).tolist(),
            curvature_category=category,
            rigid_basis_shape=list(external_basis.shape),
            internal_basis_shape=list(internal_basis.shape),
            interpretation=(
                "Local curvature at the unchanged saved geometry only. Nonzero Fmax means "
                "stationarity is not established; a positive spectrum is not a certified "
                "minimum and negative curvature is not a certified first-order saddle. "
                "Bond order, radicals, and electronic state are not validated."
            ),
        )
        np.savez(
            output / "hessian.npz",
            raw_hessian=raw_hessian,
            hessian_matrix=matrix,
            symmetrized_hessian=symmetric,
            external_basis=external_basis,
            internal_basis=internal_basis,
            projected_internal_hessian=projected,
            internal_eigenvalues=eigenvalues,
            lowest_internal_mode_cartesian=lowest_mode,
        )
    except Exception as error:
        row.update(status="failed", error=repr(error), traceback=traceback.format_exc())
    row["case_wall_seconds"] = time.perf_counter() - started_wall
    row["case_cpu_seconds"] = time.process_time() - started_cpu
    dump_json(output / "result.json", row)
    return row


def arm_class_counts(rows: list[dict], expected: dict) -> dict:
    counts = {
        label: {
            "selected_connected_classes": expected[label],
        "strictly_positive_spectrum_representatives": 0,
        "negative_curvature_representatives": 0,
        "exact_zero_sign_ambiguous_representatives": 0,
        "fresh_numerically_unqualified_evaluations": 0,
        "completed_hessian_numerically_unqualified": 0,
        "incomplete_classes": 0,
        }
        for label in expected
    }
    for row in rows:
        categories = counts[row["frame"]["arm_seed"]]
        if row.get("fresh_numerically_qualified") is False:
            categories["fresh_numerically_unqualified_evaluations"] += 1
        if row["status"] != "completed":
            categories["incomplete_classes"] += 1
        elif not row.get("fresh_numerically_qualified", False):
            categories["completed_hessian_numerically_unqualified"] += 1
        elif row["curvature_category"] == "strictly_positive_spectrum":
            categories["strictly_positive_spectrum_representatives"] += 1
        elif row["curvature_category"] == "negative_curvature":
            categories["negative_curvature_representatives"] += 1
        elif row["curvature_category"] == "exact_zero_sign_ambiguous":
            categories["exact_zero_sign_ambiguous_representatives"] += 1
        else:
            raise ValueError(f"unexpected curvature category in {row['case']}")
    return counts


def main() -> None:
    manifest_path = HERE / "manifest.json"
    result_path = HERE / "results.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")
    manifest = json.loads(manifest_path.read_text())
    manifest_sha256 = sha256_file(manifest_path)
    checks = (
        (HERE / "plan.md", manifest["protocol_sha256"]),
        (HERE / manifest["manifest_builder"], manifest["manifest_builder_sha256"]),
        (COVERAGE / "analysis.json", manifest["selection_input_sha256"]),
        (COVERAGE / "plan.json", manifest["coverage_plan_sha256"]),
        (CURVATURE_HELPER, manifest["curvature_helper_sha256"]),
    )
    for path, expected_hash in checks:
        if sha256_file(path) != expected_hash:
            raise ValueError(f"source changed after manifest preparation: {path}")
    frames = manifest.get("frames")
    if manifest.get("total_frames") != 42 or not isinstance(frames, list) or len(frames) != 42:
        raise ValueError("manifest must contain exactly 42 fixed-prefix class representatives")
    if manifest.get("class_counts_by_arm_seed") != EXPECTED_COUNTS:
        raise ValueError("manifest arm/seed class counts differ from the frozen matrix")
    actual_counts = {}
    for frame in frames:
        label = frame["arm_seed"]
        actual_counts[label] = actual_counts.get(label, 0) + 1
    if actual_counts != EXPECTED_COUNTS:
        raise ValueError(f"manifest frame counts differ from expected matrix: {actual_counts}")
    existing = [str(output_path_for(frame)) for frame in frames if output_path_for(frame).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing frame outputs: {existing[:3]}")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    helper_hash_started = time.perf_counter()
    actual_model_sha256 = sha256_file(MODEL)
    model_hash_seconds = time.perf_counter() - helper_hash_started
    if actual_model_sha256 != manifest["model_sha256"]:
        raise ValueError("MH-1 model hash differs from the prepared manifest")

    calculator_kwargs = {
        "model_paths": str(MODEL),
        "head": "omol",
        "device": "cpu",
        "default_dtype": "float64",
        "enable_cueq": False,
        "enable_oeq": False,
    }
    calculator_setup_started = time.perf_counter()
    calculator = MACECalculator(**calculator_kwargs)
    calculator_setup_seconds = time.perf_counter() - calculator_setup_started
    model_config = {
        **calculator_kwargs,
        "model_sha256": actual_model_sha256,
        "energy_units_to_eV": float(calculator.energy_units_to_eV),
        "length_units_to_A": float(calculator.length_units_to_A),
        "hessian_units": "eV/Angstrom^2 (raw output; conversion factors are 1)",
        "active_head": calculator.head,
        "available_heads": list(calculator.available_heads),
        "model_type": calculator.model_type,
        "model_count": int(calculator.num_models),
        "mace_torch_version": importlib.metadata.version("mace-torch"),
        "torch_version": torch.__version__,
        "ase_version": importlib.metadata.version("ase"),
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "omp_num_threads": os.environ["OMP_NUM_THREADS"],
        "mkl_num_threads": os.environ["MKL_NUM_THREADS"],
        "openblas_num_threads": os.environ["OPENBLAS_NUM_THREADS"],
        "calculator_setup_seconds": calculator_setup_seconds,
        "model_hash_seconds": model_hash_seconds,
    }
    if (calculator.head != "omol" or calculator.num_models != 1
            or calculator.model_type != "MACE"
            or calculator.energy_units_to_eV != 1.0
            or calculator.length_units_to_A != 1.0):
        raise ValueError("calculator head/model/unit configuration differs from the manifest")
    rigid_basis_fn, hessian_matrix_fn = load_curvature_helpers()

    started = time.perf_counter()
    cpu_started = time.process_time()
    rows = []
    dump_json(result_path, {
        "status": "running",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "model": model_config,
        "expected_frames": 42,
        "frames": rows,
    })
    for arm_seed, iterator in itertools.groupby(frames, key=lambda row: row["arm_seed"]):
        group = list(iterator)
        source_path = result_path_for(group[0])
        try:
            expected_source_hash = manifest["source_json_sha256_by_arm_seed"][arm_seed]
            actual_source_hash = sha256_file(source_path)
            if actual_source_hash != expected_source_hash:
                raise ValueError(f"source result hash mismatch for {arm_seed}")
            source_result = json.loads(source_path.read_text())
            if source_result.get("status") != "completed":
                raise ValueError(f"source result is not complete for {arm_seed}")
        except Exception as error:
            details = traceback.format_exc()
            for frame in group:
                rows.append(write_failed_frame(frame, manifest_sha256, error, details))
                dump_json(result_path, progress_summary(
                    rows, manifest, model_config, started, cpu_started
                ))
            continue

        for frame in group:
            try:
                index = frame["minimum_index"]
                minimum = source_result["minima"][index]
                source_hash = manifest["source_json_sha256_by_arm_seed"][arm_seed]
                row = analyze_frame(
                    frame, minimum, source_path, source_hash, manifest_sha256,
                    calculator, model_config, rigid_basis_fn, hessian_matrix_fn,
                )
            except Exception as error:
                row = write_failed_frame(
                    frame, manifest_sha256, error, traceback.format_exc()
                )
            rows.append(row)
            dump_json(result_path, progress_summary(
                rows, manifest, model_config, started, cpu_started
            ))
        del source_result

    summary = progress_summary(rows, manifest, model_config, started, cpu_started)
    summary["scope"] = (
        "42 fixed-prefix connected graph-class representatives; 42 fresh E/F pairs and "
        "42 analytic Hessians maximum; no geometry changes or automatic expansion"
    )
    summary["interpretation_limit"] = (
        "Curvature belongs to the saved geometry. A nonzero force is not a stationary-point "
        "certificate. Graph connectivity does not validate bond order or electronic state."
    )
    dump_json(result_path, summary)
    print(json.dumps({
        "status": summary["status"],
        "completed_frames": summary["completed_frames"],
        "failed_frames": summary["failed_frames"],
        "requests": summary["actual_requests"],
        "class_curvature_counts_by_arm_seed": summary["class_curvature_counts_by_arm_seed"],
        "elapsed_wall_seconds": summary["elapsed_wall_seconds"],
    }, indent=2))
    if summary["status"] != "completed":
        raise SystemExit(1)


def progress_summary(
    rows: list[dict], manifest: dict, model_config: dict,
    started_wall: float, started_cpu: float,
) -> dict:
    failed = sum(row["status"] != "completed" for row in rows)
    return {
        "status": "running" if len(rows) < manifest["total_frames"]
        else "completed" if failed == 0 else "completed_with_failures",
        "manifest": str(HERE / "manifest.json"),
        "manifest_sha256": sha256_file(HERE / "manifest.json"),
        "model": model_config,
        "expected_frames": manifest["total_frames"],
        "processed_frames": len(rows),
        "completed_frames": len(rows) - failed,
        "failed_frames": failed,
        "expected_requests": {"fresh_energy_force_pairs": 42, "analytic_hessians": 42},
        "actual_requests": {
            "fresh_energy_force_pairs": sum(row.get("fresh_energy_force_requests", 0) for row in rows),
            "analytic_hessians": sum(row.get("analytic_hessian_requests", 0) for row in rows),
        },
        "class_curvature_counts_by_arm_seed": arm_class_counts(
            rows, manifest["class_counts_by_arm_seed"]
        ),
        "elapsed_wall_seconds": time.perf_counter() - started_wall,
        "elapsed_cpu_seconds": time.process_time() - started_cpu,
        "frames": rows,
    }


if __name__ == "__main__":
    main()
