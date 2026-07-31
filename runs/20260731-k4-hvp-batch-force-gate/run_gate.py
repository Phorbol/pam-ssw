#!/usr/bin/env python3
"""Benchmark serial and graph-batched K4 central-HVP MACE workloads."""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
SOURCE_ROOT = Path("/mnt/d/download/trae-research-code/ssw")
MODEL_PATH = Path("/root/.cache/mace/mace-omat-0-small.model")
MODEL_SHA256 = (
    "0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5"
)
SYSTEMS = {
    "c60": {
        "path": SOURCE_ROOT
        / "runs"
        / "20260428-c60-mace-production"
        / "prerelaxed_c60.xyz",
        "sha256": (
            "c63788c18cbed305963213b47eabd9fdc4d06dac118da6a1a9e16621d5e32bf9"
        ),
        "pbc": (False, False, False),
        "direction_seed": 4200,
    },
    "pdo": {
        "path": SOURCE_ROOT / "PdO.xyz",
        "sha256": (
            "68243ceb7c0fbb6ba7a9454d680287eb98c4e5210efbd9ebb63517ba79aaa8b0"
        ),
        "pbc": (True, True, False),
        "direction_seed": 4300,
    },
}
CALCULATOR_CONFIG = {
    "device": "cuda",
    "default_dtype": "float32",
    "inference_precision": "float32",
    "enable_cueq": False,
}
EPSILON = 1.0e-3
DIRECTION_COUNT = 4
MODES = ("serial", "batch2", "batch4", "batch8")
BATCH_SIZES = {"serial": 1, "batch2": 2, "batch4": 4, "batch8": 8}
REPETITIONS = 5
EXPECTED_FORCE_EVALUATIONS = 384


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


protocol = _load_module(
    RUN_ROOT / "protocol.py",
    "_k4_hvp_batch_force_protocol_run",
)


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _current_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tracked_clean() -> bool:
    return not subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _preflight(expected_commit: str) -> dict[str, Any]:
    import torch

    if _current_commit() != expected_commit:
        raise RuntimeError("execution commit differs from --expected-commit")
    if not _tracked_clean():
        raise RuntimeError("tracked worktree must be clean")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if _sha256(MODEL_PATH) != MODEL_SHA256:
        raise RuntimeError("MACE model checksum drifted")
    for spec in SYSTEMS.values():
        if _sha256(spec["path"]) != spec["sha256"]:
            raise RuntimeError(f"input checksum drifted: {spec['path']}")
    return {
        "execution_commit": expected_commit,
        "python": platform.python_version(),
        "cuda_runtime": str(torch.version.cuda),
        "cuda_device": str(torch.cuda.get_device_name(0)),
        "model_path": str(MODEL_PATH),
        "model_sha256": MODEL_SHA256,
        "calculator": CALCULATOR_CONFIG,
        "inputs": {
            system: {
                "path": str(spec["path"]),
                "sha256": spec["sha256"],
            }
            for system, spec in SYSTEMS.items()
        },
    }


def _load_atoms(system: str):
    from ase.io import read

    atoms = read(SYSTEMS[system]["path"])
    atoms.pbc = SYSTEMS[system]["pbc"]
    return atoms


def _fixed_mask(atoms, system: str) -> np.ndarray:
    if system == "c60":
        return np.zeros(len(atoms), dtype=bool)
    z = np.asarray(atoms.positions, dtype=float)[:, 2]
    return z <= float(np.quantile(z, 0.35))


def _directions(atoms, system: str) -> list[np.ndarray]:
    fixed = _fixed_mask(atoms, system)
    rng = np.random.default_rng(int(SYSTEMS[system]["direction_seed"]))
    directions = []
    for _ in range(DIRECTION_COUNT):
        direction = rng.normal(size=(len(atoms), 3))
        direction[fixed] = 0.0
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            raise RuntimeError("deterministic direction has zero norm")
        directions.append(direction / norm)
    return directions


def _geometries(atoms, directions: Sequence[np.ndarray]):
    geometries = []
    for direction in directions:
        for sign in (1.0, -1.0):
            displaced = atoms.copy()
            displaced.positions = (
                np.asarray(atoms.positions, dtype=float)
                + sign * EPSILON * np.asarray(direction, dtype=float)
            )
            geometries.append(displaced)
    return geometries


def _set_les_periodicity(calculator, atoms) -> None:
    pbc = tuple(bool(value) for value in np.asarray(atoms.get_pbc()))
    count = sum(pbc)
    for model in calculator.models:
        if not hasattr(model, "les") or not hasattr(model.les, "ewald"):
            continue
        if count == 2:
            model.les.ewald.periodicity = "2d"
            model.les.ewald.slab_axis = next(
                axis for axis, periodic in enumerate(pbc) if not periodic
            )
        elif count == 3:
            model.les.ewald.periodicity = "3d"


def _serial_evaluate(calculator, geometries):
    energies = []
    forces = []
    for atoms in geometries:
        calculator.calculate(atoms)
        energies.append(float(calculator.results["energy"]))
        forces.append(np.asarray(calculator.results["forces"], dtype=float))
    return np.asarray(energies), np.asarray(forces)


def _batch_chunk(calculator, geometries):
    import torch
    from mace import data
    from mace.tools import torch_geometric

    calculator.arrays_keys.update({calculator.charges_key: "charges"})
    keyspec = data.KeySpecification(
        info_keys=calculator.info_keys,
        arrays_keys=calculator.arrays_keys,
    )
    atomic_data = []
    for atoms in geometries:
        config = data.config_from_atoms(
            atoms,
            key_specification=keyspec,
            head_name=calculator.head,
        )
        atomic_data.append(
            data.AtomicData.from_config(
                config,
                z_table=calculator.z_table,
                cutoff=calculator.r_max,
                heads=calculator.available_heads,
            )
        )
    batch_base = torch_geometric.Batch.from_data_list(atomic_data).to(
        calculator.device
    )
    energy_models = []
    force_models = []
    for model in calculator.models:
        batch = calculator._clone_batch(batch_base)
        out = model(
            batch.to_dict(),
            compute_stress=not calculator.use_compile,
            training=calculator.use_compile,
            compute_edge_forces=calculator.compute_atomic_stresses,
            compute_atomic_stresses=calculator.compute_atomic_stresses,
        )
        energy_models.append(out["energy"].detach())
        force_models.append(out["forces"].detach())
    energies = (
        torch.stack(energy_models).mean(dim=0).cpu().numpy()
        * calculator.energy_units_to_eV
    )
    forces_flat = (
        torch.stack(force_models).mean(dim=0).cpu().numpy()
        * calculator.energy_units_to_eV
    )
    forces = [
        forces_flat[int(batch_base.ptr[index]) : int(batch_base.ptr[index + 1])]
        for index in range(len(geometries))
    ]
    return np.asarray(energies, dtype=float), np.asarray(forces, dtype=float)


def _batched_evaluate(calculator, geometries, batch_size: int):
    energies = []
    forces = []
    for start in range(0, len(geometries), batch_size):
        chunk_energy, chunk_forces = _batch_chunk(
            calculator,
            geometries[start : start + batch_size],
        )
        energies.extend(chunk_energy.tolist())
        forces.extend(list(chunk_forces))
    return np.asarray(energies), np.asarray(forces)


def _evaluate_mode(calculator, geometries, mode: str):
    import torch

    _set_les_periodicity(calculator, geometries[0])
    torch.cuda.synchronize()
    started = perf_counter()
    if mode == "serial":
        energies, forces = _serial_evaluate(calculator, geometries)
    else:
        energies, forces = _batched_evaluate(
            calculator,
            geometries,
            BATCH_SIZES[mode],
        )
    torch.cuda.synchronize()
    wall_time_s = perf_counter() - started
    force_pairs = [
        (forces[2 * index], forces[2 * index + 1])
        for index in range(DIRECTION_COUNT)
    ]
    return {
        "energies": energies,
        "forces": forces,
        "hvps": np.asarray(
            protocol.central_hvps_from_forces(
                force_pairs,
                epsilon=EPSILON,
            )
        ),
        "wall_time_s": wall_time_s,
    }


def _json_result(
    result: Mapping[str, Any],
    directions: Sequence[np.ndarray],
) -> dict[str, Any]:
    hvps = np.asarray(result["hvps"], dtype=float)
    return {
        "energies": np.asarray(result["energies"], dtype=float).tolist(),
        "forces": np.asarray(result["forces"], dtype=float).tolist(),
        "hvps": hvps.tolist(),
        "curvatures": protocol.directional_curvatures(
            directions,
            list(hvps),
        ),
        "wall_time_s": float(result["wall_time_s"]),
    }


def run(*, output_dir: Path, expected_commit: str) -> dict[str, Any]:
    from mace.calculators import MACECalculator

    if output_dir.exists():
        raise FileExistsError(output_dir)
    preflight = _preflight(expected_commit)
    output_dir.mkdir(parents=True)
    rows = []
    direction_oracle_fe = 0
    for system in ("c60", "pdo"):
        atoms = _load_atoms(system)
        directions = _directions(atoms, system)
        geometries = _geometries(atoms, directions)
        calculator = MACECalculator(
            model_paths=str(MODEL_PATH),
            **CALCULATOR_CONFIG,
        )
        for mode in MODES:
            _evaluate_mode(calculator, geometries, mode)
            direction_oracle_fe += len(geometries)
        for repetition in range(REPETITIONS):
            shift = repetition % len(MODES)
            order = MODES[shift:] + MODES[:shift]
            for mode in order:
                result = _evaluate_mode(calculator, geometries, mode)
                direction_oracle_fe += len(geometries)
                rows.append(
                    {
                        "system": system,
                        "repetition": repetition,
                        "mode": mode,
                        "batch_size": BATCH_SIZES[mode],
                        "force_evaluations": len(geometries),
                        "result": _json_result(result, directions),
                    }
                )
    if direction_oracle_fe != EXPECTED_FORCE_EVALUATIONS:
        raise RuntimeError("batch gate FE ledger does not close")
    summary = {
        "schema_version": 1,
        **preflight,
        "protocol": {
            "epsilon_A": EPSILON,
            "direction_count": DIRECTION_COUNT,
            "geometry_count_per_workload": 2 * DIRECTION_COUNT,
            "modes": list(MODES),
            "repetitions": REPETITIONS,
            "warmup_workloads_per_mode_system": 1,
        },
        "evaluation_counts": {
            "direction_oracle": direction_oracle_fe,
            "unattributed": 0,
            "total": direction_oracle_fe,
        },
        "row_count": len(rows),
        "rows_path": "rows.json",
    }
    (output_dir / "rows.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    result = run(
        output_dir=args.output.resolve(),
        expected_commit=args.expected_commit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
