"""One bounded OMAT-small all-DOF quench of COD alpha-quartz; no SSW/MC."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ase.io import read, write

from pamssw.standalone.vc_geometry import SymmetricLogStrainChart
from pamssw.standalone.cell_relax import cell_quench
from research.ga_ssw.compare_vc_arms import serial
from research.ga_ssw.run_vc_e2e_optimizer_panel import (
    CappedSurface, _environment, _fresh_checks, _git_provenance,
)


HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def factory_from_plan(plan):
    import torch
    torch.set_num_threads(plan["calculator"]["torch_num_threads"])
    torch.set_num_interop_threads(plan["calculator"]["torch_num_interop_threads"])
    model = Path(plan["model"])
    actual = sha256(model)
    if actual != plan["model_sha256"]:
        raise ValueError(f"model SHA mismatch: {actual}")
    from mace.calculators import MACECalculator

    def factory():
        return MACECalculator(
            model_paths=str(model), device=plan["device"],
            default_dtype=plan["dtype"], head=plan["calculator"]["head"],
            enable_cueq=plan["calculator"]["enable_cueq"],
            enable_oeq=plan["calculator"]["enable_oeq"],
        )
    return factory


def certificate(energy, forces, stress, atoms, plan):
    fmax = float(np.linalg.norm(forces, axis=1).max())
    stress_residual = float(np.abs(stress + plan["quench"]["pressure_eV_A3"]
                                   * np.eye(3)).max())
    return {
        "energy_eV": float(energy),
        "volume_A3": float(atoms.get_volume()),
        "fmax_eV_A": fmax,
        "stress_residual_max_eV_A3": stress_residual,
        "force_pass": fmax <= plan["quench"]["fmax_eV_A"],
        "stress_pass": stress_residual <= plan["quench"]["stress_tol_eV_A3"],
        "certified": (fmax <= plan["quench"]["fmax_eV_A"] and
                      stress_residual <= plan["quench"]["stress_tol_eV_A3"]),
    }


def run(plan_path: Path, out: Path):
    plan = json.loads(plan_path.read_text())
    source = (REPO / plan["input_source"]).resolve()
    if not source.is_file() or sha256(source) != plan["input_source_sha256"]:
        raise ValueError("COD source missing or checksum mismatch")
    out.mkdir(parents=True, exist_ok=False)
    source_copy = out / source.name
    source_copy.write_bytes(source.read_bytes())
    atoms = read(source)
    symbols = sorted(atoms.get_chemical_symbols())
    if len(atoms) != 9 or symbols != ["O"] * 6 + ["Si"] * 3 or not atoms.pbc.all():
        raise ValueError("unexpected COD CIF expansion/stoichiometry/PBC")
    import spglib
    symmetry = spglib.get_spacegroup(
        (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers), symprec=1e-3)
    if symmetry != "P3_121 (152)":
        raise ValueError(f"unexpected expanded CIF symmetry: {symmetry}")

    effective = dict(plan)
    effective["runtime"] = {
        "hostname": subprocess.getoutput("hostname"),
        "environment": _environment(),
        "git_provenance": _git_provenance(),
        "worker_sha256": sha256(Path(__file__).resolve()),
        "source_sha256_verified": sha256(source),
        "expanded_formula": atoms.get_chemical_formula(),
        "expanded_atoms": len(atoms),
        "cell_A": atoms.cell.array.tolist(),
        "cellpar_A_deg": atoms.cell.cellpar().tolist(),
        "volume_A3": float(atoms.get_volume()),
        "spacegroup_symprec_1e-3": symmetry,
    }
    (out / "effective-plan.json").write_text(json.dumps(effective, indent=2) + "\n")
    write(out / "input.extxyz", atoms, format="extxyz")
    config = SimpleNamespace(
        pressure=plan["quench"]["pressure_eV_A3"],
        fmax=plan["quench"]["fmax_eV_A"],
        stress_tol=plan["quench"]["stress_tol_eV_A3"],
    )

    factory = factory_from_plan(plan)
    started = time.monotonic()
    surface = CappedSurface(factory(),
        request_cap=plan["budget"]["initial_plus_quench_search_efs"],
        deadline=started + plan["budget"]["search_wall_limit_seconds"],
        ledger_path=out / "search-ledger.jsonl")
    chart = SymmetricLogStrainChart(atoms, strain_length=plan["quench"]["strain_length_A"])
    initial_record = None
    initial = None
    quench = None
    error = None
    stage_boundary = {"initial_efs_request_start": surface.requests + 1}
    try:
        initial = chart.evaluate(chart.pack(atoms), surface.evaluate,
                                 pressure=config.pressure)
        initial_record = serial(initial)
        stage_boundary["quench_request_start"] = surface.requests + 1
        quench = cell_quench(
            atoms, surface, strain_length=plan["quench"]["strain_length_A"],
            pressure=config.pressure, fmax=config.fmax,
            stress_tol=config.stress_tol, max_step=plan["quench"]["max_step_A"],
            maxiter=plan["quench"]["maxiter"],
            lbfgs_memory=plan["quench"]["lbfgs_memory"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    stage_boundary["search_requests_paid"] = surface.requests
    stage_boundary["search_wall_seconds"] = time.monotonic() - started
    (out / "stage-boundaries.json").write_text(
        json.dumps(stage_boundary, indent=2, allow_nan=False) + "\n")

    final_eval = None
    final_atoms = None
    if quench is not None:
        final_eval = quench.evaluation
        if final_eval is not None:
            final_atoms = final_eval.atoms.copy()
        elif quench.optimizer is not None and np.isfinite(quench.optimizer.q).all():
            final_atoms = chart.unpack(quench.optimizer.q)
            if quench.optimizer.energy is not None and np.isfinite(quench.optimizer.energy):
                final_eval = SimpleNamespace(atoms=final_atoms, energy=float(quench.optimizer.energy))
    if final_atoms is not None:
        write(out / "relaxed-or-last-accepted.extxyz", final_atoms, format="extxyz")

    search_result = {
        "status": "completed" if quench is not None else "error",
        "error": error,
        "initial": initial_record,
        "cell_quench": None if quench is None else {
            "requests": quench.requests,
            "optimizer": serial(quench.optimizer),
            "physical_certificate": serial(quench.certificate),
            "result_converged": bool(quench.converged),
            "evaluation": None if quench.evaluation is None else serial(quench.evaluation),
        },
        "search_requests": surface.requests,
        "budget_censor": surface.budget_censor,
        "censor_reason": surface.censor_reason,
    }
    (out / "search-result.json").write_text(
        json.dumps(search_result, indent=2, allow_nan=False) + "\n")

    candidates = []
    if initial is not None:
        candidates.append({"source": "initial", "record_index": -1,
                           "accepted": True, "evaluation": initial})
    if final_eval is not None:
        candidates.append({"source": ("quench_final" if quench and quench.evaluation is not None
                                        else "last_accepted_partial"),
                           "record_index": 0, "accepted": False,
                           "evaluation": final_eval})
    if len(candidates) > plan["budget"]["independent_fresh_efs_max"]:
        raise RuntimeError("fresh endpoint count exceeds frozen cap")
    fresh = _fresh_checks(candidates, factory, config, out / "fresh-ledger.jsonl")
    (out / "fresh-checks.json").write_text(
        json.dumps(fresh, indent=2, allow_nan=False) + "\n")
    summary = {
        "status": search_result["status"],
        "algorithm_status": None if quench is None else quench.optimizer.status,
        "search_requests": surface.requests,
        "fresh_requests": fresh["requests"],
        "total_requests": surface.requests + fresh["requests"],
        "budget_censor": surface.budget_censor,
        "censor_reason": surface.censor_reason,
        "initial_certificate": None if initial is None else certificate(
            initial.energy, initial.forces, initial.stress, initial.atoms, plan),
        "quench_certificate": None if quench is None else serial(quench.certificate),
        "fresh_candidate_count": fresh["candidate_count"],
        "fresh_qualified_count": fresh["qualified_count"],
        "run_error": error,
        "wall_seconds": time.monotonic() - started,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=HERE / "qualification.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("Plan only. Pass --execute inside the reviewed Slurm allocation.")
        return 0
    run(args.plan, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
