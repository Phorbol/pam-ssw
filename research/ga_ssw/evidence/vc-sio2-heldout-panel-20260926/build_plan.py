"""Create the frozen 2x2x2 SiO2 panel plan; run on a scheduled CPU node."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from ase.io import read


HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
DEFAULT_START = (REPO / "research/ga_ssw/evidence/vc-sio2-qualification-20260926"
                 / "archive-1501259/qualified-start.extxyz")

CONFIG = {
    "strain_length": 5.0,
    "width": 0.6,
    "rotation_bias": 100.0,
    "pressure": 0.0,
    "temperature_K": 300.0,
    "forward_force": 0.1,
    "max_gaussians": 10,
    "gradient_tol": 0.05,
    "fmax": 0.05,
    "stress_tol": 0.001,
    "max_step": 0.2,
    "relax_steps": 300,
    "fd_step": 1e-4,
    "rotation_hvp": 100,
    "rotation_tol": 0.02,
    "lbfgs_memory": 500,
    "bias_release": "strict",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build(start_path: Path, output_path: Path):
    start_path = start_path.resolve()
    start = read(start_path)
    if len(start) != 9 or start.get_chemical_formula() != "O6Si3" or not start.pbc.all():
        raise ValueError("qualified starting structure is not periodic Si3O6")
    tiled = start.repeat((2, 2, 2))
    if len(tiled) != 72 or tiled.get_chemical_formula() != "O48Si24":
        raise ValueError("2x2x2 tiling did not produce the frozen 72-atom SiO2 input")
    if not np.isfinite(tiled.positions).all() or not np.isfinite(tiled.cell.array).all():
        raise ValueError("non-finite tiled geometry")
    if not math.isclose(tiled.get_volume(), 8.0 * start.get_volume(), rel_tol=1e-8):
        raise ValueError("supercell volume is not eight times the qualified input")

    input_data = {
        "numbers": tiled.numbers.tolist(),
        "positions": tiled.positions.tolist(),
        "cell": tiled.cell.array.tolist(),
        "pbc": tiled.pbc.tolist(),
    }
    plan = {
        "question": "Does the frozen Safe-total cost candidate remain operationally robust on a held-out 72-atom alpha-quartz-derived SiO2 supercell?",
        "purpose": "short held-out operational optimizer comparison; not paper BKS reproduction, phase-ordering study, or global-search validation",
        "model": "/home/gengjianrui/.cache/mace/mace-omat-0-small.model",
        "model_sha256": "0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5",
        "device": "cuda",
        "dtype": "float64",
        "calculator": {
            "head": "omat_pbe", "enable_cueq": False, "enable_oeq": False,
            "torch_num_threads": 1, "torch_num_interop_threads": 1,
        },
        "methods": ["safe_total", "ase", "scipy"],
        "seeds": [71, 83],
        "steps": 3,
        "budget": {
            "search_requests_per_arm": 6000,
            "search_seconds_per_arm": 1650,
            "external_seconds_per_arm": 1740,
            "fresh_requests_per_arm": 4,
            "total_search_requests": 36000,
            "total_fresh_requests": 24,
            "max_parallel_gpus": 2,
            "gpu_minutes_per_arm": 30,
            "max_gpu_hours": 3,
        },
        "metrics": {
            "primary": "paid search E/F/stress requests per attempted outer record",
            "qualification": "fresh-valid physical landings per attempted outer record",
            "physical_certificate": {
                "fmax_eV_A": 0.05, "stress_residual_eV_A3": 0.001,
                "pressure_eV_A3": 0.0,
            },
            "denominators": "18 planned outer slots; report actual attempted outer records separately. Slots unattempted after censor/early termination are unobserved, not optimizer failures. All six planned arms remain in the arm denominator.",
            "censoring": "failed or censored attempted proposals retain their paid cost and status; do not treat absent/unattempted slots as zero-cost successes or failures",
        },
        "cases": [{
            "name": "SiO2-COD1011097-qualified-2x2x2-72",
            "source_result": str(start_path),
            "source_field": "single COD alpha-quartz cell after job 1501259 Safe cell_quench and independent fresh physical qualification",
            "input_transform": {"repeat": [2, 2, 2], "atoms_before": 9, "atoms_after": 72},
            "input_sha256": sha256(start_path),
            "input": input_data,
            "config": CONFIG,
            "config_source": "existing frozen VC e2e plan TiO2-phase87 case (research/ga_ssw/evidence/vc-e2e-optimizer-panel-20260926/plan.json); AlOH26 uses width=0.2/max_gaussians=14 and is not the source preset",
        }],
        "protocol_limits": {
            "method_variation": "only optimizer adapter (Safe-total, ASE LBFGSLineSearch, SciPy L-BFGS-B); same input, model, full VC chart, pressure, config, seeds, proposal count, request and time budgets",
            "stopping_fairness": "operational optimizer comparison, not an isolated line-search/cost comparison: native stopping criteria differ; existing norm conversions are retained, and ASE/SciPy step-cap behavior is not fully equivalent",
            "metric_size_dependence": "strain_length=5.0 is held fixed as operational transfer, not a size-invariant cell metric; at fixed stress the log-strain cell gradient scales with replicated volume",
            "claim_ceiling": "three steps, two seeds and a replicated starting basin cannot establish stable optimizer superiority, global search ranking, alpha-quartz phase stability, or paper/BKS reproduction",
        },
        "provenance": {
            "starting_geometry": "COD 1011097 qualified endpoint archived under vc-sio2-qualification-20260926/archive-1501259",
            "qualification_job": 1501259,
            "plan_builder_sha256": sha256(Path(__file__).resolve()),
            "builder_git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite plan: {output_path}")
    output_path.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "plan": str(output_path.resolve()), "atoms": len(tiled),
        "formula": tiled.get_chemical_formula(),
        "cell_A": tiled.cell.array.tolist(),
        "volume_A3": float(tiled.get_volume()),
        "starting_structure_sha256": plan["cases"][0]["input_sha256"],
        "plan_sha256": sha256(output_path),
    }, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=Path, default=DEFAULT_START)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.start, args.out)


if __name__ == "__main__":
    main()
