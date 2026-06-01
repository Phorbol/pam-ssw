from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
RUN_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = RUN_DIR / "output"
LOG_DIR = RUN_DIR / "logs"
EVENT_LOG = RUN_DIR / "event_log.jsonl"
RESULTS_CSV = RUN_DIR / "results.csv"
RESULTS_JSON = RUN_DIR / "results.json"
SUMMARY_MD = RUN_DIR / "summary.md"
ARTIFACTS_JSON = RUN_DIR / "artifacts.json"

REFERENCE_VARIANT = "reference_original"
REFERENCE_ROOT = Path("/tmp/ssw-reference")

SOURCE_ROOT = Path("/mnt/d/Download/trae-research-code/SSW")
DEFAULT_MODEL = Path("/root/.cache/mace/mace-omat-0-small.model")
SYSTEMS: dict[str, dict[str, Any]] = {
    "c60": {
        "input": SOURCE_ROOT / "runs/20260428-c60-mace-production/prerelaxed_c60.xyz",
        "model": DEFAULT_MODEL,
        "pbc": (False, False, False),
        "fix_bottom_fraction": 0.0,
        "dedup_rmsd_tol": 0.15,
        "proposal_relax_steps": 80,
        "local_softening_active_count": 3,
        "quench_fmax": 0.01,
        "oracle_candidates": 12,
        "walk_trust_radius": 5.0,
        "proposal_trust_radius": 1.5,
    },
    "cuo": {
        "input": SOURCE_ROOT / "runs/20260506-cuo-200t-production/input/Cu110_Cu10O8/CuO_opt_input.arc",
        "model": SOURCE_ROOT / "runs/20260506-cuo-200t-production/input/Cu110_Cu10O8/CuO-OMAT_finetune.model",
        "pbc": (True, True, False),
        "fix_bottom_fraction": 0.35,
        "dedup_rmsd_tol": 0.4,
        "proposal_relax_steps": 300,
        "local_softening_active_count": 5,
        "quench_fmax": 0.03,
        "oracle_candidates": 8,
        "walk_trust_radius": 4.0,
        "proposal_trust_radius": 1.5,
    },
    "pdo": {
        "input": SOURCE_ROOT / "PdO.xyz",
        "model": DEFAULT_MODEL,
        "pbc": (True, True, False),
        "fix_bottom_fraction": 0.35,
        "dedup_rmsd_tol": 0.4,
        "proposal_relax_steps": 120,
        "local_softening_active_count": 5,
        "quench_fmax": 0.03,
        "oracle_candidates": 8,
        "walk_trust_radius": 4.0,
        "proposal_trust_radius": 1.5,
    },
}

VARIANTS: dict[str, dict[str, Any]] = {
    "paw_current_bias_relax": {
        "proposal_step_mode": "bias_relax",
        "seed_selection_mode": "archive_ucb",
        "direction_engine": "scored_pool",
    },
    "paw_metropolis_pool_bias_relax": {
        "proposal_step_mode": "bias_relax",
        "seed_selection_mode": "metropolis_chain",
        "direction_engine": "scored_pool",
    },
    "paw_ucb_reference_dimer_bias_relax": {
        "proposal_step_mode": "bias_relax",
        "seed_selection_mode": "archive_ucb",
        "direction_engine": "reference_dimer",
        "reference_dimer_max_steps": 15,
        "reference_dimer_rotation_tol": 0.03,
    },
    "paw_ucb_pool_no_momentum_bias_relax": {
        "proposal_step_mode": "bias_relax",
        "seed_selection_mode": "archive_ucb",
        "direction_engine": "scored_pool",
        "direction_pool_disable_momentum": True,
    },
    "paw_ucb_reference_dimer_direct_qp_adaptive50": {
        "proposal_step_mode": "direct_qp",
        "seed_selection_mode": "archive_ucb",
        "direction_engine": "reference_dimer",
        "direct_qp_hessian": "rank1",
        "direct_qp_gamma": 1.0,
        "direct_qp_gamma_mode": "model_error_gated_history",
        "direct_qp_gamma_history_quantile": 0.25,
        "direct_qp_gamma_history_min_samples": 4,
        "direct_qp_gamma_history_maxlen": 128,
        "direct_qp_gamma_model_error_threshold": 2.0,
        "direct_qp_gamma_model_error_streak": 2,
        "direct_qp_kappa": 240.0,
        "direct_qp_micro_steps": 3,
        "direct_qp_micro_mode": "adaptive_model_error",
        "direct_qp_micro_max_steps": 50,
        "direct_qp_micro_model_error_threshold": 2.0,
        "direct_qp_micro_model_error_high": 12.0,
        "direct_qp_micro_optimizer": "ase-fire",
        "direct_qp_micro_fmax": 0.05,
        "direct_qp_micro_trust_radius": 0.25,
    },
}
ALL_VARIANTS = {REFERENCE_VARIANT, *VARIANTS}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "value"):
        return obj.value
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def safe_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, default=_json_default)


def append_event(payload: dict[str, Any]) -> None:
    EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with EVENT_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=_json_default) + "\n")


def case_id(system: str, variant: str, seed: int, trials: int, device: str) -> str:
    return f"{system}_{variant}_seed{seed}_trials{trials}_{device}"


def reference_command(system: str, seed: int, trials: int, device: str) -> list[str]:
    if system != "c60":
        raise ValueError("reference_original is currently implemented only for c60")
    spec = SYSTEMS[system]
    outdir = OUTPUT_DIR / case_id(system, REFERENCE_VARIANT, seed, trials, device)
    return [
        sys.executable,
        str(REFERENCE_ROOT / "c60" / "run_c60_global_opt.py"),
        "--methods",
        "ssw",
        "--input",
        str(spec["input"]),
        "--model",
        str(spec["model"]),
        "--outdir",
        str(outdir),
        "--seed",
        str(seed),
        "--device",
        device,
        "--default-dtype",
        "float32",
        "--ssw-steps",
        str(trials),
        "--ssw-dimer-steps",
        "15",
        "--ssw-dimer-tol",
        "0.03",
        "--ssw-optimizer",
        "fire",
    ]


def pam_case_metadata(system: str, variant: str, seed: int, trials: int, device: str, steps_per_walk: int) -> dict[str, Any]:
    spec = SYSTEMS[system]
    return {
        "runner": "pamssw",
        "case_id": case_id(system, variant, seed, trials, device),
        "system": system,
        "variant": variant,
        "seed": seed,
        "trials": trials,
        "steps_per_walk": steps_per_walk,
        "device": device,
        "input": spec["input"],
        "model": spec["model"],
        "variant_params": VARIANTS[variant],
        "output_dir": OUTPUT_DIR / case_id(system, variant, seed, trials, device),
    }


def planned_case(system: str, variant: str, seed: int, trials: int, device: str, steps_per_walk: int) -> dict[str, Any]:
    if variant == REFERENCE_VARIANT:
        return {
            "runner": "external_reference",
            "case_id": case_id(system, variant, seed, trials, device),
            "system": system,
            "variant": variant,
            "seed": seed,
            "trials": trials,
            "device": device,
            "command": reference_command(system, seed, trials, device),
        }
    return pam_case_metadata(system, variant, seed, trials, device, steps_per_walk)


def parse_csv(value: str, allowed: set[str], label: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(items) - allowed)
    if unknown:
        raise ValueError(f"unknown {label} {unknown}; allowed={sorted(allowed)}")
    return items


def bottom_fixed_mask(positions: Any, fraction: float) -> Any:
    import numpy as np

    if fraction <= 0.0:
        return np.zeros(len(positions), dtype=bool)
    z = np.asarray(positions, dtype=float)[:, 2]
    threshold = float(np.min(z) + fraction * (np.max(z) - np.min(z)))
    return z <= threshold


def state_from_input(system: str, spec: dict[str, Any]) -> Any:
    from ase.io import read
    from pamssw import State

    atoms = read(spec["input"])
    input_pbc = tuple(bool(x) for x in atoms.pbc.tolist())
    atoms.pbc = spec["pbc"]
    fixed_mask = bottom_fixed_mask(atoms.positions, float(spec["fix_bottom_fraction"]))
    cell = atoms.cell.array if atoms.cell.rank > 0 else None
    if not any(spec["pbc"]):
        cell = None
    return State(
        numbers=atoms.numbers,
        positions=atoms.positions,
        cell=cell,
        pbc=spec["pbc"],
        fixed_mask=fixed_mask,
        metadata={
            "system": system,
            "input": str(spec["input"]),
            "input_pbc": input_pbc,
            "run_pbc": spec["pbc"],
        },
    )


def config_for(system: str, spec: dict[str, Any], variant: str, seed: int, trials: int, steps_per_walk: int, case_dir: Path) -> Any:
    from pamssw import LSSSWConfig

    params: dict[str, Any] = {
        "max_trials": trials,
        "max_steps_per_walk": steps_per_walk,
        "target_uphill_energy": 0.8,
        "target_negative_curvature": 0.05,
        "quench_fmax": spec["quench_fmax"],
        "quench_maxiter": 400,
        "dedup_rmsd_tol": spec["dedup_rmsd_tol"],
        "dedup_energy_tol": 1e-3,
        "rng_seed": seed,
        "oracle_candidates": spec["oracle_candidates"],
        "proposal_relax_steps": spec["proposal_relax_steps"],
        "proposal_fmax": 0.05,
        "proposal_optimizer": "ase-fire",
        "quench_optimizer": "scipy-lbfgsb",
        "min_step_scale": 0.05 if system != "c60" else 0.1,
        "max_step_scale": 1.2,
        "proposal_trust_radius": spec["proposal_trust_radius"],
        "walk_trust_radius": spec["walk_trust_radius"],
        "fragment_guard_factor": 3.0,
        "n_bond_pairs": 2,
        "stagnation_bond_pair_boost": 2,
        "max_stagnation_bond_pairs": 10,
        "proposal_pool_size": 1,
        "same_seed_max_consecutive": 3,
        "max_prototypes": 500,
        "max_energy_drop_per_atom": 5.0,
        "direction_curvature_source": "inner",
        "direction_score_sigma_mode": "adaptive",
        "accepted_structures_log": str(case_dir / "accepted_structures.jsonl"),
        "accepted_structures_dir": str(case_dir / "accepted_minima"),
        "write_proposal_minima": False,
        "write_relaxation_trajectories": False,
        "local_softening_mode": "active_neighbors",
        "local_softening_cutoff_scale": 1.15,
        "local_softening_active_count": spec["local_softening_active_count"],
        "local_softening_strength": 0.15,
        "local_softening_penalty": "buckingham_repulsive",
        "local_softening_xi": 0.3,
        "local_softening_cutoff": 2.0,
    }
    params.update(VARIANTS[variant])
    return LSSSWConfig(**params)


def make_calculator(model: Path, device: str) -> Any:
    from mace.calculators import MACECalculator
    from pamssw.calculators import ASECalculator

    calc = MACECalculator(
        model_paths=str(model),
        device=device,
        default_dtype="float32",
        inference_precision="float32",
        enable_cueq=False,
    )
    return ASECalculator(calc)


def state_to_atoms(state: Any) -> Any:
    from ase import Atoms

    return Atoms(numbers=state.numbers, positions=state.positions, cell=state.cell, pbc=state.pbc)


def write_pam_outputs(
    case_dir: Path,
    result: Any,
    config: Any,
    system: str,
    variant: str,
    seed: int,
    spec: dict[str, Any],
    device: str,
    wall_time_s: float,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from ase.io import write

    case_dir.mkdir(parents=True, exist_ok=True)
    archive_entries = sorted(result.archive.entries, key=lambda entry: entry.energy)
    minima_atoms = []
    for rank, entry in enumerate(archive_entries):
        atoms = state_to_atoms(entry.state)
        atoms.info.update({"entry_id": entry.entry_id, "rank": rank, "energy": entry.energy})
        minima_atoms.append(atoms)
    if minima_atoms:
        write(case_dir / "archive_minima.xyz", minima_atoms)
        write(case_dir / "best_minimum.xyz", minima_atoms[0])
    fixed_mask = archive_entries[0].state.fixed_mask if archive_entries else None
    summary = {
        "system": system,
        "variant": variant,
        "seed": seed,
        "input": str(spec["input"]),
        "model": str(spec["model"]),
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "dtype": "float32",
        "enable_cueq": False,
        "n_atoms": int(archive_entries[0].state.n_atoms if archive_entries else 0),
        "n_fixed": int(np.count_nonzero(fixed_mask)) if fixed_mask is not None else 0,
        "pbc": tuple(bool(x) for x in spec["pbc"]),
        "config": asdict(config),
        "best_energy": float(result.best_energy),
        "n_minima": len(result.archive.entries),
        "stats": result.stats,
        "archive_energies": [float(entry.energy) for entry in archive_entries],
        "wall_time_s": float(wall_time_s),
        "outputs": {
            "summary": str(case_dir / "ssw_summary.json"),
            "archive_minima_xyz": str(case_dir / "archive_minima.xyz"),
            "best_minimum_xyz": str(case_dir / "best_minimum.xyz"),
        },
    }
    (case_dir / "ssw_summary.json").write_text(safe_json(summary), encoding="utf-8")
    return summary


def row_from_pam_summary(summary: dict[str, Any]) -> dict[str, Any]:
    stats = summary.get("stats", {})
    return {
        "system": summary["system"],
        "variant": summary["variant"],
        "seed": int(summary["seed"]),
        "device": summary["device"],
        "best_energy": float(summary["best_energy"]),
        "n_minima": int(summary["n_minima"]),
        "force_evaluations": int(stats.get("force_evaluations", 0)),
        "energy_evaluations": int(stats.get("energy_evaluations", 0)),
        "duplicate_rate": float(stats.get("duplicate_rate", 0.0)),
        "bias_steps": int(stats.get("bias_steps", 0)),
        "reference_dimer_steps": int(stats.get("reference_dimer_steps", 0)),
        "reference_dimer_converged": int(stats.get("reference_dimer_converged", 0)),
        "direct_qp_steps": int(stats.get("direct_qp_steps", 0)),
        "direct_qp_rejected": int(stats.get("direct_qp_rejected", 0)),
        "wall_time_s": float(summary.get("wall_time_s", 0.0)),
        "summary_path": summary["outputs"]["summary"],
    }


def run_pam_case(system: str, variant: str, seed: int, trials: int, steps_per_walk: int, device: str) -> dict[str, Any]:
    from ase.io import write
    from pamssw import run_ls_ssw

    spec = SYSTEMS[system]
    case = case_id(system, variant, seed, trials, device)
    case_dir = OUTPUT_DIR / case
    summary_path = case_dir / "ssw_summary.json"
    if summary_path.exists():
        return row_from_pam_summary(json.loads(summary_path.read_text(encoding="utf-8")))
    append_event({"event": "start", "runner": "pamssw", "case": case, "time": now()})
    case_dir.mkdir(parents=True, exist_ok=True)
    state = state_from_input(system, spec)
    write(case_dir / "initial_structure.xyz", state_to_atoms(state))
    start = time.time()
    calculator = make_calculator(spec["model"], device=device)
    config = config_for(system, spec, variant, seed, trials, steps_per_walk, case_dir)
    result = run_ls_ssw(state, calculator, config)
    summary = write_pam_outputs(case_dir, result, config, system, variant, seed, spec, device, time.time() - start)
    row = row_from_pam_summary(summary)
    append_event({"event": "done", "runner": "pamssw", "case": case, "time": now(), "row": row})
    return row


def reference_summary_path(outdir: Path) -> Path:
    candidates = [
        outdir / "summary.json",
        outdir / "ssw_summary.json",
        outdir / "results.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(outdir.rglob("summary.json")) if outdir.exists() else []
    if matches:
        return matches[0]
    raise FileNotFoundError(f"no reference summary found under {outdir}")


def row_from_reference_summary(system: str, seed: int, trials: int, device: str, outdir: Path, wall_time_s: float) -> dict[str, Any]:
    path = reference_summary_path(outdir)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        rows = [item for item in data if isinstance(item, dict)]
        matches = [item for item in rows if item.get("method") == "ssw"]
        if not matches:
            methods = sorted(str(item.get("method")) for item in rows)
            raise ValueError(f"reference summary {path} has no method='ssw' row; methods={methods}")
        data = matches[0]
    elif not isinstance(data, dict):
        raise TypeError(f"reference summary {path} must be a dict or list of dict rows")

    extra = data.get("extra") if isinstance(data.get("extra"), dict) else {}
    best_energy = data.get("best_energy", data.get("best_energy_ev", data.get("minimum_energy", 0.0)))
    n_minima = data.get("n_minima", data.get("num_minima", data.get("accepted_minima", extra.get("accepted", 0))))
    steps = data.get("ssw_steps", extra.get("steps", trials))
    return {
        "system": system,
        "variant": REFERENCE_VARIANT,
        "seed": seed,
        "device": device,
        "best_energy": float(best_energy),
        "n_minima": int(n_minima),
        "force_evaluations": int(data.get("force_calls", data.get("force_evaluations", data.get("n_force_calls", 0))) or 0),
        "energy_evaluations": int(data.get("energy_evaluations", data.get("n_energy_calls", 0))),
        "duplicate_rate": float(data.get("duplicate_rate", 0.0)),
        "bias_steps": int(steps),
        "reference_dimer_steps": int(data.get("dimer_steps", steps)),
        "reference_dimer_converged": int(data.get("dimer_converged", 0)),
        "direct_qp_steps": 0,
        "direct_qp_rejected": 0,
        "wall_time_s": float(data.get("elapsed_s", wall_time_s)),
        "summary_path": str(path),
    }


def run_reference_case(system: str, seed: int, trials: int, device: str) -> dict[str, Any]:
    command = reference_command(system, seed, trials, device)
    outdir = OUTPUT_DIR / case_id(system, REFERENCE_VARIANT, seed, trials, device)
    append_event({"event": "start", "runner": "external_reference", "case": outdir.name, "time": now(), "command": command})
    outdir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    row = row_from_reference_summary(system, seed, trials, device, outdir, time.time() - start)
    append_event({"event": "done", "runner": "external_reference", "case": outdir.name, "time": now(), "row": row})
    return row


def run_case(system: str, variant: str, seed: int, trials: int, steps_per_walk: int, device: str) -> dict[str, Any]:
    if variant == REFERENCE_VARIANT:
        return run_reference_case(system, seed, trials, device)
    return run_pam_case(system, variant, seed, trials, steps_per_walk, device)


def write_aggregate(rows: list[dict[str, Any]], final: bool) -> None:
    if rows:
        with RESULTS_CSV.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    RESULTS_JSON.write_text(safe_json({"rows": rows}), encoding="utf-8")
    lines = [
        "# Reference Dimer Component Ablation",
        "",
        f"- Status: {'completed' if final else 'running'}",
        f"- Completed cases: {len(rows)}",
        "",
        "## Rows",
        "",
    ]
    for row in rows:
        lines.append(
            "- {system} / {variant} / seed {seed}: best={best:.9g}, n_minima={n_minima}, force={force}, wall={wall:.1f}s".format(
                system=row["system"],
                variant=row["variant"],
                seed=row["seed"],
                best=row["best_energy"],
                n_minima=row["n_minima"],
                force=row["force_evaluations"],
                wall=row["wall_time_s"],
            )
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ARTIFACTS_JSON.write_text(
        safe_json(
            {
                "artifacts": [
                    {"artifact_id": "results_csv", "type": "csv", "path": RESULTS_CSV},
                    {"artifact_id": "results_json", "type": "json", "path": RESULTS_JSON},
                    {"artifact_id": "summary_md", "type": "markdown", "path": SUMMARY_MD},
                ]
            }
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan the reference-dimer component ablation matrix. Add --execute to run calculations."
    )
    parser.add_argument("--systems", default="c60,cuo,pdo", help=f"Comma-separated systems. Choices: {','.join(sorted(SYSTEMS))}")
    parser.add_argument(
        "--variants",
        default="paw_current_bias_relax,paw_metropolis_pool_bias_relax,paw_ucb_reference_dimer_bias_relax,paw_ucb_pool_no_momentum_bias_relax,paw_ucb_reference_dimer_direct_qp_adaptive50",
        help=f"Comma-separated variants. Choices: {','.join(sorted(ALL_VARIANTS))}",
    )
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated integer seeds.")
    parser.add_argument("--trials", type=int, default=40, help="Trials per seed.")
    parser.add_argument("--steps-per-walk", type=int, default=8, help="PAM-SSW max steps per walk.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--execute", action="store_true", help="Run planned cases. Without this flag, only planned cases are printed.")
    parser.add_argument("--dry-run", action="store_true", help="Compatibility alias for the default planning-only mode.")
    args = parser.parse_args()

    systems = parse_csv(args.systems, set(SYSTEMS), "systems")
    variants = parse_csv(args.variants, ALL_VARIANTS, "variants")
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]

    planned = [
        planned_case(system, variant, seed, args.trials, args.device, args.steps_per_walk)
        for system in systems
        for variant in variants
        for seed in seeds
    ]
    if args.dry_run or not args.execute:
        print(safe_json({"execute": False, "case_count": len(planned), "cases": planned}))
        return 0

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for item in planned:
        rows.append(run_case(item["system"], item["variant"], item["seed"], args.trials, args.steps_per_walk, args.device))
        write_aggregate(rows, final=False)
    write_aggregate(rows, final=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
