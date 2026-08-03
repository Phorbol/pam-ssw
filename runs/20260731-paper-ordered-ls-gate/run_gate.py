#!/usr/bin/env python3
"""Fixed-starter gate for the documented LS-SSW application order."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence

from pamssw.accounting import EvaluationPurpose
from pamssw.archive import MinimaArchive
from pamssw.fingerprint import descriptor_distance, structural_descriptor
from pamssw.io import read_state, write_state
from pamssw.relax import has_force_convergence_certificate


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
SOURCE_GATE = (
    REPO_ROOT / "runs" / "20260730-ls-softening-scope-gate" / "run_gate.py"
)
CONFIG_RUNNER = (
    REPO_ROOT / "runs" / "20260730-starter-cell-online-gate" / "run_gate.py"
)
SOURCE_ROOT = (
    REPO_ROOT
    / "runs"
    / "20260730-starter-cell-online-gate"
    / "production-20k-seed42-output"
)
ARMS = ("none", "current_active", "paper_ordered")
SEEDS = (42, 43, 44)
STATE_FILES = {
    "bootstrap": SOURCE_ROOT
    / "c60/seed-00000042/uniform/archive_minima/entry-00000.xyz",
    "mid": SOURCE_ROOT
    / "c60/seed-00000042/uniform/archive_minima/entry-00020.xyz",
    "late": SOURCE_ROOT
    / "c60/seed-00000042/uniform/archive_minima/entry-00041.xyz",
}
CC_STANDARD_BOND_ENERGY_EV = 3.61
PAPER_INITIAL_STRENGTH_EV = 0.03 * CC_STANDARD_BOND_ENERGY_EV


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _arm_config(base, arm: str):
    if arm == "none":
        return replace(
            base,
            local_softening_protocol="moving_reference",
            local_softening_scope="none",
            choice_aligned_softening_enabled=False,
        )
    if arm == "current_active":
        return replace(
            base,
            local_softening_protocol="moving_reference",
            local_softening_scope="both",
            choice_aligned_softening_enabled=False,
        )
    if arm == "paper_ordered":
        return replace(
            base,
            local_softening_protocol="paper_ordered",
            local_softening_mode="neighbor_auto",
            local_softening_scope="both",
            local_softening_active_count=None,
            local_softening_strength=PAPER_INITIAL_STRENGTH_EV,
            local_softening_xi=0.2,
            local_softening_cutoff=None,
            local_softening_adaptive_strength=False,
            choice_aligned_softening_enabled=False,
        )
    raise ValueError(f"unknown arm: {arm}")


def preflight(expected_git_commit: str) -> dict[str, Any]:
    source = _load_module(SOURCE_GATE, "_paper_ls_source_gate")
    actual = _current_commit()
    if actual != expected_git_commit:
        raise RuntimeError(
            f"execution commit mismatch: expected {expected_git_commit}, got {actual}"
        )
    if not _tracked_clean():
        raise RuntimeError("tracked worktree is not clean")
    for path in STATE_FILES.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    production = _load_module(
        source.PRODUCTION_RUNNER,
        "_paper_ls_production_runner",
    )
    if not production.MODEL_PATH.is_file():
        raise FileNotFoundError(production.MODEL_PATH)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return {
        "schema_version": 1,
        "execution_commit": actual,
        "gpu": torch.cuda.get_device_name(0),
        "model_path": str(production.MODEL_PATH),
        "model_sha256": _sha256(production.MODEL_PATH),
        "system": "c60",
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "paper_initial_strength_eV": PAPER_INITIAL_STRENGTH_EV,
        "paper_xi_dimensionless": 0.2,
        "states": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in STATE_FILES.items()
        },
    }


def _run_case(
    *,
    state_id: str,
    state,
    seed: int,
    arm: str,
    calculator,
    source,
    config_builder,
    case_directory: Path,
) -> dict[str, Any]:
    base = replace(
        config_builder.build_production_config(
            "c60",
            case_directory,
            master_seed=seed,
        ),
        max_trials=1,
        max_force_evals=None,
    )
    config = _arm_config(base, arm)
    walker = source.ObservingWalker(
        calculator=calculator,
        config=config,
        softening_enabled=True,
    )
    archive = MinimaArchive(
        energy_tol=config.dedup_energy_tol,
        rmsd_tol=config.dedup_rmsd_tol,
        max_prototypes=config.max_prototypes,
    )
    started = perf_counter()
    with walker.calculator.purpose(EvaluationPurpose.ESCAPE_TRUE_PES_CHECK):
        starter_evaluation = walker.calculator.evaluate(state)
    starter_energy = float(starter_evaluation.energy)
    seed_entry = archive.add(state, starter_energy, parent_id=None)
    proposal = walker._proposal_pool(
        state,
        archive,
        trial_index=0,
        step_target=walker.step_target_controller.target(archive),
        seed_entry_id=seed_entry.entry_id,
        allow_duplicate_rescue=False,
    )[0]
    escape_state = proposal.state
    with walker.calculator.purpose(EvaluationPurpose.ESCAPE_TRUE_PES_CHECK):
        escape_evaluation = walker.calculator.evaluate(escape_state)
    landing = walker.relax_true_minimum(escape_state)
    wall_time_s = float(perf_counter() - started)
    before = len(archive.entries)
    archive.add(landing.state, float(landing.energy), parent_id=seed_entry.entry_id)
    counts = walker.calculator.snapshot()
    purpose_counts = counts.as_dict()
    if counts.total != sum(purpose_counts.values()):
        raise RuntimeError("purpose accounting does not close")
    if purpose_counts["unattributed"] != 0:
        raise RuntimeError("unattributed force evaluations are not allowed")
    case_directory.mkdir(parents=True, exist_ok=True)
    write_state(case_directory / "escape.xyz", escape_state)
    write_state(case_directory / "landing.xyz", landing.state)
    row = {
        "system": "c60",
        "state_id": state_id,
        "seed": seed,
        "arm": arm,
        "starter_energy_eV": starter_energy,
        "escape_energy_eV": float(escape_evaluation.energy),
        "landing_energy_eV": float(landing.energy),
        "landing_delta_eV": float(landing.energy) - starter_energy,
        "is_new_basin": len(archive.entries) > before,
        "descriptor_delta": float(
            descriptor_distance(
                structural_descriptor(state),
                structural_descriptor(landing.state),
            )
        ),
        "fragmented": bool(walker._is_fragmented_cluster(state, landing.state)),
        "quench_certificate": bool(
            has_force_convergence_certificate(landing, config.quench_fmax)
        ),
        "force_evaluations": counts.total,
        "purpose_counts": purpose_counts,
        "wall_time_s": wall_time_s,
        "directions": walker.mechanism_directions,
        "softening_builds": walker.mechanism_softening,
        "softening_diagnostics": walker.local_softening_diagnostics(),
        "effective_config": asdict(config),
    }
    _write_json(case_directory / "summary.json", row)
    return row


def _paired_effects(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> list[dict[str, Any]]:
    by_key = {
        (str(row["state_id"]), int(row["seed"]), str(row["arm"])): row
        for row in rows
    }
    effects = []
    for state_id in STATE_FILES:
        for seed in SEEDS:
            none = by_key[(state_id, seed, "none")]
            current = by_key[(state_id, seed, "current_active")]
            paper = by_key[(state_id, seed, "paper_ordered")]
            effects.append(
                {
                    "state_id": state_id,
                    "seed": seed,
                    "paper_minus_none": float(paper[metric]) - float(none[metric]),
                    "paper_minus_current": (
                        float(paper[metric]) - float(current[metric])
                    ),
                }
            )
    return effects


def build_evidence(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = {
        (state_id, seed, arm)
        for state_id in STATE_FILES
        for seed in SEEDS
        for arm in ARMS
    }
    observed = {
        (str(row["state_id"]), int(row["seed"]), str(row["arm"]))
        for row in rows
    }
    if observed != expected or len(rows) != len(expected):
        raise ValueError("case cohort does not match the fixed 27-case design")
    for row in rows:
        if not row["directions"]:
            raise ValueError("direction record is missing")
        if not row["quench_certificate"]:
            raise ValueError("landing true quench lacks a force certificate")
        if row["arm"] == "none" and row["softening_builds"]:
            raise ValueError("none arm constructed a softening model")
        if row["arm"] != "none" and not row["softening_builds"]:
            raise ValueError("softened arm did not construct a softening model")
        diagnostics = row["softening_diagnostics"]
        pre_count = int(row["purpose_counts"]["local_softening_pre_relax"])
        if row["arm"] == "paper_ordered":
            if (
                diagnostics["pre_relaxations"] != 1
                or diagnostics["pre_relax_converged"] != 1
                or diagnostics["pre_relax_force_evaluations"] != pre_count
            ):
                raise ValueError("paper-ordered pre-relaxation is uncertified")
        elif pre_count != 0:
            raise ValueError("non-paper arm used the pre-relaxation purpose")

    summaries = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        summaries[arm] = {
            "cases": len(arm_rows),
            "new_basin_count": sum(bool(row["is_new_basin"]) for row in arm_rows),
            "median_landing_delta_eV": statistics.median(
                float(row["landing_delta_eV"]) for row in arm_rows
            ),
            "median_force_evaluations": statistics.median(
                int(row["force_evaluations"]) for row in arm_rows
            ),
            "median_wall_time_s": statistics.median(
                float(row["wall_time_s"]) for row in arm_rows
            ),
            "median_first_direction_inner_curvature": statistics.median(
                float(row["directions"][0]["curvature"]) for row in arm_rows
            ),
            "median_first_direction_true_curvature": statistics.median(
                float(row["directions"][0]["true_curvature"]) for row in arm_rows
            ),
            "median_first_direction_participation_ratio": statistics.median(
                float(row["directions"][0]["participation_ratio"])
                for row in arm_rows
            ),
            "median_pre_relax_force_evaluations": statistics.median(
                int(
                    row["purpose_counts"][
                        "local_softening_pre_relax"
                    ]
                )
                for row in arm_rows
            ),
            "median_pre_relax_pls_eV_per_atom": statistics.median(
                float(
                    row["softening_diagnostics"][
                        "pre_relax_pls_eV_per_atom"
                    ]
                )
                for row in arm_rows
            ),
        }
    return {
        "schema_version": 1,
        "cohort": {
            "system": "c60",
            "states": list(STATE_FILES),
            "seeds": list(SEEDS),
            "arms": list(ARMS),
            "cases": len(rows),
        },
        "arm_summaries": summaries,
        "landing_effects": _paired_effects(rows, "landing_delta_eV"),
        "force_evaluation_effects": _paired_effects(rows, "force_evaluations"),
        "inner_curvature_effects": _paired_effects(
            [
                {
                    **row,
                    "first_inner_curvature": row["directions"][0]["curvature"],
                }
                for row in rows
            ],
            "first_inner_curvature",
        ),
        "participation_ratio_effects": _paired_effects(
            [
                {
                    **row,
                    "first_participation_ratio": (
                        row["directions"][0]["participation_ratio"]
                    ),
                }
                for row in rows
            ],
            "first_participation_ratio",
        ),
        "claim_ceiling": (
            "C60 fixed-starter mechanism evidence only; no system-general or "
            "equal-budget production claim"
        ),
        "cases": list(rows),
    }


def run_gate(output: Path, expected_git_commit: str) -> dict[str, Any]:
    manifest = preflight(expected_git_commit)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    _write_json(output / "manifest.json", manifest)
    source = _load_module(SOURCE_GATE, "_paper_ls_execution_source")
    calculator, _ = source._calculator()
    config_builder = _load_module(CONFIG_RUNNER, "_paper_ls_config_builder")
    rows = []
    for state_id, state_path in STATE_FILES.items():
        state = read_state(state_path)
        for seed in SEEDS:
            for arm in ARMS:
                case_directory = output / state_id / f"seed-{seed}" / arm
                rows.append(
                    _run_case(
                        state_id=state_id,
                        state=state,
                        seed=seed,
                        arm=arm,
                        calculator=calculator,
                        source=source,
                        config_builder=config_builder,
                        case_directory=case_directory,
                    )
                )
    evidence = build_evidence(rows)
    _write_json(output / "evidence.json", evidence)
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    args = parser.parse_args()
    evidence = run_gate(args.output, args.expected_git_commit)
    print(json.dumps(evidence["arm_summaries"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
