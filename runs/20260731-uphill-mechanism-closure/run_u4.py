#!/usr/bin/env python3
"""Run the bounded, frozen-task U4 uphill mechanism audit on C60."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any

import numpy as np

from pamssw.accounting import EvaluationCounts, EvaluationPurpose
from pamssw.archive import MinimaArchive
from pamssw.fingerprint import descriptor_distance, structural_descriptor
from pamssw.io import read_state, write_state
from pamssw.pbc import mic_displacement
from pamssw.relax import has_force_convergence_certificate
from pamssw.result import RelaxResult
from pamssw.walker import ProposalRelaxationTask, SurfaceWalker


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
PROTOCOL_PATH = RUN_ROOT / "protocol.py"
CONFIG_RUNNER = (
    REPO_ROOT / "runs" / "20260730-starter-cell-online-gate" / "run_gate.py"
)
LS_GATE_RUNNER = (
    REPO_ROOT / "runs" / "20260730-ls-softening-scope-gate" / "run_gate.py"
)
STATE_ROOT = (
    REPO_ROOT
    / "runs"
    / "20260730-starter-cell-online-gate"
    / "production-20k-seed42-output"
    / "c60"
    / "seed-00000042"
    / "uniform"
    / "archive_minima"
)
STATE_FILES = {
    "late": STATE_ROOT / "entry-00041.xyz",
    "mid": STATE_ROOT / "entry-00020.xyz",
    "bootstrap": STATE_ROOT / "entry-00000.xyz",
}
SEEDS = (42, 43, 44)
ARM_ORDER = ("baseline80", "maxiter300", "newest_only", "oracle_only")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


protocol = _load_module(PROTOCOL_PATH, "_uphill_mechanism_protocol")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


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


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _counts_delta(
    before: EvaluationCounts,
    after: EvaluationCounts,
) -> dict[str, int]:
    values = tuple(
        right - left for left, right in zip(before.values, after.values)
    )
    delta = EvaluationCounts(values)
    payload = delta.as_dict()
    payload["total"] = delta.total
    protocol.purpose_delta(
        {**before.as_dict(), "total": before.total},
        {**after.as_dict(), "total": after.total},
    )
    return payload


@dataclass
class CapturedTask:
    task: ProposalRelaxationTask
    optimizer: str
    result: RelaxResult
    proposal_counts: dict[str, int]


class CapturingWalker(SurfaceWalker):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.captured_tasks: list[CapturedTask] = []

    def _relax_proposal_task(
        self,
        task,
        *,
        optimizer,
        trajectory_callback,
    ):
        before = self.calculator.snapshot()
        result = super()._relax_proposal_task(
            task,
            optimizer=optimizer,
            trajectory_callback=trajectory_callback,
        )
        after = self.calculator.snapshot()
        self.captured_tasks.append(
            CapturedTask(
                task=task,
                optimizer=optimizer,
                result=result,
                proposal_counts=_counts_delta(before, after),
            )
        )
        return result


def _preflight(expected_commit: str) -> dict[str, Any]:
    actual = _current_commit()
    if actual != expected_commit:
        raise RuntimeError(
            f"execution commit mismatch: expected {expected_commit}, got {actual}"
        )
    if not _tracked_clean():
        raise RuntimeError("tracked worktree is not clean")
    for path in STATE_FILES.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    ls_gate = _load_module(LS_GATE_RUNNER, "_u4_ls_gate_preflight")
    production = _load_module(
        ls_gate.PRODUCTION_RUNNER,
        "_u4_production_preflight",
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
        "state_files": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in STATE_FILES.items()
        },
        "seeds": list(SEEDS),
        "arm_order": list(ARM_ORDER),
        "selection_rule": (
            "max two proposal maxiter=80 tasks per fixed-starter walk, "
            "requiring at least two Gaussian terms and ordered by "
            "descending Gaussian-history length"
        ),
    }


def _make_walker(
    seed: int,
    state_id: str,
    output: Path,
    calculator,
):
    config_builder = _load_module(
        CONFIG_RUNNER,
        f"_u4_config_builder_{state_id}_{seed}",
    )
    config = replace(
        config_builder.build_production_config(
            "c60",
            output / "generation" / state_id / f"seed-{seed}",
            master_seed=seed,
        ),
        max_trials=1,
        max_force_evals=None,
        local_softening_scope="both",
        direction_diagnostics_enabled=False,
        direction_diagnostics_path=None,
    )
    return CapturingWalker(
        calculator=calculator,
        config=config,
        softening_enabled=True,
    )


def _capture_tasks(
    *,
    state_id: str,
    seed: int,
    output: Path,
    calculator,
) -> tuple[CapturingWalker, list[CapturedTask], dict[str, int], float]:
    state = read_state(STATE_FILES[state_id])
    walker = _make_walker(seed, state_id, output, calculator)
    before = walker.calculator.snapshot()
    started = perf_counter()
    walker._walk_candidate_from_seed(
        state,
        trial_index=0,
        proposal_index=0,
    )
    wall_time = float(perf_counter() - started)
    after = walker.calculator.snapshot()
    censored = [
        item
        for item in walker.captured_tasks
        if item.result.telemetry.termination_reason == "maxiter"
        and item.task.maxiter == 80
        and len(item.task.biases) >= 2
    ]
    censored.sort(key=lambda item: len(item.task.biases), reverse=True)
    return walker, censored[:2], _counts_delta(before, after), wall_time


def _execute_task(
    walker: SurfaceWalker,
    task: ProposalRelaxationTask,
    optimizer: str,
) -> tuple[RelaxResult, dict[str, int], float]:
    before = walker.calculator.snapshot()
    started = perf_counter()
    with walker.calculator.purpose(
        EvaluationPurpose.BIASED_PROPOSAL_RELAX
    ):
        result = SurfaceWalker._relax_proposal_task(
            walker,
            task,
            optimizer=optimizer,
            trajectory_callback=None,
        )
    wall_time = float(perf_counter() - started)
    after = walker.calculator.snapshot()
    return result, _counts_delta(before, after), wall_time


def _validate_endpoint(
    walker: SurfaceWalker,
    result: RelaxResult,
) -> tuple[dict[str, Any], Any]:
    before = walker.calculator.snapshot()
    started = perf_counter()
    with walker.calculator.purpose(EvaluationPurpose.ESCAPE_TRUE_PES_CHECK):
        endpoint = walker.calculator.evaluate(result.state)
    landing = walker.relax_true_minimum(result.state)
    wall_time = float(perf_counter() - started)
    after = walker.calculator.snapshot()
    return (
        {
            "true_endpoint_energy_eV": float(endpoint.energy),
            "landing_energy_eV": float(landing.energy),
            "landing_gradient_norm_eV_per_A": float(
                landing.gradient_norm
            ),
            "landing_iterations": int(landing.n_iter),
            "landing_certificate": bool(
                has_force_convergence_certificate(
                    landing,
                    walker.config.quench_fmax,
                )
            ),
            "validation_counts": _counts_delta(before, after),
            "validation_wall_time_s": wall_time,
        },
        landing,
    )


def _arm_tasks(
    frozen: CapturedTask,
) -> dict[str, ProposalRelaxationTask]:
    maxiter = protocol.maxiter_arms(
        frozen.task,
        extended_maxiter=300,
    )
    history = protocol.history_arms(frozen.task)
    softening = protocol.softening_arms(frozen.task)
    return {
        "baseline80": maxiter["maxiter80"],
        "maxiter300": maxiter["maxiter300"],
        "newest_only": history["newest_only"],
        "oracle_only": softening["oracle_only"],
    }


def _rmsd(lhs, rhs) -> float:
    return MinimaArchive._rmsd(lhs, rhs)


def _run_frozen_task(
    *,
    walker: CapturingWalker,
    frozen: CapturedTask,
    task_id: str,
    state_id: str,
    seed: int,
    output: Path,
) -> dict[str, Any]:
    tasks = _arm_tasks(frozen)
    expected_diffs = {
        "maxiter300": {"maxiter"},
        "newest_only": {"bias_history"},
        "oracle_only": {"proposal_softening"},
    }
    for arm, expected in expected_diffs.items():
        observed = protocol.task_component_diff(
            tasks["baseline80"],
            tasks[arm],
        )
        if observed != expected:
            raise RuntimeError(
                f"{task_id} {arm} changed {sorted(observed)}, "
                f"expected {sorted(expected)}"
            )

    rows: dict[str, dict[str, Any]] = {}
    landings = {}
    proposal_states = {}
    for arm in ARM_ORDER:
        task = tasks[arm]
        result, proposal_counts, proposal_wall_time = _execute_task(
            walker,
            task,
            frozen.optimizer,
        )
        validation, landing = _validate_endpoint(walker, result)
        landings[arm] = landing
        proposal_states[arm] = result.state
        arm_dir = output / "tasks" / task_id / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        write_state(arm_dir / "proposal_endpoint.xyz", result.state)
        write_state(arm_dir / "landing.xyz", landing.state)
        rows[arm] = {
            "arm": arm,
            "task_id": task_id,
            "state_id": state_id,
            "seed": seed,
            "bias_count": len(task.biases),
            "proposal_softening": task.softening is not None,
            "proposal_maxiter": task.maxiter,
            "physical_task_fingerprint": (
                protocol.physical_task_fingerprint(task)
            ),
            "proposal_energy_eV": float(result.energy),
            "proposal_gradient_norm_eV_per_A": float(
                result.gradient_norm
            ),
            "proposal_iterations": int(result.n_iter),
            "proposal_certificate": bool(
                has_force_convergence_certificate(result, task.fmax)
            ),
            "proposal_termination": (
                result.telemetry.termination_reason
            ),
            "proposal_telemetry": asdict(result.telemetry),
            "proposal_counts": proposal_counts,
            "proposal_wall_time_s": proposal_wall_time,
            **validation,
        }

    baseline = landings["baseline80"]
    baseline_descriptor = structural_descriptor(baseline.state)
    for arm in ARM_ORDER:
        landing = landings[arm]
        row = rows[arm]
        rmsd_to_baseline = _rmsd(
            landing.state,
            baseline.state,
        )
        row["landing_rmsd_to_baseline_A"] = (
            float(rmsd_to_baseline)
            if np.isfinite(rmsd_to_baseline)
            else None
        )
        row["landing_descriptor_delta_to_baseline"] = float(
            descriptor_distance(
                structural_descriptor(landing.state),
                baseline_descriptor,
            )
        )
        row["landing_energy_delta_to_baseline_eV"] = float(
            landing.energy - baseline.energy
        )
        row["same_landing_as_baseline"] = bool(
            abs(landing.energy - baseline.energy)
            <= walker.config.dedup_energy_tol
            and row["landing_rmsd_to_baseline_A"] is not None
            and (
                row["landing_rmsd_to_baseline_A"]
                <= walker.config.dedup_rmsd_tol
            )
        )
    replay_rmsd = _rmsd(
        frozen.result.state,
        proposal_states["baseline80"],
    )
    return {
        "task_id": task_id,
        "state_id": state_id,
        "seed": seed,
        "frozen_bias_count": len(frozen.task.biases),
        "frozen_optimizer": frozen.optimizer,
        "baseline_physical_task_fingerprint": (
            protocol.physical_task_fingerprint(frozen.task)
        ),
        "selection_run": {
            "proposal_energy_eV": float(frozen.result.energy),
            "proposal_gradient_norm_eV_per_A": float(
                frozen.result.gradient_norm
            ),
            "proposal_iterations": int(frozen.result.n_iter),
            "proposal_certificate": bool(
                has_force_convergence_certificate(
                    frozen.result,
                    frozen.task.fmax,
                )
            ),
            "proposal_termination": (
                frozen.result.telemetry.termination_reason
            ),
            "proposal_counts": frozen.proposal_counts,
        },
        "selection_to_baseline_proposal_rmsd_A": (
            float(replay_rmsd) if np.isfinite(replay_rmsd) else None
        ),
        "selection_to_baseline_proposal_energy_delta_eV": float(
            rows["baseline80"]["proposal_energy_eV"]
            - frozen.result.energy
        ),
        "rows": rows,
    }


def _factor_evidence(
    tasks: list[dict[str, Any]],
    *,
    baseline: str,
    variant: str,
) -> dict[str, Any]:
    pairs = []
    for task in tasks:
        left = task["rows"][baseline]
        right = task["rows"][variant]
        pairs.append(
            {
                "task_id": task["task_id"],
                "baseline": left,
                "variant": right,
                "proposal_certificate_gain": int(
                    right["proposal_certificate"]
                )
                - int(left["proposal_certificate"]),
                "landing_certificate_gain": int(
                    right["landing_certificate"]
                )
                - int(left["landing_certificate"]),
                "landing_energy_delta_eV": (
                    right["landing_energy_eV"]
                    - left["landing_energy_eV"]
                ),
                "proposal_force_delta": (
                    right["proposal_counts"]["total"]
                    - left["proposal_counts"]["total"]
                ),
                "same_landing": right["same_landing_as_baseline"],
            }
        )
    return {
        "baseline_arm": baseline,
        "variant_arm": variant,
        "pair_count": len(pairs),
        "proposal_certificate_gains": sum(
            pair["proposal_certificate_gain"] > 0 for pair in pairs
        ),
        "proposal_certificate_losses": sum(
            pair["proposal_certificate_gain"] < 0 for pair in pairs
        ),
        "landing_certificate_gains": sum(
            pair["landing_certificate_gain"] > 0 for pair in pairs
        ),
        "landing_certificate_losses": sum(
            pair["landing_certificate_gain"] < 0 for pair in pairs
        ),
        "same_landing_count": sum(pair["same_landing"] for pair in pairs),
        "landing_energy_wins": sum(
            pair["landing_energy_delta_eV"] < -1.0e-3 for pair in pairs
        ),
        "landing_energy_losses": sum(
            pair["landing_energy_delta_eV"] > 1.0e-3 for pair in pairs
        ),
        "proposal_force_delta_total": sum(
            pair["proposal_force_delta"] for pair in pairs
        ),
        "pairs": pairs,
    }


def _write_conclusion(
    path: Path,
    title: str,
    evidence: dict[str, Any],
) -> None:
    lines = [
        f"# {title}",
        "",
        f"- Frozen paired tasks: {evidence['pair_count']}",
        (
            "- Proposal certificate gains/losses: "
            f"{evidence['proposal_certificate_gains']}/"
            f"{evidence['proposal_certificate_losses']}"
        ),
        (
            "- Landing certificate gains/losses: "
            f"{evidence['landing_certificate_gains']}/"
            f"{evidence['landing_certificate_losses']}"
        ),
        (
            "- Same landing basin as baseline: "
            f"{evidence['same_landing_count']}/{evidence['pair_count']}"
        ),
        (
            "- Landing energy wins/losses: "
            f"{evidence['landing_energy_wins']}/"
            f"{evidence['landing_energy_losses']}"
        ),
        (
            "- Additional proposal force evaluations: "
            f"{evidence['proposal_force_delta_total']}"
        ),
        "",
        (
            "This is a fixed-task mechanism audit. It does not establish a "
            "global-search or cross-system production default by itself."
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(
    *,
    output: Path,
    expected_commit: str,
    max_tasks: int,
) -> dict[str, Any]:
    preflight = _preflight(expected_commit)
    ls_gate = _load_module(LS_GATE_RUNNER, "_u4_ls_gate_execution")
    calculator, _production = ls_gate._calculator()
    tasks: list[dict[str, Any]] = []
    generation_rows = []
    for state_id in ("late", "mid", "bootstrap"):
        for seed in SEEDS:
            walker, frozen_tasks, counts, wall_time = _capture_tasks(
                state_id=state_id,
                seed=seed,
                output=output,
                calculator=calculator,
            )
            generation_rows.append(
                {
                    "state_id": state_id,
                    "seed": seed,
                    "captured_maxiter80_tasks": len(frozen_tasks),
                    "counts": counts,
                    "wall_time_s": wall_time,
                }
            )
            for frozen in frozen_tasks:
                task_id = (
                    f"{state_id}-seed{seed}-"
                    f"bias{len(frozen.task.biases):02d}"
                )
                tasks.append(
                    _run_frozen_task(
                        walker=walker,
                        frozen=frozen,
                        task_id=task_id,
                        state_id=state_id,
                        seed=seed,
                        output=output,
                    )
                )
                if len(tasks) >= max_tasks:
                    break
            if len(tasks) >= max_tasks:
                break
        if len(tasks) >= max_tasks:
            break
    if not tasks:
        raise RuntimeError("no proposal maxiter=80 tasks were captured")

    u4_0 = _factor_evidence(
        tasks,
        baseline="baseline80",
        variant="maxiter300",
    )
    u4_a = _factor_evidence(
        tasks,
        baseline="baseline80",
        variant="newest_only",
    )
    u4_b = _factor_evidence(
        tasks,
        baseline="baseline80",
        variant="oracle_only",
    )
    payload = {
        "schema_version": 1,
        "preflight": preflight,
        "generation": generation_rows,
        "tasks": tasks,
        "u4_0": u4_0,
        "u4_a": u4_a,
        "u4_b": u4_b,
    }
    _write_json(output / "evidence.json", payload)
    _write_json(output / "u4_0_evidence.json", u4_0)
    _write_json(output / "u4_a_evidence.json", u4_a)
    _write_json(output / "u4_b_evidence.json", u4_b)
    _write_conclusion(
        output / "u4_0_conclusion.md",
        "U4-0 Proposal Relaxation Capacity",
        u4_0,
    )
    _write_conclusion(
        output / "u4_a_conclusion.md",
        "U4-A Gaussian History Retention",
        u4_a,
    )
    _write_conclusion(
        output / "u4_b_conclusion.md",
        "U4-B Proposal Local Softening",
        u4_b,
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--max-tasks", type=int, default=6)
    args = parser.parse_args()
    if args.max_tasks <= 0 or args.max_tasks > 6:
        parser.error("--max-tasks must be in [1, 6]")
    payload = run(
        output=args.output,
        expected_commit=args.expected_git_commit,
        max_tasks=args.max_tasks,
    )
    print(
        json.dumps(
            {
                "task_count": len(payload["tasks"]),
                "u4_0": {
                    key: value
                    for key, value in payload["u4_0"].items()
                    if key != "pairs"
                },
                "u4_a": {
                    key: value
                    for key, value in payload["u4_a"].items()
                    if key != "pairs"
                },
                "u4_b": {
                    key: value
                    for key, value in payload["u4_b"].items()
                    if key != "pairs"
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
