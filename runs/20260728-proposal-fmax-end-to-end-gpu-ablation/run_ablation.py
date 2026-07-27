#!/usr/bin/env python3
"""Fixed-total-FE GPU ablation of the proposal force certificate only.

This is intentionally a thin wrapper around the frozen posterior starter-policy
harness.  It fixes that harness to its full-support ``uniform`` policy and
delegates calculator ownership, exact FE accounting, and campaign execution to
its existing implementation.  The two arms differ only in ``proposal_fmax``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import importlib.util
import json
from math import isfinite
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from pamssw.config import SSWConfig


RUN_ROOT = Path(__file__).resolve().parent
POSTERIOR_RUNNER_PATH = (
    RUN_ROOT.parent / "20260728-posterior-starter-policy-gpu-ablation" / "run_ablation.py"
)
_POSTERIOR_MODULE_NAME = "_proposal_fmax_end_to_end_frozen_posterior_harness"

SYSTEMS = ("c60", "pdo")
POLICY_NAME = "uniform"
SERIAL_BATCH_SIZE = 1
SERIAL_MAX_WORKERS = 1
SUPPORTED_ARMS = (0.05, 0.10)
DEFAULT_ARMS = SUPPORTED_ARMS
SCHEMA_VERSION = 1
_NONOPERATIVE_CONFIG_OUTPUTS = {
    "accepted_structures_log": None,
    "accepted_structures_dir": None,
    "write_proposal_minima": False,
    "proposal_minima_dir": None,
    "write_relaxation_trajectories": False,
    "relaxation_trajectory_dir": None,
    "direction_diagnostics_enabled": False,
    "direction_diagnostics_path": None,
    "direction_archive_enabled": False,
    "direction_archive_path": None,
}
_CAMPAIGN_ARTIFACT_PATHS = {
    "action_metrics_path": "action_metrics.jsonl",
    "event_log_path": "events.jsonl",
}


def _posterior_runner():
    """Load the frozen posterior harness once, without copying its machinery."""
    module = sys.modules.get(_POSTERIOR_MODULE_NAME)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(
        _POSTERIOR_MODULE_NAME, POSTERIOR_RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen posterior starter-policy harness")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_POSTERIOR_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_lbfgs_history_limit() -> int:
    """Confirm that the production history setting still matches the runtime."""
    posterior = _posterior_runner()
    production = posterior._load_production_runner()
    expected = int(production.SAFE_LBFGS_DEFAULT_HISTORY_LIMIT)
    from pamssw.relax import _SAFE_LBFGS_MEMORY

    actual = int(_SAFE_LBFGS_MEMORY)
    if actual != expected:
        raise RuntimeError(
            "safe L-BFGS history differs from the frozen production configuration: "
            f"expected {expected}, got {actual}"
        )
    return actual


def _validated_arm(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("proposal_fmax arms must be finite numeric values")
    arm = float(value)
    if not isfinite(arm) or arm not in SUPPORTED_ARMS:
        raise ValueError("proposal_fmax arms must be exactly 0.05 and 0.10 eV/A")
    return arm


def _validated_arms(arms: object) -> tuple[float, ...]:
    try:
        values = tuple(_validated_arm(value) for value in arms)
    except TypeError as exc:
        raise TypeError("proposal_fmax arms must be iterable") from exc
    if len(values) != len(SUPPORTED_ARMS) or len(set(values)) != len(values):
        raise ValueError(
            "proposal_fmax arms must contain each prespecified arm once"
        )
    if set(values) != set(SUPPORTED_ARMS):
        raise ValueError("proposal_fmax arms must be exactly 0.05 and 0.10 eV/A")
    # Canonical ordering gives every matrix row a stable arm index.
    return SUPPORTED_ARMS


def _arm_id(proposal_fmax: float) -> str:
    return f"proposal-fmax-{proposal_fmax:.2f}"


def _arm_records(arms: Sequence[float]) -> list[dict[str, object]]:
    return [
        {
            "arm_index": index,
            "arm_id": _arm_id(proposal_fmax),
            "proposal_fmax": proposal_fmax,
        }
        for index, proposal_fmax in enumerate(arms)
    ]


def _protocol_or_fail(config: SSWConfig, *, history_limit: int) -> None:
    if config.proposal_optimizer != "safe-lbfgs-total":
        raise RuntimeError("proposal optimizer is not frozen to safe-lbfgs-total")
    if config.proposal_pool_size != 1:
        raise RuntimeError("proposal pool size is not frozen to one")
    if config.quench_optimizer != "ase-lbfgs":
        raise RuntimeError("true quench is not frozen to ase-lbfgs")
    if config.quench_fallback_optimizer != "ase-fire":
        raise RuntimeError("true quench fallback is not frozen to one ase-fire pass")
    if config.quench_fmax != 0.01 or config.quench_maxiter != 400:
        raise RuntimeError("true quench certificate is not the frozen 0.01/400 protocol")
    if history_limit <= 0:
        raise RuntimeError("safe L-BFGS history limit must be positive")
    if _nonoperative_config_outputs(config) != _NONOPERATIVE_CONFIG_OUTPUTS:
        raise RuntimeError("configuration enables an untracked campaign output")


def _nonoperative_config_outputs(config: SSWConfig) -> dict[str, object]:
    return {
        name: getattr(config, name)
        for name in _NONOPERATIVE_CONFIG_OUTPUTS
    }


def build_ssw_config(
    system: str,
    case_directory: Path,
    *,
    proposal_fmax: float,
) -> tuple[SSWConfig, dict[str, object]]:
    """Apply the one arm-specific override to the frozen posterior projection."""
    if system not in SYSTEMS:
        raise ValueError(f"unknown system: {system}")
    arm = _validated_arm(proposal_fmax)
    history_limit = _safe_lbfgs_history_limit()
    posterior = _posterior_runner()
    frozen_config, frozen_projection = posterior.build_ssw_config(
        system, Path(case_directory)
    )
    if type(frozen_config) is not SSWConfig:
        raise TypeError("frozen posterior harness did not return exactly an SSWConfig")
    if frozen_projection.get("softening_enabled") is not False:
        raise RuntimeError("proposal-fmax ablation requires softening disabled")

    config = replace(frozen_config, proposal_fmax=arm)
    _protocol_or_fail(config, history_limit=history_limit)
    projection = {
        "source_config_type": frozen_projection["source_config_type"],
        "source_config": frozen_projection["source_config"],
        "removed_ls_fields": frozen_projection["removed_ls_fields"],
        "frozen_posterior_overrides": frozen_projection["overrides"],
        "frozen_posterior_effective_ssw_config": asdict(frozen_config),
        "effective_ssw_config": asdict(config),
        "arm_overrides": {"proposal_fmax": arm},
        "proposal_protocol": {
            "optimizer": config.proposal_optimizer,
            "history_limit": history_limit,
            "proposal_pool_size": config.proposal_pool_size,
            "softening_enabled": False,
        },
        "true_quench_protocol": {
            "optimizer": config.quench_optimizer,
            "fallback_optimizer": config.quench_fallback_optimizer,
            "fmax_eV_per_A": config.quench_fmax,
            "maxiter": config.quench_maxiter,
        },
        "nonoperative_config_outputs": _nonoperative_config_outputs(config),
    }
    return config, projection


def _only_proposal_fmax_differs(configs: Sequence[SSWConfig]) -> None:
    if len(configs) != len(SUPPORTED_ARMS):
        raise RuntimeError("expected exactly the two proposal-fmax configurations")
    first, second = (asdict(config) for config in configs)
    changed = {
        name
        for name in first
        if first[name] != second[name]
    }
    if changed != {"proposal_fmax"}:
        raise RuntimeError(
            "proposal-fmax arms differ in fields other than proposal_fmax: "
            f"{sorted(changed)!r}"
        )


def _matrix(
    *, systems: Sequence[str], master_seeds: Sequence[int], arms: Sequence[float]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for system in systems:
        for master_seed in master_seeds:
            for arm in _arm_records(arms):
                relative_directory = (
                    Path(system)
                    / f"seed-{master_seed:08d}"
                    / str(arm["arm_id"])
                )
                rows.append(
                    {
                        "matrix_index": len(rows),
                        "system": system,
                        "master_seed": master_seed,
                        "policy_name": POLICY_NAME,
                        "batch_size": SERIAL_BATCH_SIZE,
                        "max_workers": SERIAL_MAX_WORKERS,
                        "arm_index": arm["arm_index"],
                        "arm_id": arm["arm_id"],
                        "proposal_fmax": arm["proposal_fmax"],
                        "relative_case_directory": str(relative_directory),
                    }
                )
    return rows


def preflight(
    *,
    expected_git_commit: str,
    systems: Sequence[str],
    master_seeds: Sequence[int],
    action_force_budget: int,
    total_force_budget: int,
    arms: Sequence[float] = DEFAULT_ARMS,
) -> dict[str, object]:
    """Fail closed through the frozen runtime and identity checks."""
    selected_arms = _validated_arms(arms)
    posterior = _posterior_runner()
    frozen_manifest = posterior.preflight(
        expected_git_commit=expected_git_commit,
        systems=systems,
        master_seeds=master_seeds,
        action_force_budget=action_force_budget,
        total_force_budget=total_force_budget,
    )
    selected_systems = tuple(frozen_manifest["systems"])
    selected_seeds = tuple(frozen_manifest["master_seeds"])
    history_limit = _safe_lbfgs_history_limit()
    projections: dict[str, list[dict[str, object]]] = {}
    for system in selected_systems:
        configs: list[SSWConfig] = []
        arm_projections: list[dict[str, object]] = []
        for arm in selected_arms:
            config, projection = build_ssw_config(
                system,
                RUN_ROOT / ".preflight" / system / _arm_id(arm),
                proposal_fmax=arm,
            )
            configs.append(config)
            arm_projections.append(
                {
                    "arm_index": len(arm_projections),
                    "arm_id": _arm_id(arm),
                    "proposal_fmax": arm,
                    **projection,
                }
            )
        _only_proposal_fmax_differs(configs)
        projections[system] = arm_projections

    matrix = _matrix(
        systems=selected_systems,
        master_seeds=selected_seeds,
        arms=selected_arms,
    )
    if not matrix:
        raise RuntimeError("preflight yielded an empty campaign matrix")
    return {
        "schema_version": SCHEMA_VERSION,
        "execution_commit": frozen_manifest["execution_commit"],
        "frozen_posterior_harness": {
            "path": str(POSTERIOR_RUNNER_PATH),
            "sha256": _sha256(POSTERIOR_RUNNER_PATH),
            "preflight": frozen_manifest,
        },
        "systems": list(selected_systems),
        "master_seeds": list(selected_seeds),
        "policy_name": POLICY_NAME,
        "policy_support": "full-support random baseline",
        "batch_size": SERIAL_BATCH_SIZE,
        "max_workers": SERIAL_MAX_WORKERS,
        "action_force_budget": action_force_budget,
        "total_force_budget": total_force_budget,
        "arms": _arm_records(selected_arms),
        "proposal_optimizer": "safe-lbfgs-total",
        "safe_lbfgs_history_limit": history_limit,
        "softening_enabled": False,
        "projections": projections,
        "matrix": matrix,
    }


def _staging_root(output_root: Path) -> Path:
    return output_root.with_name(f".{output_root.name}.partial")


def _preflight_output_root(output_root: Path) -> None:
    staging_root = _staging_root(output_root)
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"output root already exists: {output_root}")
    if staging_root.exists() or staging_root.is_symlink():
        raise FileExistsError(f"staging output already exists: {staging_root}")
    if not output_root.parent.is_dir():
        raise FileNotFoundError(f"output root parent does not exist: {output_root.parent}")


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    posterior = _posterior_runner()
    posterior._write_json_exclusive(path, payload)


def _json_value(payload: Mapping[str, Any]) -> dict[str, object]:
    """Return the exact JSON shape that will be published to disk."""
    return json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))


def _replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace one already-published JSON file inside staging."""
    temporary_path = path.with_name(f".{path.name}.rewrite")
    if temporary_path.exists() or temporary_path.is_symlink():
        raise FileExistsError(f"campaign summary rewrite path exists: {temporary_path}")
    with temporary_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_path, path)


def _rewrite_campaign_summary_paths(
    *,
    case_directory: Path,
    staging_root: Path,
    output_root: Path,
) -> None:
    """Rebase only the two delegated summary paths that move with staging."""
    summary_path = case_directory / "campaign_summary.json"
    if not summary_path.is_file():
        raise RuntimeError("delegated campaign did not write campaign_summary.json")
    with summary_path.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    if not isinstance(summary, dict):
        raise RuntimeError("campaign_summary.json must contain a JSON object")

    for field_name, artifact_name in _CAMPAIGN_ARTIFACT_PATHS.items():
        recorded = summary.get(field_name)
        if not isinstance(recorded, str):
            raise RuntimeError(f"campaign summary lacks {field_name}")
        staged_artifact = Path(recorded)
        expected_staged_artifact = case_directory / artifact_name
        if staged_artifact != expected_staged_artifact:
            raise RuntimeError(
                f"campaign summary {field_name} is not its staging artifact"
            )
        if not staged_artifact.is_file():
            raise RuntimeError(
                f"campaign summary {field_name} target does not exist in staging"
            )
        try:
            relative_artifact = staged_artifact.relative_to(staging_root)
        except ValueError as exc:
            raise RuntimeError(
                f"campaign summary {field_name} escapes the staging root"
            ) from exc
        published_artifact = output_root / relative_artifact
        summary[field_name] = str(published_artifact)

    _replace_json(summary_path, summary)


def _verify_published_campaign_summary_paths(
    *, campaigns: Sequence[Mapping[str, object]], output_root: Path, staging_root: Path
) -> None:
    """Fail closed if any rebased summary path does not resolve after publication."""
    for campaign in campaigns:
        summary_path = Path(str(campaign["campaign_summary_path"]))
        if not summary_path.is_file():
            raise RuntimeError("published campaign summary is missing")
        with summary_path.open("r", encoding="utf-8") as stream:
            summary = json.load(stream)
        if not isinstance(summary, dict):
            raise RuntimeError("published campaign summary must contain a JSON object")
        for field_name in _CAMPAIGN_ARTIFACT_PATHS:
            recorded = summary.get(field_name)
            if not isinstance(recorded, str):
                raise RuntimeError(f"published campaign summary lacks {field_name}")
            published_artifact = Path(recorded)
            try:
                published_artifact.relative_to(staging_root)
            except ValueError:
                pass
            else:
                raise RuntimeError(
                    f"published campaign summary {field_name} retains staging path"
                )
            try:
                published_artifact.relative_to(output_root)
            except ValueError as exc:
                raise RuntimeError(
                    f"published campaign summary {field_name} escapes output root"
                ) from exc
            if not published_artifact.is_file():
                raise RuntimeError(
                    f"published campaign summary {field_name} does not resolve"
                )


def _closed_index(
    manifest: Mapping[str, object], campaigns: Sequence[Mapping[str, object]]
) -> None:
    planned = manifest["matrix"]
    if not isinstance(planned, list) or len(campaigns) != len(planned):
        raise RuntimeError("campaign index does not close against the planned matrix")
    expected = [
        (row["matrix_index"], row["system"], row["master_seed"], row["arm_id"])
        for row in planned
    ]
    observed = [
        (row["matrix_index"], row["system"], row["master_seed"], row["arm_id"])
        for row in campaigns
    ]
    if observed != expected or len(set(observed)) != len(observed):
        raise RuntimeError("campaign index identities do not exactly match the manifest")


def run_ablation(
    *,
    output_root: Path,
    expected_git_commit: str,
    systems: Sequence[str],
    master_seeds: Sequence[int],
    action_force_budget: int,
    total_force_budget: int,
    arms: Sequence[float] = DEFAULT_ARMS,
    preflight_only: bool = False,
) -> dict[str, object]:
    """Run the complete arm matrix, publishing it only after exact closure."""
    output_root = Path(output_root)
    _preflight_output_root(output_root)
    manifest = preflight(
        expected_git_commit=expected_git_commit,
        systems=systems,
        master_seeds=master_seeds,
        action_force_budget=action_force_budget,
        total_force_budget=total_force_budget,
        arms=arms,
    )
    if preflight_only:
        return manifest

    posterior = _posterior_runner()
    production = posterior._load_production_runner()
    staging_root = _staging_root(output_root)
    staging_root.mkdir()
    _write_json_exclusive(staging_root / "manifest.json", manifest)
    campaigns: list[dict[str, object]] = []
    for row in manifest["matrix"]:
        case_directory = staging_root / str(row["relative_case_directory"])
        case_directory.parent.mkdir(parents=True, exist_ok=True)
        config, projection = build_ssw_config(
            str(row["system"]),
            case_directory,
            proposal_fmax=float(row["proposal_fmax"]),
        )
        exploration = posterior.build_exploration_config(
            policy_name=POLICY_NAME,
            run_directory=case_directory,
            master_seed=int(row["master_seed"]),
            action_force_budget=action_force_budget,
            total_force_budget=total_force_budget,
            batch_size=SERIAL_BATCH_SIZE,
            max_workers=SERIAL_MAX_WORKERS,
        )
        owner = posterior.ThreadOwnedCalculatorFactory(
            posterior._mace_calculator_factory
        )
        factory = posterior.InstrumentedCalculatorFactory(owner)
        posterior.run_campaign(
            initial_state=production.load_state(str(row["system"])),
            calculator_factory=factory,
            ssw_config=config,
            exploration_config=exploration,
        )
        _rewrite_campaign_summary_paths(
            case_directory=case_directory,
            staging_root=staging_root,
            output_root=output_root,
        )
        final_case_directory = output_root / str(row["relative_case_directory"])
        campaigns.append(
            {
                **row,
                "projection": projection,
                "campaign_summary_path": str(final_case_directory / "campaign_summary.json"),
            }
        )
    _closed_index(manifest, campaigns)
    index = {
        "schema_version": SCHEMA_VERSION,
        "manifest": manifest,
        "campaigns": campaigns,
    }
    published_index = _json_value(index)
    _write_json_exclusive(staging_root / "index.json", published_index)
    os.replace(staging_root, output_root)
    _verify_published_campaign_summary_paths(
        campaigns=campaigns,
        output_root=output_root,
        staging_root=staging_root,
    )
    return published_index


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=SYSTEMS)
    parser.add_argument(
        "--seeds",
        "--master-seeds",
        dest="master_seeds",
        nargs="+",
        type=int,
        required=True,
    )
    parser.add_argument("--action-force-budget", type=int, required=True)
    parser.add_argument("--total-force-budget", type=int, required=True)
    parser.add_argument("--arms", nargs="+", type=float, default=list(DEFAULT_ARMS))
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = run_ablation(
        output_root=args.output,
        expected_git_commit=args.expected_git_commit,
        systems=tuple(args.systems),
        master_seeds=tuple(args.master_seeds),
        action_force_budget=args.action_force_budget,
        total_force_budget=args.total_force_budget,
        arms=tuple(args.arms),
        preflight_only=args.preflight_only,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
