#!/usr/bin/env python3
"""Build compact evidence for the two-operator Stage B gate."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import importlib.util
import json
import platform
from pathlib import Path
from statistics import mean, median
import subprocess
import sys


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
RUNNER_PATH = RUN_ROOT / "run_stage_b.py"
PROTOCOL_PATH = RUN_ROOT / "protocol.py"


def _load_module(path: Path, name: str):
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_module(RUNNER_PATH, "_two_operator_stage_b_analysis_runner")
protocol = _load_module(PROTOCOL_PATH, "_two_operator_stage_b_analysis_protocol")


def _distribution(values) -> dict[str, float | int | None]:
    data = [float(value) for value in values]
    if not data:
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "mean": None,
            "maximum": None,
        }
    return {
        "count": len(data),
        "minimum": min(data),
        "median": median(data),
        "mean": mean(data),
        "maximum": max(data),
    }


def _family_summary(rows) -> dict[str, object]:
    loaded_fe = sum(int(row["fully_loaded_force_evaluations"]) for row in rows)
    purpose_counts = Counter()
    for row in rows:
        purpose_counts.update(
            {str(key): int(value) for key, value in row["purpose_counts"].items()}
        )
    certified_nonstarter = sum(
        bool(row["certified"]) and not bool(row["same_starter_basin"])
        for row in rows
    )
    global_improvements = sum(bool(row["improved_global_best"]) for row in rows)
    return {
        "action_count": len(rows),
        "certified_count": sum(bool(row["certified"]) for row in rows),
        "certified_nonstarter_count": certified_nonstarter,
        "global_improvement_count": global_improvements,
        "geometry_invalid_count": sum(
            not bool(row["geometry_valid"]) and not bool(row["budget_censored"])
            for row in rows
        ),
        "fragmented_count": sum(bool(row["fragmented"]) for row in rows),
        "budget_censored_count": sum(bool(row["budget_censored"]) for row in rows),
        "exclusive_force_evaluations": sum(
            int(row["force_evaluations"]) for row in rows
        ),
        "fully_loaded_force_evaluations": loaded_fe,
        "purpose_counts": dict(sorted(purpose_counts.items())),
        "certified_nonstarter_per_1000_loaded_fe": (
            1000.0 * certified_nonstarter / loaded_fe if loaded_fe else None
        ),
        "global_improvements_per_1000_loaded_fe": (
            1000.0 * global_improvements / loaded_fe if loaded_fe else None
        ),
        "exclusive_force_evaluation_distribution": _distribution(
            row["force_evaluations"] for row in rows
        ),
        "fully_loaded_force_evaluation_distribution": _distribution(
            row["fully_loaded_force_evaluations"] for row in rows
        ),
        "exclusive_wall_time_s_distribution": _distribution(
            row["wall_time_s"] for row in rows
        ),
        "fully_loaded_wall_time_s_distribution": _distribution(
            row["fully_loaded_wall_time_s"] for row in rows
        ),
        "landing_delta_eV_distribution": _distribution(
            row["landing_delta_eV"]
            for row in rows
            if row["landing_delta_eV"] is not None
        ),
    }


def _paired_support(pair_records) -> dict[str, int]:
    counts = Counter()
    for pair in pair_records:
        by_family = {row["operator_family"]: row for row in pair["actions"]}
        direct = by_family["direct"]
        ssw = by_family["ssw"]
        direct_escape = bool(direct["certified"]) and not bool(
            direct["same_starter_basin"]
        )
        ssw_escape = bool(ssw["certified"]) and not bool(ssw["same_starter_basin"])
        if direct_escape and ssw_escape:
            counts["both_certified_nonstarter"] += 1
        elif direct_escape:
            counts["direct_only_certified_nonstarter"] += 1
        elif ssw_escape:
            counts["ssw_only_certified_nonstarter"] += 1
        else:
            counts["neither_certified_nonstarter"] += 1
    return {
        key: counts[key]
        for key in (
            "both_certified_nonstarter",
            "direct_only_certified_nonstarter",
            "ssw_only_certified_nonstarter",
            "neither_certified_nonstarter",
        )
    }


def _paired_regrets(pair_records) -> list[float]:
    values = []
    for pair in pair_records:
        by_family = {row["operator_family"]: row for row in pair["actions"]}
        direct_delta = by_family["direct"]["landing_delta_eV"]
        ssw_delta = by_family["ssw"]["landing_delta_eV"]
        if direct_delta is not None and ssw_delta is not None:
            values.append(float(direct_delta) - float(ssw_delta))
    return values


def _canonical_sha256(payload) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def classify_basin_split(
    *,
    same_starter_basin: bool,
    landing_delta_eV: float,
    energy_tolerance_eV: float,
    archive_rmsd_A: float,
    rmsd_tolerance_A: float,
) -> str:
    """Explain which archive predicate separated a landing from its starter."""
    if same_starter_basin:
        return "archive_match"
    energy_split = abs(float(landing_delta_eV)) > float(energy_tolerance_eV)
    geometry_split = float(archive_rmsd_A) > float(rmsd_tolerance_A)
    if energy_split and geometry_split:
        return "energy_and_geometry_split"
    if energy_split:
        return "energy_only_split"
    if geometry_split:
        return "geometry_only_split"
    return "inconsistent_with_archive_predicate"


def _geometry_audit(input_directory: Path, pairs) -> dict[str, object]:
    """Audit basin labels from saved coordinates without new PES evaluations."""
    from pamssw.archive import MinimaArchive
    from pamssw.io import read_state

    rows = []
    missing = []
    for pair in pairs:
        case_directory = (
            Path(input_directory)
            / pair["system"]
            / pair["starter_context"]
            / f"seed-{int(pair['seed']):08d}"
        )
        starter_path = case_directory / "starter.xyz"
        for action in pair["actions"]:
            landing_path = case_directory / f"{action['operator_family']}-landing.xyz"
            if not starter_path.is_file() or not landing_path.is_file():
                missing.append(str(landing_path.relative_to(input_directory)))
                continue
            starter = read_state(starter_path)
            landing = read_state(landing_path)
            rmsd = MinimaArchive._rmsd(starter, landing)
            config = pair["effective_config"]
            rows.append(
                {
                    "system": pair["system"],
                    "starter_context": pair["starter_context"],
                    "seed": int(pair["seed"]),
                    "operator_family": action["operator_family"],
                    "same_starter_basin": bool(action["same_starter_basin"]),
                    "landing_delta_eV": float(action["landing_delta_eV"]),
                    "archive_rmsd_A": float(rmsd),
                    "classification": classify_basin_split(
                        same_starter_basin=bool(action["same_starter_basin"]),
                        landing_delta_eV=float(action["landing_delta_eV"]),
                        energy_tolerance_eV=float(config["dedup_energy_tol"]),
                        archive_rmsd_A=rmsd,
                        rmsd_tolerance_A=float(config["dedup_rmsd_tol"]),
                    ),
                }
            )
    counts = Counter(row["classification"] for row in rows)
    direct_rows = [row for row in rows if row["operator_family"] == "direct"]
    direct_nonstarter = [row for row in direct_rows if not row["same_starter_basin"]]
    return {
        "new_force_evaluations": 0,
        "complete": not missing and len(rows) == 2 * len(pairs),
        "missing_artifacts": missing,
        "classification_counts": dict(sorted(counts.items())),
        "direct_nonstarter_count": len(direct_nonstarter),
        "direct_energy_only_split_count": sum(
            row["classification"] == "energy_only_split"
            for row in direct_nonstarter
        ),
        "direct_nonstarter_archive_rmsd_A_distribution": _distribution(
            row["archive_rmsd_A"] for row in direct_nonstarter
        ),
        "rows": rows,
    }


def build_evidence(pair_records, provenance) -> dict[str, object]:
    pairs = sorted(
        pair_records,
        key=lambda row: (
            protocol.SYSTEMS.index(row["system"]),
            protocol.STARTERS.index(row["starter_context"]),
            protocol.SEEDS.index(int(row["seed"])),
        ),
    )
    expected_keys = [
        (item.system, item.starter_context, item.seed)
        for item in protocol.case_matrix()
    ]
    observed_keys = [
        (row["system"], row["starter_context"], int(row["seed"]))
        for row in pairs
    ]
    if observed_keys != expected_keys:
        raise ValueError("Stage B does not contain the exact 18-pair matrix")
    for pair in pairs:
        runner.validate_pair_record(pair)

    actions = sorted(
        [action for pair in pairs for action in pair["actions"]],
        key=lambda row: (
            protocol.SYSTEMS.index(row["system"]),
            protocol.STARTERS.index(row["starter_context"]),
            protocol.SEEDS.index(int(row["seed"])),
            protocol.FAMILIES.index(row["operator_family"]),
        ),
    )
    if len(actions) != 36:
        raise ValueError("Stage B does not contain exactly 36 actions")

    purpose_counts = Counter()
    global_force_evaluations = 0
    for pair in pairs:
        global_force_evaluations += int(pair["pair_force_evaluations"])
        purpose_counts.update(
            {
                str(key): int(value)
                for key, value in pair[
                    "starter_validation_purpose_counts"
                ].items()
            }
        )
        purpose_counts["direction_oracle"] += int(
            pair["shared_direction_force_evaluations"]
        )
        for action in pair["actions"]:
            purpose_counts.update(
                {
                    str(key): int(value)
                    for key, value in action["purpose_counts"].items()
                }
            )
    purpose_total = sum(purpose_counts.values())
    if purpose_total != global_force_evaluations:
        raise ValueError("global purpose ledger does not close")
    unattributed = int(purpose_counts.get("unattributed", 0))
    if unattributed != 0:
        raise ValueError("global purpose ledger contains unattributed work")

    system_summaries = {}
    for system in protocol.SYSTEMS:
        system_rows = [row for row in actions if row["system"] == system]
        system_pairs = [row for row in pairs if row["system"] == system]
        family_summaries = {
            family: _family_summary(
                [row for row in system_rows if row["operator_family"] == family]
            )
            for family in protocol.FAMILIES
        }
        system_summaries[system] = {
            **family_summaries,
            "paired_support": _paired_support(system_pairs),
            "paired_regret_eV_distribution": _distribution(
                _paired_regrets(system_pairs)
            ),
        }
    pooled_summary = {
        "action_count": len(actions),
        "families": {
            family: _family_summary(
                [row for row in actions if row["operator_family"] == family]
            )
            for family in protocol.FAMILIES
        },
        "paired_support": _paired_support(pairs),
        "paired_regret_eV_distribution": _distribution(_paired_regrets(pairs)),
    }
    effective_config_manifest = [
        {
            "system": pair["system"],
            "starter_context": pair["starter_context"],
            "seed": int(pair["seed"]),
            "sha256": _canonical_sha256(pair["effective_config"]),
        }
        for pair in pairs
    ]
    starter_manifest = [
        {
            "system": pair["system"],
            "starter_context": pair["starter_context"],
            "sha256": pair["starter_sha256"],
        }
        for pair in pairs
        if int(pair["seed"]) == protocol.SEEDS[0]
    ]
    return {
        "schema_version": 1,
        "cohort": {
            "systems": list(protocol.SYSTEMS),
            "starters": list(protocol.STARTERS),
            "seeds": list(protocol.SEEDS),
            "pair_count": len(pairs),
            "action_count": len(actions),
            "max_force_evaluations": protocol.MAX_FORCE_EVALUATIONS,
        },
        "global_force_evaluations": global_force_evaluations,
        "purpose_counts": dict(sorted(purpose_counts.items())),
        "purpose_ledger_closes": True,
        "unattributed_force_evaluations": unattributed,
        "system_summaries": system_summaries,
        "pooled_summary": pooled_summary,
        "decision": protocol.decide_stage_b(actions),
        "provenance": {
            **dict(provenance),
            "effective_config_manifest": effective_config_manifest,
            "starter_manifest": starter_manifest,
        },
    }


def _file_sha256(path: Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def analyze_directory(
    input_directory: Path,
    output: Path,
    *,
    provenance,
):
    input_directory = Path(input_directory)
    paths = sorted(input_directory.glob("**/pair.json"))
    if not paths:
        raise ValueError(f"no pair records found under {input_directory}")
    manifest = [
        {
            "path": str(path.relative_to(input_directory)),
            "sha256": _file_sha256(path),
        }
        for path in paths
    ]
    pairs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    evidence = build_evidence(
        pairs,
        provenance={
            **dict(provenance),
            "raw_pair_manifest_count": len(manifest),
            "raw_pair_manifest": manifest,
        },
    )
    evidence["posthoc_geometry_audit"] = _geometry_audit(
        input_directory,
        pairs,
    )
    _write_json(output, evidence)
    return evidence


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def runtime_provenance() -> dict[str, object]:
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        device = torch.cuda.get_device_name(0) if cuda_available else "cpu"
        torch_version = str(torch.__version__)
    except Exception:
        cuda_available = False
        device = "unavailable"
        torch_version = None
    common_model = Path("/root/.cache/mace/mace-omat-0-small.model")
    cuo_archive = runner.observability.base.source.source.CUO_ARCHIVE_PATH
    return {
        "git_commit": git_commit,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": cuda_available,
        "device": device,
        "torch_version": torch_version,
        "package_versions": {
            name: _package_version(name)
            for name in ("ase", "mace-torch", "numpy", "scipy")
        },
        "common_model_path": str(common_model),
        "common_model_sha256": (
            _file_sha256(common_model) if common_model.is_file() else None
        ),
        "cuo_archive_path": str(cuo_archive),
        "cuo_archive_sha256": (
            _file_sha256(cuo_archive) if cuo_archive.is_file() else None
        ),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=RUN_ROOT / "output")
    parser.add_argument(
        "--output",
        type=Path,
        default=RUN_ROOT / "output/evidence.json",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    evidence = analyze_directory(
        args.input,
        args.output,
        provenance=runtime_provenance(),
    )
    print(json.dumps(evidence["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
