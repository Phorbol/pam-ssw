#!/usr/bin/env python3
"""Run paired C60 random-only versus bond-only terminal labeling."""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
PROTOCOL_PATH = RUN_ROOT / "protocol.py"
LEGACY_RUNNER_PATH = (
    REPO_ROOT
    / "runs"
    / "20260729-staged-direction-efficiency-ablation"
    / "run_stage.py"
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_protocol():
    return _load_module(
        PROTOCOL_PATH,
        "_fixed_direction_family_protocol_runtime",
    )


def _load_legacy_runner():
    return _load_module(
        LEGACY_RUNNER_PATH,
        "_fixed_direction_family_legacy_runner",
    )


def _current_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _tracked_worktree_clean() -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not completed.stdout.strip()


def _write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CaseBaseRunner:
    """Inject one ablation seam while preserving the locked runtime."""

    def __init__(self, base_runner, case) -> None:
        self._base_runner = base_runner
        self._case = case

    def build_config(self, system: str, case_dir: Path):
        source = self._base_runner.build_config(system, case_dir)
        return replace(
            source,
            n_bond_pairs=self._case.settings.n_bond_pairs,
            stagnation_bond_pair_boost=(
                self._case.settings.stagnation_bond_pair_boost
            ),
        )

    def write_state(self, path: Path, state):
        return self._base_runner.write_state(path, state)


class LegacyCertificateCapture:
    """Capture the real certificate while letting the legacy writer finish."""

    def __init__(self, checker) -> None:
        self._checker = checker
        self.actual_certificate: bool | None = None

    def __call__(self, landing, fmax: float) -> bool:
        self.actual_certificate = bool(self._checker(landing, fmax))
        return True


def annotate_case_row(
    row: dict[str, Any],
    *,
    case,
    baseline_n_bond_pairs: int,
    baseline_stagnation_bond_pair_boost: int,
) -> dict[str, Any]:
    annotated = dict(row)
    config_diff = dict(
        annotated.get("source_to_effective_config_diff", {})
    )
    config_diff["n_bond_pairs"] = [
        int(baseline_n_bond_pairs),
        int(case.settings.n_bond_pairs),
    ]
    config_diff["stagnation_bond_pair_boost"] = [
        int(baseline_stagnation_bond_pair_boost),
        int(case.settings.stagnation_bond_pair_boost),
    ]
    annotated["source_to_effective_config_diff"] = config_diff
    annotated["ablation_control"] = {
        "field": "n_bond_pairs",
        "source_value": int(baseline_n_bond_pairs),
        "effective_value": int(case.settings.n_bond_pairs),
        "expected_direction_kind": case.settings.expected_kind,
    }
    annotated["frozen_feedback_controls"] = {
        "stagnation_bond_pair_boost": int(
            case.settings.stagnation_bond_pair_boost
        ),
    }
    return annotated


def _execute_case(
    *,
    legacy_runner,
    case,
    state,
    state_provenance,
    shared_calculator,
    base_runner,
    case_dir,
    execution_commit,
):
    from pamssw.relax import has_force_convergence_certificate

    baseline_config = base_runner.build_config("c60", case_dir)
    certificate_capture = LegacyCertificateCapture(
        has_force_convergence_certificate,
    )
    row = legacy_runner._run_case(
        case=case,
        state=state,
        state_provenance=state_provenance,
        shared_calculator=shared_calculator,
        base_runner=CaseBaseRunner(base_runner, case),
        case_dir=case_dir,
        execution_commit=execution_commit,
        certificate_checker=certificate_capture,
    )
    if certificate_capture.actual_certificate is None:
        raise RuntimeError("legacy runner did not evaluate the certificate")
    row["certificate"] = certificate_capture.actual_certificate
    row["terminal_failure"] = (
        None
        if certificate_capture.actual_certificate
        else "strict_quench_nonconvergence"
    )
    row["meaningful"] = _load_protocol().is_meaningful(row)
    row = annotate_case_row(
        row,
        case=case,
        baseline_n_bond_pairs=baseline_config.n_bond_pairs,
        baseline_stagnation_bond_pair_boost=(
            baseline_config.stagnation_bond_pair_boost
        ),
    )
    _load_protocol().validate_case_row(case, row)
    _write_json(Path(case_dir) / "summary.json", row)
    return row


def _validate_saved_case(
    *,
    legacy_runner,
    protocol,
    row,
    case,
    execution_commit: str,
    config_path: Path,
) -> None:
    if row.get("certificate") is True:
        legacy_runner._validate_saved_case(
            protocol,
            row=row,
            case=case,
            execution_commit=execution_commit,
            config_path=config_path,
        )
        return
    protocol.validate_case_row(case, row)
    if row.get("execution_commit") != execution_commit:
        raise RuntimeError(f"saved case commit drifted: {case.key}")
    for path_field, hash_field in (
        ("starter_path", "starter_file_sha256"),
        ("escape_path", "escape_sha256"),
        ("landing_path", "landing_sha256"),
    ):
        path = Path(row[path_field])
        if not path.is_file() or _sha256(path) != row[hash_field]:
            raise RuntimeError(
                f"saved case artifact does not revalidate: {case.key}"
            )
    if (
        not config_path.is_file()
        or json.loads(config_path.read_text(encoding="utf-8"))
        != row["effective_config"]
    ):
        raise RuntimeError(
            f"saved case config does not revalidate: {case.key}"
        )


def _pin_sources(output_dir: Path) -> dict[str, str]:
    sources = {
        "protocol.py": PROTOCOL_PATH.resolve(),
        "run_experiment.py": Path(__file__).resolve(),
        "legacy_run_stage.py": LEGACY_RUNNER_PATH.resolve(),
    }
    hashes = {}
    for name, source in sources.items():
        target = output_dir / "pinned_sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        source_hash = _sha256(source)
        if target.exists():
            if _sha256(target) != source_hash:
                raise RuntimeError(f"pinned source drifted: {name}")
        else:
            shutil.copy2(source, target)
        hashes[name] = source_hash
    return hashes


def conclusion(evidence: dict[str, Any]) -> str:
    gate = evidence["posterior_gate"]
    counts = evidence["stable_meaningful_by_family"]
    audit = evidence["held_out_audit"]
    entered = str(bool(gate["enter_posterior_stage"])).lower()
    return (
        "# Fixed C60 direction-family terminal labels\n\n"
        f"Completed cases: {evidence['cohort']['completed_cases']}.\n\n"
        "Stable meaningful labels:\n\n"
        f"- random_only: {counts['random_only']}\n"
        f"- bond_only: {counts['bond_only']}\n\n"
        "Uncertified terminal outcomes: "
        f"{evidence['totals']['noncertified_terminal_outcomes']}.\n\n"
        "Leave-one-seed-out Brier scores:\n\n"
        f"- starter only: {audit['starter_only_brier']:.9f}\n"
        "- starter plus direction family: "
        f"{audit['starter_plus_family_brier']:.9f}\n\n"
        f"enter_posterior_stage: `{entered}`\n\n"
        f"Reason: `{gate['reason']}`.\n\n"
        "This paired fixed-starter experiment does not change production "
        "defaults and does not itself promote a posterior selector.\n"
    )


def run_experiment(
    *,
    output_dir: Path,
    expected_git_commit: str,
) -> dict[str, Any]:
    actual_commit = _current_commit()
    if actual_commit != expected_git_commit:
        raise RuntimeError(
            f"execution commit mismatch: expected {expected_git_commit}, "
            f"got {actual_commit}"
        )
    if not _tracked_worktree_clean():
        raise RuntimeError("tracked worktree is not clean")

    protocol = _load_protocol()
    legacy = _load_legacy_runner()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_hashes = _pin_sources(output_dir)
    (
        states,
        shared_calculator,
        base_runner,
        state_provenance,
        shared_provenance,
    ) = legacy._load_locked_runtime()

    rows = []
    for case in protocol.case_matrix():
        case_dir = output_dir / "cases" / case.key
        summary_path = case_dir / "summary.json"
        config_path = (
            output_dir / "effective_configs" / f"{case.key}.json"
        )
        if summary_path.exists():
            row = json.loads(summary_path.read_text(encoding="utf-8"))
            _validate_saved_case(
                legacy_runner=legacy,
                protocol=protocol,
                row=row,
                case=case,
                execution_commit=actual_commit,
                config_path=config_path,
            )
            protocol.validate_case_row(case, row)
        else:
            print(
                f"[fixed-family] case={case.key}",
                flush=True,
            )
            row = _execute_case(
                legacy_runner=legacy,
                case=case,
                state=states[case.state_id],
                state_provenance=state_provenance[case.state_id],
                shared_calculator=shared_calculator,
                base_runner=base_runner,
                case_dir=case_dir,
                execution_commit=actual_commit,
            )
            _write_json(config_path, row["effective_config"])
        rows.append(row)
        _write_json(
            output_dir / "raw.json",
            {
                "schema_version": 1,
                "execution_commit": actual_commit,
                "runner_source_sha256": source_hashes,
                "shared_provenance": shared_provenance,
                "state_provenance": state_provenance,
                "cases": rows,
            },
        )

    evidence = protocol.build_evidence(rows)
    evidence.update(
        {
            "execution_commit": actual_commit,
            "runner_source_sha256": source_hashes,
            "shared_provenance": shared_provenance,
            "state_provenance": state_provenance,
        }
    )
    _write_json(output_dir / "evidence.json", evidence)
    (output_dir / "conclusion.md").write_text(
        conclusion(evidence),
        encoding="utf-8",
    )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the preregistered fixed C60 random/bond direction-family "
            "terminal-label experiment."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    args = parser.parse_args()
    run_experiment(
        output_dir=args.output_dir,
        expected_git_commit=args.expected_git_commit,
    )


if __name__ == "__main__":
    main()
