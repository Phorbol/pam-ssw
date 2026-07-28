"""Contract tests for the strict-quench 200-step production evidence."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

import pytest


ROOT = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260728-safe-lbfgs-strict-quench-200-production"
)
ANALYZER_PATH = ROOT / "analyze_production.py"
RAW_ROOT = Path(os.environ.get("PAMSSW_STRICT_EVIDENCE_RAW_ROOT", ROOT))
NEW_OUTPUT_ROOT = RAW_ROOT
BASELINE_ROOT = RAW_ROOT

REQUIRED_RAW_CASES = tuple(
    RAW_ROOT / name
    for name in (
        "output-c60-seed42",
        "output-pdo-seed42",
        "output-baseline-current-c60-seed42",
        "output-baseline-current-pdo-seed42",
    )
)
MISSING_RAW_CASES = tuple(path for path in REQUIRED_RAW_CASES if not path.is_dir())
requires_raw = pytest.mark.skipif(
    bool(MISSING_RAW_CASES),
    reason=(
        "local gitignored strict-quench production evidence is required: "
        + ", ".join(path.name for path in MISSING_RAW_CASES)
    ),
)


def load_analyzer(name: str = "strict_quench_200_analysis"):
    spec = importlib.util.spec_from_file_location(name, ANALYZER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def copy_raw_pairs(tmp_path: Path) -> tuple[Path, Path]:
    copied_new = tmp_path / "new"
    copied_old = tmp_path / "old"
    for system in ("c60", "pdo"):
        for source_name, target_root in (
            (f"output-{system}-seed42", copied_new),
            (f"output-baseline-current-{system}-seed42", copied_old),
        ):
            source = RAW_ROOT / source_name
            target = target_root / source_name
            target.mkdir(parents=True)
            for name in (
                "summary.json",
                "energy_trace.json",
                "walk_records.json",
                "optimizer_diagnostics.json",
            ):
                shutil.copy2(source / name, target / name)
    return copied_new, copied_old


def _config(system: str, root: str, *, strict: bool) -> dict:
    config = {
        "max_trials": 200,
        "rng_seed": 42,
        "proposal_optimizer": "safe-lbfgs-total",
        "proposal_fmax": 0.05,
        "quench_optimizer": "scipy-lbfgsb",
        "quench_fmax": 0.01 if system == "c60" else 0.03,
        "quench_maxiter": 400,
        "quench_fallback_optimizer": None,
        "target_uphill_energy": 0.5,
        "accepted_structures_dir": f"{root}/accepted",
        "accepted_structures_log": f"{root}/accepted.jsonl",
        "direction_diagnostics_path": f"{root}/directions.jsonl",
    }
    if strict:
        config.update(
            quench_optimizer="ase-lbfgs",
            quench_fmax=0.01,
            quench_fallback_optimizer="ase-fire",
        )
    return config


def _identity(system: str) -> dict:
    return {
        "input_path": (
            "/portable/repo/runs/20260428-c60-mace-production/prerelaxed_c60.xyz"
            if system == "c60"
            else "/portable/repo/PdO.xyz"
        ),
        "input_sha256": "1" * 64 if system == "c60" else "2" * 64,
        "model_path": "/portable/cache/model.model",
        "model_sha256": "3" * 64,
        "runtime_versions": {"python": "3.12", "mace": "0.3.14"},
        "cuda": {"available": True, "device_name": "synthetic"},
        "calculator": {"device": "cuda", "default_dtype": "float32"},
        "safe_lbfgs_default_history_limit": 10,
    }


def _write_synthetic_case(root: Path, system: str, *, strict: bool) -> None:
    name = (
        f"output-{system}-seed42"
        if strict
        else f"output-baseline-current-{system}-seed42"
    )
    case_dir = root / name
    case_dir.mkdir(parents=True)
    recorded = 200 if strict or system == "pdo" else 199
    fragment_rejections = 200 - recorded
    walk_records = [{"trial": trial} for trial in range(1, recorded + 1)]
    trace = [
        {"trial": trial, "best_energy_eV": 0.0 if trial == 0 else -1.0}
        for trial in range(recorded + 1)
    ]
    if strict:
        diagnostics = {
            "true_quench_count": 201,
            "true_quench_termination_converged": 201,
            "true_quench_termination_unconverged": 0,
            "true_quench_unconverged": 0,
            "true_quench_max_gradient": 0.009,
            "quench_fallback_attempts": 1,
            "quench_fallback_converged": 1,
        }
    elif system == "c60":
        diagnostics = {
            "true_quench_count": 201,
            "true_quench_termination_converged": 40,
            "true_quench_termination_unconverged": 161,
            "true_quench_unconverged": 161,
            "true_quench_max_gradient": 0.2,
            "quench_fallback_attempts": 0,
            "quench_fallback_converged": 0,
        }
    else:
        diagnostics = {
            "true_quench_count": 201,
            "true_quench_termination_converged": 201,
            "true_quench_termination_unconverged": 0,
            "true_quench_unconverged": 0,
            "true_quench_max_gradient": 0.02,
            "quench_fallback_attempts": 0,
            "quench_fallback_converged": 0,
        }
    source_config = _config(system, f"/synthetic/{system}/strict", strict=False)
    effective_config = _config(
        system,
        f"/synthetic/{system}/{'strict' if strict else 'baseline'}",
        strict=strict,
    )
    purpose_counts = {
        "landing_true_quench": 10,
        "biased_proposal_relax": 20,
        "direction_oracle": 5,
        "unattributed": 0,
    }
    identity = _identity(system)
    summary = {
        "system": system,
        "execution_commit": "32980ad9154ec6481b310477dd7b85597cefd49a",
        **identity,
        "effective_config": effective_config,
        "stats": {
            "n_trials": 200,
            "configured_max_trials": 200,
            "fragment_rejections": fragment_rejections,
            "force_evaluations": 35,
            "n_minima": 2,
            "duplicate_rate": 0.0,
        },
        "force_evaluations": 35,
        "purpose_counts": purpose_counts,
        "optimizer_telemetry": diagnostics,
        "walk_records": walk_records,
        "initial_energy_eV": 0.0,
        "best_energy_eV": -1.0,
        "energy_drop_eV": 1.0,
        "timing": {"total_wall_time_s": 1.0},
    }
    if strict:
        summary["strict_quench_wrapper"] = {
            "schema_version": 1,
            "base_runner": {
                "path": "/portable/repo/runs/20260728-safe-lbfgs-200-production/run_production.py",
                "sha256": "4" * 64,
            },
            "base_preflight": {
                "execution_commit": "32980ad9154ec6481b310477dd7b85597cefd49a",
                "system": system,
                **identity,
            },
            "source_config": source_config,
            "effective_config": effective_config,
            "config_diff": {
                key: [source_config[key], effective_config[key]]
                for key in sorted(source_config)
                if source_config[key] != effective_config[key]
            },
            "overrides": {
                "quench_optimizer": "ase-lbfgs",
                "quench_fmax": 0.01,
                "quench_fallback_optimizer": "ase-fire",
                "quench_maxiter": 400,
            },
            "fallback_counts": {"attempts": 1, "converged": 1},
        }
    for filename, payload in (
        ("summary.json", summary),
        ("energy_trace.json", trace),
        ("walk_records.json", walk_records),
        ("optimizer_diagnostics.json", diagnostics),
    ):
        (case_dir / filename).write_text(
            json.dumps(payload, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def synthetic_raw_root(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    for system in ("c60", "pdo"):
        _write_synthetic_case(root, system, strict=True)
        _write_synthetic_case(root, system, strict=False)
    return root


@requires_raw
def test_analyze_requires_full_strict_certificates_and_reports_exact_deltas():
    analyzer = load_analyzer("strict_quench_200_analysis_contract")

    evidence = analyzer.analyze(NEW_OUTPUT_ROOT, BASELINE_ROOT)

    assert evidence["schema_version"] == 1
    for system in ("c60", "pdo"):
        strict = evidence["strict_production"][system]
        assert strict["trials"] == 200
        assert strict["walk_records"] == 200
        assert strict["true_quench"]["count"] == 201
        assert strict["true_quench"]["certified"] == 201
        assert strict["true_quench"]["unconverged"] == 0
        assert strict["cost"]["unattributed"] == 0
        assert strict["cost"]["purpose_total"] == strict["force_evaluations"]
        assert strict["strict_quench_wrapper"]["fallback_counts"] == strict[
            "true_quench"
        ]["fallback_counts"]
        assert strict["raw_sha256"]["summary"]

    assert evidence["baseline_production"]["c60"]["trials"] == 200
    assert evidence["baseline_production"]["c60"]["recorded_trials"] == 199
    assert evidence["baseline_production"]["c60"]["unlogged_trials"] == 1
    assert evidence["baseline_production"]["c60"]["unlogged_accounted_by"] == {
        "fragment_rejections": 1
    }
    assert evidence["baseline_production"]["pdo"]["recorded_trials"] == 200
    assert evidence["baseline_production"]["pdo"]["unlogged_trials"] == 0
    assert evidence["baseline_production"]["pdo"]["unlogged_accounted_by"] == {}
    assert evidence["comparison"]["c60"]["delta"]["force_evaluations"] == -1217
    assert evidence["comparison"]["c60"]["delta"]["energy_drop_eV"] == pytest.approx(
        -6.44195556640625
    )
    assert evidence["comparison"]["pdo"]["delta"]["force_evaluations"] == 10872
    assert evidence["comparison"]["pdo"]["delta"]["energy_drop_eV"] == pytest.approx(
        -0.87860107421875
    )
    assert evidence["strict_production"]["c60"]["cost"]["purpose_fractions"][
        "biased_proposal_relax"
    ] == pytest.approx(0.5159143075745983)
    assert evidence["strict_production"]["pdo"]["cost"]["purpose_fractions"][
        "landing_true_quench"
    ] == pytest.approx(0.4297355450092323)


@requires_raw
def test_analyze_rejects_missing_strict_true_quench_certificate(tmp_path):
    analyzer = load_analyzer("strict_quench_200_analysis_reject_certificate")
    copied_new, copied_old = copy_raw_pairs(tmp_path)

    for path in (
        copied_new / "output-c60-seed42" / "optimizer_diagnostics.json",
        copied_new / "output-c60-seed42" / "summary.json",
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        diagnostics = payload if path.name == "optimizer_diagnostics.json" else payload[
            "optimizer_telemetry"
        ]
        diagnostics["true_quench_termination_converged"] = 200
        path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="strict true-quench certificate"):
        analyzer.analyze(copied_new, copied_old)


@requires_raw
def test_analyze_derives_the_complete_config_diff_instead_of_trusting_stored_diff(
    tmp_path,
):
    analyzer = load_analyzer("strict_quench_200_analysis_reject_config_mutation")
    copied_new, copied_old = copy_raw_pairs(tmp_path)
    summary_path = copied_new / "output-c60-seed42" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["strict_quench_wrapper"]["source_config"]["target_uphill_energy"] = 0.9
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="derived strict config diff"):
        analyzer.analyze(copied_new, copied_old)


@requires_raw
def test_analyze_rejects_nonquench_baseline_config_mutation(tmp_path):
    analyzer = load_analyzer("strict_quench_200_analysis_reject_baseline_config")
    copied_new, copied_old = copy_raw_pairs(tmp_path)
    summary_path = copied_old / "output-baseline-current-c60-seed42" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["effective_config"]["target_uphill_energy"] = 987.654321
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="baseline config differs from strict source config"):
        analyzer.analyze(copied_new, copied_old)


@requires_raw
def test_analyze_rejects_c60_baseline_transition_completeness_drift(tmp_path):
    analyzer = load_analyzer("strict_quench_200_analysis_reject_c60_completeness")
    copied_new, copied_old = copy_raw_pairs(tmp_path)
    case_dir = copied_old / "output-baseline-current-c60-seed42"
    summary_path = case_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    walk_records = json.loads((case_dir / "walk_records.json").read_text(encoding="utf-8"))[:-1]
    energy_trace = json.loads((case_dir / "energy_trace.json").read_text(encoding="utf-8"))[:-1]
    summary["walk_records"] = walk_records
    summary["stats"]["fragment_rejections"] = 2
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    (case_dir / "walk_records.json").write_text(json.dumps(walk_records), encoding="utf-8")
    (case_dir / "energy_trace.json").write_text(json.dumps(energy_trace), encoding="utf-8")

    with pytest.raises(ValueError, match="transition completeness mismatch"):
        analyzer.analyze(copied_new, copied_old)


@requires_raw
def test_write_outputs_records_raw_and_evidence_provenance(tmp_path):
    analyzer = load_analyzer("strict_quench_200_analysis_write")

    evidence = analyzer.write_outputs(NEW_OUTPUT_ROOT, BASELINE_ROOT, tmp_path)

    evidence_path = tmp_path / "production_evidence.json"
    conclusion_path = tmp_path / "production_conclusion.md"
    assert evidence_path.is_file()
    assert conclusion_path.is_file()
    persisted = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert persisted == evidence
    assert persisted["artifact_provenance"][
        "content_sha256_excluding_self"
    ] == analyzer.content_sha256_excluding_self(persisted)
    conclusion = conclusion_path.read_text(encoding="utf-8")
    assert "ASE-LBFGS+certificate FIRE" in conclusion
    assert "不能纯归因" in conclusion


@requires_raw
def test_write_outputs_is_byte_identical_for_absolute_and_relative_roots(
    tmp_path, monkeypatch
):
    analyzer = load_analyzer("strict_quench_200_analysis_portable_paths")
    repo_root = Path(__file__).resolve().parents[2]
    relative_run_root = RAW_ROOT.relative_to(repo_root)
    absolute_output = tmp_path / "absolute"
    relative_output = tmp_path / "relative"
    monkeypatch.chdir(repo_root)

    absolute_evidence = analyzer.write_outputs(
        RAW_ROOT.resolve(), RAW_ROOT.resolve(), absolute_output
    )
    relative_evidence = analyzer.write_outputs(
        relative_run_root, relative_run_root, relative_output
    )

    assert absolute_evidence == relative_evidence
    assert (
        absolute_output / "production_evidence.json"
    ).read_bytes() == (
        relative_output / "production_evidence.json"
    ).read_bytes()
    assert (
        absolute_output / "production_conclusion.md"
    ).read_bytes() == (
        relative_output / "production_conclusion.md"
    ).read_bytes()
    evidence_text = (absolute_output / "production_evidence.json").read_text(
        encoding="utf-8"
    )
    assert str(repo_root.resolve()) not in evidence_text
    assert "/tmp/SSW-worktrees/" not in evidence_text
    assert absolute_evidence["artifact_provenance"][
        "content_sha256_excluding_self"
    ] == relative_evidence["artifact_provenance"][
        "content_sha256_excluding_self"
    ]


def test_synthetic_contract_rejects_missing_artifact_and_completeness_drift(
    tmp_path,
):
    analyzer = load_analyzer("strict_quench_200_synthetic_missing")
    root = synthetic_raw_root(tmp_path)
    (root / "output-c60-seed42" / "walk_records.json").unlink()

    with pytest.raises(ValueError, match="missing raw artifact"):
        analyzer.analyze(root, root)

    root = synthetic_raw_root(tmp_path / "completeness")
    case_dir = root / "output-baseline-current-c60-seed42"
    summary_path = case_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    walk_records = json.loads(
        (case_dir / "walk_records.json").read_text(encoding="utf-8")
    )[:-1]
    trace = json.loads(
        (case_dir / "energy_trace.json").read_text(encoding="utf-8")
    )[:-1]
    summary["walk_records"] = walk_records
    summary["stats"]["fragment_rejections"] = 2
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    (case_dir / "walk_records.json").write_text(
        json.dumps(walk_records), encoding="utf-8"
    )
    (case_dir / "energy_trace.json").write_text(json.dumps(trace), encoding="utf-8")
    with pytest.raises(ValueError, match="transition completeness mismatch"):
        analyzer.analyze(root, root)


def test_synthetic_contract_rejects_full_config_and_wrapper_diff_drift(tmp_path):
    analyzer = load_analyzer("strict_quench_200_synthetic_config")
    root = synthetic_raw_root(tmp_path)
    path = root / "output-baseline-current-c60-seed42" / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["effective_config"]["target_uphill_energy"] = 9.0
    path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="baseline config differs"):
        analyzer.analyze(root, root)

    for mutation, expected in (
        ("source", "derived strict config diff"),
        ("stored", "derived strict config diff"),
    ):
        root = synthetic_raw_root(tmp_path / mutation)
        path = root / "output-c60-seed42" / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        wrapper = summary["strict_quench_wrapper"]
        if mutation == "source":
            wrapper["source_config"]["target_uphill_energy"] = 9.0
        else:
            wrapper["config_diff"]["target_uphill_energy"] = [0.5, 9.0]
        path.write_text(json.dumps(summary), encoding="utf-8")
        with pytest.raises(ValueError, match=expected):
            analyzer.analyze(root, root)


def test_synthetic_contract_rejects_ledger_and_certificate_drift(tmp_path):
    analyzer = load_analyzer("strict_quench_200_synthetic_ledger")
    root = synthetic_raw_root(tmp_path)
    summary_path = root / "output-c60-seed42" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["purpose_counts"]["landing_true_quench"] = 9
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="purpose counts do not close"):
        analyzer.analyze(root, root)

    root = synthetic_raw_root(tmp_path / "certificate")
    for filename in ("summary.json", "optimizer_diagnostics.json"):
        path = root / "output-c60-seed42" / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        diagnostics = (
            payload["optimizer_telemetry"] if filename == "summary.json" else payload
        )
        diagnostics["true_quench_termination_converged"] = 200
        path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="strict true-quench certificate"):
        analyzer.analyze(root, root)


def test_synthetic_contract_self_hash_and_portable_paths(tmp_path, monkeypatch):
    analyzer = load_analyzer("strict_quench_200_synthetic_portable")
    root = synthetic_raw_root(tmp_path)
    absolute_output = tmp_path / "absolute"
    relative_output = tmp_path / "relative"
    monkeypatch.chdir(tmp_path)

    absolute = analyzer.write_outputs(root.resolve(), root.resolve(), absolute_output)
    relative_root = root.relative_to(tmp_path)
    relative = analyzer.write_outputs(relative_root, relative_root, relative_output)

    assert absolute == relative
    assert (absolute_output / "production_evidence.json").read_bytes() == (
        relative_output / "production_evidence.json"
    ).read_bytes()
    assert absolute["artifact_provenance"][
        "content_sha256_excluding_self"
    ] == analyzer.content_sha256_excluding_self(absolute)


def test_rendered_conclusion_does_not_claim_landing_basin_identity(tmp_path):
    analyzer = load_analyzer("strict_quench_200_synthetic_claim_boundary")
    root = synthetic_raw_root(tmp_path)

    conclusion = analyzer.render_conclusion(analyzer.analyze(root, root))

    assert "landing basin 已因 true quench 改变而分叉" not in conclusion
    assert "不能继续视为 paired trajectory comparison" in conclusion


def test_synthetic_outputs_are_published_with_atomic_replace(tmp_path, monkeypatch):
    analyzer = load_analyzer("strict_quench_200_synthetic_atomic")
    root = synthetic_raw_root(tmp_path)
    output = tmp_path / "published"
    replacements = []
    original_replace = analyzer.os.replace

    def record_replace(source, target):
        replacements.append((Path(source), Path(target)))
        original_replace(source, target)

    monkeypatch.setattr(analyzer.os, "replace", record_replace)
    analyzer.write_outputs(root, root, output)

    assert [target.name for _, target in replacements] == [
        "production_evidence.json",
        "production_conclusion.md",
    ]
    assert all(not source.exists() for source, _ in replacements)
