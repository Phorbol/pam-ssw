from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    REPO_ROOT
    / "runs"
    / "20260731-pdo-matcher-numerical-gate"
    / "protocol.py"
)
ANALYZER_PATH = (
    REPO_ROOT
    / "runs"
    / "20260731-pdo-matcher-numerical-gate"
    / "analyze.py"
)
RESCUE_RUNNER_PATH = (
    REPO_ROOT
    / "runs"
    / "20260731-pdo-matcher-numerical-gate"
    / "run_certificate_rescue.py"
)


def _protocol():
    spec = importlib.util.spec_from_file_location(
        "_pdo_matcher_numerical_protocol",
        PROTOCOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _analyzer():
    spec = importlib.util.spec_from_file_location(
        "_pdo_matcher_numerical_analyzer",
        ANALYZER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rescue_runner():
    spec = importlib.util.spec_from_file_location(
        "_pdo_matcher_certificate_rescue",
        RESCUE_RUNNER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fixed_mask_matches_frozen_first_passage_runtime() -> None:
    analyzer = _analyzer()

    mask = analyzer._pdo_fixed_mask()

    assert mask.shape == (115,)
    assert int(mask.sum()) == 40
    assert int((~mask).sum()) == 75


def test_decomposition_uses_existing_matcher_thresholds_only() -> None:
    protocol = _protocol()

    row = protocol.decompose_pair(
        energy_delta_eV=0.4,
        energy_tol_eV=0.001,
        indexed_mic_rmsd_A=0.05,
        rmsd_tol_A=0.15,
        descriptor_delta=0.03,
        descriptor_tol=0.1,
    )

    assert row["energy_same"] is False
    assert row["indexed_geometry_same"] is True
    assert row["descriptor_same"] is True
    assert row["mechanism"] == "energy_only_archive_split"


def test_geometry_split_identifies_descriptor_collision() -> None:
    protocol = _protocol()

    row = protocol.decompose_pair(
        energy_delta_eV=0.4,
        energy_tol_eV=0.001,
        indexed_mic_rmsd_A=0.25,
        rmsd_tol_A=0.15,
        descriptor_delta=0.03,
        descriptor_tol=0.1,
    )

    assert row["indexed_geometry_same"] is False
    assert row["mechanism"] == "descriptor_collision_geometry_split"


def test_gate_allows_relabel_only_for_uniform_geometry_supported_escape() -> None:
    protocol = _protocol()
    geometry_split = [
        {
            "certificate": True,
            "geometry_valid": True,
            "fragmented": False,
            "energy_same": False,
            "indexed_geometry_same": False,
            "descriptor_same": True,
            "mechanism": "descriptor_collision_geometry_split",
        }
        for _ in range(5)
    ]

    decision = protocol.evaluate_gate(geometry_split)

    assert decision["classification"] == "descriptor_collision"
    assert decision["relabel_as_escaped_allowed"] is True
    assert decision["strict_requench_gate_required"] is False


def test_gate_routes_uniform_energy_only_split_to_numerical_requench() -> None:
    protocol = _protocol()
    energy_only = [
        {
            "certificate": True,
            "geometry_valid": True,
            "fragmented": False,
            "energy_same": False,
            "indexed_geometry_same": True,
            "descriptor_same": True,
            "mechanism": "energy_only_archive_split",
        }
        for _ in range(5)
    ]

    decision = protocol.evaluate_gate(energy_only)

    assert decision["classification"] == "energy_only_archive_split"
    assert decision["relabel_as_escaped_allowed"] is False
    assert decision["strict_requench_gate_required"] is True


def test_mixed_disagreement_keeps_ambiguity_and_opens_no_component() -> None:
    protocol = _protocol()
    rows = [
        {
            "certificate": True,
            "geometry_valid": True,
            "fragmented": False,
            "energy_same": False,
            "indexed_geometry_same": geometry_same,
            "descriptor_same": True,
            "mechanism": (
                "energy_only_archive_split"
                if geometry_same
                else "descriptor_collision_geometry_split"
            ),
        }
        for geometry_same in (True, False)
    ]

    decision = protocol.evaluate_gate(rows)

    assert decision["classification"] == "mixed_unresolved"
    assert decision["relabel_as_escaped_allowed"] is False
    assert decision["strict_requench_gate_required"] is False


def test_local_region_uses_same_existing_rmsd_threshold() -> None:
    protocol = _protocol()

    split = protocol.decompose_local_region(
        movable_indexed_mic_rmsd_A=0.41,
        rmsd_tol_A=0.4,
    )
    same = protocol.decompose_local_region(
        movable_indexed_mic_rmsd_A=0.40,
        rmsd_tol_A=0.4,
    )

    assert split["movable_geometry_same"] is False
    assert split["local_region_mechanism"] == "local_event_global_dilution"
    assert same["movable_geometry_same"] is True
    assert same["local_region_mechanism"] == "energy_only_local_same"


def test_local_region_resolves_mixed_gate_without_changing_matcher() -> None:
    protocol = _protocol()
    rows = [
        {
            "certificate": True,
            "geometry_valid": True,
            "fragmented": False,
            "energy_same": False,
            "indexed_geometry_same": False,
            "descriptor_same": True,
            "mechanism": "descriptor_collision_geometry_split",
        },
        {
            "certificate": True,
            "geometry_valid": True,
            "fragmented": False,
            "energy_same": False,
            "indexed_geometry_same": True,
            "descriptor_same": True,
            "mechanism": "energy_only_archive_split",
            "movable_geometry_same": False,
            "local_region_mechanism": "local_event_global_dilution",
        },
    ]

    decision = protocol.evaluate_local_region_gate(
        rows,
        initial_gate=protocol.evaluate_gate(rows),
    )

    assert (
        decision["classification"]
        == "descriptor_collision_with_local_dilution"
    )
    assert decision["relabel_as_escaped_allowed"] is True
    assert decision["strict_requench_gate_required"] is False
    assert decision["production_matcher_change_allowed"] is False


def test_local_region_routes_residual_same_pair_to_strict_requench() -> None:
    protocol = _protocol()
    rows = [
        {
            "certificate": True,
            "geometry_valid": True,
            "fragmented": False,
            "energy_same": False,
            "indexed_geometry_same": False,
            "descriptor_same": True,
            "mechanism": "descriptor_collision_geometry_split",
        },
        {
            "certificate": True,
            "geometry_valid": True,
            "fragmented": False,
            "energy_same": False,
            "indexed_geometry_same": True,
            "descriptor_same": True,
            "mechanism": "energy_only_archive_split",
            "movable_geometry_same": True,
            "local_region_mechanism": "energy_only_local_same",
        },
    ]

    decision = protocol.evaluate_local_region_gate(
        rows,
        initial_gate=protocol.evaluate_gate(rows),
    )

    assert decision["classification"] == "residual_energy_only_local_same"
    assert decision["relabel_as_escaped_allowed"] is False
    assert decision["strict_requench_gate_required"] is True
    assert decision["strict_requench_pair_count"] == 1


def test_strict_requench_resolves_same_matcher_endpoint_as_return() -> None:
    protocol = _protocol()

    decision = protocol.evaluate_strict_requench(
        starter_converged=True,
        landing_converged=True,
        strict_energy_delta_eV=0.0005,
        energy_tol_eV=0.001,
        strict_indexed_mic_rmsd_A=0.3,
        rmsd_tol_A=0.4,
    )

    assert decision["classification"] == (
        "return_starter_after_strict_requench"
    )
    assert decision["offline_label"] == "RETURN_STARTER"


def test_strict_requench_certifies_persistent_matcher_split_as_escape() -> None:
    protocol = _protocol()

    decision = protocol.evaluate_strict_requench(
        starter_converged=True,
        landing_converged=True,
        strict_energy_delta_eV=0.2,
        energy_tol_eV=0.001,
        strict_indexed_mic_rmsd_A=0.3,
        rmsd_tol_A=0.4,
    )

    assert decision["classification"] == (
        "escaped_certified_after_strict_requench"
    )
    assert decision["offline_label"] == "ESCAPED_CERTIFIED"
    assert decision["production_matcher_change_allowed"] is False


def test_strict_requench_requires_both_endpoint_certificates() -> None:
    protocol = _protocol()

    decision = protocol.evaluate_strict_requench(
        starter_converged=True,
        landing_converged=False,
        strict_energy_delta_eV=0.2,
        energy_tol_eV=0.001,
        strict_indexed_mic_rmsd_A=0.3,
        rmsd_tol_A=0.4,
    )

    assert decision["classification"] == "strict_requench_unresolved"
    assert decision["offline_label"] == "AMBIGUOUS_MATCH"


def test_certificate_rescue_uses_raw_force_not_optimizer_success() -> None:
    runner = _rescue_runner()
    scipy_success_without_force_certificate = {
        "termination_reason": "unconverged",
        "final": {"max_active_force_eV_per_A": 0.013},
        "telemetry": {"optimizer_success": True},
    }

    assert (
        runner.has_strict_certificate(
            scipy_success_without_force_certificate,
            fmax=0.01,
        )
        is False
    )


def test_certificate_rescue_accepts_only_converged_raw_force() -> None:
    runner = _rescue_runner()
    certified = {
        "termination_reason": "converged",
        "final": {"max_active_force_eV_per_A": 0.009},
        "telemetry": {"optimizer_success": True},
    }

    assert runner.has_strict_certificate(certified, fmax=0.01) is True
