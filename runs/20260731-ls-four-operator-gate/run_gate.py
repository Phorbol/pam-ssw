#!/usr/bin/env python3
"""Execute the preregistered C60 LS four-operator mechanism gate."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from pamssw.accounting import EvaluationPurpose
from pamssw.archive import MinimaArchive
from pamssw.io import read_state, write_state
from pamssw.walker import DirectionCandidate, ProposalPotential


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
PROTOCOL_PATH = RUN_ROOT / "protocol.py"
PAPER_GATE_PATH = REPO_ROOT / "runs" / "20260731-paper-ordered-ls-gate" / "run_gate.py"
SCOPE_GATE_PATH = REPO_ROOT / "runs" / "20260730-ls-softening-scope-gate" / "run_gate.py"
CONFIG_RUNNER_PATH = REPO_ROOT / "runs" / "20260730-starter-cell-online-gate" / "run_gate.py"
STATES = ("bootstrap", "mid", "late")
SEEDS = (42, 43, 44)
MAX_NEW_FORCE_EVALUATIONS = 500


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PROTOCOL = _load_module(PROTOCOL_PATH, "_ls_four_operator_protocol_runtime")


def execution_contract(candidate_count: int) -> dict[str, object]:
    if isinstance(candidate_count, bool) or candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    return {
        "true_pes_hvp_geometries": 2,
        "central_fd_force_evaluations": 4 * int(candidate_count),
        "operator_a_b_share_true_hvp": True,
        "operator_c_d_share_true_hvp": True,
        "analytic_ls_hvp_force_evaluations": 0,
        "proposal_side_ls": False,
    }


def select_candidate_indices(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    if not rows:
        raise ValueError("candidate rows cannot be empty")
    selected: dict[str, int] = {}
    for operator in ("a", "b", "c", "d"):
        key = f"score_{operator}"
        best = max(rows, key=lambda row: float(row[key]))
        selected[operator] = int(best["candidate_index"])
    return selected


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


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    return sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _paper_inputs():
    paper = _load_module(PAPER_GATE_PATH, "_ls_four_operator_paper_gate")
    return paper, {
        state_id: Path(paper.STATE_FILES[state_id]) for state_id in STATES
    }


def preflight(expected_git_commit: str) -> dict[str, object]:
    actual_commit = _current_commit()
    if actual_commit != expected_git_commit:
        raise RuntimeError(
            f"execution commit mismatch: expected {expected_git_commit}, got {actual_commit}"
        )
    if not _tracked_clean():
        raise RuntimeError("tracked worktree must be clean before GPU execution")
    paper, state_files = _paper_inputs()
    for path in state_files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    scope = _load_module(SCOPE_GATE_PATH, "_ls_four_operator_scope_gate")
    production = _load_module(
        scope.PRODUCTION_RUNNER,
        "_ls_four_operator_production_runner",
    )
    model_path = Path(production.MODEL_PATH)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return {
        "schema_version": 1,
        "execution_commit": actual_commit,
        "gpu": torch.cuda.get_device_name(0),
        "model_path": str(model_path),
        "model_sha256": _sha256_file(model_path),
        "paper_initial_strength_eV": float(paper.PAPER_INITIAL_STRENGTH_EV),
        "paper_xi_dimensionless": 0.2,
        "states": {
            key: {"path": str(path), "sha256": _sha256_file(path)}
            for key, path in state_files.items()
        },
        "seeds": list(SEEDS),
        "force_evaluation_ceiling": MAX_NEW_FORCE_EVALUATIONS,
    }


def _calculator():
    scope = _load_module(SCOPE_GATE_PATH, "_ls_four_operator_calculator_source")
    production = _load_module(
        scope.PRODUCTION_RUNNER,
        "_ls_four_operator_calculator_production",
    )
    from mace.calculators import MACECalculator
    from pamssw.mace_batch import MACEBatchCalculator

    raw = MACECalculator(
        model_paths=str(production.MODEL_PATH),
        **production.CALCULATOR_CONFIG,
    )
    return MACEBatchCalculator(raw)


def _paper_config(config_builder, paper, case_directory: Path, seed: int):
    base = replace(
        config_builder.build_production_config(
            "c60",
            case_directory,
            master_seed=seed,
        ),
        max_trials=1,
        max_force_evals=None,
    )
    return replace(
        paper._arm_config(base, "paper_ordered"),
        local_softening_scope="oracle",
    )


def _candidate_pool_hash(candidates: Sequence[DirectionCandidate]) -> str:
    digest = sha256()
    for candidate in candidates:
        digest.update(candidate.kind.value.encode("ascii"))
        digest.update(np.asarray(candidate.direction, dtype="<f8").tobytes())
    return digest.hexdigest()


def _normalized_projected_candidate(walker, state, candidate, direction):
    return walker.oracle.generator._candidate(state, candidate.kind, direction)


def _score_candidates(
    *,
    walker,
    state,
    candidates: Sequence[DirectionCandidate],
    curvatures: Sequence[float],
    anchor_direction: np.ndarray,
    archive,
    step_target: float,
) -> tuple[list[float], list[float]]:
    score_sigma_fn = walker._direction_score_sigma_fn(
        1.0,
        step_target=step_target,
    )
    scores: list[float] = []
    sigmas: list[float] = []
    for candidate, curvature in zip(candidates, curvatures):
        sigma = walker.oracle._candidate_score_sigma(
            curvature=float(curvature),
            score_sigma=None,
            score_sigma_fn=score_sigma_fn,
            step_scale_fn=None,
        )
        score = walker.oracle.scorer.score_candidate(
            state=state,
            candidate=candidate,
            curvature=float(curvature),
            sigma=sigma,
            previous_direction=None,
            anchor_direction=anchor_direction,
            archive=archive,
        )
        scores.append(float(score))
        sigmas.append(float(sigma))
    return scores, sigmas


def _selected_cosines(
    selected: Mapping[str, int],
    candidates_x0: Sequence[DirectionCandidate],
    candidates_xr: Sequence[DirectionCandidate],
    alignment,
) -> dict[str, float]:
    directions = {
        "a": np.asarray(candidates_x0[selected["a"]].direction, dtype=float),
        "b": np.asarray(candidates_x0[selected["b"]].direction, dtype=float),
        "c": alignment.direction_to_reference(
            candidates_xr[selected["c"]].direction
        ),
        "d": alignment.direction_to_reference(
            candidates_xr[selected["d"]].direction
        ),
    }
    for key, value in directions.items():
        directions[key] = value / (np.linalg.norm(value) + 1.0e-30)
    return {
        "abs_cosine_b_a": abs(float(np.dot(directions["b"], directions["a"]))),
        "abs_cosine_c_a": abs(float(np.dot(directions["c"], directions["a"]))),
        "abs_cosine_d_a": abs(float(np.dot(directions["d"], directions["a"]))),
        "abs_cosine_d_c": abs(float(np.dot(directions["d"], directions["c"]))),
    }


def _run_case(
    *,
    state_id: str,
    state,
    seed: int,
    backend,
    paper,
    source,
    config_builder,
    case_directory: Path,
) -> dict[str, object]:
    config = _paper_config(config_builder, paper, case_directory, seed)
    if config.local_softening_scope != "oracle":
        raise RuntimeError("G-LS0A forbids proposal-side local softening")
    if config.direction_selection_mode != "discrete" or config.direction_synthesis_mode != "none":
        raise RuntimeError("G-LS0A requires the native discrete candidate pool")
    walker = source.ObservingWalker(
        calculator=backend,
        config=config,
        softening_enabled=True,
    )
    prestrained, softening = walker._prepare_frozen_local_softening(state)
    if softening is None:
        raise RuntimeError("paper-ordered prestrain did not construct a softening model")
    diagnostics = walker.local_softening_diagnostics()
    if diagnostics["pre_relax_converged"] != 1:
        raise RuntimeError("paper-ordered prestrain lacks a force certificate")

    archive = MinimaArchive(
        energy_tol=config.dedup_energy_tol,
        rmsd_tol=config.dedup_rmsd_tol,
        max_prototypes=config.max_prototypes,
    )
    with walker.calculator.purpose(EvaluationPurpose.ESCAPE_TRUE_PES_CHECK):
        starter_energy = float(walker.calculator.evaluate(state).energy)
    archive.add(state, starter_energy, parent_id=None)
    step_target = float(walker.step_target_controller.target(archive))

    anchor_x0, _ = walker._initialize_walk_direction_context(state, trial_index=0)
    candidates_x0 = walker.oracle.generator.generate(
        state,
        previous_direction=None,
        anchor_direction=anchor_x0,
        anchor_mixing_alpha=walker.oracle.anchor_mixing_alpha,
        n_bond_pairs=config.n_bond_pairs,
    )
    if len(candidates_x0) != config.oracle_candidates:
        raise RuntimeError("native candidate pool size drifted")
    alignment = PROTOCOL.align_prestrained_to_reference(state, prestrained)
    anchor_xr = alignment.direction_to_prestrained(anchor_x0)
    anchor_xr /= np.linalg.norm(anchor_xr) + 1.0e-30
    candidates_xr = [
        _normalized_projected_candidate(
            walker,
            prestrained,
            candidate,
            alignment.direction_to_prestrained(candidate.direction),
        )
        for candidate in candidates_x0
    ]

    true_proposal = ProposalPotential(walker.calculator)
    with walker.calculator.purpose(EvaluationPurpose.DIRECTION_ORACLE):
        hvps_x0 = walker.oracle._candidate_directional_hvps_many(
            state,
            true_proposal,
            tuple(candidate.direction for candidate in candidates_x0),
        )
        hvps_xr = walker.oracle._candidate_directional_hvps_many(
            prestrained,
            true_proposal,
            tuple(candidate.direction for candidate in candidates_xr),
        )
    true_hvps_x0 = []
    true_hvps_xr = []
    for total, true in hvps_x0:
        if true is None or not np.allclose(total, true, rtol=0.0, atol=0.0):
            raise RuntimeError("A/B true-PES HVP stencil is not shared")
        true_hvps_x0.append(np.asarray(true, dtype=float))
    for total, true in hvps_xr:
        if true is None or not np.allclose(total, true, rtol=0.0, atol=0.0):
            raise RuntimeError("C/D true-PES HVP stencil is not shared")
        true_hvps_xr.append(np.asarray(true, dtype=float))

    candidate_rows: list[dict[str, object]] = []
    for index, (candidate_x0, candidate_xr, true_x0, true_xr) in enumerate(
        zip(candidates_x0, candidates_xr, true_hvps_x0, true_hvps_xr)
    ):
        ls_x0 = PROTOCOL.pair_operator_action(
            softening,
            state,
            candidate_x0.direction,
        )
        ls_xr = PROTOCOL.pair_operator_action(
            softening,
            prestrained,
            candidate_xr.direction,
        )
        row = PROTOCOL.build_operator_row(
            direction_x0=candidate_x0.direction,
            direction_xr=candidate_xr.direction,
            true_hvp_x0=true_x0,
            true_hvp_xr=true_xr,
            ls_action_x0=ls_x0,
            ls_action_xr=ls_xr,
        )
        row.update(
            {
                "candidate_index": index,
                "candidate_kind": candidate_x0.kind.value,
                "direction_x0_sha256": _sha256_array(candidate_x0.direction),
                "direction_xr_sha256": _sha256_array(candidate_xr.direction),
            }
        )
        candidate_rows.append(row)

    for operator, geometry, candidates, anchor, curvature_key in (
        ("a", state, candidates_x0, anchor_x0, "kappa_a"),
        ("b", state, candidates_x0, anchor_x0, "kappa_b"),
        ("c", prestrained, candidates_xr, anchor_xr, "kappa_c"),
        ("d", prestrained, candidates_xr, anchor_xr, "kappa_d"),
    ):
        scores, sigmas = _score_candidates(
            walker=walker,
            state=geometry,
            candidates=candidates,
            curvatures=[float(row[curvature_key]) for row in candidate_rows],
            anchor_direction=anchor,
            archive=archive,
            step_target=step_target,
        )
        for row, score, sigma in zip(candidate_rows, scores, sigmas):
            row[f"score_{operator}"] = score
            row[f"score_sigma_{operator}"] = sigma

    selected = select_candidate_indices(candidate_rows)
    counts = walker.calculator.snapshot()
    purpose_counts = counts.as_dict()
    if counts.total != sum(purpose_counts.values()) or purpose_counts["unattributed"] != 0:
        raise RuntimeError("case force accounting does not close")
    contract = execution_contract(len(candidates_x0))
    if purpose_counts["direction_oracle"] != contract["central_fd_force_evaluations"]:
        raise RuntimeError("true-PES HVP force count violates the two-stencil contract")
    aligned_delta = alignment.positions_in_reference_frame - state.positions
    case = {
        "schema_version": 1,
        "state_id": state_id,
        "seed": seed,
        "candidate_pool_sha256": _candidate_pool_hash(candidates_x0),
        "candidate_rows": candidate_rows,
        "selected_candidate": selected,
        "selected_direction_cosines": _selected_cosines(
            selected,
            candidates_x0,
            candidates_xr,
            alignment,
        ),
        "pre_relax": {
            "rms_displacement_A": float(np.sqrt(np.mean(np.square(aligned_delta)))),
            "p_ls_eV_per_atom": float(diagnostics["pre_relax_pls_eV_per_atom"]),
            "force_evaluations": int(diagnostics["pre_relax_force_evaluations"]),
            "iterations": int(diagnostics["pre_relax_iterations"]),
            "gradient_norm_eV_per_A": float(diagnostics["pre_relax_gradient_norm"]),
            "pair_terms": int(diagnostics["terms_last"]),
        },
        "hvp_stencil_sha256": {
            "x0_true": _sha256_array(np.vstack(true_hvps_x0)),
            "xr_true": _sha256_array(np.vstack(true_hvps_xr)),
        },
        "execution_contract": contract,
        "force_evaluations": counts.total,
        "purpose_counts": purpose_counts,
        "effective_config": asdict(config),
    }
    case_directory.mkdir(parents=True, exist_ok=True)
    write_state(case_directory / "prestrained.xyz", prestrained)
    _write_json(case_directory / "summary.json", case)
    return case


def _load_valid_case(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("force_evaluations", -1)) != sum(
        int(value) for value in payload.get("purpose_counts", {}).values()
    ):
        return None
    if int(payload["purpose_counts"].get("unattributed", -1)) != 0:
        return None
    if not payload.get("candidate_rows"):
        return None
    return payload


def analyze(output: Path) -> dict[str, object]:
    cases = []
    for state_id in STATES:
        for seed in SEEDS:
            path = output / state_id / f"seed-{seed}" / "summary.json"
            case = _load_valid_case(path)
            if case is None:
                raise ValueError(f"missing or invalid case summary: {path}")
            cases.append(case)
    evidence = PROTOCOL.build_evidence(cases)
    evidence["execution_contract"] = execution_contract(
        int(evidence["cohort"]["candidates_per_block"])
    )
    evidence["cases"] = cases
    if int(evidence["force_accounting"]["total"]) > MAX_NEW_FORCE_EVALUATIONS:
        raise RuntimeError("G-LS0A exceeded its preregistered FE ceiling")
    _write_json(output / "evidence.json", evidence)
    return evidence


def run_gate(output: Path, expected_git_commit: str) -> dict[str, object]:
    manifest = preflight(expected_git_commit)
    if output.exists():
        manifest_path = output / "manifest.json"
        if not manifest_path.is_file():
            raise FileExistsError(output)
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError("existing output manifest does not match this execution")
    else:
        output.mkdir(parents=True)
        _write_json(output / "manifest.json", manifest)

    paper, state_files = _paper_inputs()
    source = _load_module(SCOPE_GATE_PATH, "_ls_four_operator_execution_source")
    config_builder = _load_module(
        CONFIG_RUNNER_PATH,
        "_ls_four_operator_config_builder",
    )
    backend = _calculator()
    for state_id in STATES:
        state = read_state(state_files[state_id])
        for seed in SEEDS:
            case_directory = output / state_id / f"seed-{seed}"
            existing = _load_valid_case(case_directory / "summary.json")
            if existing is not None:
                continue
            _run_case(
                state_id=state_id,
                state=state,
                seed=seed,
                backend=backend,
                paper=paper,
                source=source,
                config_builder=config_builder,
                case_directory=case_directory,
            )
    return analyze(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--expected-git-commit", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--expected-git-commit", required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        payload = preflight(args.expected_git_commit)
    elif args.command == "run":
        payload = run_gate(args.output, args.expected_git_commit)
    else:
        payload = analyze(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
