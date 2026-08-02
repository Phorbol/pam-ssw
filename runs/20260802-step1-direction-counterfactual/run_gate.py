#!/usr/bin/env python3
"""Run the shared micro-step-1 direction counterfactual gate."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from pamssw.accounting import EvaluationPurpose
from pamssw.walker import DirectionChoice, SurfaceWalker


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
PROTOCOL_PATH = RUN_ROOT / "protocol.py"
OBSERVABILITY_RUNNER_PATH = (
    REPO_ROOT / "runs" / "20260801-uphill-action-observability" / "run_gate.py"
)
STEP_ZERO_RUNNER_PATH = (
    REPO_ROOT
    / "runs"
    / "20260731-direction-candidate-counterfactual-gate"
    / "run_experiment.py"
)
SYSTEMS = ("c60", "pdo", "cuo")
SEEDS = (52, 53, 54)
MAX_TOTAL_FORCE_EVALUATIONS = 60_000
MAX_WALL_SECONDS = 35.0 * 60.0


PREFIX_POSITION_ATOL_A = 1.0e-10


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


protocol = _load_module(PROTOCOL_PATH, "_step1_direction_protocol_runner")
observability = _load_module(
    OBSERVABILITY_RUNNER_PATH,
    "_step1_direction_observability_runner",
)
step_zero = _load_module(
    STEP_ZERO_RUNNER_PATH,
    "_step1_direction_step_zero_runner",
)


def _array_bytes(values: np.ndarray) -> bytes:
    array = np.asarray(values, dtype="<f8")
    shape = np.asarray(array.shape, dtype="<i8")
    return shape.tobytes() + array.tobytes()


def _array_sha256(values: np.ndarray) -> str:
    return sha256(_array_bytes(values)).hexdigest()


def direction_sha256(direction: np.ndarray) -> str:
    values = np.asarray(direction, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("direction must have a finite nonzero norm")
    return _array_sha256(values / norm)


def _step_zero_direction_sha256(direction: np.ndarray) -> str:
    """Match the frozen 20260731 step-0 gate's hash representation."""

    values = np.asarray(direction, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("direction must have a finite nonzero norm")
    return sha256(np.asarray(values / norm, dtype="<f8").tobytes()).hexdigest()


def bias_sha256(biases: Sequence[Any]) -> str:
    digest = sha256()
    digest.update(np.asarray([len(biases)], dtype="<i8").tobytes())
    for bias in biases:
        digest.update(_array_bytes(np.asarray(bias.center, dtype=float)))
        digest.update(_array_bytes(np.asarray(bias.direction, dtype=float)))
        digest.update(np.asarray([bias.sigma, bias.weight], dtype="<f8").tobytes())
    return digest.hexdigest()


def prefix_certificate(state, proposal, previous_direction: np.ndarray) -> dict[str, object]:
    positions = np.asarray(state.positions, dtype=float).copy()
    cell = (
        np.zeros((0, 0), dtype=float)
        if state.cell is None
        else np.asarray(state.cell, dtype=float).copy()
    )
    return {
        "positions": positions.tolist(),
        "positions_sha256": _array_sha256(positions),
        "cell_sha256": _array_sha256(cell),
        "numbers_sha256": sha256(
            np.asarray(state.numbers, dtype="<i8").tobytes()
        ).hexdigest(),
        "bias_count": len(proposal.biases),
        "biases_sha256": bias_sha256(proposal.biases),
        "previous_direction_sha256": direction_sha256(previous_direction),
    }


def validate_replayed_pool(
    reference: Mapping[str, Any],
    replayed: Mapping[str, Any],
) -> None:
    reference_candidates = list(reference["candidates"])
    replayed_candidates = list(replayed["candidates"])
    kinds = [str(candidate["kind"]) for candidate in reference_candidates]
    if (
        len(reference_candidates) != 4
        or kinds.count("momentum") != 1
        or not any(kind != "momentum" for kind in kinds)
    ):
        raise RuntimeError("shared pool must contain exactly one momentum candidate")
    reference_identity = [
        (
            int(candidate["candidate_index"]),
            str(candidate["kind"]),
            int(candidate["static_rank"]),
            str(candidate["direction_sha256"]),
        )
        for candidate in reference_candidates
    ]
    replayed_identity = [
        (
            int(candidate["candidate_index"]),
            str(candidate["kind"]),
            int(candidate["static_rank"]),
            str(candidate["direction_sha256"]),
        )
        for candidate in replayed_candidates
    ]
    if reference_identity != replayed_identity:
        raise RuntimeError("replayed candidate identity differs from the shared pool")

    reference_prefix = reference["prefix"]
    replayed_prefix = replayed["prefix"]
    left_positions = np.asarray(reference_prefix["positions"], dtype=float)
    right_positions = np.asarray(replayed_prefix["positions"], dtype=float)
    if left_positions.shape != right_positions.shape:
        raise RuntimeError("replayed prefix positions have a different shape")
    maximum_error = float(np.max(np.abs(left_positions - right_positions)))
    if maximum_error > PREFIX_POSITION_ATOL_A:
        raise RuntimeError(
            f"replayed prefix positions differ by {maximum_error:.3e} A"
        )
    for key in (
        "cell_sha256",
        "numbers_sha256",
        "bias_count",
        "biases_sha256",
        "previous_direction_sha256",
    ):
        if reference_prefix[key] != replayed_prefix[key]:
            raise RuntimeError(f"replayed prefix differs in {key}")


def evaluate_native_pool(
    *,
    walker,
    state,
    proposal,
    previous_direction: np.ndarray,
    anchor_direction: np.ndarray | None,
    archive,
    history_gradient: np.ndarray | None,
    continuity_weight: float | None,
    n_bond_pairs: int | None,
    score_sigma: float | None,
    score_sigma_fn,
    step_scale_fn,
) -> dict[str, object]:
    """Evaluate the native discrete K=4 pool once at a shared prefix.

    This deliberately mirrors the non-synthetic branch of
    ``SoftModeOracle.choose_direction``.  The gate needs every native choice,
    rather than only the production winner, but must not change either the
    candidate generator or the static score being audited.
    """

    oracle = walker.oracle
    if oracle.direction_selection_mode != "discrete":
        raise ValueError("the step-one gate requires discrete direction selection")
    if oracle.direction_ranking_mode != "static_score":
        raise ValueError("the step-one gate requires static-score ranking")

    candidates = oracle.generator.generate(
        state,
        previous_direction,
        anchor_direction=anchor_direction,
        anchor_mixing_alpha=oracle.anchor_mixing_alpha,
        n_bond_pairs=n_bond_pairs,
    )
    if len(candidates) != 4:
        raise RuntimeError(f"the shared step-one pool must contain 4 candidates, got {len(candidates)}")

    scoring_anchor = None if oracle.anchor_mixing_alpha is not None else anchor_direction
    before = walker.calculator.snapshot().count(EvaluationPurpose.DIRECTION_ORACLE)
    with walker.calculator.purpose(EvaluationPurpose.DIRECTION_ORACLE):
        hvp_pairs = oracle._candidate_directional_hvps_many(
            state,
            proposal,
            tuple(candidate.direction for candidate in candidates),
        )
    after = walker.calculator.snapshot().count(EvaluationPurpose.DIRECTION_ORACLE)
    force_evaluations = after - before
    if force_evaluations != 2 * len(candidates):
        raise RuntimeError(
            "native pool must use one central-difference HVP per candidate; "
            f"observed {force_evaluations} evaluations for {len(candidates)} candidates"
        )

    evaluated: list[dict[str, object]] = []
    for candidate_index, (candidate, (hvp, true_hvp)) in enumerate(
        zip(candidates, hvp_pairs)
    ):
        curvature = float(np.dot(hvp, candidate.direction))
        true_curvature = (
            None
            if true_hvp is None
            else float(np.dot(true_hvp, candidate.direction))
        )
        candidate_sigma = oracle._candidate_score_sigma(
            curvature=curvature,
            score_sigma=score_sigma,
            score_sigma_fn=score_sigma_fn,
            step_scale_fn=step_scale_fn,
        )
        history_push = (
            0.0
            if history_gradient is None
            else -float(np.dot(history_gradient, candidate.direction))
        )
        score = oracle.scorer.score_candidate(
            state=state,
            candidate=candidate,
            curvature=curvature,
            sigma=candidate_sigma,
            previous_direction=previous_direction,
            anchor_direction=scoring_anchor,
            archive=archive,
            history_push=history_push,
            continuity_weight=continuity_weight,
        )
        evaluated.append(
            {
                "candidate_index": candidate_index,
                "kind": candidate.kind.value,
                "direction": np.asarray(candidate.direction, dtype=float).copy(),
                "direction_sha256": direction_sha256(candidate.direction),
                "curvature": curvature,
                "true_curvature": true_curvature,
                "score_sigma": float(candidate_sigma),
                "score": float(score),
                "damage_risk": float(candidate.damage_risk),
                "rigid_body_overlap": float(candidate.rigid_body_overlap),
                "post_projection_rigid_body_overlap": float(
                    candidate.post_projection_rigid_body_overlap
                ),
            }
        )

    ranked_indices = sorted(
        range(len(evaluated)),
        key=lambda index: (-float(evaluated[index]["score"]), index),
    )
    for static_rank, candidate_index in enumerate(ranked_indices, start=1):
        evaluated[candidate_index]["static_rank"] = static_rank

    choices: list[DirectionChoice] = []
    for record in evaluated:
        choices.append(
            DirectionChoice(
                direction=np.asarray(record["direction"], dtype=float).copy(),
                curvature=float(record["curvature"]),
                kind=candidates[int(record["candidate_index"])].kind,
                candidate_count=len(candidates),
                mean_rigid_body_overlap=float(record["rigid_body_overlap"]),
                mean_post_projection_rigid_body_overlap=float(
                    record["post_projection_rigid_body_overlap"]
                ),
                score=float(record["score"]),
                true_curvature=(
                    None
                    if record["true_curvature"] is None
                    else float(record["true_curvature"])
                ),
                diagnostics={
                    "counterfactual_shared_step1_pool": True,
                    "candidate_index": int(record["candidate_index"]),
                    "static_rank": int(record["static_rank"]),
                    "score_sigma": float(record["score_sigma"]),
                    "direction_hvp_count": 1,
                },
            )
        )

    serializable_records = [
        {key: value for key, value in record.items() if key != "direction"}
        for record in evaluated
    ]
    return {
        "choices": choices,
        "candidates": serializable_records,
        "prefix": prefix_certificate(state, proposal, previous_direction),
        "post_pool_rng_state": deepcopy(walker.rng.bit_generator.state),
        "direction_oracle_force_evaluations": force_evaluations,
    }


class StepOneController:
    """Intercept exactly the first ordinary oracle call of an H8 walk.

    Micro-step 0 is supplied through ``initial_direction_choice`` by the gate,
    so the first call observed here is micro-step 1.  A reference controller
    pays for and captures the common K4 pool.  A forced controller regenerates
    only the zero-cost candidate vectors, verifies their identity, and returns
    one already-paid choice.
    """

    def __init__(
        self,
        *,
        walker,
        original_choose,
        reference_pool: Mapping[str, Any] | None = None,
        forced_index: int | None = None,
    ) -> None:
        if reference_pool is not None and forced_index is None:
            raise ValueError("forced_index is required when replaying a reference pool")
        self.walker = walker
        self.original_choose = original_choose
        self.reference_pool = reference_pool
        self.forced_index = forced_index
        self.normal_call_count = 0

    @staticmethod
    def _require_plain_native_pool(kwargs: Mapping[str, Any]) -> None:
        if kwargs.get("direction_type_bonus_fn") is not None:
            raise RuntimeError("step-one gate requires direction-type bonus disabled")
        if bool(kwargs.get("plateau_evolution_active", False)):
            raise RuntimeError("step-one gate requires plateau evolution disabled")
        if int(kwargs.get("archive_momentum_limit", 0)) != 0:
            raise RuntimeError("step-one gate requires archive momentum disabled")
        if kwargs.get("krylov_intents") is not None:
            raise RuntimeError("step-one gate requires the native discrete pool")

    def __call__(self, state, proposal, previous_direction, **kwargs):
        if self.normal_call_count > 0:
            self.normal_call_count += 1
            return self.original_choose(
                state,
                proposal,
                previous_direction,
                **kwargs,
            )
        self.normal_call_count += 1
        self._require_plain_native_pool(kwargs)

        if self.reference_pool is None:
            pool = evaluate_native_pool(
                walker=self.walker,
                state=state,
                proposal=proposal,
                previous_direction=previous_direction,
                anchor_direction=kwargs.get("anchor_direction"),
                archive=kwargs.get("archive"),
                history_gradient=kwargs.get("history_gradient"),
                continuity_weight=kwargs.get("continuity_weight"),
                n_bond_pairs=kwargs.get("n_bond_pairs"),
                score_sigma=kwargs.get("score_sigma"),
                score_sigma_fn=kwargs.get("score_sigma_fn"),
                step_scale_fn=kwargs.get("step_scale_fn"),
            )
            self.reference_pool = pool
            selected_index = next(
                int(record["candidate_index"])
                for record in pool["candidates"]
                if int(record["static_rank"]) == 1
            )
        else:
            candidates = self.walker.oracle.generator.generate(
                state,
                previous_direction,
                anchor_direction=kwargs.get("anchor_direction"),
                anchor_mixing_alpha=self.walker.oracle.anchor_mixing_alpha,
                n_bond_pairs=kwargs.get("n_bond_pairs"),
            )
            reference_records = list(self.reference_pool["candidates"])
            if len(candidates) != len(reference_records):
                raise RuntimeError("replayed pool has a different candidate count")
            replayed_records = [
                {
                    "candidate_index": index,
                    "kind": candidate.kind.value,
                    "static_rank": int(reference_records[index]["static_rank"]),
                    "direction_sha256": direction_sha256(candidate.direction),
                }
                for index, candidate in enumerate(candidates)
            ]
            replayed_pool = {
                "candidates": replayed_records,
                "prefix": prefix_certificate(state, proposal, previous_direction),
            }
            validate_replayed_pool(self.reference_pool, replayed_pool)
            self.walker.rng.bit_generator.state = deepcopy(
                self.reference_pool["post_pool_rng_state"]
            )
            selected_index = int(self.forced_index)

        choices = list(self.reference_pool["choices"])
        if selected_index < 0 or selected_index >= len(choices):
            raise IndexError("forced step-one candidate index is outside the shared pool")
        choice = deepcopy(choices[selected_index])
        choice.direction = np.asarray(choice.direction, dtype=float).copy()
        choice.diagnostics = dict(choice.diagnostics)
        choice.diagnostics.update(
            {
                "shared_step1_direction": True,
                "candidate_index": selected_index,
                "oracle_selection_force_evaluations_shared": (
                    int(self.reference_pool["direction_oracle_force_evaluations"])
                    if self.forced_index is None
                    else 0
                ),
            }
        )
        return choice


class PostStepZeroRNGWalker(SurfaceWalker):
    """Replay the shared step-0 anchor, then enter its post-pool RNG state."""

    def __init__(
        self,
        *args,
        step_zero_anchor_sha256: str,
        post_step_zero_pool_rng_state: Mapping[str, Any],
        **kwargs,
    ) -> None:
        self._step_zero_anchor_sha256 = str(step_zero_anchor_sha256)
        self._post_step_zero_pool_rng_state = deepcopy(
            post_step_zero_pool_rng_state
        )
        super().__init__(*args, **kwargs)

    def _initialize_walk_direction_context(self, state, *, trial_index):
        anchor, intents = super()._initialize_walk_direction_context(
            state,
            trial_index=trial_index,
        )
        if _step_zero_direction_sha256(anchor) != self._step_zero_anchor_sha256:
            raise RuntimeError("replayed walk generated a different step-0 anchor")
        self.rng.bit_generator.state = deepcopy(
            self._post_step_zero_pool_rng_state
        )
        return anchor, intents


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=str)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def build_config(system: str, case_directory: Path, *, seed: int):
    config = observability.build_config(
        system,
        Path(case_directory),
        seed=seed,
        force_budget=MAX_TOTAL_FORCE_EVALUATIONS,
    )
    return replace(
        config,
        max_trials=1,
        max_force_evals=None,
        max_steps_per_walk=8,
        oracle_candidates=4,
        direction_selection_mode="discrete",
        direction_ranking_mode="static_score",
        direction_synthesis_mode="none",
        direction_probe_enabled=False,
        direction_type_ucb_enabled=False,
        direction_archive_enabled=False,
        direction_archive_path=None,
        plateau_evolution_enabled=False,
        accepted_structures_log=None,
        accepted_structures_dir=None,
        write_proposal_minima=False,
        proposal_minima_dir=None,
        write_relaxation_trajectories=False,
        relaxation_trajectory_dir=None,
        direction_diagnostics_enabled=False,
        direction_diagnostics_path=None,
        proposal_pool_size=1,
        proposal_duplicate_rescue_optimizer=None,
    )


def _new_walker(
    *,
    system: str,
    config,
    step_zero_pool: Mapping[str, Any],
    cuo_resources,
) -> PostStepZeroRNGWalker:
    walker = PostStepZeroRNGWalker(
        calculator=observability.base.source.source._calculator(
            system,
            cuo_resources,
        ),
        config=config,
        softening_enabled=True,
        step_zero_anchor_sha256=str(step_zero_pool["anchor_sha256"]),
        post_step_zero_pool_rng_state=step_zero_pool["post_pool_rng_state"],
    )
    walker.step_target_controller = observability.base.TargetModeController(
        walker.step_target_controller,
        mode="archive_scaled",
        reference_eV=observability.base.FIXED_REFERENCE_EV,
    )
    return walker


def _run_arm(
    *,
    system: str,
    seed: int,
    repeat: int,
    candidate_index: int | None,
    case_directory: Path,
    shared_bootstrap,
    config,
    step_zero_pool: Mapping[str, Any],
    reference_pool: Mapping[str, Any] | None,
    cuo_resources,
) -> tuple[dict[str, Any], Mapping[str, Any] | None]:
    from pamssw.archive import MinimaArchive
    from pamssw.relax import has_force_convergence_certificate

    case_directory.mkdir(parents=True, exist_ok=False)
    walker = _new_walker(
        system=system,
        config=config,
        step_zero_pool=step_zero_pool,
        cuo_resources=cuo_resources,
    )
    archive = MinimaArchive(
        energy_tol=config.dedup_energy_tol,
        rmsd_tol=config.dedup_rmsd_tol,
        max_prototypes=config.max_prototypes,
    )
    starter_state = shared_bootstrap.result.state
    starter_energy = float(shared_bootstrap.result.energy)
    starter_entry = archive.add(starter_state, starter_energy, parent_id=None)
    original_choose = walker.oracle.choose_direction
    controller = StepOneController(
        walker=walker,
        original_choose=original_choose,
        reference_pool=reference_pool,
        forced_index=candidate_index,
    )
    walker.oracle.choose_direction = controller
    traces = []
    started = perf_counter()
    escape_state = walker._walk_candidate_from_seed(
        starter_state,
        archive,
        walker.step_target_controller.target(archive),
        trial_index=0,
        proposal_index=0,
        seed_entry_id=starter_entry.entry_id,
        initial_direction_choice=step_zero_pool["choices"][
            next(
                index
                for index, record in enumerate(step_zero_pool["records"])
                if int(record["static_rank"]) == 1
            )
        ],
        trace_sink=traces,
    )
    if controller.reference_pool is None:
        return (
            {
                "system": system,
                "seed": seed,
                "repeat": repeat,
                "right_censored_before_step1": True,
                "wall_time_s": perf_counter() - started,
                "purpose_counts": walker.calculator.snapshot().as_dict(),
            },
            None,
        )
    landing = walker.relax_true_minimum(
        escape_state,
        trajectory_name="step1-counterfactual-landing",
    )
    wall_time_s = perf_counter() - started
    counts = walker.calculator.snapshot()
    if counts.count(EvaluationPurpose.UNATTRIBUTED) != 0:
        raise RuntimeError("counterfactual arm contains unattributed evaluations")
    if sum(counts.as_dict().values()) != counts.total:
        raise RuntimeError("counterfactual arm force ledger does not close")
    if len(traces) != 1:
        raise RuntimeError("counterfactual arm did not emit one uphill trace")

    actual_index = (
        int(candidate_index)
        if candidate_index is not None
        else next(
            int(record["candidate_index"])
            for record in controller.reference_pool["candidates"]
            if int(record["static_rank"]) == 1
        )
    )
    record = controller.reference_pool["candidates"][actual_index]
    before_entries = len(archive.entries)
    landing_entry = archive.add(
        landing.state,
        float(landing.energy),
        parent_id=starter_entry.entry_id,
    )
    fragmented = bool(
        walker._is_fragmented_cluster(starter_state, landing.state)
    )
    trace = traces[0]
    step_zero_true_energy = (
        None
        if not trace.steps
        else float(trace.steps[0].true_energy_after_eV)
    )
    row = {
        "system": system,
        "seed": seed,
        "repeat": repeat,
        "candidate_index": actual_index,
        "kind": str(record["kind"]),
        "static_rank": int(record["static_rank"]),
        "static_score": float(record["score"]),
        "direction_sha256": str(record["direction_sha256"]),
        "curvature": float(record["curvature"]),
        "true_curvature": (
            None
            if record["true_curvature"] is None
            else float(record["true_curvature"])
        ),
        "step1_prefix_positions_sha256": str(
            controller.reference_pool["prefix"]["positions_sha256"]
        ),
        "step1_prefix_biases_sha256": str(
            controller.reference_pool["prefix"]["biases_sha256"]
        ),
        "step1_prefix_true_energy_eV": step_zero_true_energy,
        "prefix_valid": True,
        "starter_energy_eV": starter_energy,
        "landing_energy_eV": float(landing.energy),
        "landing_delta_eV": float(landing.energy) - starter_energy,
        "best_energy_improvement_eV": max(
            0.0,
            starter_energy - float(landing.energy),
        ),
        "certificate": bool(
            has_force_convergence_certificate(landing, config.quench_fmax)
        ),
        "landing_force_norm_eV_per_A": float(landing.gradient_norm),
        "landing_iterations": int(landing.n_iter),
        "landing_geometry_valid": bool(
            walker.geometry_validator.is_valid_state(landing.state)
        ),
        "fragmented": fragmented,
        "same_starter_basin": bool(landing_entry.entry_id == starter_entry.entry_id),
        "new_basin": bool(len(archive.entries) > before_entries),
        "walk_termination_reason": str(trace.termination_reason),
        "walk_step_count": len(trace.steps),
        "force_evaluations": counts.total,
        "purpose_counts": counts.as_dict(),
        "wall_time_s": wall_time_s,
        "trace": asdict(trace),
        "right_censored_before_step1": False,
    }
    _write_json(case_directory / "outcome.json", row)
    return row, controller.reference_pool


def _run_group(
    *,
    system: str,
    seed: int,
    group_directory: Path,
    shared_bootstrap,
    cuo_resources,
) -> dict[str, Any]:
    config = build_config(system, group_directory, seed=seed)
    step_zero_pool = step_zero._precompute_pool(
        state=shared_bootstrap.result.state,
        calculator=observability.base.source.source._calculator(
            system,
            cuo_resources,
        ),
        config=config,
    )
    reference_row, reference_pool = _run_arm(
        system=system,
        seed=seed,
        repeat=0,
        candidate_index=None,
        case_directory=group_directory / "repeat-0-reference",
        shared_bootstrap=shared_bootstrap,
        config=config,
        step_zero_pool=step_zero_pool,
        reference_pool=None,
        cuo_resources=cuo_resources,
    )
    if reference_pool is None:
        return {
            "system": system,
            "seed": seed,
            "right_censored_before_step1": True,
            "rows": [reference_row],
            "step_zero_pool_force_evaluations": int(step_zero_pool["force_evaluations"]),
            "effective_config": asdict(config),
        }
    rows = [reference_row]
    static_winner_index = int(reference_row["candidate_index"])
    for repeat in (0, 1):
        for candidate_index in range(4):
            if repeat == 0 and candidate_index == static_winner_index:
                continue
            row, _ = _run_arm(
                system=system,
                seed=seed,
                repeat=repeat,
                candidate_index=candidate_index,
                case_directory=(
                    group_directory
                    / f"repeat-{repeat}-candidate-{candidate_index}"
                ),
                shared_bootstrap=shared_bootstrap,
                config=config,
                step_zero_pool=step_zero_pool,
                reference_pool=reference_pool,
                cuo_resources=cuo_resources,
            )
            rows.append(row)
    rows.sort(key=lambda row: (int(row["repeat"]), int(row["candidate_index"])))
    return {
        "system": system,
        "seed": seed,
        "right_censored_before_step1": False,
        "rows": rows,
        "step_zero_pool_force_evaluations": int(step_zero_pool["force_evaluations"]),
        "step_zero_pool_purpose_counts": dict(step_zero_pool["purpose_counts"]),
        "step_one_shared_pool_force_evaluations": int(
            reference_pool["direction_oracle_force_evaluations"]
        ),
        "step_one_candidates": list(reference_pool["candidates"]),
        "effective_config": asdict(config),
    }


def run_gate(
    *,
    output_directory: Path,
    expected_commit: str,
    systems: Sequence[str],
    seeds: Sequence[int],
    preflight_only: bool = False,
) -> dict[str, Any]:
    systems = tuple(str(system) for system in systems)
    seeds = tuple(int(seed) for seed in seeds)
    if any(system not in SYSTEMS for system in systems):
        raise ValueError(f"systems must be drawn from {SYSTEMS!r}")
    provenance = observability.base.source.source._preflight(
        expected_commit,
        systems,
    )
    provenance["gate"] = "shared-step1-native-K4-H8-counterfactual"
    provenance["direction_type_ucb_enabled"] = False
    if preflight_only:
        return provenance

    output_directory = Path(output_directory)
    if output_directory.exists():
        raise FileExistsError(output_directory)
    output_directory.mkdir(parents=True)
    cuo_resources = None
    if "cuo" in systems:
        cuo_resources = observability.base.source.source._materialize_cuo_resources(
            observability.base.source.source.CUO_ARCHIVE_PATH,
            output_directory / "cuo-input",
        )
    groups = []
    total_started = perf_counter()
    for system in systems:
        for seed in seeds:
            elapsed = perf_counter() - total_started
            if elapsed > MAX_WALL_SECONDS:
                raise RuntimeError("step-one gate exceeded its preregistered wall-time cap")
            group_directory = output_directory / system / f"seed-{seed:08d}"
            shared_bootstrap = observability.base.source.source._bootstrap_case(
                system=system,
                seed=seed,
                bootstrap_directory=group_directory / "bootstrap",
                force_budget=MAX_TOTAL_FORCE_EVALUATIONS,
                cuo_resources=cuo_resources,
            )
            group = _run_group(
                system=system,
                seed=seed,
                group_directory=group_directory,
                shared_bootstrap=shared_bootstrap,
                cuo_resources=cuo_resources,
            )
            group["bootstrap_force_evaluations"] = shared_bootstrap.counts.total
            group["bootstrap_purpose_counts"] = shared_bootstrap.counts.as_dict()
            groups.append(group)
            _write_json(output_directory / "partial_groups.json", groups)

    rows = [row for group in groups for row in group["rows"] if not row["right_censored_before_step1"]]
    summary = protocol.summarize_repeats(rows)
    new_force_evaluations = sum(
        int(group["bootstrap_force_evaluations"])
        + int(group["step_zero_pool_force_evaluations"])
        + sum(int(row.get("force_evaluations", 0)) for row in group["rows"])
        for group in groups
    )
    if new_force_evaluations > MAX_TOTAL_FORCE_EVALUATIONS:
        raise RuntimeError("step-one gate exceeded its force-evaluation cap")
    evidence = {
        "schema_version": 1,
        "execution_commit": _git_commit(),
        "provenance": provenance,
        "cohort": {"systems": list(systems), "seeds": list(seeds), "horizon": 8, "K": 4},
        "groups": groups,
        "rows": rows,
        "summary": summary,
        "aggregate": {
            "group_count": len(groups),
            "terminal_arm_count": len(rows),
            "new_force_evaluations": new_force_evaluations,
            "wall_time_s": perf_counter() - total_started,
        },
        "production_default_changed": False,
        "claim_ceiling": (
            "three-system three-seed shared-step1 causal gate under the frozen "
            "H8 propagator; not a production policy change"
        ),
    }
    _write_json(output_directory / "evidence.json", evidence)
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.output is None or args.expected_commit is None:
        raise SystemExit("--output and --expected-commit are required")
    evidence = run_gate(
        output_directory=args.output,
        expected_commit=args.expected_commit,
        systems=SYSTEMS if args.systems is None else args.systems,
        seeds=SEEDS if args.seeds is None else args.seeds,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
