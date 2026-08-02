from __future__ import annotations

import importlib.util
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import sys

import numpy as np
import pytest

from pamssw.bias import GaussianBiasTerm
from pamssw.archive import MinimaArchive
from pamssw.calculators import AnalyticCalculator
from pamssw.config import SSWConfig
from pamssw.state import State
from pamssw.walker import (
    CandidateDirectionGenerator,
    DirectionCandidateKind,
    DirectionChoice,
    ProposalPotential,
    SurfaceWalker,
)


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "runs" / "20260802-step1-direction-counterfactual"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


protocol = _load(RUN_ROOT / "protocol.py", "_step1_direction_protocol_test")
runner = _load(RUN_ROOT / "run_gate.py", "_step1_direction_runner_test")


def test_step_zero_hash_matches_frozen_gate_representation() -> None:
    direction = np.array([2.0, 0.0, -2.0])
    normalized = direction / np.linalg.norm(direction)
    expected = sha256(np.asarray(normalized, dtype="<f8").tobytes()).hexdigest()

    assert runner._step_zero_direction_sha256(direction) == expected


def test_gate_disables_only_archive_momentum_not_plain_momentum(tmp_path) -> None:
    config = runner.build_config("c60", tmp_path, seed=52)

    assert config.enable_momentum_candidate is True
    assert config.archive_escape_momentum_enabled is False
    assert config.direction_archive_enabled is False


def _pool_rows(
    *,
    system: str = "c60",
    seed: int = 52,
    best_by_repeat: tuple[int, int] = (1, 1),
):
    kinds = ("momentum", "bond", "bond", "random")
    rows = []
    for repeat, best_index in enumerate(best_by_repeat):
        for candidate_index, kind in enumerate(kinds):
            landing = float(candidate_index)
            if candidate_index == best_index:
                landing = -2.0
            rows.append(
                {
                    "system": system,
                    "seed": seed,
                    "repeat": repeat,
                    "candidate_index": candidate_index,
                    "kind": kind,
                    "static_rank": candidate_index + 1,
                    "static_score": float(4 - candidate_index),
                    "direction_sha256": f"direction-{candidate_index}",
                    "landing_delta_eV": landing,
                    "certificate": True,
                    "landing_geometry_valid": True,
                    "fragmented": False,
                    "prefix_valid": True,
                }
            )
    return rows


def _complete_rows(*, stable_misses: dict[str, int]):
    rows = []
    for system in ("c60", "pdo", "cuo"):
        for offset, seed in enumerate((52, 53, 54)):
            best = 3 if offset < stable_misses[system] else 0
            rows.extend(
                _pool_rows(
                    system=system,
                    seed=seed,
                    best_by_repeat=(best, best),
                )
            )
    return rows


def test_group_matrix_is_three_system_three_seed_order() -> None:
    assert protocol.group_matrix() == [
        {"system": system, "seed": seed}
        for system in ("c60", "pdo", "cuo")
        for seed in (52, 53, 54)
    ]


def test_family_terminal_medians_do_not_double_weight_bond_slots() -> None:
    rows = [row for row in _pool_rows() if row["repeat"] == 0]

    result = protocol.family_terminal_medians(rows)

    assert result == {
        "momentum": pytest.approx(0.0),
        "bond": pytest.approx(0.0),
        "random": pytest.approx(3.0),
    }


def test_repeat_summary_rejects_unstable_best_identity() -> None:
    result = protocol.summarize_repeats(
        _pool_rows(best_by_repeat=(0, 1))
    )

    assert result["decision"] == "STOP_NON_IDENTIFIABLE_DIRECTION_LABELS"
    assert result["unstable_best_pools"] == ["c60:52"]
    assert result["posterior_gate_allowed"] is False


def test_static_bottleneck_requires_every_system_to_pass() -> None:
    result = protocol.summarize_repeats(
        _complete_rows(
            stable_misses={"c60": 3, "pdo": 2, "cuo": 1}
        )
    )

    assert result["static_continuation_bottleneck"] is False
    assert result["posterior_gate_allowed"] is False


def test_cross_system_momentum_dominance_opens_only_source_gate() -> None:
    rows = []
    for system in ("c60", "pdo", "cuo"):
        for seed in (52, 53, 54):
            rows.extend(
                _pool_rows(
                    system=system,
                    seed=seed,
                    best_by_repeat=(0, 0),
                )
            )

    result = protocol.summarize_repeats(rows)

    assert result["dominant_source"] == "momentum"
    assert result["source_only_gate_allowed"] is True
    assert result["posterior_gate_allowed"] is False
    assert result["decision"] == "ALLOW_SOURCE_ONLY_PROSPECTIVE_GATE"


def _state(shift: float = 0.0) -> State:
    return State(
        numbers=np.array([6, 6]),
        positions=np.array([[shift, 0.0, 0.0], [1.4, 0.0, 0.0]]),
    )


def _proposal(weight: float = 1.0) -> ProposalPotential:
    bias = GaussianBiasTerm(
        center=np.array([0.0, 0.0, 0.0, 1.4, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0]),
        sigma=0.5,
        weight=weight,
    )
    return ProposalPotential(calculator=None, biases=[bias])


def _serialized_pool():
    return {
        "prefix": runner.prefix_certificate(
            _state(),
            _proposal(),
            np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0]),
        ),
        "candidates": [
            {
                "candidate_index": index,
                "kind": kind,
                "static_rank": index + 1,
                "direction_sha256": f"direction-{index}",
            }
            for index, kind in enumerate(
                ("momentum", "bond", "bond", "random")
            )
        ],
    }


def test_prefix_certificate_changes_with_positions_or_bias() -> None:
    direction = np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0])

    reference = runner.prefix_certificate(_state(), _proposal(), direction)
    moved = runner.prefix_certificate(_state(shift=1.0e-4), _proposal(), direction)
    changed = runner.prefix_certificate(_state(), _proposal(weight=2.0), direction)

    assert reference["positions_sha256"] != moved["positions_sha256"]
    assert reference["biases_sha256"] != changed["biases_sha256"]
    assert reference["bias_count"] == 1


def test_replayed_pool_requires_exact_identity_and_one_momentum() -> None:
    reference = _serialized_pool()
    runner.validate_replayed_pool(reference, deepcopy(reference))

    changed = deepcopy(reference)
    changed["candidates"][1]["direction_sha256"] = "changed"
    with pytest.raises(RuntimeError, match="candidate identity"):
        runner.validate_replayed_pool(reference, changed)

    no_momentum = deepcopy(reference)
    no_momentum["candidates"][0]["kind"] = "random"
    with pytest.raises(RuntimeError, match="one momentum"):
        runner.validate_replayed_pool(no_momentum, no_momentum)


def test_replayed_pool_rejects_prefix_position_drift() -> None:
    reference = _serialized_pool()
    replayed = deepcopy(reference)
    replayed["prefix"]["positions"] = np.asarray(
        replayed["prefix"]["positions"], dtype=float
    ).copy()
    replayed["prefix"]["positions"][0][0] += 2.0e-10

    with pytest.raises(RuntimeError, match="prefix positions"):
        runner.validate_replayed_pool(reference, replayed)


class _Quadratic:
    def energy_gradient(self, flat_positions, state):
        values = np.asarray(flat_positions, dtype=float)
        return 0.5 * float(values @ values), values


def test_native_step1_pool_uses_one_batched_hvp_per_candidate() -> None:
    state = State(
        numbers=np.array([1, 1, 1, 1]),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [4.0, 1.0, 0.0],
            ]
        ),
    )
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(_Quadratic()),
        config=SSWConfig(
            oracle_candidates=4,
            n_bond_pairs=1,
            direction_selection_mode="discrete",
        ),
        softening_enabled=False,
    )
    walker.oracle.generator = CandidateDirectionGenerator(
        walker.rng,
        n_random=4,
        bond_pairs=[(0, 1)],
        n_bond_pairs=1,
        bond_distance_threshold=2.0,
    )
    archive = MinimaArchive(energy_tol=1.0e-3, rmsd_tol=0.15)
    archive.add(state, 0.0, parent_id=None)
    previous = np.array(
        [1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0]
    )

    pool = runner.evaluate_native_pool(
        walker=walker,
        state=state,
        proposal=ProposalPotential(walker.calculator),
        previous_direction=previous,
        anchor_direction=previous,
        archive=archive,
        history_gradient=None,
        continuity_weight=walker.config.continuity_weight,
        n_bond_pairs=walker.config.n_bond_pairs,
        score_sigma=None,
        score_sigma_fn=walker._direction_score_sigma_fn(1.0),
        step_scale_fn=lambda curvature: walker._scaled_step_scale(curvature, 1.0),
    )

    assert [record["kind"] for record in pool["candidates"]] == [
        "momentum",
        "bond",
        "bond",
        "random",
    ]
    assert sorted(record["static_rank"] for record in pool["candidates"]) == [1, 2, 3, 4]
    assert len(pool["choices"]) == 4
    assert pool["direction_oracle_force_evaluations"] == 8


def _step1_case():
    state = State(
        numbers=np.array([1, 1, 1, 1]),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [4.0, 1.0, 0.0],
            ]
        ),
    )
    config = SSWConfig(
        oracle_candidates=4,
        n_bond_pairs=1,
        direction_selection_mode="discrete",
        rng_seed=17,
    )
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(_Quadratic()),
        config=config,
        softening_enabled=False,
    )
    walker.oracle.generator = CandidateDirectionGenerator(
        walker.rng,
        n_random=4,
        bond_pairs=[(0, 1)],
        n_bond_pairs=1,
        bond_distance_threshold=2.0,
    )
    archive = MinimaArchive(energy_tol=1.0e-3, rmsd_tol=0.15)
    archive.add(state, 0.0, parent_id=None)
    previous = np.array(
        [1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0]
    )
    kwargs = {
        "anchor_direction": previous,
        "step_scale_fn": lambda curvature: walker._scaled_step_scale(curvature, 1.0),
        "archive": archive,
        "history_gradient": None,
        "continuity_weight": walker.config.continuity_weight,
        "n_bond_pairs": walker.config.n_bond_pairs,
        "score_sigma": None,
        "score_sigma_fn": walker._direction_score_sigma_fn(1.0),
        "direction_type_bonus_fn": None,
        "plateau_evolution_active": False,
        "plateau_history": [],
        "plateau_evolution_children": 0,
        "plateau_evolution_crossover_pairs": 0,
        "plateau_evolution_mutation_count": 0,
        "archive_momentum_history": [],
        "archive_momentum_limit": 0,
        "energy_bound_step_scale": None,
        "energy_bound_target": None,
    }
    return walker, state, ProposalPotential(walker.calculator), previous, kwargs


def test_controller_branches_only_at_first_normal_oracle_call() -> None:
    reference_walker, state, proposal, previous, kwargs = _step1_case()
    reference_pool = runner.evaluate_native_pool(
        walker=reference_walker,
        state=state,
        proposal=proposal,
        previous_direction=previous,
        anchor_direction=kwargs["anchor_direction"],
        archive=kwargs["archive"],
        history_gradient=kwargs["history_gradient"],
        continuity_weight=kwargs["continuity_weight"],
        n_bond_pairs=kwargs["n_bond_pairs"],
        score_sigma=kwargs["score_sigma"],
        score_sigma_fn=kwargs["score_sigma_fn"],
        step_scale_fn=kwargs["step_scale_fn"],
    )
    replay_walker, replay_state, replay_proposal, replay_previous, replay_kwargs = _step1_case()
    later_choice = DirectionChoice(
        direction=replay_previous / np.linalg.norm(replay_previous),
        curvature=1.0,
        kind=DirectionCandidateKind.MOMENTUM,
        candidate_count=1,
    )
    later_calls = []

    def original_choose(*args, **inner_kwargs):
        later_calls.append((args, inner_kwargs))
        return later_choice

    controller = runner.StepOneController(
        walker=replay_walker,
        original_choose=original_choose,
        reference_pool=reference_pool,
        forced_index=2,
    )

    forced = controller(
        replay_state,
        replay_proposal,
        replay_previous,
        **replay_kwargs,
    )
    later = controller(
        replay_state,
        replay_proposal,
        replay_previous,
        **replay_kwargs,
    )

    assert forced.diagnostics["shared_step1_direction"] is True
    assert forced.diagnostics["candidate_index"] == 2
    assert later is later_choice
    assert len(later_calls) == 1


def test_forced_controller_charges_no_step1_selection_hvp() -> None:
    reference_walker, state, proposal, previous, kwargs = _step1_case()
    reference_pool = runner.evaluate_native_pool(
        walker=reference_walker,
        state=state,
        proposal=proposal,
        previous_direction=previous,
        anchor_direction=kwargs["anchor_direction"],
        archive=kwargs["archive"],
        history_gradient=kwargs["history_gradient"],
        continuity_weight=kwargs["continuity_weight"],
        n_bond_pairs=kwargs["n_bond_pairs"],
        score_sigma=kwargs["score_sigma"],
        score_sigma_fn=kwargs["score_sigma_fn"],
        step_scale_fn=kwargs["step_scale_fn"],
    )
    replay_walker, replay_state, replay_proposal, replay_previous, replay_kwargs = _step1_case()
    controller = runner.StepOneController(
        walker=replay_walker,
        original_choose=replay_walker.oracle.choose_direction,
        reference_pool=reference_pool,
        forced_index=2,
    )
    before = replay_walker.calculator.snapshot()

    controller(
        replay_state,
        replay_proposal,
        replay_previous,
        **replay_kwargs,
    )

    after = replay_walker.calculator.snapshot()
    assert after.total - before.total == 0


def test_uphill_walk_pause_resume_matches_uninterrupted_analytic_path() -> None:
    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    config = SSWConfig(
        max_steps_per_walk=2,
        oracle_candidates=2,
        n_bond_pairs=1,
        proposal_relax_steps=2,
        proposal_fmax=1.0e-4,
        rng_seed=29,
    )
    direction = np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0])
    direction /= np.linalg.norm(direction)
    initial_choice = DirectionChoice(
        direction=direction,
        curvature=1.0,
        true_curvature=1.0,
        kind=DirectionCandidateKind.BOND,
        candidate_count=1,
    )

    uninterrupted = SurfaceWalker(
        calculator=AnalyticCalculator(_Quadratic()),
        config=config,
        softening_enabled=False,
    )
    uninterrupted_archive = MinimaArchive(energy_tol=1.0e-3, rmsd_tol=0.15)
    uninterrupted_entry = uninterrupted_archive.add(state, 0.5, parent_id=None)
    expected = uninterrupted._walk_candidate_from_seed(
        state,
        uninterrupted_archive,
        trial_index=0,
        seed_entry_id=uninterrupted_entry.entry_id,
        initial_direction_choice=initial_choice,
    )
    uninterrupted_count = uninterrupted.calculator.snapshot().total

    prefix_walker = SurfaceWalker(
        calculator=AnalyticCalculator(_Quadratic()),
        config=config,
        softening_enabled=False,
    )
    prefix_archive = MinimaArchive(energy_tol=1.0e-3, rmsd_tol=0.15)
    prefix_entry = prefix_archive.add(state, 0.5, parent_id=None)
    continuations = []
    prefix_traces = []
    prefix_walker._walk_candidate_from_seed(
        state,
        prefix_archive,
        trial_index=0,
        seed_entry_id=prefix_entry.entry_id,
        initial_direction_choice=initial_choice,
        pause_after_step=0,
        continuation_sink=continuations,
        trace_sink=prefix_traces,
    )
    prefix_count = prefix_walker.calculator.snapshot().total

    resumed_walker = SurfaceWalker(
        calculator=AnalyticCalculator(_Quadratic()),
        config=config,
        softening_enabled=False,
    )
    resumed_traces = []
    resumed = resumed_walker._walk_candidate_from_seed(
        state,
        prefix_archive,
        trial_index=0,
        seed_entry_id=prefix_entry.entry_id,
        continuation=continuations[0],
        trace_sink=resumed_traces,
    )

    assert len(continuations) == 1
    assert prefix_traces[0].termination_reason == "paused_for_continuation"
    assert continuations[0].next_step_index == 1
    assert [step.step_index for step in resumed_traces[0].steps] == [0, 1]
    assert prefix_count + resumed_walker.calculator.snapshot().total == uninterrupted_count
    np.testing.assert_allclose(resumed.positions, expected.positions, atol=1.0e-12)
