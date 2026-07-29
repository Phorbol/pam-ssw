import numpy as np
import pytest

from pamssw import LSSSWConfig, SSWConfig, State, run_ls_ssw, run_ssw
from pamssw.calculators import AnalyticCalculator
from pamssw.potentials import CoupledPairWell


def test_ls_ssw_crosses_stiff_landscape_more_effectively_than_ssw():
    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[-0.35, 0.0, 0.0], [0.35, 0.0, 0.0]]),
    )
    calc = AnalyticCalculator(CoupledPairWell())
    base = dict(
        max_trials=1,
        max_steps_per_walk=2,
        target_uphill_energy=0.05,
        min_step_scale=0.3,
        n_bond_pairs=0,
        rng_seed=11,
    )

    ssw = run_ssw(state, calc, SSWConfig(**base))
    ls = run_ls_ssw(
        state,
        calc,
        LSSSWConfig(
            **base,
            local_softening_strength=0.9,
            local_softening_pairs=[(0, 1)],
        ),
    )

    def pair_distances(result):
        return sorted(
            np.linalg.norm(entry.state.positions[1] - entry.state.positions[0])
            for entry in result.archive.entries
        )

    assert len(ls.archive.entries) >= len(ssw.archive.entries)
    assert ls.best_energy <= ssw.best_energy + 1e-8
    assert pair_distances(ls)[-1] > 1.0


def test_ls_ssw_neighbor_auto_runs_without_manual_pairs():
    state = State(
        numbers=np.array([6, 1]),
        positions=np.array([[-0.545, 0.0, 0.0], [0.545, 0.0, 0.0]]),
    )
    calc = AnalyticCalculator(CoupledPairWell())

    result = run_ls_ssw(
        state,
        calc,
        LSSSWConfig(
            max_trials=1,
            max_steps_per_walk=2,
            target_uphill_energy=0.05,
            min_step_scale=0.3,
            n_bond_pairs=0,
            rng_seed=11,
            local_softening_mode="neighbor_auto",
        ),
    )

    assert result.stats["local_softening_terms_total"] > 0
    assert result.stats["local_softening_builds"] > 0
    assert result.stats["local_softening_terms_total"] == result.stats["local_softening_terms_built_total"]
    assert len(result.archive.entries) > 0


def test_direction_archive_enabled_is_deterministic_noop_for_analytic_run(tmp_path):
    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[-0.35, 0.0, 0.0], [0.35, 0.0, 0.0]]),
    )
    base = dict(
        max_trials=2,
        max_steps_per_walk=2,
        target_uphill_energy=0.05,
        min_step_scale=0.3,
        n_bond_pairs=0,
        rng_seed=17,
    )

    disabled = run_ssw(
        state,
        AnalyticCalculator(CoupledPairWell()),
        SSWConfig(**base, direction_archive_enabled=False),
    )
    archive_path = tmp_path / "directions.jsonl"
    enabled = run_ssw(
        state,
        AnalyticCalculator(CoupledPairWell()),
        SSWConfig(
            **base,
            direction_archive_enabled=True,
            direction_archive_path=str(archive_path),
        ),
    )

    assert enabled.best_energy == pytest.approx(disabled.best_energy)
    assert len(enabled.archive.entries) == len(disabled.archive.entries)
    for left, right in zip(enabled.archive.entries, disabled.archive.entries, strict=True):
        assert left.entry_id == right.entry_id
        assert left.parent_id == right.parent_id
        assert left.energy == pytest.approx(right.energy)
        np.testing.assert_allclose(left.state.positions, right.state.positions)
    assert enabled.walk_history == disabled.walk_history

    behavior_keys = [
        "n_trials",
        "n_minima",
        "local_relaxations",
        "force_evaluations",
        "energy_evaluations",
        "direction_choices",
        "direction_selected_momentum",
        "direction_selected_random",
        "direction_selected_bond",
    ]
    for key in behavior_keys:
        assert enabled.stats[key] == disabled.stats[key]

    assert disabled.stats["direction_archive_enabled"] == 0
    assert disabled.stats["direction_archive_records"] == 0
    assert enabled.stats["direction_archive_enabled"] == 1
    assert enabled.stats["direction_archive_records"] == enabled.stats["direction_choices"]
    assert len(archive_path.read_text(encoding="utf-8").splitlines()) == enabled.stats["direction_choices"]
