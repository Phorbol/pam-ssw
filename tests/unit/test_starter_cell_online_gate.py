from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

from pamssw.archive import MinimaArchive
from pamssw.exploration.posterior import StarterProductivityPosterior
from pamssw.state import State


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260730-starter-cell-online-gate"
    / "run_gate.py"
)


def _runner():
    name = "_starter_cell_online_gate_test"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load starter-cell online gate")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _archive() -> MinimaArchive:
    archive = MinimaArchive(energy_tol=1e-6, rmsd_tol=0.01)
    for entry_id, x in enumerate((0.0, 1.0, 2.0, 10.0)):
        state = State(
            numbers=np.array([6, 6]),
            positions=np.array([[0.0, 0.0, 0.0], [x + 1.0, 0.0, 0.0]]),
        )
        entry = archive.add(state, -float(entry_id), parent_id=None)
        assert entry.entry_id == entry_id
    return archive


def test_mace_cell_builder_caches_each_archive_entry_and_records_two_stage_probabilities():
    runner = _runner()

    class FakeMACE:
        def __init__(self):
            self.calls = 0

        def get_descriptors(self, atoms, invariants_only=True, num_layers=-1):
            assert invariants_only is True
            assert num_layers == -1
            self.calls += 1
            x = float(atoms.positions[1, 0])
            return np.array([[x, 0.0], [x, 0.0]])

    calculator = FakeMACE()
    builder = runner.MACEFPSCellSnapshotBuilder(
        calculator,
        species=(6,),
        max_cells=2,
    )
    posterior = StarterProductivityPosterior()
    archive = _archive()

    first = builder(archive, posterior, 0, 0)
    second = builder(archive, posterior, 1, 1)

    assert calculator.calls == len(archive.entries)
    assert first.probabilities == second.probabilities
    assert first.probabilities == (1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 0.5)
    assert first.support_complete is True
    records = builder.partition_records()
    assert len(records) == 2
    assert records[0]["center_ids"] == [0, 3]
    assert records[0]["members_by_center"] == [[0, 1, 2], [3]]
    assert records[0]["cell_probabilities"] == [0.5, 0.5]
    assert records[0]["conditional_probabilities"] == [
        1.0 / 3.0,
        1.0 / 3.0,
        1.0 / 3.0,
        1.0,
    ]
    assert builder.descriptor_cost()["descriptor_forward_calls"] == 4


def test_cell_count_is_derived_only_from_trial_resolution_not_system_energy():
    runner = _runner()

    assert runner.cell_count_for_trial_gate(200, observations_per_cell=3) == 66
    assert runner.cell_count_for_trial_gate(50, observations_per_cell=3) == 16


def test_production_gate_uses_system_specific_validated_action_kernels(tmp_path):
    runner = _runner()

    c60 = runner.build_production_config("c60", tmp_path / "c60", master_seed=43)
    pdo = runner.build_production_config("pdo", tmp_path / "pdo", master_seed=43)

    assert c60.oracle_candidates == 4
    assert c60.quench_optimizer == "ase-lbfgs"
    assert c60.quench_fallback_optimizer == "ase-fire"
    assert c60.quench_fmax == 0.01
    assert c60.local_softening_active_count == 3

    assert pdo.oracle_candidates == 8
    assert pdo.quench_optimizer == "scipy-lbfgsb"
    assert pdo.quench_fallback_optimizer is None
    assert pdo.quench_fmax == 0.03
    assert pdo.local_softening_active_count == 5

    for config in (c60, pdo):
        assert config.rng_seed == 43
        assert config.proposal_optimizer == "safe-lbfgs-total"
        assert config.proposal_pool_size == 1
        assert config.proposal_duplicate_rescue_optimizer is None
        assert config.accepted_structures_log is None
        assert config.accepted_structures_dir is None
        assert config.direction_diagnostics_enabled is False
