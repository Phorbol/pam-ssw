from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
from ase import Atoms
from ase.io import write


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260731-starter-selection-mechanism-gate"
    / "run_gate.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "_starter_selection_mechanism_gate",
        RUNNER_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_gate_changes_only_starter_mode_across_each_system_matrix(tmp_path):
    runner = _load_runner()

    for system in runner.SYSTEMS:
        configs = {
            mode: runner.build_config(
                system,
                tmp_path / system / mode,
                seed=43,
                starter_mode=mode,
                force_budget=20_000,
            )
            for mode in runner.STARTER_MODES
        }
        reference = configs["uniform_archive"]
        for mode, config in configs.items():
            assert config.seed_selection_mode == mode
            assert config.rng_seed == 43
            assert config.max_force_evals == 20_000
            assert config.max_trials == runner.MAX_TRIALS
            differing = {
                name
                for name in config.__dataclass_fields__
                if getattr(config, name) != getattr(reference, name)
            }
            assert differing <= {
                "seed_selection_mode",
                "accepted_structures_dir",
                "accepted_structures_log",
                "direction_diagnostics_path",
                "direction_archive_path",
                "proposal_minima_dir",
                "relaxation_trajectory_dir",
            }


def test_cuo_uses_the_frozen_generic_slab_kernel(tmp_path):
    runner = _load_runner()

    assert "cuo" in runner.SYSTEMS
    cuo = runner.build_config(
        "cuo",
        tmp_path / "cuo",
        seed=42,
        starter_mode="uniform_archive",
        force_budget=20_000,
    )
    pdo = runner.build_config(
        "pdo",
        tmp_path / "pdo",
        seed=42,
        starter_mode="uniform_archive",
        force_budget=20_000,
    )

    differing = {
        name
        for name in cuo.__dataclass_fields__
        if getattr(cuo, name) != getattr(pdo, name)
    }
    assert differing <= {
        "accepted_structures_dir",
        "accepted_structures_log",
        "direction_diagnostics_path",
        "direction_archive_path",
        "proposal_minima_dir",
        "relaxation_trajectory_dir",
    }


def test_cuo_archive_materialization_extracts_only_the_declared_resources(tmp_path):
    runner = _load_runner()
    archive = tmp_path / "Cu110_Cu10O8.zip"
    with ZipFile(archive, "w") as handle:
        handle.writestr(runner.CUO_INPUT_MEMBER, b"arc")
        handle.writestr(runner.CUO_MODEL_MEMBER, b"model")
        handle.writestr("unexpected.txt", b"do not extract")

    resources = runner._materialize_cuo_resources(archive, tmp_path / "materialized")

    assert resources["input"].read_bytes() == b"arc"
    assert resources["model"].read_bytes() == b"model"
    assert not (tmp_path / "materialized" / "unexpected.txt").exists()


def test_cuo_state_uses_slab_pbc_and_frozen_lowest_35_percent(tmp_path):
    runner = _load_runner()
    structure = tmp_path / "cuo.xyz"
    atoms = Atoms(
        "Cu10",
        positions=np.column_stack(
            (
                np.zeros(10),
                np.zeros(10),
                np.arange(10, dtype=float),
            )
        ),
        cell=(10.0, 10.0, 20.0),
        pbc=(True, True, False),
    )
    write(structure, atoms)

    state = runner._load_state(
        "cuo",
        {"input": structure, "model": tmp_path / "model.pt"},
    )

    assert tuple(state.pbc) == (True, True, False)
    np.testing.assert_array_equal(
        state.fixed_mask,
        np.array([True, True, True, True, False, False, False, False, False, False]),
    )
    assert runner._state_facts(state) == {
        "n_atoms": 10,
        "n_fixed_atoms": 4,
        "pbc": [True, True, False],
    }
