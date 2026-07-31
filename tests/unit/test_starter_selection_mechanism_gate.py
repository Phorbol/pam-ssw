from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


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
