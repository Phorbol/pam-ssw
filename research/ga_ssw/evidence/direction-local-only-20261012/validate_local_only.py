#!/usr/bin/env python3
"""Zero-PES source-ablation contract check against baseline 6c70ef4."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
from ase import Atoms
from ase.build import molecule


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
BASELINE = "6c70ef4b1a568866f084a9670a01eeae496d79fc"
SOURCE = "pamssw/standalone/native_direction_control.py"
FIXTURE = ROOT / "tests/standalone/fixtures/native_pair_generator.json"


def load_baseline_module():
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:{SOURCE}"], cwd=ROOT, text=True
    )
    module = ModuleType("pamssw.standalone._baseline_native_direction_control")
    module.__file__ = f"{BASELINE}:{SOURCE}"
    module.__package__ = "pamssw.standalone"
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def rng_state(rng):
    return copy.deepcopy(rng.bit_generator.state)


def assert_rng_equal(left, right):
    assert rng_state(left) == rng_state(right)


def state_for(cls, row, coefficients, center):
    state = cls(row["pair"], np.zeros(row["n"], dtype=int), coefficients,
                group_marker=0, geometry="nonperiodic")
    state.save_gaussian_center(center)
    return state


def main():
    from pamssw.standalone import native_direction_control as experimental

    baseline = load_baseline_module()
    fixture = json.loads(FIXTURE.read_text())["cases"][0]
    atoms = Atoms("Cu" * fixture["n"], positions=fixture["positions"])
    coefficients = np.asarray(fixture["coefficients"], dtype=float)

    # Initial direction generation must be byte-for-byte equivalent in its
    # observable geometry, route, marker, and consumed RNG state.
    init_base_rng = np.random.default_rng(811)
    init_exp_rng = np.random.default_rng(811)
    base_initial = baseline.LocalDirectionState(
        fixture["pair"], np.zeros(len(atoms), int), coefficients,
        group_marker=0, geometry="nonperiodic").initial(atoms.copy(), init_base_rng)
    exp_initial = experimental.LocalDirectionState(
        fixture["pair"], np.zeros(len(atoms), int), coefficients,
        group_marker=0, geometry="nonperiodic").initial(atoms.copy(), init_exp_rng)
    np.testing.assert_array_equal(exp_initial.direction, base_initial.direction)
    assert (exp_initial.release_all, exp_initial.local_route, exp_initial.group_marker) == (
        base_initial.release_all, base_initial.local_route, base_initial.group_marker)
    assert_rng_equal(init_exp_rng, init_base_rng)

    # At update, only the normalized displacement seed is removed. All local
    # coefficients, c9 scaling, generator route, marker transitions, and RNG
    # consumption remain the original path.
    displacement = np.arange(atoms.positions.size, dtype=float).reshape(-1, 3)
    displacement = .08 * displacement / np.linalg.norm(displacement)
    center = atoms.positions - displacement
    base = state_for(baseline.LocalDirectionState, fixture, coefficients, center)
    exp = state_for(experimental.LocalDirectionState, fixture, coefficients, center)
    base_rng = np.random.default_rng(1771)
    exp_rng = np.random.default_rng(1771)
    base_update = base.update(atoms.copy(), base_rng)
    exp_update = exp.update(atoms.copy(), exp_rng)
    assert (exp_update.release_all, exp_update.local_route, exp_update.group_marker) == (
        base_update.release_all, base_update.local_route, base_update.group_marker)
    assert exp.group_marker == base.group_marker
    assert_rng_equal(exp_rng, base_rng)

    expected_coefficients = np.zeros(10)
    expected_coefficients[4:7] = coefficients[4:7]
    expected_coefficients[9] = 1.2 * np.sum(expected_coefficients[:9])
    expected_rng = np.random.default_rng(1771)
    expected_update = experimental.generate_local_direction(
        atoms.copy(), np.zeros_like(atoms.positions), expected_coefficients,
        fixture["pair"], np.zeros(len(atoms), int), expected_rng,
        group_marker=0, c1_radius_policy="restricted")
    np.testing.assert_allclose(exp_update.direction, expected_update.direction,
                               atol=2e-14, rtol=0)
    assert exp_update.local_route == expected_update.local_route
    assert exp_update.group_marker == expected_update.group_marker
    assert_rng_equal(exp_rng, expected_rng)
    assert not np.allclose(exp_update.direction, base_update.direction)

    # A forbidden local pair with no c1 component and zero local-only seed
    # preserves the established all-zero direction / release-all behavior.
    forbidden = molecule("C6H6")
    forbidden.positions += 15.0
    zero_coefficients = np.zeros(10)
    zero_coefficients[4] = .6
    zero_center = forbidden.positions - np.full_like(forbidden.positions, .03)
    zero_state = experimental.LocalDirectionState(
        (8, 3), np.zeros(len(forbidden), int), zero_coefficients,
        group_marker=0, geometry="nonperiodic")
    zero_state.save_gaussian_center(zero_center)
    zero_result = zero_state.update(forbidden, np.random.default_rng(919))
    assert zero_result.release_all
    assert zero_result.local_route == "forbidden"
    assert not np.any(zero_result.direction)

    print(json.dumps({
        "status": "ok",
        "baseline": BASELINE,
        "checks": [
            "initial direction, route, release flag, marker, and RNG state equal baseline",
            "update route, release flag, marker, and RNG state equal baseline",
            "update direction equals generator with explicit zero displacement seed and unchanged coefficients",
            "zero local-only direction preserves release-all behavior for forbidden pair",
        ],
        "scientific_scope": "implementation contract only; no PES/search result",
    }, indent=2))


if __name__ == "__main__":
    main()
