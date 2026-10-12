#!/usr/bin/env python3
"""Verify feedback-only source ablation against its immutable parent source."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
from ase import Atoms

from pamssw.standalone import native_direction_control as variant


BASELINE = "6c70ef4b1a568866f084a9670a01eeae496d79fc"
SOURCE = "pamssw/standalone/native_direction_control.py"
FIXTURE = Path(__file__).resolve().parents[4] / "tests/standalone/fixtures/native_pair_generator.json"


def load_baseline_module():
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:{SOURCE}"], text=True
    )
    name = "pamssw.standalone._baseline_native_direction_control"
    module = types.ModuleType(name)
    module.__file__ = f"{BASELINE}:{SOURCE}"
    module.__package__ = "pamssw.standalone"
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


class CountingUniform:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.value


def make_state(module, row):
    atoms = Atoms("Cu" * row["n"], positions=row["positions"])
    coefficients = np.zeros(10)
    coefficients[1] = 1.0
    coefficients[4] = 0.6
    state = module.LocalDirectionState(
        row["pair"], np.zeros(len(atoms), int), coefficients, group_marker=0
    )
    return atoms, state


def main():
    baseline = load_baseline_module()
    rows = json.loads(FIXTURE.read_text())["cases"]
    row = next(r for r in rows if r["n"] == 13 and r["coefficients"][9] == 0.72)

    # The initial axis and all initial-generation side effects remain baseline.
    atoms_base, state_base = make_state(baseline, row)
    atoms_variant, state_variant = make_state(variant, row)
    rng_base, rng_variant = CountingUniform(row["uniform"]), CountingUniform(row["uniform"])
    initial_base = state_base.initial(atoms_base, rng_base)
    initial_variant = state_variant.initial(atoms_variant, rng_variant)
    np.testing.assert_array_equal(initial_variant.direction, initial_base.direction)
    assert initial_variant.local_route == initial_base.local_route
    assert state_variant.group_marker == state_base.group_marker
    assert rng_variant.calls == rng_base.calls

    center = np.asarray(row["positions"], dtype=float)
    displacement = 0.05 * np.asarray(row["input_seed"], dtype=float)
    atoms_base.positions = center + displacement
    atoms_variant.positions = center + displacement
    state_base.save_gaussian_center(center)
    state_variant.save_gaussian_center(center)
    rng_base, rng_variant = CountingUniform(row["uniform"]), CountingUniform(row["uniform"])
    result_base = state_base.update(atoms_base, rng_base)
    result_variant = state_variant.update(atoms_variant, rng_variant)

    # The full generator still runs: route, marker and random consumption match.
    assert result_variant.local_route == result_base.local_route
    assert state_variant.group_marker == state_base.group_marker
    assert rng_variant.calls == rng_base.calls

    # Only the normalized saved-center displacement, scaled by original c9, returns.
    seed = variant._native_normalize(displacement)
    c9 = 1.2 * np.sum(state_variant.last_coefficients[:9])
    expected = variant._native_normalize(seed * c9)
    np.testing.assert_allclose(result_variant.direction, expected, atol=4e-14, rtol=0)
    assert not result_variant.release_all
    assert not np.allclose(result_variant.direction, result_base.direction)

    # A zero saved-center displacement discards locally generated vectors and releases.
    atoms_base, state_base = make_state(baseline, row)
    atoms_variant, state_variant = make_state(variant, row)
    center_base, center_variant = atoms_base.positions.copy(), atoms_variant.positions.copy()
    state_base.save_gaussian_center(center_base)
    state_variant.save_gaussian_center(center_variant)
    rng_base, rng_variant = CountingUniform(row["uniform"]), CountingUniform(row["uniform"])
    zero_base = state_base.update(atoms_base, rng_base)
    zero_variant = state_variant.update(atoms_variant, rng_variant)
    assert zero_variant.local_route == zero_base.local_route
    assert state_variant.group_marker == state_base.group_marker
    assert rng_variant.calls == rng_base.calls
    assert zero_variant.release_all
    assert not np.any(zero_variant.direction)

    print("PASS: initial axis and update RNG/marker/route match parent; feedback formula and zero-seed release verified")


if __name__ == "__main__":
    main()
