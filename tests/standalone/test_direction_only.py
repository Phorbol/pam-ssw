"""Regression of the real Cu13 direction-only ablation through the public driver."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
from ase.io import read
from ase.calculators.emt import EMT
from pamssw.standalone import ASESurface, SSWConfig, run_ssw
from pamssw.standalone import dimer, paper_reference
from pamssw.standalone.cluster_frame import ClusterFrame


def _projected_solver(solver):
    """The archived run's legacy wrapper, independent of direction_only."""
    def rotate(atoms, anchor, *, evaluate, **kwargs):
        frame = ClusterFrame(atoms)
        internal = frame.project(anchor)
        norm = np.linalg.norm(internal)
        if norm <= np.finfo(float).eps * internal.size:
            raise ValueError('anchor has no resolvable internal component')
        internal /= norm

        def projected(candidate):
            candidate = candidate.copy()
            candidate.positions = frame.positions(candidate.positions)
            energy, forces = evaluate(candidate)
            return energy, frame.project(forces)

        return solver(atoms, internal, evaluate=projected, **kwargs)
    return rotate


@pytest.mark.parametrize("solver", ["ritz", "dimer"])
def test_public_direction_only_matches_legacy_wrapper(solver, monkeypatch):
    base = Path(__file__).resolve().parents[2] / 'research/ga_ssw/evidence/cu13-direction-only'
    reference = json.loads((base / f'3-{solver}.json').read_text())
    legacy_config = SSWConfig(**reference['config'])
    assert legacy_config.cluster_frame == 'cartesian'
    public_config = replace(legacy_config, cluster_frame='direction_only')
    solver_module = dimer if solver == 'dimer' else paper_reference
    solver_name = 'paper_dimer_direction' if solver == 'dimer' else 'paper_biased_direction'
    original = getattr(solver_module, solver_name)
    monkeypatch.setattr(solver_module, solver_name, _projected_solver(original))
    legacy = run_ssw(read(base / 'initial.extxyz'), ASESurface(EMT()), steps=1,
        config=legacy_config, rng=np.random.default_rng(3))
    monkeypatch.setattr(solver_module, solver_name, original)
    public = run_ssw(read(base / 'initial.extxyz'), ASESurface(EMT()), steps=1,
        config=public_config, rng=np.random.default_rng(3))
    wrapped = legacy.records[0]
    actual = public.records[0]
    assert wrapped.status == actual.status == 'biased_quench_failed'
    assert len(wrapped.climb) == len(actual.climb)
    np.testing.assert_allclose(actual.climb[0]['direction'], wrapped.climb[0]['direction'], atol=1e-10, rtol=0)
    np.testing.assert_allclose(actual.last_atoms.positions, wrapped.last_atoms.positions, atol=1e-7, rtol=0)
    assert actual.evaluation_requests == wrapped.evaluation_requests
    assert len(legacy.minima) == len(public.minima) == 1  # A failed move never becomes an archive entry.
    assert actual.climb[0]['force_certificate'] == wrapped.climb[0]['force_certificate'] == 'modified'
