"""Public LS height-policy wiring on real ASE/EMT systems."""
from dataclasses import replace

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.cluster.icosahedron import Icosahedron

from pamssw.standalone import paper_reference as paper
from pamssw.standalone.ls_native_reference import NativeLSSettings, run_native_ls_ssw
from pamssw.standalone.native_height_policy import ConservativeNativeHeightPolicy
from pamssw.standalone.native_mc import NativeMCSettings
from pamssw.standalone.minimal_angle_height import MinimalAngleHeightPolicy
from pamssw.standalone.paper_reference import LSSettings, SSWConfig, run_ls_ssw
from pamssw.standalone.surface import ASESurface


POLICY = ConservativeNativeHeightPolicy(2.0, 0.2, 1, 10.0, 1.05, 1.25)
LS = LSSettings({(29, 29): 1.0}, {(29, 29): 2.9}, target_per_atom=0.001)


def cluster_config():
    return SSWConfig(width=0.2, rotation_bias=0.5, max_gaussians=1,
                     temperature_K=0.0, fmax=0.01, relax_steps=150,
                     fd_step=1e-4, rotation_hvp=20, rotation_tol=0.02,
                     direction_sampling='global', rotation_solver='ritz',
                     cluster_frame='direction_only',
                     quench_optimizer='safe-lbfgs-total')


def periodic_config():
    return replace(cluster_config(), width=0.1, rotation_bias=100.0,
                   rotation_solver='dimer', cluster_frame='translation_only')


@pytest.mark.parametrize('mc', [None, NativeMCSettings(0.1, 2)])
def test_public_paper_ls_forwards_height_keywords_exactly(monkeypatch, mc):
    seen = {}
    sentinel = object()

    def fake_run_ssw(*args, **kwargs):
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr(paper, 'run_ssw', fake_run_ssw)
    assert run_ls_ssw('atoms', 'surface', steps=0, config='config', rng='rng',
                      ls=LS, height_policy=POLICY,
                      height_update_budget=37, mc=mc) is sentinel
    assert seen == {'steps': 0, 'config': 'config', 'rng': 'rng', 'ls': LS,
                    'reconnect_distance': None, 'height_policy': POLICY,
                    'height_update_budget': 37, 'gaussian_policy': None,
                    'checkpoint': None, 'checkpoint_path': None,
                    'structure_matcher': None, 'mc': mc}

    matcher = object()
    seen.clear()
    assert run_ls_ssw('atoms', 'surface', steps=0, config='config', rng='rng',
                      ls=LS, structure_matcher=matcher) is sentinel
    assert seen['structure_matcher'] is matcher


@pytest.mark.parametrize('mc', [None, NativeMCSettings(0.1, 2)])
def test_public_native_ls_forwards_height_keywords_exactly(monkeypatch, mc):
    seen = {}
    sentinel = object()

    def fake_run_ssw(*args, **kwargs):
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr('pamssw.standalone.paper_reference.run_ssw', fake_run_ssw)
    native = NativeLSSettings({(29, 29): 1.0}, {(29, 29): 2.9})
    assert run_native_ls_ssw('atoms', 'surface', steps=0, config='config', rng='rng',
                             ls=native, height_policy=POLICY,
                             height_update_budget=41, mc=mc) is sentinel
    assert seen == {'steps': 0, 'config': 'config', 'rng': 'rng', 'ls': native,
                    'height_policy': POLICY, 'height_update_budget': 41, 'gaussian_policy': None,
                    'checkpoint': None, 'checkpoint_path': None,
                    'structure_matcher': None, 'mc': mc}

    matcher = object()
    seen.clear()
    assert run_native_ls_ssw('atoms', 'surface', steps=0, config='config', rng='rng',
                             ls=native, structure_matcher=matcher) is sentinel
    assert seen['structure_matcher'] is matcher


def test_paper_ls_height_policy_reaches_real_cu13_landing():
    atoms = Icosahedron('Cu', 2)
    surface = ASESurface(EMT())
    result = run_ls_ssw(atoms, surface, steps=1, config=cluster_config(),
                        rng=np.random.default_rng(7), ls=LS,
                        height_policy=POLICY)
    assert result.status == 'completed'
    assert len(result.records) == 1 and result.records[0].climb
    assert result.records[0].landing is not None
    assert result.records[0].landing.converged
    assert result.records[0].landing.surface == 'true'
    assert len(result.minima) >= 2
    assert all(m.surface == 'true' and m.converged for m in result.minima)
    assert result.evaluation_requests == surface.requests


def test_paper_ls_height_policy_completes_two_gaussians_on_real_emt():
    """Exercise the nonempty prepared-height history on the second Gaussian."""
    atoms = Icosahedron('Cu', 2)
    surface = ASESurface(EMT())
    result = run_ls_ssw(atoms, surface, steps=1,
                        config=replace(cluster_config(), max_gaussians=2),
                        rng=np.random.default_rng(7), ls=LS,
                        height_policy=POLICY)
    assert result.status == 'completed'
    assert len(result.records) == 1
    climb = result.records[0].climb
    assert len(climb) == 2
    assert [event['index'] for event in climb] == [0, 1]
    assert all(event['status'] == 'converged' for event in climb)
    assert result.evaluation_requests == surface.requests


def test_paper_ls_minimal_angle_height_completes_two_gaussians_on_real_emt():
    atoms = Icosahedron('Cu', 2)
    surface = ASESurface(EMT())
    result = run_ls_ssw(atoms, surface, steps=1,
                        config=replace(cluster_config(), max_gaussians=2),
                        rng=np.random.default_rng(7), ls=LS,
                        height_policy=MinimalAngleHeightPolicy())
    assert result.status == 'completed'
    assert len(result.records) == 1
    climb = result.records[0].climb
    assert len(climb) == 2
    assert [event['index'] for event in climb] == [0, 1]
    assert all(event['status'] == 'converged' for event in climb)
    assert result.evaluation_requests == surface.requests


def test_paper_ls_height_policy_reaches_fixed_cell_cu_landing():
    atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
    original_cell = atoms.cell.array.copy()
    surface = ASESurface(EMT())
    result = run_ls_ssw(atoms, surface, steps=1, config=periodic_config(),
                        rng=np.random.default_rng(17), ls=LS,
                        height_policy=POLICY)
    assert result.status == 'completed'
    assert len(result.records) == 1 and result.records[0].climb
    assert result.records[0].landing is not None
    assert result.records[0].landing.converged
    assert result.records[0].landing.surface == 'true'
    assert len(result.minima) >= 2
    assert all(m.surface == 'true' and m.converged for m in result.minima)
    assert all(np.array_equal(m.atoms.cell.array, original_cell)
               for m in result.minima)
    assert result.evaluation_requests == surface.requests


def test_pair_sampling_uses_actual_soft_quenched_geometry(monkeypatch):
    prepared_positions = []
    sampled_positions = []
    prepare = paper.prepare_ls_step
    sample = paper.sample_initial_direction
    def capture_prepare(*args, **kwargs):
        result = prepare(*args, **kwargs)
        prepared_positions.append(result.atoms.positions.copy())
        return result
    def capture_sample(atoms, *args, **kwargs):
        sampled_positions.append(atoms.positions.copy())
        return sample(atoms, *args, **kwargs)
    monkeypatch.setattr(paper, 'prepare_ls_step', capture_prepare)
    monkeypatch.setattr(paper, 'sample_initial_direction', capture_sample)
    result = run_ls_ssw(Icosahedron('Cu', 2), ASESurface(EMT()), steps=1,
        config=replace(cluster_config(), direction_sampling='paper'),
        rng=np.random.default_rng(11), ls=LS)
    assert len(prepared_positions) == len(sampled_positions) == 1
    assert np.linalg.norm(prepared_positions[0]-result.initial.atoms.positions) > 1e-4
    np.testing.assert_array_equal(sampled_positions[0], prepared_positions[0])
