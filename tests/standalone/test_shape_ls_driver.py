"""Shape LS driver and checkpoint contract, without a physical PES run."""
from dataclasses import replace

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import Hookean

from pamssw.standalone.paper_reference import (
    LSSettings, SSWConfig, _prepare_pool_restart, _validate_ssw_checkpoint,
    load_ssw_checkpoint, run_ssw,
)
from pamssw.standalone.shape_softening import FrozenShapeBondSoftening


class FlatSurface:
    def __init__(self):
        self.requests = 0

    def evaluate(self, atoms):
        self.requests += 1
        return 0., np.zeros_like(atoms.positions)


def case():
    atoms = Atoms('Cu2', positions=[[0., 0., 0.], [2.5, 0., 0.]])
    config = SSWConfig(width=.1, rotation_bias=1., max_gaussians=1,
        temperature_K=0., fmax=.03, relax_steps=2, fd_step=.001,
        rotation_hvp=2, rotation_tol=.02, direction_sampling='global')
    ls = LSSettings({(29, 29): 1.}, {(29, 29): 3.}, .001,
                    bias_geometry='shape')
    return atoms, config, ls


def run(atoms, surface, config, ls, **kwargs):
    return run_ssw(atoms, surface, steps=0, config=config,
                   rng=np.random.default_rng(4), ls=ls, **kwargs)


def test_shape_zero_step_checkpoint_and_resume(tmp_path):
    atoms, config, ls = case()
    path = tmp_path / 'shape.pkl'
    first = run(atoms, FlatSurface(), config, ls, checkpoint_path=path)
    saved = load_ssw_checkpoint(path)
    assert first.status == 'completed'
    assert saved.schema_version == 7 and saved.base_schema_version == 1
    assert isinstance(saved.frozen, FrozenShapeBondSoftening)
    assert np.isfinite(saved.frozen.radius0) and saved.frozen.radius0 > 0
    surface = FlatSurface()
    resumed = run(atoms, surface, config, ls, checkpoint=saved)
    assert surface.requests == 0
    assert resumed.checkpoint.schema_version == 7


def test_shape_initialization_failure_checkpoint_allows_null_frozen(tmp_path):
    atoms, config, ls = case()
    atoms.positions[1, 0] = 3.5  # No eligible frozen pair under the explicit 3 A cutoff.
    path = tmp_path / 'failed.pkl'
    result = run(atoms, FlatSurface(), config, ls, checkpoint_path=path)
    saved = load_ssw_checkpoint(path)
    assert result.status == saved.status == 'ls_initialization_failed'
    assert saved.schema_version == 7 and saved.frozen is None


@pytest.mark.parametrize('invalid', ['pbc', 'hookean'])
def test_shape_rejects_unsupported_geometry_before_pes(invalid):
    atoms, config, ls = case()
    if invalid == 'pbc':
        atoms.pbc = [True, False, False]
    else:
        atoms.set_constraint(Hookean(0, 1, k=1., rt=3.))
    surface = FlatSurface()
    with pytest.raises(NotImplementedError, match='shape LS'):
        run(atoms, surface, config, ls)
    assert surface.requests == 0


def test_cross_geometry_resume_rejected_before_pes(tmp_path):
    atoms, config, shape = case()
    pair = replace(shape, bias_geometry='pair')
    shape_path = tmp_path / 'shape.pkl'
    pair_path = tmp_path / 'pair.pkl'
    run(atoms, FlatSurface(), config, shape, checkpoint_path=shape_path)
    run(atoms, FlatSurface(), config, pair, checkpoint_path=pair_path)
    for saved, requested in ((shape_path, pair), (pair_path, shape)):
        surface = FlatSurface()
        with pytest.raises(ValueError, match='LS settings'):
            run(atoms, surface, config, requested, checkpoint=load_ssw_checkpoint(saved))
        assert surface.requests == 0


def test_shape_checkpoint_rejects_wrong_schema_and_frozen_state(tmp_path):
    atoms, config, shape = case()
    path = tmp_path / 'shape.pkl'
    run(atoms, FlatSurface(), config, shape, checkpoint_path=path)
    checkpoint = load_ssw_checkpoint(path)
    for mutate in (
        lambda cp: setattr(cp, 'schema_version', 1),
        lambda cp: setattr(cp, 'frozen', None),
        lambda cp: object.__setattr__(cp.frozen, 'radius0', float('nan')),
    ):
        from copy import deepcopy
        damaged = deepcopy(checkpoint)
        mutate(damaged)
        with pytest.raises(ValueError):
            _validate_ssw_checkpoint(damaged)


def test_old_pair_settings_without_geometry_field_resume(tmp_path):
    atoms, config, shape = case()
    pair = replace(shape, bias_geometry='pair')
    path = tmp_path / 'pair.pkl'
    run(atoms, FlatSurface(), config, pair, checkpoint_path=path)
    checkpoint = load_ssw_checkpoint(path)
    checkpoint.ls.__dict__.pop('bias_geometry')
    pair.__dict__.pop('bias_geometry')
    surface = FlatSurface()
    resumed = run(atoms, surface, config, pair, checkpoint=checkpoint)
    assert surface.requests == 0
    assert resumed.checkpoint.ls.bias_geometry == 'pair'


def test_other_ls_drivers_reject_shape_before_pes():
    from types import SimpleNamespace
    from pamssw.standalone.constrained_reference import run_constrained_ssw
    from pamssw.standalone.vc_reference import run_vc_ssw
    atoms, _, shape = case()
    surface = FlatSurface()
    for call, kwargs in (
        (run_constrained_ssw, {'ls': shape}),
        (run_constrained_ssw, {'checkpoint': SimpleNamespace(ls_settings=shape)}),
        (run_vc_ssw, {'ls': shape}),
    ):
        with pytest.raises(NotImplementedError, match='shape LS'):
            call(atoms, surface, steps=0, config=None,
                 rng=np.random.default_rng(4), **kwargs)
        assert surface.requests == 0


def test_pool_restart_freezes_new_radius():
    atoms, _, shape = case()
    first, _, _ = _prepare_pool_restart(atoms, ls=shape,
        recovered_direction=None, rng=np.random.default_rng(4))
    changed = atoms.copy()
    changed.positions *= 1.2
    next_frozen, _, _ = _prepare_pool_restart(changed, ls=shape,
        recovered_direction=None, rng=np.random.default_rng(4))
    assert next_frozen.radius0 == pytest.approx(1.2 * first.radius0)
    assert next_frozen.reference_distances[0] == pytest.approx(1.2 * first.reference_distances[0])


def test_shape_emt_split_resume_matches_continuous(tmp_path):
    """Integration check for the actual outer walk; run in the CPU test allocation."""
    from ase.calculators.emt import EMT
    from pamssw.standalone.surface import ASESurface
    atoms = Atoms('Cu3', positions=[[0., 0., 0.], [2.5, 0., 0.], [1.2, 2.2, 0.]])
    config = SSWConfig(width=.1, rotation_bias=2., max_gaussians=1,
        temperature_K=300., fmax=.005, relax_steps=100, fd_step=1e-4,
        rotation_hvp=8, rotation_tol=.001, direction_sampling='global')
    ls = LSSettings({(29, 29): 1.}, {(29, 29): 3.}, .001,
                    bias_geometry='shape')
    continuous = run_ssw(atoms, ASESurface(EMT()), steps=2, config=config,
        rng=np.random.default_rng(19), ls=ls,
        checkpoint_path=tmp_path / 'continuous.pkl')
    path = tmp_path / 'split.pkl'
    run_ssw(atoms, ASESurface(EMT()), steps=1, config=config,
        rng=np.random.default_rng(19), ls=ls, checkpoint_path=path)
    checkpoint = load_ssw_checkpoint(path)
    resumed = run_ssw(atoms, ASESurface(EMT()), steps=1, config=config,
        rng=np.random.default_rng(999), ls=ls, checkpoint=checkpoint)
    assert resumed.status == continuous.status == 'completed'
    assert resumed.evaluation_requests == continuous.evaluation_requests
    assert [record.status for record in resumed.records] == [
        record.status for record in continuous.records]
    np.testing.assert_allclose(resumed.current.positions, continuous.current.positions)
    assert resumed.checkpoint.frozen.radius0 == pytest.approx(
        continuous.checkpoint.frozen.radius0)
