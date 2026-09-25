import copy
import pickle

import numpy as np
import pytest
from ase.cluster import Icosahedron
from ase.constraints import FixAtoms, Hookean

from pamssw.standalone import SSWConfig, run_ssw
from pamssw.standalone.ase_constraints import normalize_constraints, bind_hookean_surface


class FlatSurface:
    def __init__(self):
        self.requests = 0

    def evaluate(self, atoms):
        assert not atoms.constraints
        self.requests += 1
        return 0., np.zeros_like(atoms.positions)


def config():
    return SSWConfig(width=.1, rotation_bias=1., max_gaussians=2,
        temperature_K=0., fmax=.03, relax_steps=2, fd_step=.001,
        rotation_hvp=2, rotation_tol=.02, direction_sampling='global',
        cluster_frame='direction_only')


def fixture():
    atoms = Icosahedron('Cu', 2)
    atoms.set_constraint(Hookean(0, 1, k=1., rt=10.))
    return atoms


def run(atoms, surface=None, **kwargs):
    return run_ssw(atoms, surface or FlatSurface(), steps=0, config=config(),
                   rng=kwargs.pop('rng', np.random.default_rng(4)),
                   progress_callback=lambda event: False, **kwargs)


def test_pair_hookean_saved_roundtrip_and_idempotent_binding():
    atoms = fixture()
    specs = normalize_constraints(atoms).hookean_specs
    surface = FlatSurface()
    result = run(atoms, bind_hookean_surface(surface, specs))
    assert len(atoms.constraints) == 1
    assert not result.current.constraints
    checkpoint = pickle.loads(pickle.dumps(result.checkpoint))
    assert checkpoint.schema_version == 6
    assert checkpoint.base_schema_version == 1
    assert checkpoint.hookean_specs == specs
    fresh = FlatSurface()
    resumed = run(atoms, fresh, checkpoint=checkpoint)
    assert fresh.requests == 0
    assert resumed.checkpoint.hookean_specs == specs
    assert resumed.checkpoint.rng_state == checkpoint.rng_state


@pytest.mark.parametrize('change', ['k', 'rt', 'pair', 'remove'])
def test_changed_restraint_rejected_before_rng_and_pes(change):
    atoms = fixture()
    checkpoint = run(atoms).checkpoint
    if change == 'remove':
        atoms.set_constraint()
    else:
        atoms.set_constraint(Hookean(0, 2 if change == 'pair' else 1,
                                    k=2. if change == 'k' else 1.,
                                    rt=11. if change == 'rt' else 10.))
    rng = np.random.default_rng(456)
    before = copy.deepcopy(rng.bit_generator.state)
    surface = FlatSurface()
    with pytest.raises(ValueError, match='Hookean|constraint|restraint'):
        run(atoms, surface, rng=rng, checkpoint=checkpoint)
    assert surface.requests == 0
    assert rng.bit_generator.state == before


def test_legacy_checkpoint_cannot_silently_gain_restraint():
    atoms = fixture()
    bare = atoms.copy()
    bare.set_constraint()
    cp = run(bare).checkpoint
    assert cp.schema_version == 1
    cp.__dict__.pop('hookean_specs', None)
    cp = pickle.loads(pickle.dumps(cp))
    surface = FlatSurface()
    run(bare, surface, checkpoint=cp)
    assert surface.requests == 0
    with pytest.raises(ValueError, match='Hookean|constraint|restraint'):
        run(atoms, surface, checkpoint=cp)
    assert surface.requests == 0


@pytest.mark.parametrize('constraint', [FixAtoms(indices=[0]),
    Hookean(0, (0., 0., 0.), k=1., rt=10.),
    Hookean(0, (0., 0., 1., -10.), k=1.)])
def test_nonpair_constraints_rejected_before_evaluation(constraint):
    atoms = fixture()
    atoms.set_constraint(constraint)
    surface = FlatSurface()
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        run(atoms, surface)
    assert surface.requests == 0


def test_wrong_prewrapped_surface_rejected_before_rng_and_pes():
    atoms = fixture()
    other = atoms.copy()
    other.set_constraint(Hookean(0, 1, k=2., rt=10.))
    physical = FlatSurface()
    surface = bind_hookean_surface(physical, normalize_constraints(other).hookean_specs)
    rng = np.random.default_rng(9)
    before = copy.deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match='Hookean|constraint|restraint'):
        run(atoms, surface, rng=rng)
    assert physical.requests == 0 and rng.bit_generator.state == before


def test_periodic_hookean_requires_other_entry():
    atoms = fixture()
    atoms.set_cell([20., 20., 20.])
    atoms.pbc = True
    surface = FlatSurface()
    with pytest.raises((ValueError, NotImplementedError)):
        run(atoms, surface)
    assert surface.requests == 0


@pytest.mark.parametrize('damage', ['old_schema', 'empty', 'bad_json', 'mc_type'])
def test_schema_six_does_not_skip_integrity_checks(damage):
    from pamssw.standalone.paper_reference import _validate_ssw_checkpoint
    cp = run(fixture()).checkpoint
    if damage == 'old_schema':
        cp.schema_version = 1
    elif damage == 'empty':
        cp.hookean_specs = ()
    elif damage == 'bad_json':
        cp.hookean_specs = ('bad json',)
    else:
        cp.mc_settings = 'wrong settings'
        cp.native_mc_state = 'wrong state'
    with pytest.raises((ValueError, TypeError)):
        _validate_ssw_checkpoint(cp)


def test_ls_initialization_failure_keeps_restraint_identity(monkeypatch):
    import pamssw.standalone.paper_reference as driver
    from pamssw.standalone.ls_native_reference import NativeLSSettings
    def fail(*args, **kwargs):
        raise ValueError('intentional missing LS domain')
    monkeypatch.setattr(driver, '_initialize_ls_state', fail)
    atoms = fixture()
    result = run(atoms, ls=NativeLSSettings({(29, 29): 3.}, {(29, 29): 2.8}, scale=.1))
    assert result.status == 'ls_initialization_failed'
    assert result.checkpoint.schema_version == 6
    assert result.checkpoint.hookean_specs == normalize_constraints(atoms).hookean_specs


class StatefulSelector:
    def checkpoint_contract(self): return {'identity': 'hookean-test', 'version': 1}
    def export_state(self): return {}
    def restore_state(self, state): pass
    def __call__(self, snapshot, rng): return None


@pytest.mark.parametrize('feature', ['direction', 'pool'])
def test_schema_six_retains_required_feature_state_guards(feature):
    from pamssw.standalone.paper_reference import _validate_ssw_checkpoint
    from pamssw.standalone.recovered_direction import RecoveredDirectionSettings
    if feature == 'direction':
        settings = RecoveredDirectionSettings(50,.5,.5,1,1,.01,.01,'euclidean',8)
        cp = run(fixture(), recovered_direction=settings).checkpoint
        cp.recovered_direction_state = None
    else:
        cp = run(fixture(), starter_selector=StatefulSelector(),
                 selector_rng=np.random.default_rng(34)).checkpoint
        cp.pool_state = None
    with pytest.raises(ValueError, match='direction|pool'):
        _validate_ssw_checkpoint(cp)


@pytest.mark.parametrize('base', [None, 0, 6, True, '4'])
def test_schema_six_requires_valid_original_capability_version(base):
    from pamssw.standalone.paper_reference import _validate_ssw_checkpoint
    cp = run(fixture()).checkpoint
    cp.base_schema_version = base
    with pytest.raises(ValueError, match='base|schema|capability'):
        _validate_ssw_checkpoint(cp)
