"""Real EMT and PAM adapter resume qualification; not search-efficiency evidence."""
import numpy as np
import pytest
from ase.calculators.emt import EMT
from pamssw.standalone.surface import ASESurface
from pamssw.standalone.paper_reference import run_ssw, load_ssw_checkpoint
from research.ga_ssw.pool_starter_adapter import PoolStarterAdapter
from test_ssw_checkpoint import _case


@pytest.mark.parametrize('mode', ['uniform', 'pam'])
def test_real_ls_pool_resume_matches_uninterrupted(mode, tmp_path):
    atoms, config, ls = _case()
    def adapter():
        return PoolStarterAdapter(mode=mode, energy_tol=1e-5, rmsd_tol=1e-3)
    continuous_policy, partial_policy = adapter(), adapter()
    main_rng, pool_rng = np.random.default_rng(19), np.random.default_rng(23)
    full = run_ssw(atoms, ASESurface(EMT()), steps=4, config=config, ls=ls,
        rng=main_rng, starter_selector=continuous_policy, selector_rng=pool_rng)
    path = tmp_path / 'pool.pkl'
    first = run_ssw(atoms, ASESurface(EMT()), steps=2, config=config, ls=ls,
        rng=np.random.default_rng(19), starter_selector=partial_policy,
        selector_rng=np.random.default_rng(23), checkpoint_path=path)
    checkpoint = load_ssw_checkpoint(path)
    assert checkpoint.schema_version == 5
    resumed_policy = adapter()
    resumed_rng, resumed_pool_rng = np.random.default_rng(999), np.random.default_rng(888)
    resumed_surface = ASESurface(EMT())
    resumed = run_ssw(atoms, resumed_surface, steps=2, config=config, ls=ls,
        rng=resumed_rng, starter_selector=resumed_policy, selector_rng=resumed_pool_rng,
        checkpoint=checkpoint)
    assert resumed.status == full.status == 'completed'
    assert len(resumed.records) == len(full.records) == 4
    assert resumed.evaluation_requests == full.evaluation_requests
    assert first.evaluation_requests + resumed_surface.requests == full.evaluation_requests
    np.testing.assert_array_equal(resumed.current.positions, full.current.positions)
    assert [r.starter_selection for r in resumed.records] == [r.starter_selection for r in full.records]
    assert main_rng.bit_generator.state == resumed_rng.bit_generator.state
    assert pool_rng.bit_generator.state == resumed_pool_rng.bit_generator.state
    np.testing.assert_equal(continuous_policy.export_state(), resumed_policy.export_state())
    assert continuous_policy.finalize(full) == resumed_policy.finalize(resumed)


def test_changed_pool_configuration_rejected_without_pes(tmp_path):
    atoms, config, ls = _case()
    path = tmp_path/'pool.pkl'
    run_ssw(atoms, ASESurface(EMT()), steps=0, config=config, ls=ls,
        rng=np.random.default_rng(19), starter_selector=PoolStarterAdapter(mode='pam', energy_tol=1e-5, rmsd_tol=1e-3),
        selector_rng=np.random.default_rng(23), checkpoint_path=path)
    surface = ASESurface(EMT())
    with pytest.raises(ValueError):
        run_ssw(atoms, surface, steps=1, config=config, ls=ls,
            rng=np.random.default_rng(19), starter_selector=PoolStarterAdapter(mode='uniform', energy_tol=1e-5, rmsd_tol=1e-3),
            selector_rng=np.random.default_rng(23), checkpoint=load_ssw_checkpoint(path))
    assert surface.requests == 0


def test_real_native_ls_mc_pool_resume_matches_uninterrupted(tmp_path):
    from ase.cluster import Icosahedron
    from test_ls_pool_restart import _native_config, _native_ls
    from pamssw.standalone.native_mc import NativeMCSettings

    atoms = Icosahedron('Cu', 2)
    common = dict(config=_native_config(), ls=_native_ls(), mc=NativeMCSettings(.1, 2))
    policies = [PoolStarterAdapter(mode='pam', energy_tol=1e-5, rmsd_tol=1e-3)
                for _ in range(3)]
    full_rng, full_pool_rng = np.random.default_rng(7), np.random.default_rng(8)
    full = run_ssw(atoms, ASESurface(EMT()), steps=2, **common,
        rng=full_rng, starter_selector=policies[0], selector_rng=full_pool_rng,
        checkpoint_path=tmp_path/'full.pkl')
    first = run_ssw(atoms, ASESurface(EMT()), steps=1, **common,
        rng=np.random.default_rng(7), starter_selector=policies[1],
        selector_rng=np.random.default_rng(8), checkpoint_path=tmp_path/'partial.pkl')
    resumed_rng, resumed_pool_rng = np.random.default_rng(999), np.random.default_rng(888)
    surface = ASESurface(EMT())
    resumed = run_ssw(atoms, surface, steps=1, **common, rng=resumed_rng,
        starter_selector=policies[2], selector_rng=resumed_pool_rng,
        checkpoint=load_ssw_checkpoint(tmp_path/'partial.pkl'),
        checkpoint_path=tmp_path/'resumed.pkl')
    assert full.status == resumed.status == 'completed'
    assert len(full.records) == len(resumed.records) == 2
    assert policies[0].decisions  # Exercise actual pool selection, not only an empty snapshot.
    assert full.evaluation_requests == first.evaluation_requests + surface.requests
    assert resumed.evaluation_requests == full.evaluation_requests
    np.testing.assert_array_equal(full.current.positions, resumed.current.positions)
    assert [r.starter_selection for r in full.records] == [r.starter_selection for r in resumed.records]
    assert [r.ls_update for r in full.records] == [r.ls_update for r in resumed.records]
    np.testing.assert_equal(policies[0].export_state(), policies[2].export_state())
    assert full_rng.bit_generator.state == resumed_rng.bit_generator.state
    assert full_pool_rng.bit_generator.state == resumed_pool_rng.bit_generator.state
    assert full.checkpoint.native_mc_state == resumed.checkpoint.native_mc_state
    np.testing.assert_equal(vars(full.checkpoint.response), vars(resumed.checkpoint.response))
