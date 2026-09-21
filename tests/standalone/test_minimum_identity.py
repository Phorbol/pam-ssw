import numpy as np
from types import SimpleNamespace
from ase import Atoms
from ase.calculators.emt import EMT

from pamssw.standalone.minimum_identity import MinimumIdentityView, update_identity_view
from pamssw.standalone.paper_reference import SSWConfig, run_ssw
from pamssw.standalone.surface import ASESurface
from pamssw.standalone.surface import quench


def _config():
    return SSWConfig(width=.1, rotation_bias=2., max_gaussians=1,
        temperature_K=300., fmax=1e-5, relax_steps=100, fd_step=1e-4,
        rotation_hvp=8, rotation_tol=1e-3, direction_sampling='global')

def _atoms():
    seed=quench(Atoms('Cu2', positions=[[0,0,0],[2.7,0,0]]), ASESurface(EMT()), fmax=1e-6, steps=100)
    return seed.atoms


def test_identity_view_keeps_raw_emt_result_and_merges_only_caller_matches():
    atoms=_atoms()
    matcher=lambda a,b: np.allclose(a.positions, b.positions)
    class Counted:
        def __init__(self): self.surface=ASESurface(EMT()); self.calls=[]
        @property
        def requests(self): return self.surface.requests
        def evaluate(self, a):
            e,f=self.surface.evaluate(a); self.calls.append((a.positions.copy(),e,f.copy())); return e,f
    p=Counted(); q=Counted()
    plain=run_ssw(atoms, p, steps=2, config=_config(), rng=np.random.default_rng(19))
    identified=run_ssw(atoms, q, steps=2, config=_config(), rng=np.random.default_rng(19), structure_matcher=matcher)
    assert len(identified.minima)==len(plain.minima)
    assert p.requests==q.requests and len(p.calls)==len(q.calls)
    for (xp,ep,fp),(xq,eq,fq) in zip(p.calls,q.calls):
        np.testing.assert_allclose(xp,xq); assert ep==eq; np.testing.assert_allclose(fp,fq)
    assert [(r.index,r.status,r.accepted) for r in identified.records] == [(r.index,r.status,r.accepted) for r in plain.records]
    view=identified.identity_view
    assert view.processed_count==len(identified.minima) and view.match_calls > 0
    assert len(view.observation_to_representative)==len(identified.minima)


def test_identity_matcher_exception_keeps_observation_and_maps_none():
    atoms=_atoms()
    def failing(a,b): raise RuntimeError('geometry matcher failure')
    result=run_ssw(atoms, ASESurface(EMT()), steps=1, config=_config(),
                   rng=np.random.default_rng(19), structure_matcher=failing)
    assert len(result.minima)==2 and result.identity_view.observation_to_representative[-1] is None
    assert result.identity_view.failures and result.identity_view.match_calls==1


def test_identity_checkpoint_resume_does_not_rematch_old_observations(tmp_path):
    atoms=_atoms()
    def make_matcher(calls):
        def matcher(a,b):
            calls.append((a.positions.copy(),b.positions.copy()))
            return np.allclose(a.positions,b.positions)
        return matcher

    continuous_calls=[]
    continuous=run_ssw(atoms, ASESurface(EMT()), steps=2, config=_config(),
                        rng=np.random.default_rng(19),
                        structure_matcher=make_matcher(continuous_calls))
    path=tmp_path/'id.pkl'
    split_calls=[]
    first=run_ssw(atoms, ASESurface(EMT()), steps=1, config=_config(),
                  rng=np.random.default_rng(19), structure_matcher=make_matcher(split_calls),
                  checkpoint_path=path)
    from pamssw.standalone.paper_reference import load_ssw_checkpoint
    cp=load_ssw_checkpoint(path); old_calls=len(split_calls)
    resumed_calls=[]
    resumed=run_ssw(atoms, ASESurface(EMT()), steps=1, config=_config(),
                    rng=np.random.default_rng(99), structure_matcher=make_matcher(resumed_calls),
                    checkpoint=cp)
    split_calls.extend(resumed_calls)
    assert len(resumed_calls) > 0
    assert len(split_calls) == len(continuous_calls)
    for (xc, yc), (xs, ys) in zip(continuous_calls, split_calls):
        np.testing.assert_array_equal(xc, xs)
        np.testing.assert_array_equal(yc, ys)
    assert resumed.identity_view.observation_to_representative == continuous.identity_view.observation_to_representative
    assert len(resumed_calls) == resumed.identity_view.match_calls - cp.identity_view.match_calls
    assert old_calls == cp.identity_view.match_calls


def test_identity_tool_keeps_first_representative_and_counts_matcher_failure():
    from pamssw.standalone.minimum_identity import MinimumIdentityView, update_identity_view
    a=_atoms(); translated=a.copy(); translated.positions += [3., -2., 1.]
    different=a.copy(); different.positions[1,0] += .2
    minima=[SimpleNamespace(atoms=x) for x in (a, translated, different)]
    matcher=lambda x,y: np.allclose(x.get_all_distances(mic=False), y.get_all_distances(mic=False))
    view=update_identity_view(MinimumIdentityView([],[],[],0,0), minima, matcher)
    assert view.representative_indices == [0,2]
    assert view.observation_to_representative == [0,0,2]
    def fail(x,y): raise RuntimeError('matcher failure')
    extra=[SimpleNamespace(atoms=a.copy())]
    before=view.match_calls; update_identity_view(view, minima+extra, fail)
    assert view.observation_to_representative[-1] is None and view.match_calls == before+1
