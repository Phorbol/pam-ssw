"""Exact extraction parity against the unchanged periodic Cu EMT climb."""
import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from pamssw.standalone.atomic_climb import atomic_climb
from pamssw.standalone.paper_reference import run_ssw,SSWConfig
from pamssw.standalone.surface import ASESurface

class StatelessEMTSurface(ASESurface):
    # Identical fresh neighbor construction on every evaluation removes
    # irrelevant differences from the preceding initial-quench neighbor cache.
    def evaluate(self, atoms):
        self.calculator = EMT()
        return super().evaluate(atoms)

@pytest.mark.parametrize('solver',['dimer','ritz'])
def test_existing_cu_climb_geometry_events_and_exact_cost(solver):
    a=bulk('Cu','fcc',a=3.65,cubic=True);a.positions[0]+=[.03,-.02,.01]
    c=SSWConfig(width=.2,rotation_bias=.5,max_gaussians=2,temperature_K=300.,fmax=.01,relax_steps=150,fd_step=1e-4,rotation_hvp=41,rotation_tol=.02,direction_sampling='global',rotation_solver=solver,cluster_frame='translation_only',quench_optimizer='safe-lbfgs-total')
    baseline_surface=StatelessEMTSurface(EMT());baseline=run_ssw(a,baseline_surface,steps=1,config=c,rng=np.random.default_rng(7))
    surface=StatelessEMTSurface(EMT());r=atomic_climb(baseline.initial.atoms,surface,reference_energy=baseline.initial.energy,config=c,rng=np.random.default_rng(7))
    old=baseline.records[0]
    assert len(r.climb) == len(old.climb)
    # Full walker additionally records stage status/cost; the extracted climb
    # must preserve every other field, including the original telemetry.
    for actual, expected in zip(r.climb, old.climb):
        assert actual == {k: v for k, v in expected.items()
                          if k not in {'status', 'requests'}}
        if 'optimizer_telemetry' in actual:
            assert actual['optimizer_telemetry'] == expected['optimizer_telemetry']
            assert actual['termination_reason'] == actual['optimizer_telemetry'].termination_reason
            assert actual['optimizer_telemetry'].converged == (actual['max_force'] <= c.fmax)
    np.testing.assert_array_equal(r.atoms.positions,old.last_atoms.positions)
    np.testing.assert_array_equal(r.atoms.cell.array,baseline.initial.atoms.cell.array)
    np.testing.assert_array_equal(r.initial_direction,old.initial_direction)
    expected=old.evaluation_requests-(0 if old.landing is None else old.landing.evaluation_requests)
    assert r.requests==surface.requests==expected
    assert r.status==old.status

def test_failure_cost_and_no_initial_or_final_quench():
    a=bulk('Cu','fcc',cubic=True)
    c=SSWConfig(width=.2,rotation_bias=.5,max_gaussians=1,temperature_K=300.,fmax=.01,relax_steps=150,fd_step=1e-4,rotation_hvp=2,rotation_tol=1e-12,direction_sampling='global',rotation_solver='dimer',cluster_frame='translation_only',quench_optimizer='safe-lbfgs-total')
    s=ASESurface(EMT());r=atomic_climb(a,s,reference_energy=0.,config=c,rng=np.random.default_rng(7))
    assert r.status=='rotation_failed' and r.requests==2
    np.testing.assert_array_equal(r.atoms.positions,a.positions)
    class Broken:
        requests=0
        def evaluate(self,a):self.requests+=1;raise RuntimeError('backend failure')
    s=Broken();r=atomic_climb(a,s,reference_energy=0.,config=c,rng=np.random.default_rng(7))
    assert r.status=='evaluation_failed' and r.requests==1 and 'backend failure' in r.error


def test_real_cu_checkpoint_boundary_and_interrupted_bias_replay():
    from pamssw.standalone.atomic_climb import resume_atomic_climb
    from dataclasses import replace
    atoms=bulk('Cu','fcc',a=3.65,cubic=True)
    c=SSWConfig(width=.2,rotation_bias=.5,max_gaussians=2,temperature_K=300.,fmax=.01,relax_steps=150,fd_step=1e-4,rotation_hvp=41,rotation_tol=.02,direction_sampling='global',cluster_frame='translation_only',quench_optimizer='safe-lbfgs-total')
    def start(surface, **kw):
        return atomic_climb(atoms,surface,reference_energy=-1e6,config=c,
                            rng=np.random.default_rng(7),**kw)
    full=start(StatelessEMTSurface(EMT()))
    assert full.status=='gaussian_limit' and len(full.checkpoint.climb)==2
    paused=start(StatelessEMTSurface(EMT()),max_completed_gaussians=1)
    cp=paused.checkpoint
    assert paused.status=='checkpoint_boundary' and cp.next_index==1
    assert cp.pending is None and cp.terminal_status is None
    surface=StatelessEMTSurface(EMT())
    resumed=resume_atomic_climb(cp,surface,c)
    assert resumed.climb==full.climb
    np.testing.assert_array_equal(resumed.atoms.positions,full.atoms.positions)
    np.testing.assert_array_equal(resumed.initial_direction,full.initial_direction)
    assert resumed.total_requests==full.requests==paused.requests+resumed.requests
    assert resumed.requests==surface.requests
    # A finished checkpoint costs nothing and does not silently add Gaussians.
    terminal=resume_atomic_climb(resumed.checkpoint,surface,c)
    assert terminal.requests==0 and terminal.climb==full.climb
    with pytest.raises(ValueError,match='config'):
        resume_atomic_climb(cp,surface,replace(c,width=.3))
    assert surface.requests==resumed.requests

    class Interrupted(StatelessEMTSurface):
        def evaluate(self, atoms):
            # First biased-quench request after mode and height evaluations.
            if self.requests==full.climb[0]['rotation_force_requests']+1:
                self.requests+=1
                raise RuntimeError('intentional interrupted biased quench')
            return super().evaluate(atoms)
    interrupted=start(Interrupted(EMT()))
    saved=interrupted.checkpoint
    assert saved.next_index==0 and saved.climb==()
    assert saved.pending['stage']=='biased_quench'
    assert saved.pending['weight']==full.climb[0]['weight']
    assert saved.pending['center']==full.climb[0]['center']
    assert saved.pending['direction']==full.climb[0]['direction']
    assert saved.pending['displaced'].calc is None
    np.testing.assert_array_equal(saved.atoms.positions,atoms.positions)
    replay=resume_atomic_climb(saved,StatelessEMTSurface(EMT()),c)
    assert replay.climb==full.climb
    np.testing.assert_array_equal(replay.atoms.positions,full.atoms.positions)
    assert replay.requests==full.requests
    assert replay.total_requests==interrupted.requests+full.requests
    assert interrupted.status!='gaussian_limit' # Original truncated run stays failed.
