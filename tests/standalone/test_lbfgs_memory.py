import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from pamssw.relax import Relaxer
from pamssw.state import State
from pamssw.standalone.surface import ASESurface,quench
from pamssw.standalone.paper_reference import SSWConfig

@pytest.mark.parametrize('memory',[0,-1,True,1.5,'10'])
def test_invalid_memory_rejected_before_evaluation(memory):
 with pytest.raises(ValueError):Relaxer(lambda *_:pytest.fail('evaluated'),optimizer='safe-lbfgs-total',lbfgs_memory=memory)

def test_unsupported_backend_and_conflict():
 with pytest.raises(ValueError):Relaxer(lambda *_:None,optimizer='ase-lbfgs',lbfgs_memory=400)
 r=Relaxer(lambda *_:pytest.fail('evaluated'),optimizer='safe-lbfgs-total',lbfgs_memory=400)
 with pytest.raises(ValueError):r.relax(State([29],[[0.,0.,0.]]),fmax=.01,maxiter=1,_safe_lbfgs_history_limit=10)
 a=Atoms('Cu2',positions=[[0,0,0],[2.4,0,0]]);s=ASESurface(EMT())
 with pytest.raises(ValueError):quench(a,s,fmax=.01,steps=1,lbfgs_memory=400)
 assert s.requests==0

def test_default_and_explicit10_real_emt_identical():
 a=Atoms('Cu3',positions=[[0,0,0],[2.4,0,0],[1.2,2.2,0]])
 results=[quench(a,ASESurface(EMT()),fmax=.01,steps=30,optimizer='safe-lbfgs-total',lbfgs_memory=m) for m in [None,10]]
 assert np.array_equal(results[0].atoms.positions,results[1].atoms.positions)
 assert results[0].evaluation_requests==results[1].evaluation_requests

@pytest.mark.parametrize('variant',['ordinary','paper','native'])
def test_real_cu_memory_reaches_every_quench(monkeypatch,variant):
 from ase.cluster.icosahedron import Icosahedron
 from pamssw.standalone import paper_reference as paper,ls_cycle
 from pamssw.standalone.ls_native_reference import NativeLSSettings
 config=paper.SSWConfig(width=.2,rotation_bias=.5,max_gaussians=1,temperature_K=0.,fmax=.01,
   relax_steps=150,fd_step=1e-4,rotation_hvp=41,rotation_tol=.02,direction_sampling='global',
   rotation_solver='ritz',cluster_frame='direction_only',quench_optimizer='safe-lbfgs-total',lbfgs_memory=400)
 original=paper.quench;calls=[]
 def observed(*args,**kw):
  calls.append(kw.get('lbfgs_memory'));return original(*args,**kw)
 monkeypatch.setattr(paper,'quench',observed);monkeypatch.setattr(ls_cycle,'quench',observed)
 ls=None if variant=='ordinary' else (paper.LSSettings({(29,29):3.},{(29,29):2.9},target_per_atom=.02) if variant=='paper' else NativeLSSettings({(29,29):3.},{(29,29):2.8},scale=.1))
 surface=ASESurface(EMT());r=paper.run_ssw(Icosahedron('Cu',2),surface,steps=1,config=config,rng=np.random.default_rng(7),ls=ls)
 assert len(calls)>=(3 if variant=='ordinary' else 4)
 assert set(calls)=={400}
 assert surface.requests<500
 assert r.evaluation_requests==surface.requests
