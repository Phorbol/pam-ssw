from types import SimpleNamespace
import numpy as np
from ase import Atoms
import research.ga_ssw.run_fe7c3_ls_comparison as runner


def test_runner_keeps_failed_preparation_cost_without_search_landing(monkeypatch):
    atoms=Atoms('FeC',positions=[[0,0,0],[1.9,0,0]],cell=[6,6,6],pbc=True)
    ev=SimpleNamespace(atoms=atoms,energy=-1.,objective=-1.,forces=np.zeros((2,3)),stress=np.zeros((3,3)))
    records=[dict(requests=2),dict(stage='ls_initialization',status='ls_initialization_failed',requests=0)]
    result=SimpleNamespace(minima=[ev],records=records,requests=2,current=ev,best=ev,status='ls_initialization_failed')
    monkeypatch.setattr(runner,'ls_settings',lambda arm: arm)
    monkeypatch.setattr(runner,'run_vc_ssw',lambda *args,**kwargs: result)
    report=runner.run_ls_arm(atoms,SimpleNamespace(requests=2),arm='ls_filter',joint=None,seed=7)
    assert report['requests']==2 and report['valid_proposals']==0
    assert report['landings'][0]['index']==-1
    assert report['status']=='ls_initialization_failed'
