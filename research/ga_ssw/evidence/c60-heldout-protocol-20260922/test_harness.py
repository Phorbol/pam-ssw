from types import SimpleNamespace
import numpy as np
from ase import Atoms
import runner
import validator

def minimum(energy, converged=True):
    return SimpleNamespace(energy=energy,converged=converged,evaluation_requests=7,
        optimizer_steps=4,status='failed',surface={'requests':7},
        atoms=Atoms('C60',positions=np.zeros((60,3)),cell=[50]*3,pbc=False))

def test_failed_initial_is_saved(tmp_path):
    failed=minimum(float('nan'),False)
    saved=runner.save_failed_initial(failed,tmp_path)
    assert saved['json'] and saved['traj'] and saved['serialization_fallback']
    import json
    row=json.loads((tmp_path/'failed-initial.json').read_text())
    assert row['energy'] is None and row['evaluation_requests']==7 and row['optimizer_steps']==4 and row['surface']=={'requests':7}

def test_conditional_cage_requires_one_extra(monkeypatch):
    minima=[minimum(2.),minimum(1.),minimum(3.)]
    monkeypatch.setattr(validator,'graph_row',lambda n,p,c,reference_graph=None:{'graph_cage_candidate':p[0,0]>0})
    minima[2].atoms.positions[0,0]=1
    index,rows=validator.conditional_candidate(minima)
    assert index==2 and len(rows)==3

def test_initial_or_best_cage_needs_no_extra(monkeypatch):
    minima=[minimum(2.),minimum(1.)]
    monkeypatch.setattr(validator,'graph_row',lambda n,p,c,reference_graph=None:{'graph_cage_candidate':True})
    index,_=validator.conditional_candidate(minima)
    assert index is None
