"""Synthetic readout-interface fixtures using the saved real C60 geometry; no PES."""
import json
from pathlib import Path
import numpy as np
from ase.io import read, write
import analyze as target

def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+"\n")

def atoms_dict(atoms):
    return dict(numbers=atoms.numbers.tolist(),positions=atoms.positions.tolist(),
                cell=atoms.cell.array.tolist(),pbc=atoms.pbc.tolist())

def plan(root,case='fixture'):
    put(root/'plan.json',{'cases':[case],'seeds':{case:7},'search_cap_per_arm':10,
        'fresh_per_arm':2,'conditional_first_cage_extra_per_arm':1,
        'acceptance':{'energy_reference_eV':0.,'energy_target_margin_eV':.01}})

def valid_fixture(tmp_path,monkeypatch,bad_counts=False):
    root=tmp_path/'synthetic-heldout';plan(root); atoms=read(target.REFERENCE)
    saved=atoms_dict(atoms)
    minima=[{'energy':-1.,'atoms':saved},{'energy':0.,'atoms':saved}]
    result={'initial':{'evaluation_requests':2},'records':[{'evaluation_requests':3}],
            'evaluation_requests':5,'minima':minima}
    fresh=[{'label':'initial','numerical_qualified':True,'energy_eV':-1.},
           {'label':'best','numerical_qualified':True,'energy_eV':-1.},
           {'label':'first_cage','numerical_qualified':True,'energy_eV':0.}]
    for arm in ('baseline_broyden','recovered_rotation'):
        folder=root/arm/'fixture-seed7';folder.mkdir(parents=True)
        summary={'search_requests':5,'denials':0,'fresh_requests':3,'fresh':fresh,
                 'fresh_initial_best_requests':1 if bad_counts else 2,
                 'fresh_conditional_requests':1,'conditional_cage_index':1}
        put(folder/'summary.json',summary);put(folder/'result.json',result)
        write(folder/'first_cage.traj',atoms)
    # The first cage is deliberately saved-force-unqualified. Selection must
    # still use converged+graph and then apply the independent fresh check.
    def fake_arm(root_for_arm,case,seed,new,graph,reference_graph):
        return {'ledger':{'present':True,'ids_contiguous':True,'ids_unique':True,
                 'summary_request_match':True,'summary_denial_match':True,'errors':[]},
                'result':{'minima':[{'index':0,'converged':True,'numerical_qualified':True,
                                     'graph_cage_candidate':False},
                                    {'index':1,'converged':True,'numerical_qualified':False,
                                     'graph_cage_candidate':True}]},
                'search_requests':5,'actual_calculate':4,'fresh_requests':3,'fresh_actual_calculate':3}
    monkeypatch.setattr(target.prior,'arm',fake_arm)
    return target.analyze(root)

def test_real_analyze_selects_first_converged_graph_cage(monkeypatch,tmp_path):
    out=valid_fixture(tmp_path,monkeypatch)
    assert out['errors']==[]
    assert all(r['first_cage_index']==1 and r['cage_independently_qualified'] for r in out['arms'])

def test_fresh_subcount_mismatch_is_an_error(monkeypatch,tmp_path):
    out=valid_fixture(tmp_path,monkeypatch,bad_counts=True)
    assert sum('conditional fresh selection/count mismatch' in e for e in out['errors'])==2
    assert sum('fresh subcounts not closed' in e for e in out['errors'])==2

def test_missing_failed_initial_artifacts_are_errors(monkeypatch,tmp_path):
    root=tmp_path/'synthetic-failed';plan(root)
    for arm in ('baseline_broyden','recovered_rotation'):
        folder=root/arm/'fixture-seed7';folder.mkdir(parents=True)
        put(folder/'summary.json',{'search_requests':3,'denials':0,'fresh_requests':0,'fresh':[],
            'fresh_initial_best_requests':0,'fresh_conditional_requests':0,
            'error':'InitialQuenchError()', 'failed_initial_saved':{'json':True,'traj':True}})
    def fake_arm(*args,**kwargs):
        return {'ledger':{'present':True,'ids_contiguous':True,'ids_unique':True,
                 'summary_request_match':True,'summary_denial_match':True,'errors':[]},
                'result':{'present':False},'search_requests':3,'actual_calculate':2,
                'fresh_requests':0,'fresh_actual_calculate':0}
    monkeypatch.setattr(target.prior,'arm',fake_arm)
    out=target.analyze(root)
    assert sum('missing failed initial certificate/structure' in e for e in out['errors'])==2
