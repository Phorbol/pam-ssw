"""Qualification bookkeeping tests; no MACE model or physical efficacy claim."""
import json
from pathlib import Path
import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT
from pamssw.standalone.vc_geometry import ASEStressSurface
from research.ga_ssw.qualify_material_gate import collect,qualify,PLAN,Limit,key


def endpoint(atoms,domain='fixed',index=-1,run='a',accepted=True):
    return dict(atoms=atoms,domain=domain,index=index,run=run,accepted=accepted,
        pressure=0.,strain_length=3.6,source_energy=0.,source_fmax=.01,source_stress_tol=.001)


def runs():return [dict(name=str(i),source_status='complete') for i in range(8)]


class Zero:
    def __init__(self,cap=1000):self.requests=0;self.seen=[];self.cap=cap
    def evaluate(self,a):
        if self.requests>=self.cap:raise Limit('test global cap')
        self.requests+=1;self.seen.append(key(a))
        return 0.,np.zeros((len(a),3)),np.zeros((3,3))


def test_full_denominator_all_fresh_first_reject_and_domain_dimensions():
    a=Atoms('Cu2',positions=[[0,0,0],[1,0,0]],cell=[5,5,5],pbc=True)
    b=a.copy();b.positions[1,0]=1.1
    endpoints=[endpoint(a),endpoint(a,'variable',run='b'),endpoint(b,index=0,accepted=False)]
    s=Zero();report=qualify(runs(),endpoints,s)
    assert len(report['runs'])==8 and len(report['endpoints'])==3
    assert report['status']=='completed'
    assert report['endpoints'][2]['accepted'] is False
    assert s.seen[:2]==[key(a),key(b)] # No refinement until both unique fresh checks.
    assert report['endpoints'][0]['fresh']['request']==report['endpoints'][1]['fresh']['request']==1
    assert [t['spectra'][0]['dimension'] for t in report['tasks']]==[3,9,3]
    assert report['requests']==2+sum(t['requests'] for t in report['tasks'])==68


def test_global_cap_retains_all_rows_no_fresh_retry_no_refinement():
    a=Atoms('Cu',cell=[5,5,5],pbc=True);b=a.copy();b.positions[0,0]=1
    s=Zero(cap=1)
    report=qualify(runs(),[endpoint(a),endpoint(b,index=0),endpoint(b,index=1,accepted=False)],s)
    assert report['requests']==1 and len(report['runs'])==8 and len(report['endpoints'])==3
    assert [r['fresh_status'] for r in report['endpoints']]==['checked','pending','pending']
    assert report['tasks'][0]['status']=='pending_all_fresh_incomplete'
    assert report['status']=='partial'


def test_hessian_budget_preserves_partial_columns_and_requests():
    a=Atoms('Cu2',positions=[[0,0,0],[1,0,0]],cell=[5,5,5],pbc=True)
    s=Zero(cap=6);report=qualify(runs(),[endpoint(a)],s)
    assert report['requests']==6
    t=report['tasks'][0]
    assert t['requests']==5 and t['status']=='pending_or_failed'
    assert len(t['spectra'][0]['columns'])==1
    assert report['endpoints'][0]['fresh_status']=='checked'


def test_gate_guard_before_models_and_missing_slots(tmp_path):
    commands=[['python','--output',f'/x/run{i}','--arm','fixed','--input','x.extxyz','--model','m.model'] for i in range(8)]
    (tmp_path/'manifest.json').write_text(json.dumps(dict(commands=commands)))
    (tmp_path/'execution.json').write_text(json.dumps(dict(status='running',runs=[])))
    with pytest.raises(ValueError,match='all eight'):collect(tmp_path)
    (tmp_path/'execution.json').write_text(json.dumps(dict(status='completed',runs=[dict(name=f'run{i}',returncode=1) for i in range(8)])))
    rr,ee=collect(tmp_path)
    assert len(rr)==8 and ee==[] and qualify(rr,ee,Zero())['status']=='partial'


def test_real_emt_fixed_cell_stress_diagnostic_separate():
    # A periodic primitive fcc cell is force stationary even away from zero stress.
    a=bulk('Cu',a=3.8)
    s=ASEStressSurface(EMT());report=qualify(runs(),[endpoint(a)],s)
    r=report['endpoints'][0];t=report['tasks'][0]
    assert r['source_tolerance_force_pass'] and not r['source_tolerance_stress_pass']
    assert r['source_domain_certificate_pass']
    assert t['refined_certificate']['certified']
    assert t['spectra'][0]['dimension']==0
    assert s.requests==3


def test_real_emt_variable_cell_refinement_and_two_hessians():
    a=bulk('Cu',a=3.8);s=ASEStressSurface(EMT())
    report=qualify(runs(),[endpoint(a,'variable')],s)
    t=report['tasks'][0]
    assert report['status']=='completed' and s.requests<130
    assert t['refined_certificate']['stress_residual']<=PLAN['stress_tol']
    assert t['geometry_change']['cell_frobenius']>0
    assert [spec['dimension'] for spec in t['spectra']]==[6,6]
    assert all(np.min(spec['eigenvalues'])>0 for spec in t['spectra'])
