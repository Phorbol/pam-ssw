"""Runner state/cost protocol only; numerical search efficacy is separate."""
from dataclasses import replace
import time
import numpy as np
from ase.build import bulk
from research.ga_ssw import compare_vc_arms as runner

def configs():
 f=runner.SSWConfig(width=.2,rotation_bias=.5,max_gaussians=1,temperature_K=300.,fmax=.01,relax_steps=150,fd_step=1e-4,rotation_hvp=41,rotation_tol=.02,direction_sampling='global',rotation_solver='dimer',cluster_frame='translation_only',quench_optimizer='safe-lbfgs-total')
 v=runner.VCSSWConfig(strain_length=3.6,width=.2,rotation_bias=.5,max_gaussians=1,relax_steps=150,rotation_hvp=41)
 return f,v

def test_deadline_and_request_caps_preserve_censored_records(tmp_path):
 f,v=configs();a=bulk('Cu','fcc',a=3.65,cubic=True)
 for cap,deadline,expected in [(2,None,2),(300,time.monotonic()-1,0)]:
  r=runner.run_arm(a,arm='joint-vc',fixed_config=f,vc_config=v,steps=2,request_cap=cap,seed=7,deadline=deadline,output=tmp_path/f'{cap}.json')
  assert r['status']=='censored' and r['requests']==expected
  assert sum(s['requests'] for s in r['stages'])==r['requests']
  assert (tmp_path/f'{cap}.json').exists()

def test_extracts_landing_even_if_kernel_rejected_it(monkeypatch):
 f,v=configs();a=bulk('Cu','fcc',a=3.65,cubic=True);original=runner.run_ssw
 def rejected(*args,**kwargs):
  result=original(*args,**kwargs)
  return replace(result,current=result.initial.atoms.copy(),records=tuple(replace(r,accepted=False) for r in result.records))
 monkeypatch.setattr(runner,'run_ssw',rejected)
 r=runner.run_arm(a,arm='posterior-cell-quench',fixed_config=f,vc_config=v,steps=1,request_cap=300,seed=7)
 assert r['status']=='completed' and len(r['landings'])==2
 assert not r['records'][0]['kernel'].records[0].accepted
 assert r['landings'][-1]['certificate']['stress_required']
 names=[s['name'] for s in r['stages']]
 assert names.index('0:fixed_proposal')<names.index('0:posterior_cell_quench')
 assert r['records'][0]['status']=='valid_landing'
 assert sum(s['requests'] for s in r['stages'])==r['requests']
 assert sum(r['requests'] for r in r['records'])+r['stages'][0]['requests']==r['requests']
