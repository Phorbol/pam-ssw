"""Derive projected center/endpoint force sensitivity from saved logs only."""
import json
from pathlib import Path
import numpy as np
from ase import Atoms
from pamssw.standalone.cluster_frame import ClusterFrame
from pamssw.standalone.softening import FrozenBondSoftening

ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'research/ga_ssw/evidence/hard-c60-gfn2-paper-ls-memory400-single-step'; O=B/'ritz-results-v2'
frozen=json.load(open(O/'frozen-objective-v2.json')); frame=ClusterFrame(Atoms(**frozen['atoms'])); ls=frozen['ls']
soft=FrozenBondSoftening(tuple(ls['numbers']),tuple(map(tuple,ls['cell'])),tuple(ls['pbc']),tuple(map(tuple,ls['pairs'])),tuple(ls['reference_distances']),tuple(ls['strengths']),ls['xi'])
old=[json.loads(x) for x in open(B/'results/paper-seed3/evaluations.jsonl') if 917<=json.loads(x)['call']<=1016]; new=[json.loads(x) for x in open(O/'evaluations.jsonl')]
def old_projected(i):
    a=Atoms(**old[i]['atoms']); return frame.project(np.asarray(old[i]['forces'])+soft.evaluate(a)[1])
dr=np.asarray(json.load(open(O/'result.json'))['direction']); dd=np.asarray(json.load(open(B/'results/paper-seed3/offline-stage10/summary.json'))['direction']); h=1e-4
def one(d,i):
    delta=np.asarray(new[i]['forces'])-old_projected(i); z=delta.ravel()/h; n=d.ravel(); par=float(z@n); return par,float(np.linalg.norm(z-par*n))
d0=np.asarray(new[0]['forces'])-old_projected(0); d1=np.asarray(new[1]['forces'])-old_projected(1)
out={'center_projected_total_delta_over_h':float(np.linalg.norm(d0)/h),'center_parallel_over_h':{'ritz':one(dr,0)[0],'dimer':one(dd,0)[0]},'center_tangent_norm_over_h':{'ritz':one(dr,0)[1],'dimer':one(dd,0)[1]},'endpoint_projected_total_delta_over_h':float(np.linalg.norm(d1)/h),'endpoint_parallel_over_h':{'ritz':one(dr,1)[0],'dimer':one(dd,1)[0]},'endpoint_tangent_norm_over_h':{'ritz':one(dr,1)[1],'dimer':one(dd,1)[1]},'mixed_sensitivity_residual':0.0230064773,'mixed_sensitivity_curvature':-64.5899149464,'new_residual':0.013392524741931065,'new_curvature':-64.58809083337435,'h':h,'extra_PES':0}
json.dump(out,open(O/'sensitivity-diagnostic.json','w'),indent=2); print(json.dumps(out,indent=2))
