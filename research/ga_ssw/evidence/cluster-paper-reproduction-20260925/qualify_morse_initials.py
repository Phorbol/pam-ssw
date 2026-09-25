"""Qualify a prospective Morse input ensemble, not global-search performance."""
from pathlib import Path
import importlib.util,json,time,sys
import numpy as np
import networkx as nx
from ase.calculators.morse import MorsePotential
from ase.io import write
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT))
from pamssw.standalone.surface import ASESurface,quench
spec=importlib.util.spec_from_file_location('pilot',HERE/'run_lj_pilot.py')
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)
OUT=HERE/'morse-initials'
OUT.mkdir(exist_ok=False)
rows=[]
for n in (29,80):
 for seed in (25092501,25092502):
  start=time.monotonic()
  class Bounded(ASESurface):
   def evaluate(self,atoms):
    if self.requests>=1500 or time.monotonic()-start>=90:
     raise RuntimeError('initial qualification cap')
    return super().evaluate(atoms)
  atoms,_=pilot.uniform_volume_cluster(n,seed)
  folder=OUT/f'm{n}-{seed}';folder.mkdir()
  write(folder/'initial.extxyz',atoms)
  surface=Bounded(MorsePotential(epsilon=1.,rho0=14.,r0=2.7,rcut1=100.,rcut2=101.))
  row=dict(n=n,seed=seed,scope='random-ensemble qualification only',r0_A=2.7,rho0=14.,radius_A=5.5*2.7)
  try:
   result=quench(atoms,surface,fmax=.01,steps=1000,optimizer='safe-lbfgs-total',lbfgs_memory=500)
   final=result.atoms
   write(folder/'final.extxyz',final)
   d=final.get_all_distances()
   components={}
   for cutoff in (1.3,1.5,2.):
    g=nx.Graph();g.add_nodes_from(range(n))
    g.add_edges_from(zip(*np.where(np.triu((d>0)&(d<cutoff*2.7),1))))
    components[str(cutoff)]=sorted((len(c) for c in nx.connected_components(g)),reverse=True)
   fresh=ASESurface(MorsePotential(epsilon=1.,rho0=14.,r0=2.7,rcut1=100.,rcut2=101.))
   e,f=fresh.evaluate(final)
   row.update(status=('force_converged' if result.converged else 'not_converged'),converged=bool(result.converged),energy=e,
     fmax=float(np.linalg.norm(f,axis=1).max()),components_by_cutoff_r0=components,
     diameter_A=float(d.max()),fresh_requests=fresh.requests)
  except Exception as exc: row.update(error=repr(exc))
  row.update(search_requests=surface.requests,wall_seconds=time.monotonic()-start)
  rows.append(row)
  (OUT/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
