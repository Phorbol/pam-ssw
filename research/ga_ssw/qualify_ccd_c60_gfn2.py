"""Predeclared source-first C60/GFN2 input qualification; no surface walking."""
import json, shutil, time, signal, importlib.metadata
from pathlib import Path
import numpy as np
import networkx as nx
from ase.build import molecule
from ase.io import read, write
from pamssw.standalone.surface import ASESurface, quench
from research.ga_ssw.compare_vc_arms import serial

OUT=Path('research/ga_ssw/evidence/ccd-c60-gfn2-qualification')
DATA=Path('research/ga_ssw/datasets/ccd-c60')
class Budget(RuntimeError): pass

def dump(path, x): path.write_text(json.dumps(serial(x), indent=2, allow_nan=False))
def geometry(a):
    d=a.get_all_distances(); graph=nx.Graph();graph.add_nodes_from(range(len(a)))
    graph.add_edges_from(zip(*np.where(np.triu((d<1.8)&(d>0),1))))
    radius=np.linalg.norm(a.positions-a.positions.mean(0),axis=1)
    return dict(atoms=a,edges=list(graph.edges),degrees=sorted(dict(graph.degree()).values()),
        components=nx.number_connected_components(graph),min_distance=float(d[np.triu_indices(len(a),1)].min()),
        radius_min=float(radius.min()),radius_max=float(radius.max()),radius_gyration=float(np.sqrt(np.mean(radius**2))))
def graph_of(g):
    x=nx.Graph();x.add_nodes_from(range(len(g['degrees'])));x.add_edges_from(g['edges']);return x

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    inputs=[('ccd1809asym',read(DATA/'c60-1809asym-source.extxyz')),('ASE_Ih',molecule('C60'))]
    dump(OUT/'plan.json',dict(order=[x[0] for x in inputs],total_requests_cap=900,wall_seconds_cap=480,
        threads=1,backend='tblite GFN2-xTB accuracy .001',fmax=.01,steps=400,optimizer='safe-lbfgs-total',
        graph_cutoff_A=1.8,qualification='fresh E/F and carbon network, no Hessian or GM proof',
        reservation='quench denied at 898 charged requests; fresh allowed up to 900; shared wall cap includes fresh',
        failure='no retries or extensions; every failed request charged; last evaluated geometry retained',
        versions={k:importlib.metadata.version(k) for k in ['ase','tblite','numpy']}))
    shutil.copytree('pamssw',OUT/'source'/'pamssw',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(__file__,OUT/'script.py')
    for path in DATA.glob('*provenance*.json'):shutil.copy2(path,OUT/path.name)
    for name,a in inputs:write(OUT/f'{name}-before.extxyz',a)
    from tblite.ase import TBLite
    start=time.monotonic(); deadline=start+480; used=0; rows=[]
    def alarm(*_):raise Budget('shared wall 480 seconds')
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,480)
    try:
        for name,a in inputs:
            row=dict(name=name,before=geometry(a),status='pending',search_requests=0,fresh_requests=0)
            rows.append(row);last={}; t0=time.monotonic()
            with (OUT/f'{name}-evaluations.jsonl').open('w') as log:
                class Counted(ASESurface):
                    def __init__(self,fresh=False):
                        super().__init__(TBLite(method='GFN2-xTB',accuracy=.001,verbosity=0));self.fresh=fresh
                    def evaluate(self,atoms):
                        nonlocal used
                        if used >= (900 if self.fresh else 898) or time.monotonic()>=deadline:raise Budget('shared EF/wall limit')
                        used+=1;row['fresh_requests' if self.fresh else 'search_requests']+=1
                        last.update(atoms=atoms.copy())
                        try:
                            e,f=super().evaluate(atoms);entry=dict(request=used,fresh=self.fresh,atoms=atoms,energy=e,forces=f)
                            if 'initial_energy' not in row:row.update(initial_energy=e,initial_fmax=float(np.linalg.norm(f,axis=1).max()))
                            last.update(energy=e,forces=f);log.write(json.dumps(serial(entry))+'\n');log.flush();return e,f
                        except Exception as exc:
                            log.write(json.dumps(serial(dict(request=used,fresh=self.fresh,atoms=atoms,error=repr(exc))))+'\n');log.flush();raise
                try:
                    q=quench(a,Counted(),fmax=.01,steps=400,optimizer='safe-lbfgs-total')
                    dump(OUT/f'{name}-quench.json',q);write(OUT/f'{name}-after.extxyz',q.atoms)
                    row.update(status='converged' if q.converged else 'not_converged',quench_energy=q.energy,optimizer_steps=q.optimizer_steps,after=geometry(q.atoms))
                    e,f=Counted(True).evaluate(q.atoms)
                    row.update(fresh_energy=e,fresh_fmax=float(np.linalg.norm(f,axis=1).max()),fresh_pass=bool(np.linalg.norm(f,axis=1).max()<=.01),fresh_energy_error=e-q.energy)
                    row['same_graph_as_input']=nx.is_isomorphic(graph_of(row['before']),graph_of(row['after']))
                    row['same_graph_as_ASE_Ih']=nx.is_isomorphic(graph_of(geometry(inputs[1][1])),graph_of(row['after']))
                except Exception as exc:
                    row.update(status='censored' if used>=898 or time.monotonic()>=deadline or isinstance(exc,Budget) else 'error',error=repr(exc))
                    if last:dump(OUT/f'{name}-last-evaluation.json',last)
                finally:
                    row['wall_seconds']=time.monotonic()-t0
                    dump(OUT/'summary.json',dict(runs=rows,total_requests=used,wall_seconds=time.monotonic()-start))
                    print(json.dumps({k:v for k,v in row.items() if k not in ['before','after']}),flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if len(rows)==2 and all('fresh_energy' in r for r in rows):rows[0]['relative_energy_to_Ih_eV']=rows[0]['fresh_energy']-rows[1]['fresh_energy']
        dump(OUT/'summary.json',dict(runs=rows,total_requests=used,wall_seconds=time.monotonic()-start))
if __name__=='__main__':main()
