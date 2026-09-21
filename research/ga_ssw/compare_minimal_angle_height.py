"""Frozen two-system paired conservative-native-height feasibility comparison."""
from pathlib import Path
from dataclasses import asdict
import json,time,shutil
import numpy as np
from ase.collections import g2
from ase.io import read,write
from ase.calculators.emt import EMT
from pamssw.standalone import ASESurface,SSWConfig,run_ssw
from pamssw.standalone.minimal_angle_height import MinimalAngleHeightPolicy
from research.ga_ssw.compare_vc_arms import serial

OUT=Path('research/ga_ssw/evidence/minimal-angle-height-two-system')
BASELINE=Path('research/ga_ssw/evidence/conservative-native-height-two-system')
POLICY=MinimalAngleHeightPolicy()
CAP=6000;STEPS=2;SECONDS=600

def dump(path,data):path.write_text(json.dumps(serial(data),indent=2,allow_nan=False))

def main():
    from tblite.ase import TBLite
    import importlib.metadata
    out=OUT;out.mkdir(parents=True,exist_ok=False)
    oldplan=json.loads((BASELINE/'plan.json').read_text())
    systems=[(name,read(BASELINE/f'{name}.extxyz'),SSWConfig(**oldplan['configs'][name])) for name in ('c4h6','cu13')]
    # Require identical numerical/PES components; only explicit height branch is new.
    paths=['standalone/surface.py','standalone/direction.py','standalone/dimer.py','standalone/gaussian.py','standalone/cluster_frame.py','standalone/native_rotation.py','relax.py']
    for relative in paths:
        assert (BASELINE/'source'/'pamssw'/relative).read_bytes()==(Path('pamssw')/relative).read_bytes(),relative
    import difflib
    (out/'paper_reference-change.diff').write_text(''.join(difflib.unified_diff((BASELINE/'source/pamssw/standalone/paper_reference.py').read_text().splitlines(True),Path('pamssw/standalone/paper_reference.py').read_text().splitlines(True),fromfile='frozen_baseline',tofile='minimal_angle_branch')))
    plan=dict(seeds=[3,17],arms=['minimal_angle'],steps=STEPS,
        runs=4,per_run_total_cap=CAP,search_cap=CAP-STEPS-1,fresh_reserve=STEPS+1,
        total_cap=4*CAP,total_wall_seconds=SECONDS,threads=1,policy=dict(name="minimal87degree analytic positive height",no_initial_growth_or_cap=True,zero_height="existing nonpositive_height failure"),
        policy_source='analytic minimal positive height for recovered87degree criterion; old8runs preserved, new4 paired bysystem/seed/config/initial',baseline=str(BASELINE),baseline_unchanged_numeric_paths=paths,separate_development_cost=True,
        configs={name:config for name,a,config in systems},
        intervention='height/history only, conservative old-force single count; rotation/quench/MC unchanged',
        metric='all certified observations including rejects, fresh E/F, fragmentation, total cost; no Hessian/global-minimum claim',
        rule='no retuning/retry; censored runs remain in denominator; development systems, not held-out validation',
        versions={p:importlib.metadata.version(p) for p in ('ase','numpy','tblite')})
    dump(out/'plan.json',plan);shutil.copy2(__file__,out/'script.py')
    shutil.copytree('pamssw',out/'source'/'pamssw',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    started=time.monotonic();rows=[]
    for name,atoms,config in systems:
        write(out/f'{name}.extxyz',atoms)
        def calculator():return TBLite(method='GFN2-xTB',accuracy=.001,verbosity=0) if name=='c4h6' else EMT()
        for seed in (3,17):
            for arm in ('minimal_angle',):
                directory=out/f'{name}-{arm}-seed{seed}';directory.mkdir()
                row=dict(system=name,seed=seed,arm=arm,status='running',checks=[]);rows.append(row)
                used=[0];t0=time.monotonic();calls=(directory/'calls.jsonl').open('w')
                class Counted(ASESurface):
                    def __init__(self,fresh=False):super().__init__(calculator());self.fresh=fresh
                    def evaluate(self,a):
                        if used[0]>=(CAP if self.fresh else CAP-STEPS-1) or time.monotonic()-started>=SECONDS:
                            row['censored']=True;raise RuntimeError('declared shared wall/per-run EF cap')
                        used[0]+=1
                        try:
                            e,f=super().evaluate(a)
                            calls.write(json.dumps(serial(dict(request=used[0],fresh=self.fresh,atoms=a,energy=e,forces=f)))+'\n');calls.flush();return e,f
                        except Exception as error:
                            calls.write(json.dumps(dict(request=used[0],fresh=self.fresh,error=repr(error)))+'\n');calls.flush();raise
                surface=Counted()
                try:
                    result=run_ssw(atoms,surface,steps=STEPS,config=config,rng=np.random.default_rng(seed),
                        height_policy=POLICY)
                    dump(directory/'result.json',result)
                    row.update(status=result.status,search_requests=surface.requests,
                        steps=[r.status for r in result.records],accepted=[r.accepted for r in result.records],
                        costs_reconciled=result.evaluation_requests==surface.requests)
                    for i,m in enumerate(result.minima):
                        e,f=Counted(True).evaluate(m.atoms)
                        # Same element-specific C/H cutoffs as prior C4H6 runs.
                        import networkx as nx
                        if name=='c4h6':
                            from pamssw.standalone.native_ls import HC_BOND_LENGTHS
                            from research.ga_ssw.compare_c4h6_safe_ls import graph as molecular_graph
                            cutoff={k:v+.1 for k,v in HC_BOND_LENGTHS.items()}
                            graph,_=molecular_graph(m.atoms,cutoff)
                        else:
                            from ase.neighborlist import neighbor_list
                            cutoff=3.3
                            ii,jj=neighbor_list('ij',m.atoms,cutoff)
                            graph=nx.Graph();graph.add_nodes_from(range(len(m.atoms)));graph.add_edges_from(zip(ii,jj))
                        row['checks'].append(dict(index=i,energy=e,energy_error=e-m.energy,
                            fmax=float(np.linalg.norm(f,axis=1).max()),components=nx.number_connected_components(graph),
                            connectivity_cutoff_A=cutoff,atoms=m.atoms))
                    dump(directory/'fresh.json',row['checks'])
                except Exception as error:row.update(status='error',error=repr(error))
                finally:
                    if row.get('censored'):row['status']='censored'
                    row.update(total_requests=used[0],seconds=time.monotonic()-t0)
                    calls.close();dump(directory/'summary.json',row)
                    dump(out/'summary.json',dict(runs=rows,total_requests=sum(r.get('total_requests',0) for r in rows),seconds=time.monotonic()-started))
                    print(json.dumps({k:v for k,v in row.items() if k!='checks'}),flush=True)

if __name__=='__main__':main()
