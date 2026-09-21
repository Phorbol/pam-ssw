"""Preregistered bounded C4H6/GFN2 SSW-versus-LS CPU experiment.
Paper LS lifecycle plus explicitly native-raw pair parameters: not paper/native
numerical reproduction. No production module files are changed.
"""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import signal
import time
import numpy as np
import networkx as nx
from ase.collections import g2
from ase.io import write
from tblite.ase import TBLite
from pamssw.standalone import paper_reference as paper
from pamssw.standalone import ls_cycle
from pamssw.standalone.surface import ASESurface
from .run_independent_water_ga import serial

OUT=Path('research/ga_ssw/evidence/c4h6-safe-ssw-ls')


def dump(path,value):
    path.write_text(json.dumps(serial(value),indent=2)+'\n')


def calc():
    return TBLite(method='GFN2-xTB',accuracy=.001,verbosity=0)


class Deadline(RuntimeError):pass


def graph(atoms,cutoffs):
    g=nx.Graph();g.add_nodes_from((i,{'Z':int(z)}) for i,z in enumerate(atoms.numbers))
    distances=[]
    for i in range(len(atoms)):
        for j in range(i+1,len(atoms)):
            pair=tuple(sorted((int(atoms.numbers[i]),int(atoms.numbers[j]))))
            d=float(np.linalg.norm(atoms.positions[i]-atoms.positions[j]))
            if d<cutoffs[pair]:g.add_edge(i,j)
            distances.append(dict(i=i,j=j,pair=pair,distance=d,connected=d<cutoffs[pair],threshold_margin=d-cutoffs[pair]))
    return g,distances


def main():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=False)
    def timeout(*_):raise Deadline('600 second total CPU-run wall deadline')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,600)
    atoms=g2['butadiene'].copy();write(OUT/'input.extxyz',atoms)
    raw=json.loads(Path('research/ga_ssw/evidence/native-ls-pair-table/result.json').read_text())
    energy={tuple(sorted(r['pair'])):r['raw_return'] for r in raw['rows'] if r['function']=='bondeneval_'}
    lengths={tuple(sorted(r['pair'])):r['raw_return']+.1 for r in raw['rows'] if r['function']=='bondlenval_'}
    ls=paper.LSSettings(energy,lengths,target_per_atom=.7)
    config=paper.SSWConfig(width=.1,rotation_bias=100.,max_gaussians=25,temperature_K=150.,
        fmax=.01,relax_steps=400,fd_step=1e-4,rotation_hvp=100,rotation_tol=.02,
        rotation_solver='dimer',cluster_frame='direction_only',quench_optimizer='safe-lbfgs-total')
    plan=dict(system='ASE G2 trans-butadiene C4H6',seeds=[3,17],arms=['SSW','LS'],
        backend='tblite 0.7.0 GFN2-xTB CPU single thread accuracy=.001',config=config,ls=ls,
        parameter_mode='paper lifecycle/feedback and xi/initial fraction; native RAW energy lookup and raw length + .1 Angstrom cutoffs; NOT native B/A initialization',
        sources=dict(paper='10.1021/acs.jctc.4c01081 SI sections 7.1/7.2: T150 NG25 ds.1 fmax.01 target.7 eV/atom',
            raw_table='research/ga_ssw/evidence/native-ls-pair-table/result.json',
            numerical='existing standalone C60 rotation settings and Safe-total; relax_steps400 differs from SI MaxOptstep3000'),
        stages='four independent 1-step pilots; restart from same input and seed for equal outer-step full runs',
        full_steps_rule='min(10,max(0,floor((remaining_wall_seconds-60)/sum(four pilot wall_seconds))))',
        wall_limit_seconds=600,validation_reserve_seconds=60,
        budget_comparison='completed certified landings at common cumulative search request budget; pilot cost excluded from discovery curves but included total cost',
        failure_policy='no retuning, no retry; terminal LS update failure retained; deadline persists every E/F and quench certificate, incomplete run not treated as complete trajectory',
        scientific_target='new true stationary molecular structures/connectivity and LS response; no GM or PBE-reproduction claim')
    dump(OUT/'plan.json',plan);(OUT/'runner.py').write_text(Path(__file__).read_text())
    sourcepaths=['pamssw/standalone/paper_reference.py','pamssw/standalone/surface.py','pamssw/standalone/ls_cycle.py','pamssw/standalone/softening.py','pamssw/relax.py']
    dump(OUT/'source_sha256.json',{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sourcepaths})
    runs=[];completed=[];current={};original_quench=paper.quench
    def one(phase,seed,arm,nsteps):
        runout=OUT/f'{phase}-{seed}-{arm}';runout.mkdir()
        efile=(runout/'evaluations.jsonl').open('w');qfile=(runout/'quenches.jsonl').open('w')
        pfile=(runout/'ls-preparation.jsonl').open('w');t0=time.monotonic()
        class LoggedSurface(ASESurface):
            def evaluate(self,a):
                begin=self.requests+1
                try:
                    e,f=super().evaluate(a)
                except Exception as exc:
                    efile.write(json.dumps(dict(request=begin,positions=a.positions.tolist(),error=repr(exc)))+'\n');efile.flush();raise
                efile.write(json.dumps(dict(request=self.requests,positions=a.positions.tolist(),energy=e,forces=f.tolist()))+'\n');efile.flush()
                return e,f
        surface=LoggedSurface(calc());current.update(phase=phase,seed=seed,arm=arm)
        def observed_quench(*args,**kwargs):
            r=original_quench(*args,**kwargs)
            qfile.write(json.dumps(serial(dict(cumulative_requests=surface.requests,result=r)))+'\n');qfile.flush()
            return r
        original_prepare=paper.prepare_ls_step
        def observed_prepare(*args,**kwargs):
            soft=kwargs['softening'];r=original_prepare(*args,**kwargs)
            pfile.write(json.dumps(serial(dict(pairs=soft.pairs,strengths=soft.strengths,
                sum_A_per_atom=sum(soft.strengths)/len(atoms),response=r.energy_response,
                before=r.energy_before,after=r.energy_after,requests=r.evaluation_requests)))+'\n');pfile.flush()
            return r
        paper.quench=observed_quench;ls_cycle.quench=observed_quench;paper.prepare_ls_step=observed_prepare
        row=dict(phase=phase,seed=seed,arm=arm,requested_steps=nsteps)
        try:
            r=paper.run_ssw(atoms,surface,steps=nsteps,config=config,rng=np.random.default_rng(seed),ls=ls if arm=='LS' else None)
            dump(runout/'result.json',r);completed.append((row,runout,r))
            row.update(status=r.status,records=len(r.records),statuses=[x.status for x in r.records],
                minima_records=len(r.minima),responses=[x.energy_response for x in r.records])
        except Exception as exc:
            row.update(status=type(exc).__name__,error=str(exc))
            if isinstance(exc,Deadline):raise
        finally:
            paper.quench=original_quench;ls_cycle.quench=original_quench;paper.prepare_ls_step=original_prepare
            row.update(search_requests=surface.requests,wall_seconds=time.monotonic()-t0)
            runs.append(row);dump(runout/'summary.json',row);dump(OUT/'progress.json',dict(runs=runs,elapsed=time.monotonic()-started))
            efile.close();qfile.close();pfile.close();print(json.dumps(row),flush=True)
    overall='completed';validation=[]
    try:
        for seed in [3,17]:
            for arm in ['SSW','LS']:one('pilot',seed,arm,1)
        pilot_time=sum(r['wall_seconds'] for r in runs)
        count=min(10,max(0,math.floor((600-(time.monotonic()-started)-60)/pilot_time)))
        dump(OUT/'pilot-decision.json',dict(full_steps=count,pilot_wall_seconds=pilot_time,
            pilot_search_requests=sum(r['search_requests'] for r in runs),selection='cost-only preregistered rule'))
        if count:
            for seed in [3,17]:
                for arm in ['SSW','LS']:one('full',seed,arm,count)
        else:overall='pilot_only_budget'
        # Independently initialized calculator for EACH recorded true minimum.
        baseline_graph,_=graph(atoms,lengths)
        for row,runout,result in completed:
            checks=[];cumulative=result.initial.evaluation_requests
            arrivals=[(None,cumulative,result.initial)]
            for record in result.records:
                cumulative+=record.evaluation_requests
                if record.landing is not None and record.landing.converged:
                    arrivals.append((record.index,cumulative,record.landing))
            for index,calls,m in arrivals:
                fresh=ASESurface(calc());e,f=fresh.evaluate(m.atoms);g,distances=graph(m.atoms,lengths)
                checks.append(dict(index=index,cumulative_search_requests=calls,energy=e,stored_energy=m.energy,
                    force_max=float(np.linalg.norm(f,axis=1).max()),force_pass=bool(np.linalg.norm(f,axis=1).max()<=config.fmax),
                    validation_requests=fresh.requests,atoms=serial(m.atoms),distances=distances,
                    components=nx.number_connected_components(g),edges=list(g.edges),
                    initial_connectivity_isomorphic=nx.is_isomorphic(g,baseline_graph,node_match=lambda a,b:a['Z']==b['Z']),
                    carbon_chain_dihedral=m.atoms.get_dihedral(0,1,2,3)))
            dump(runout/'fresh-checks.json',checks)
            validation.append(dict(phase=row['phase'],seed=row['seed'],arm=row['arm'],
                certificates=len(checks),force_pass=sum(x['force_pass'] for x in checks),
                connectivity_changed=sum(not x['initial_connectivity_isomorphic'] for x in checks),
                fragmented=sum(x['components']>1 for x in checks),requests=len(checks)))
    except Deadline as exc:overall='deadline';dump(OUT/'deadline.json',dict(active=current,error=str(exc)))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        dump(OUT/'summary.json',dict(status=overall,runs=runs,validation=validation,
            search_requests=sum(r['search_requests'] for r in runs),validation_requests=sum(v['requests'] for v in validation),
            wall_seconds=time.monotonic()-started,limits='no Hessian or permutation-aware geometry basin certification; GFN2 model not PBE'))

if __name__=='__main__':main()
