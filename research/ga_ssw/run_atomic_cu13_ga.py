"""Predeclared Cu13/EMT TYPE0 full-loop diagnostic, not an efficiency benchmark."""
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
from ase import Atoms
from ase.io import write
from ase.calculators.emt import EMT
from pamssw.standalone import ASESurface,SSWConfig,PaperGAConfig,run_ga_ssw
from pamssw.standalone.legacy_descriptor import cluster_descriptor
from .compare_dimer_ritz_cu13_escape import serial


def main():
    out=Path('research/ga_ssw/evidence/atomic-cu13-ga');out.mkdir(exist_ok=False)
    (out/'script.py').write_text(Path(__file__).read_text())
    sources=out/'source';sources.mkdir()
    for name in ['atomic_ga','paper_ga','paper_reference','surface','ga_operators','legacy_descriptor','population']:
        (sources/f'{name}.py').write_text(Path(f'pamssw/standalone/{name}.py').read_text())
    base=Path('research/ga_ssw/evidence/cu13-safe-total/strict-validation')
    rows=json.loads((base/'17-dimer.json').read_text());initial=[];identities=[]
    for r in rows:
        if r['qualified'] and r['fingerprint_group'] not in identities:
            initial.append(Atoms('Cu13',positions=r['positions']));identities.append(r['fingerprint_group'])
        if len(initial)==3:break
    assert len(initial)==3
    write(out/'initial.extxyz',initial)
    # ElementPara.getAtomR(29)=2.259876/2; BasicInfo bond is radius sum.
    bonds={(29,29):2.259876};neighbor=2.
    refs=[cluster_descriptor(a.numbers,a.positions,bonds,neighbor) for a in initial]
    ssw=SSWConfig(width=.2,rotation_bias=100.,max_gaussians=14,temperature_K=300.,
        fmax=.01,relax_steps=200,fd_step=1e-4,rotation_hvp=100,rotation_tol=.02,
        rotation_solver='dimer',cluster_frame='direction_only',quench_optimizer='safe-lbfgs-total')
    ga=PaperGAConfig(quick_steps=1,generations=1,generation_steps=1,fine_steps=1,
        ga_candidates=8,regions=1,fine_regions=1,quench_fmax=.01,quench_steps=200,
        proposal_max_batches=2,proposal_max_cut_attempts=1000,proposal_max_pair_attempts=10000,
        partition_max_draws=10000,projection_tolerance=1e-4,energy_window=10.,proposal_type=0,
        proposal_max_insertion_attempts=10000)
    plan=dict(input_source=str(base/'17-dimer.json'),input_selection='first three distinct certified fingerprint groups in source order',
        groups=identities,backend='ASE EMT',seeds=[3,17],ga=asdict(ga),ssw=asdict(ssw),
        bond_lengths=[[29,29,2.259876]],neighbor_range=neighbor,descriptor_weights=[.3,.2,.2,.1,.1,.1],
        proposal_bond_limits={},limits='bounded full-loop diagnostic on already-known minima; no GM speed or native trajectory claim; legacy descriptor is not a structural identity certificate',
        mutation='corrected collision acceptance; no original origin fallback',budget='single CPU, external 300 second cap')
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    summaries=[]
    for seed in plan['seeds']:
        surface=ASESurface(EMT());start=time.monotonic()
        result=run_ga_ssw(initial,surface,groups=None,references=refs,descriptor_bonds=bonds,
            descriptor_weights=plan['descriptor_weights'],neighbor_range=neighbor,proposal_bond_limits={},
            config=ga,ssw_config=ssw,rng=np.random.default_rng(seed))
        (out/f'{seed}.json').write_text(json.dumps(serial(result),indent=2)+'\n')
        checks=[]
        for o in result.observations:
            a=o.result.atoms;e,f=ASESurface(EMT()).evaluate(a)
            checks.append(dict(id=o.id,phase=o.phase,eligible=o.eligible_for_archive,
                energy=e,max_force=float(np.linalg.norm(f,axis=1).max()),
                force_passed=bool(np.linalg.norm(f,axis=1).max()<=ga.quench_fmax),
                composition_passed=bool(len(a)==13 and np.all(a.numbers==29)),
                min_distance=float(a.get_all_distances()[np.triu_indices(13,1)].min())))
        (out/f'{seed}-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
        row=dict(seed=seed,status=result.status,search_requests=surface.requests,validation_requests=len(checks),
            observations=len(result.observations),archive=len(result.archive),failures=len(result.failures),
            stages=[dict(phase=s.phase,status=s.status,requests=s.evaluation_requests) for s in result.stages],
            eligible=sum(c['eligible'] for c in checks),eligible_fresh_passed=sum(c['eligible'] and c['force_passed'] and c['composition_passed'] for c in checks),
            wall_seconds=time.monotonic()-start)
        summaries.append(row);print(json.dumps(row),flush=True)
        (out/'summary.json').write_text(json.dumps(dict(runs=summaries,complete=len(summaries)==2),indent=2)+'\n')
if __name__=='__main__':main()
