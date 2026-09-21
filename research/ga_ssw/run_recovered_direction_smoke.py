"""Prepare or execute a bounded shared-controller implementation check.

ASE icosahedra are interface fixtures, not literature structures. Runs qualify
returned minima independently and do not rank paper and recovered directions.
Preparation performs no calculator evaluation.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib, json, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
from ase.cluster import Icosahedron
ROOT=Path(__file__).resolve().parents[2]
if (ROOT/'research').is_dir():
    sys.path.insert(0,str(ROOT))
try:
    from ledger_helpers import CountedSurface, dump
except ImportError:
    from research.ga_ssw.run_public_broyden_ssw import CountedSurface, dump
DEFAULT_MODEL=Path('/home/gengjianrui/.cache/mace/mace-omat-0-small.model')
CASES=(('cu13','Cu',2),('cu55','Cu',3),('au13','Au',2))
WATER15=ROOT/'research/ga_ssw/evidence/hco-omat-small-ls-exit-20260912/water15-ls_force-seed11/inputs/structure.extxyz'
WATER15_ARC=Path('/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_examples_run/global_exploration/input-templates/TYPE3-(H2O)15/addition/add.arc')

def deposited_gaussians(result):
    def prepared(event):
        weight=event.get('weight')
        return ('center' in event and 'direction' in event and
                isinstance(weight,(int,float)) and np.isfinite(weight) and weight>0)
    return sum(prepared(event) for record in result.records for event in record.climb)

def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def encoded_ls(settings):
    payload=asdict(settings)
    for name in ('bond_energies','bond_lengths'):
        payload[name]=[{'elements':list(key),'value':value} for key,value in sorted(getattr(settings,name).items())]
    return payload

def decoded_ls(payload):
    from pamssw.standalone.ls_native_reference import NativeLSSettings
    from pamssw.standalone.ls_prequench import LSPrequenchSettings
    value=dict(payload)
    for name in ('bond_energies','bond_lengths'):
        value[name]={tuple(row['elements']):row['value'] for row in value[name]}
    value['prequench']=LSPrequenchSettings(**value['prequench'])
    return NativeLSSettings(**value)

def protocol(model,backend,campaign='smoke'):
    from pamssw.standalone.paper_reference import SSWConfig
    from pamssw.standalone.recovered_direction import RecoveredDirectionSettings
    material=campaign=='ls-materials'
    config=SSWConfig(width=.1,rotation_bias=1.,max_gaussians=25 if material else 4,temperature_K=150.,fmax=.03,bias_fmax=.1,
        relax_steps=300 if material else 150,fd_step=.001,rotation_hvp=39,rotation_tol=.02,direction_sampling='global',
        rotation_solver='broyden-euclidean',cluster_frame='direction_only',rotation_exit_policy='force_or_budget',
        quench_optimizer='safe-lbfgs-total' if material else 'ase-lbfgs-linesearch',lbfgs_memory=10 if material else None)
    recovered=RecoveredDirectionSettings(ratio_local=50,local_probability=.5,group_threshold=.5,
        pre_rotmax=2,rotmax=8,pre_ftol=.01,ftol=.01,metric='euclidean',max_force_calls=40)
    plan=dict(status='prepared_not_executed',campaign=campaign,purpose='shared-controller implementation and numerical qualification',
        scientific_scope='No performance ranking; ASE icosahedra are not paper initial structures.',
        cases=[x[0] for x in CASES],arms=['paper','recovered'],seed=19,steps=2,per_arm_request_cap=2000,
        per_arm_wall_seconds=120,backend=dict(name=('MACE-OMAT-0-small' if backend=='mace' else 'ASE EMT'),
        kind=backend,model=(str(model) if backend=='mace' else None),model_sha256=(sha256(model) if backend=='mace' else None),
        device=('cuda' if backend=='mace' else 'cpu'),dtype='float64'),config=asdict(config),recovered=asdict(recovered),
        qualification='fresh calculator evaluation of every returned minimum; rejected landings retained by run_ssw',
        expected='Both explicit rules traverse the shared driver or retain a bounded failure.',
        parameter_basis='Existing EMT smoke settings; bounded interface check, not tuned native defaults.',
        code_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    if material:
        plan.pop('seed')  # The campaign uses the explicit seeds list below.
        from pamssw.standalone.ls_native_reference import NativeLSSettings
        from pamssw.standalone.ls_prequench import LSPrequenchSettings
        from pamssw.standalone.native_ls import HCO_BOND_ENERGIES,HCO_BOND_LENGTHS
        pre=LSPrequenchSettings(.1,50,'force_or_step_limit')
        water_ls=NativeLSSettings(HCO_BOND_ENERGIES,HCO_BOND_LENGTHS,target_mev_per_atom=20.,prequench=pre)
        copper_ls=NativeLSSettings({(29,29):3.6298000812530518},{(29,29):1.875},target_mev_per_atom=20.,prequench=pre)
        plan.update(cases=['cu55','water15'],arms=['paper','recovered','recovered_native_ls'],seeds=[11,29],steps=20,
            per_arm_request_cap=6000,per_arm_wall_seconds=240,fresh_cap=21,fresh_request_cap=21,
            scientific_scope='Component integration and NativeLS nonzero/update acceptance only; no method ranking.',
            parameter_basis='Frozen shared-controller smoke direction settings plus existing NativeLS Safe-total contract; no tuning.',
            input_sources={'cu55':{'kind':'ASE Icosahedron Cu, 3 shells'},'water15':{'path':str(WATER15),'sha256':sha256(WATER15),
                'upstream_arc':str(WATER15_ARC),'upstream_arc_sha256':sha256(WATER15_ARC),'frame':0,'pbc':False,'cell':'zero'}},
            native_ls={'water15':encoded_ls(water_ls),'cu55':encoded_ls(copper_ls)},
            generator_domain='c1 all-near support requires selected-atom radius <=12 A; failures retained without fallback')
    return config,recovered,plan

def prepare(out,model,backend,campaign='smoke'):
    if out.exists(): raise FileExistsError(out)
    if backend=='mace' and not model.is_file(): raise FileNotFoundError(model)
    out.mkdir(parents=True)
    shutil.copytree(ROOT/'pamssw',out/'source/pamssw',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(__file__,out/'runner.py'); shutil.copy2(ROOT/'research/ga_ssw/run_public_broyden_ssw.py',out/'ledger_helpers.py')
    tests=out/'tests'; tests.mkdir()
    for name in ('test_paper_reference.py','test_recovered_direction.py','test_recovered_cbd.py','test_recovered_direction_driver.py',
                 'test_native_ls_driver.py','test_ls_prequench_settings.py'):
        shutil.copy2(ROOT/'tests/standalone'/name,tests/name)
    sys.path.insert(0,str(out/'source')); _,_,plan=protocol(model,backend,campaign); dump(out/'plan.json',plan)
    inputs=out/'inputs'; inputs.mkdir(); from ase.io import write
    if campaign=='smoke':
        prepared_inputs=((name,Icosahedron(element,shells)) for name,element,shells in CASES)
    else:
        from ase.io import read
        prepared_inputs=(('cu55',Icosahedron('Cu',3)),('water15',read(WATER15)))
    for name,atoms in prepared_inputs:
        write(inputs/f'{name}.extxyz',atoms); dump(inputs/f'{name}.json',atoms)
    manifest={str(p.relative_to(out/'source')):sha256(p) for p in sorted((out/'source').rglob('*.py'))}
    dump(out/'source-manifest.json',dict(sha256=manifest))
    job=f'''#!/bin/bash
#SBATCH --partition=4V100
#SBATCH --qos=rush-1o2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time={'01:00:00' if campaign=='ls-materials' else '00:15:00'}
#SBATCH --job-name={'recovered-ls-materials' if campaign=='ls-materials' else 'recovered-dir-e2e'}
#SBATCH --output={out}/slurm-%j.out
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1
cd {out}
PYTHONPATH={out}/source /home/gengjianrui/.conda/envs/mace_env/bin/python -m pytest tests -q
/home/gengjianrui/.conda/envs/mace_env/bin/python {out}/runner.py --execute --campaign {campaign} --backend {backend} --output {out}
'''
    (out/'job.sh').write_text(job)
    dump(out/'prepared.json',dict(plan_sha256=sha256(out/'plan.json'),source_manifest_sha256=sha256(out/'source-manifest.json'),job=str(out/'job.sh')))

def calculator(backend,model):
    if backend=='emt':
        from ase.calculators.emt import EMT
        return EMT()
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=str(model),device='cuda',default_dtype='float64',enable_cueq=False,enable_oeq=False)

def execute(out,backend,model_override=None):
    plan=json.loads((out/'plan.json').read_text())
    if backend!=plan['backend']['kind']: raise ValueError('execution backend differs from prepared protocol')
    model=Path(model_override or plan['backend']['model']) if backend=='mace' else None
    if backend=='mace' and sha256(model)!=plan['backend']['model_sha256']: raise ValueError('model hash differs from prepared protocol')
    sys.path.insert(0,str(out/'source')); import pamssw
    if not Path(pamssw.__file__).resolve().is_relative_to(out/'source'): raise RuntimeError('execution must import frozen source package')
    from ase.io import read
    from pamssw.standalone.paper_reference import SSWConfig,run_ssw
    from pamssw.standalone.recovered_direction import RecoveredDirectionSettings
    from pamssw.standalone.surface import ASESurface
    config=SSWConfig(**plan['config']); recovered=RecoveredDirectionSettings(**plan['recovered']); rows=[]
    seeds=plan.get('seeds',[plan.get('seed')])
    for case in plan['cases']:
      atoms=read(out/'inputs'/f'{case}.extxyz')
      for seed in seeds:
       for arm in plan['arms']:
        name=f'{case}-{arm}-seed{seed}'
        if name in plan.get('excluded_runs', {}): continue
        directory=out/name; directory.mkdir(exist_ok=False)
        arm_started=time.monotonic(); init_started=time.monotonic(); search_calc=calculator(backend,model)
        search_initialization_seconds=time.monotonic()-init_started
        init_started=time.monotonic(); validation_calc=calculator(backend,model)
        validation_initialization_seconds=time.monotonic()-init_started
        surface=CountedSurface(search_calc,directory/'requests.jsonl',cap=plan['per_arm_request_cap'],wall=plan['per_arm_wall_seconds'])
        row=dict(case=case,arm=arm,seed=seed,backend=backend,execution='started',numerical='unassessed',scientific='implementation check only')
        checks=[]; fresh_requests=0
        try:
            kwargs=dict(steps=plan['steps'],config=config,rng=np.random.default_rng(seed),
                        recovered_direction=recovered if arm!='paper' else None)
            ls_settings=None
            if arm=='recovered_native_ls':
                ls_settings=decoded_ls(plan['native_ls'][case]); kwargs['ls']=ls_settings
            result=run_ssw(atoms,surface,**kwargs)
            dump(directory/'result.json',result)
            fresh=ASESurface(validation_calc)
            for index,minimum in enumerate(result.minima[:plan.get('fresh_cap',len(result.minima))]):
              try:
                energy,forces=fresh.evaluate(minimum.atoms)
                fmax=float(np.linalg.norm(forces,axis=1).max()); checks.append(dict(index=index,energy=energy,energy_error=energy-minimum.energy,
                    fmax=fmax,qualified=fmax<=config.fmax,composition_match=bool(np.array_equal(minimum.atoms.numbers,atoms.numbers)),
                    pbc_unchanged=bool(np.array_equal(minimum.atoms.pbc,atoms.pbc)),fixed_cell=bool(np.array_equal(minimum.atoms.cell.array,atoms.cell.array))))
              except Exception as error:
                checks.append(dict(index=index,error=repr(error),qualified=False,
                    composition_match=bool(np.array_equal(minimum.atoms.numbers,atoms.numbers)),
                    pbc_unchanged=bool(np.array_equal(minimum.atoms.pbc,atoms.pbc)),
                    fixed_cell=bool(np.array_equal(minimum.atoms.cell.array,atoms.cell.array))))
            fresh_requests=fresh.requests
            dump(directory/'qualification.json',checks)
            ls_diagnostics=None
            if ls_settings is not None:
                from pamssw.standalone.ls_native_reference import NativeLSRuntime
                replay=NativeLSRuntime(result.initial.atoms,ls_settings)
                updates=[r.ls_update for r in result.records if r.ls_update is not None]
                ls_diagnostics={'source':'derived initialization replay; zero PES requests',
                    'initial_bond_count':replay.state.old_bond_count,'initial_table':replay.state.table,
                    'initial_pair_count':len(replay.frozen.pairs),'initial_strengths':list(replay.frozen.strengths),
                    'initial_strength_sum':float(sum(replay.frozen.strengths)),
                    'initial_nonzero':bool(replay.frozen.pairs and any(x != 0. for x in replay.frozen.strengths)),
                    'updates':updates,'update_count':len(updates),
                    'real_response_update':any(
                        'normal_update' in update.get('actions', ()) and
                        update.get('observed_response_mev_per_atom') is not None and
                        np.isfinite(update['observed_response_mev_per_atom']) for update in updates),
                    'preparations':[r.ls_preparation for r in result.records if r.ls_preparation is not None],
                    'cycle_scope':'20 steps can cover normal response updates; does not cover post-presteps save_zero/restore cycle'}
                dump(directory/'native-ls-diagnostics.json',ls_diagnostics)
            row.update(execution=result.status,numerical='qualified minima' if checks and all(c['qualified'] and c['fixed_cell'] and c['composition_match'] and c['pbc_unchanged'] for c in checks) else 'failed qualification',
                minima=len(result.minima),fresh_denominator=min(len(result.minima),plan.get('fresh_cap',len(result.minima))),
                outer_statuses=dict(Counter(r.status for r in result.records)),gaussians=deposited_gaussians(result),
                native_ls=ls_diagnostics)
        except Exception as error: row.update(execution='exception',error=repr(error))
        row.update(search_requests=surface.requests,fresh_requests=fresh_requests,denials=surface.denials,boundary=surface.boundary,
                   search_initialization_seconds=search_initialization_seconds,validation_initialization_seconds=validation_initialization_seconds,
                   search_seconds=time.monotonic()-surface.started,total_execution_seconds=time.monotonic()-arm_started,
                   physical_qualification='not independently validated',checks=checks)
        dump(directory/'summary.json',row); rows.append(row); dump(out/'summary.json',rows); print(case,arm,row['execution'],surface.requests,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__); action=p.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare',action='store_true'); action.add_argument('--execute',action='store_true')
    p.add_argument('--output',type=Path,required=True); p.add_argument('--backend',choices=('emt','mace'),default='mace'); p.add_argument('--model',type=Path)
    p.add_argument('--campaign',choices=('smoke','ls-materials'),default='smoke')
    a=p.parse_args(); out=a.output.resolve(); model=(a.model or DEFAULT_MODEL).resolve()
    prepare(out,model,a.backend,a.campaign) if a.prepare else execute(out,a.backend,a.model.resolve() if a.model else None)
if __name__=='__main__': main()
