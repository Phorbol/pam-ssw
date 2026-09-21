"""Prepare the approved 2x2 rotation-parameter allocation diagnostic."""
import argparse, hashlib, json, shutil, sys
from dataclasses import asdict
from pathlib import Path
from ase import Atoms
from ase.io import write
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT)); REPLAY=ROOT/'research/ga_ssw/evidence/rotation-parameter-replay-20260918'; MODEL=Path('/home/gengjianrui/.cache/mace/mace-omat-0-small.model')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x): Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--group',choices=['c60','controls'],required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    plan=json.loads((REPLAY/'plan.json').read_text()); states={s['name']:s for s in plan['states']}
    names=['c60_17093','c60_17094'] if a.group=='c60' else ['cu55','water15']; seeds={'c60_17093':17093,'c60_17094':17094,'cu55':17093,'water15':17093}
    a.output.mkdir(parents=True); (a.output/'inputs').mkdir(); (a.output/'source').mkdir()
    shutil.copytree(ROOT/'pamssw',a.output/'source/pamssw',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(ROOT/'research/ga_ssw/run_ssw_allocation_factorial_20260918.py',a.output/'runner.py'); shutil.copy2(ROOT/'research/ga_ssw/run_public_broyden_ssw.py',a.output/'ledger_helpers.py')
    from pamssw.standalone.paper_reference import SSWConfig
    configs={}
    for name in names:
        s=states[name]; atoms=Atoms(numbers=s['numbers'],positions=s['positions'],cell=s['cell'],pbc=s.get('pbc',False)); write(a.output/'inputs'/f'{name}.traj',atoms,format='traj')
        c=SSWConfig(width=.6 if name.startswith('c60') else .1,rotation_bias=1.,max_gaussians=12 if name.startswith('c60') else 25,temperature_K=150.,fmax=.03,relax_steps=1000,fd_step=.001,rotation_hvp=39,rotation_tol=.02,forward_force=.1,direction_sampling='global',rotation_solver='broyden-euclidean',cluster_frame='direction_only',quench_optimizer='safe-lbfgs-total',lbfgs_memory=500,bias_fmax=.1,rotation_exit_policy='force_or_budget')
        configs[name]=asdict(c)
    arms=[{'name':'strict','rotation_tol':.02,'bias_stage_steps':None},{'name':'loose','rotation_tol':2.,'bias_stage_steps':None},{'name':'cap15','rotation_tol':.02,'bias_stage_steps':15},{'name':'loose_cap15','rotation_tol':2.,'bias_stage_steps':15}]
    frozen={'group':a.group,'cases':names,'seeds':seeds,'arms':arms,'steps':100,'per_arm_request_cap':6000,'per_arm_wall_seconds':600,'fresh_request_cap':101,'backend':{'name':'MACE-OMAT-0-small','model':str(MODEL),'model_sha256':sha(MODEL),'device':'cuda','dtype':'float64'},'configs':configs,'scope':'saved-minimum allocation diagnostic; not random-cloud acceptance; no recovered/LS/PAM/height/reconnect','parameter_basis':'rotation-parameter-replay-20260918; loose tol=0.1/(10*0.005), cap15 borrowed native scale but not native reverse-communication stop','input_source_plan':str(REPLAY/'plan.json')}
    dump(a.output/'plan.json',frozen); dump(a.output/'source-manifest.json',{'sha256':{str(p.relative_to(a.output/'source')):sha(p) for p in sorted((a.output/'source').rglob('*.py'))}})
    out_abs=a.output.resolve()
    job=f'''#!/bin/bash\n#SBATCH --partition=4V100\n#SBATCH --qos=rush-1o2gpu\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --gres=gpu:1\n#SBATCH --time=00:45:00\n#SBATCH --job-name=ssw-{a.group}-factorial\n#SBATCH --output={out_abs}/slurm-%j.out\nset -euo pipefail\nexport OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONNOUSERSITE=1\ncd {out_abs}\nPYTHONPATH=source /home/gengjianrui/.conda/envs/mace_env/bin/python runner.py --execute --output .\n'''; (a.output/'job.sh').write_text(job)
    dump(a.output/'prepared.json',{'plan_sha256':sha(a.output/'plan.json'),'source_manifest_sha256':sha(a.output/'source-manifest.json'),'job':str(a.output/'job.sh')})
if __name__=='__main__': main()
