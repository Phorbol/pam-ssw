import argparse,hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; FACT=ROOT/'research/ga_ssw/evidence/ssw-allocation-factorial-20260918'; MODEL=Path('/home/gengjianrui/.cache/mace/mace-omat-0-small.model')
sys.path.insert(0,str(ROOT))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x): Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--group',choices=['c60','controls'],required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.output.exists(): raise FileExistsError(a.output)
 base=json.loads((FACT/a.group/'plan.json').read_text()); a.output.mkdir(parents=True); shutil.copytree(FACT/a.group/'source',a.output/'source'); shutil.copytree(FACT/a.group/'inputs',a.output/'inputs'); shutil.copy2(ROOT/'research/ga_ssw/run_pam_gaussian_matched_20260918.py',a.output/'runner.py'); shutil.copy2(FACT/a.group/'ledger_helpers.py',a.output/'ledger_helpers.py')
 from pamssw.standalone.pam_gaussian import PAMCurvatureGaussian
 policy=PAMCurvatureGaussian(); frozen=dict(campaign='pam-gaussian-matched-20260918',group=a.group,cases=base['cases'],seeds=base['seeds'],arms=[{'name':'height_only','policy':dict(target_uphill_energy=policy.target_uphill_energy,target_negative_curvature=policy.target_negative_curvature,min_width=policy.min_width,max_width=policy.max_width,min_weight=policy.min_weight,max_weight=policy.max_weight,curvature_floor=policy.curvature_floor,mode='height_only')},{'name':'height_width','policy':policy.parameters()}],steps=100,per_arm_request_cap=6000,per_arm_wall_seconds=600,fresh_request_cap=101,backend=base['backend'],configs=base['configs'],scope='PAM Gaussian policy isolation from strict factorial baseline; saved-minimum diagnostic, not random-cloud acceptance',parameter_basis='Existing PAMCurvatureGaussian defaults recorded explicitly; provisional experimental values, not optimal; only gaussian_policy differs from strict baseline',excluded=['recovered_direction','LS','native_height','reconnect_distance'])
 for c in frozen['configs'].values(): c['rotation_tol']=.02;c['bias_stage_steps']=None
 dump(a.output/'plan.json',frozen); dump(a.output/'source-manifest.json',{'sha256':{str(p.relative_to(a.output/'source')):sha(p) for p in sorted((a.output/'source').rglob('*.py'))}})
 out=a.output.resolve(); job=f'''#!/bin/bash\n#SBATCH --partition=4V100\n#SBATCH --qos=rush-1o2gpu\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --gres=gpu:1\n#SBATCH --time=00:45:00\n#SBATCH --job-name=pam-{a.group}\n#SBATCH --output={out}/slurm-%j.out\nset -euo pipefail\nexport OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONNOUSERSITE=1\ncd {out}\nPYTHONPATH=source /home/gengjianrui/.conda/envs/mace_env/bin/python runner.py --execute --output .\n''';(a.output/'job.sh').write_text(job);dump(a.output/'prepared.json',{'plan_sha256':sha(a.output/'plan.json'),'source_manifest_sha256':sha(a.output/'source-manifest.json'),'job':str(a.output/'job.sh')})
if __name__=='__main__':main()
