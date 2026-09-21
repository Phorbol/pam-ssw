"""Freeze matched C60 one-step component controls; preparation uses zero PES."""
import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
EVID=ROOT/'research/ga_ssw/evidence'
OLD=EVID/'hard-c60-mace-omat-ordinary-single-step'
def main(variant):
    base=EVID/f'hard-c60-mace-omat-{variant}-single-step';base.mkdir(exist_ok=False)
    shutil.copy2(OLD/'input.extxyz',base/'input.extxyz')
    shutil.copytree(OLD/'pamssw',base/'pamssw',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    if variant=='native-cooperative': shutil.copy2(ROOT/'pamssw/standalone/native_local_pair.py',base/'pamssw/standalone/native_local_pair.py')
    code=(OLD/'runner-executed.py').read_text()
    hook='''
# Research-only one-step common-random-number component ablation.
import copy,os,sys
VARIANT = VARIANT_LITERAL
_original_sample = paper.sample_initial_direction

def component_sample(atoms,rng,*,mode):
    if mode != 'paper': return _original_sample(atoms,rng,mode=mode)
    global_mode=rng.normal(size=(len(atoms),3))/np.sqrt(atoms.get_masses()[:,None])
    global_mode/=np.linalg.norm(global_mode)
    eligible=[(i,j) for i in range(len(atoms)) for j in range(i+1,len(atoms))
              if np.linalg.norm(atoms.positions[j]-atoms.positions[i])>3.]
    pair=eligible[int(rng.integers(len(eligible)))];lam=float(rng.uniform(.1,1.5))
    local=np.zeros_like(global_mode);local[pair[0]]=atoms.positions[pair[1]]-atoms.positions[pair[0]];local[pair[1]]=-local[pair[0]]
    record=dict(variant=VARIANT,pair=pair,coefficient=lam,global_direction=global_mode,raw_pair_direction=local.copy(),outer_rng_state=rng.bit_generator.state)
    if VARIANT=='native-cooperative':
        from pamssw.standalone.native_local_pair import native_local_pair
        helper=native_local_pair(atoms,pair,copy.deepcopy(rng))
        local=helper.raw_direction.copy();record['helper']=helper
    local/=np.linalg.norm(local)
    mixed=global_mode+lam*local;mixed/=np.linalg.norm(mixed)
    record.update(unit_local=local,initial_direction=mixed)
    dump(OUT/'direction-component.json',record)
    return mixed
paper.sample_initial_direction=component_sample
'''.replace('VARIANT_LITERAL',repr(variant))
    code=code.replace('def main():',hook+'\ndef main():',1)
    start=code.index("    dump(out/'plan.json'");end=code.index('    shutil.copytree',start)
    code=code[:start]+'''    dump(out/'plan.json',dict(variant=VARIANT,seeds=[3],arms=['ssw'],steps=1,config=config,
        total_EF_cap=2000,search_cap=1998,fresh_reserve=2,wall_cap_seconds=900,
        input_source='same input.extxyz as ordinary paper SSW control',
        backend='MACE-OMAT-small CPU float64',
        changes='local contribution only; same global/pair/lambda draws; native-cooperative helper uses cloned RNG to keep one-step MC stream common',
        basis='docs/research/c60-local-direction-ablation-plan.md',
        scope='component ablation; not complete native sampler, no default change, no efficacy claim from one seed'))
    import ase,torch,mace
    dump(out/'environment.json',dict(python=sys.executable,numpy=np.__version__,ase=ase.__version__,torch=torch.__version__,mace=mace.__version__,ase_file=ase.__file__,numpy_file=np.__file__,env={k:os.environ.get(k) for k in ['PYTHONNOUSERSITE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES']}))
'''+code[end:]
    code=code.replace("limits='two arms, one seed, one attempt; force/network/cage checks not Hessian or C60 GM proof; GFN2 not original NN/DFT; unequal completed work cannot imply efficiency gain'","limits='one component-control arm, seed3, one step; MACE not original NN/DFT; force/network/cage checks not Hessian or C60 GM proof; unequal completed work cannot imply efficiency gain'")
    (base/'runner.py').write_text(code);compile(code,str(base/'runner.py'),'exec')
    source_diffs=[]
    for f in (OLD/'pamssw').rglob('*.py'):
        rel=f.relative_to(OLD);assert f.read_bytes()==(base/rel).read_bytes(),rel
    (base/'preparation.json').write_text(json.dumps(dict(variant=variant,PES_calls=0,input_sha256=hashlib.sha256((base/'input.extxyz').read_bytes()).hexdigest(),runner_sha256=hashlib.sha256((base/'runner.py').read_bytes()).hexdigest(),baseline_package='byte-identical to original ordinary control except optional added native_local_pair.py',plan='docs/research/c60-local-direction-ablation-plan.md'),indent=2)+'\n')
    print(base)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('variant',choices=['unit-local','native-cooperative']);main(p.parse_args().variant)
