"""Bounded real-system constraint lifecycle audit, not search efficiency."""
from pathlib import Path
from dataclasses import asdict, is_dataclass
import argparse, json, time, shutil, subprocess
import numpy as np
from ase import Atoms
from ase.io import read
from ase.constraints import FixAtoms, Hookean
from ase.calculators.emt import EMT
from pamssw.standalone import ASESurface, LSSettings, NativeLSSettings
from pamssw.standalone.native_ls import HCO_BOND_ENERGIES, HCO_BOND_LENGTHS
from pamssw.standalone.constrained_reference import ConstrainedSSWConfig, run_constrained_ssw
from pamssw.standalone.ase_constraints import normalize_constraints
from pamssw.standalone.pam_gaussian import PAMCurvatureGaussian


def serial(x):
    if isinstance(x, Atoms):
        return dict(numbers=x.numbers.tolist(), positions=x.positions.tolist(),
                    cell=x.cell.array.tolist(), pbc=x.pbc.tolist(),
                    constraints=[c.todict() for c in x.constraints])
    if is_dataclass(x): return {k:serial(v) for k,v in vars(x).items()}
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,dict): return {str(k):serial(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [serial(v) for v in x]
    return x


def dump(p,x): p.write_text(json.dumps(serial(x),indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--gaussian-policy', choices=['reference','pam'],default='reference')
    args=parser.parse_args()
    policy=PAMCurvatureGaussian() if args.gaussian_policy=='pam' else None
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    src=Path('research/ga_ssw/evidence')
    molecule=read(src/'hco-fixed-cell-20260912/water_dimer.extxyz')
    oxygen=np.flatnonzero(molecule.numbers==8)
    molecule.set_constraint(Hookean(int(oxygen[0]),int(oxygen[1]),k=1.,rt=2.7))
    d=json.loads((src/'constrained-cu111-emt/plan.json').read_text())
    slab=Atoms(symbols=d['symbols'],positions=d['positions'],cell=d['cell'],pbc=d['pbc'])
    # A weak laboratory-frame ceiling on the adatom: exercises a plane restraint
    # and reactions on a constrained substrate; not a material force field.
    slab.set_constraint([Hookean(12,(0.,0.,1.,-13.9),k=1.),FixAtoms(indices=d['fixed_indices'])])
    cases={'water_dimer':molecule,'cu111':slab}
    cfg=ConstrainedSSWConfig(width=.1,rotation_bias=100.,max_gaussians=2,
        gradient_tol=.1,fmax=.03,relax_steps=300,rotation_hvp=100,rotation_solver='dimer')
    ls_settings={
        'water_dimer':(LSSettings(HCO_BOND_ENERGIES,{k:v+.1 for k,v in HCO_BOND_LENGTHS.items()},target_per_atom=.02),
                       NativeLSSettings(HCO_BOND_ENERGIES,HCO_BOND_LENGTHS,target_mev_per_atom=20.)),
        'cu111':(LSSettings({(29,29):1.},{(29,29):2.9},target_per_atom=.02),
                 NativeLSSettings({(29,29):1.},{(29,29):2.9},target_mev_per_atom=20.))}
    plan=dict(purpose='FixAtoms/Hookean objective consistency through SSW and paper/native LS; no efficiency claim',
        seed=11,steps=1,variants=['ssw','ls_paper','ls_native'],config=asdict(cfg),
        gaussian_policy=None if policy is None else policy.parameters(),
        search_cap=3998,fresh_cap=2,search_seconds=90,inputs=cases,ls_settings=ls_settings,
        sources=['saved ASE S22 water dimer from hco-fixed-cell-20260912', 'saved Cu111 EMT fixture from constrained-cu111-emt/plan.json'],
        parameter_basis='Inner active-gradient L2 tolerance .1 eV/A and outer max-atom .03 eV/A follow user requested loose ranges; width .1 and rotation100 inherited diagnostics. Hookean k1 eV/A^2, O-O rt2.7A and plane z13.9A explicitly exercise persistent restraints, not optimized search parameters. HCO LS tables recovered native; Cu pair D1eV r2.9A diagnostic only. Optional PAM uses existing unmodified height_width defaults: target energy .6 eV, negative curvature .05 eV/A^2, width [.15,1.5] A, weight [0,10] eV, floor1e-4 eV/A^2. No new fitting.',
        backend='water: tblite0.7 GFN2-xTB accuracy .001; Cu: ASE EMT; CPU one thread',
        qualification='fresh objective energy/force match, active fmax<=.03, exact fixed positions/cell, native constraints retained; no TS/Hessian or global efficiency claim')
    dump(out/'plan.json',plan)
    if not args.execute:return
    shutil.copytree('pamssw',out/'source/pamssw',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(__file__,out/'runner.py')
    dump(out/'provenance.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),dirty_source_snapshot='source/pamssw'))
    from tblite.ase import TBLite
    rows=[]
    for name,atoms in cases.items():
      for i,variant in enumerate(plan['variants']):
        directory=out/f'{name}-{variant}';directory.mkdir();start=time.monotonic()
        spec=normalize_constraints(atoms);fixed=np.array(spec.fixed_indices,dtype=int)
        active=np.array([j for j in range(len(atoms)) if j not in spec.fixed_indices],dtype=int)
        def calculator(): return EMT() if name=='cu111' else TBLite(method='GFN2-xTB',accuracy=.001,verbosity=0)
        class Counted(ASESurface):
            exhausted=False
            def evaluate(self,a):
                if self.requests>=3998 or time.monotonic()-start>=90:
                    self.exhausted=True;raise RuntimeError('bounded search budget exhausted')
                try:
                    e,f=super().evaluate(a)
                except Exception as error:
                    with (directory/'evaluations.jsonl').open('a') as h:h.write(json.dumps(dict(request=self.requests,error=repr(error)))+'\n')
                    raise
                with (directory/'evaluations.jsonl').open('a') as h:h.write(json.dumps(serial(dict(request=self.requests,atoms=a,energy=e,forces=f)))+'\n')
                return e,f
        surface=Counted(calculator());result=None
        row=dict(case=name,variant=variant,checks=[],status='running')
        try:
            result=run_constrained_ssw(atoms,surface,steps=1,config=cfg,rng=np.random.default_rng(11),
                ls=None if i==0 else ls_settings[name][i-1],
                **({} if policy is None else dict(gaussian_policy=policy)))
            dump(directory/'result.json',result)
            row.update(status=result.status,records=[r['status'] for r in result.records],
                       ledger_consistent=result.requests==surface.requests==sum(r['requests'] for r in result.records))
        except Exception as error:row.update(status='exception',error=repr(error))
        row['search_requests']=surface.requests
        fresh=0
        if result is not None:
          for minimum in result.minima:
            fresh+=1
            if fresh>2:raise RuntimeError('unexpected extra minima beyond frozen validation budget')
            try:
                a=minimum.atoms.copy();a.calc=calculator()
                physical_e=a.get_potential_energy(apply_constraint=False)
                physical_f=a.get_forces(apply_constraint=False)
                e=a.get_potential_energy();f=a.get_forces()
                force=float(np.linalg.norm(f[active],axis=1).max())
                delta=float(e-minimum.energy)
                restored=normalize_constraints(a)==spec
                invariant=np.array_equal(a.positions[fixed],atoms.positions[fixed]) and np.array_equal(a.cell.array,atoms.cell.array)
                check=dict(energy=e,physical_energy=physical_e,hookean_energy=e-physical_e,
                    physical_forces=physical_f,forces=f,active_fmax=force,energy_error=delta,
                    constraint_metadata=restored,fixed_cell_exact=invariant,
                    qualified=bool(force<=cfg.fmax and abs(delta)<1e-7 and restored and invariant))
                row['checks'].append(check)
            except Exception as error:row['checks'].append(dict(error=repr(error),qualified=False))
        row.update(fresh_requests=fresh,total_requests=surface.requests+fresh,wall_seconds=time.monotonic()-start)
        dump(directory/'summary.json',row);rows.append(row);dump(out/'summary.json',rows)
        print(name,variant,row['status'],row['search_requests'],flush=True)

if __name__=='__main__':main()
