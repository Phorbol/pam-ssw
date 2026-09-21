"""Bounded Cu28/EMT end-to-end constrained LS contract check, not efficacy."""
import argparse,json,time
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from pamssw.standalone.surface import ASESurface
from pamssw.standalone.constrained_reference import ConstrainedSSWConfig,run_constrained_ssw
from pamssw.standalone.paper_reference import LSSettings
from research.ga_ssw.compare_vc_arms import serial

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root
    plan=json.loads((root/'plan.json').read_text());source=json.loads((root/'input-plan.json').read_text())
    atoms=Atoms(**source['atoms']);atoms.pbc=True
    fixed=np.array(source['physical_fixed']);active=np.setdiff1d(np.arange(len(atoms)),fixed)
    atoms.set_constraint(FixAtoms(indices=fixed))
    config=ConstrainedSSWConfig(**source['config'])
    settings=LSSettings(bond_energies={(29,29):1.},bond_lengths={(29,29):2.9},target_per_atom=.001)
    out=root/'emt';out.mkdir(exist_ok=False)
    (out/'settings.json').write_text(json.dumps(serial(dict(atoms=atoms,config=config,ls=settings,plan=plan)),indent=2)+'\n')
    for arm in ('plain','ls','ls_adatom_mode'):
        work=out/arm;work.mkdir();started=time.monotonic()
        class Recorded(ASESurface):
            def evaluate(self,a):
                if self.requests>=1000 or time.monotonic()-started>60:raise RuntimeError('declared1000EF/60second search cap')
                e,f=super().evaluate(a)
                with (work/'calls.jsonl').open('a') as stream:stream.write(json.dumps(serial(dict(request=self.requests,atoms=a,energy=e,forces=f)))+'\n')
                return e,f
        surface=Recorded(EMT())
        kwargs={} if arm=='plain' else dict(ls=settings)
        if arm=='ls_adatom_mode':kwargs['direction_fixed_indices']=list(range(len(atoms)-1))
        result=run_constrained_ssw(atoms,surface,steps=2,config=config,rng=np.random.default_rng(29),**kwargs)
        (work/'search-result.json').write_text(json.dumps(serial(result),indent=2)+'\n')
        fresh=[]
        for minimum in result.minima:
            a=minimum.atoms.copy();a.set_constraint();e,f=ASESurface(EMT()).evaluate(a)
            fresh.append(dict(energy=e,energy_error=e-minimum.energy,active_fmax=float(np.linalg.norm(f[active],axis=1).max()),
              fixed_exact=bool(np.array_equal(a.positions[fixed],atoms.positions[fixed])),cell_exact=bool(np.array_equal(a.cell.array,atoms.cell.array)),atoms=a,forces=f))
        record=dict(result=result,fresh=fresh,search_requests=surface.requests,fresh_requests=len(fresh),total_requests=surface.requests+len(fresh),wall_seconds=time.monotonic()-started)
        (work/'result.json').write_text(json.dumps(serial(record),indent=2)+'\n')
        assert result.requests==surface.requests==sum(r['requests'] for r in result.records)
        assert all(f['fixed_exact'] and f['cell_exact'] and f['active_fmax']<=config.fmax and abs(f['energy_error'])<1e-10 for f in fresh)
        if arm=='ls_adatom_mode':
            for event in result.records[1:]:
                for stage in event.get('climb',[]):
                    if 'mode' in stage:assert np.array_equal(stage['mode'].direction.reshape(-1,3)[:-1],np.zeros((len(active)-1,3)))
        print(arm,result.status,surface.requests,len(result.minima),flush=True)
if __name__=='__main__':main()
