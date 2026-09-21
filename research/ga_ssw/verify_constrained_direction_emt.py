"""Cu111 supported adatom: old/new default trace and separate mode support.

Three complete attempts, each <=800 E/F and60s; this is feasibility evidence.
"""
import hashlib
import importlib.util
import json
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from ase.build import fcc111,add_adsorbate
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from pamssw.standalone.surface import ASESurface
from pamssw.standalone.constrained_reference import ConstrainedSSWConfig,run_constrained_ssw
from research.ga_ssw.compare_vc_arms import serial

OUT=Path('research/ga_ssw/evidence/constrained-direction-subspace')


def old_module(name,path):
    import sys
    spec=importlib.util.spec_from_file_location('pamssw.standalone.'+name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def main():
    prior_shared=old_module('_prior_rc_reference',OUT/'rc_reference-before.py')
    prior_constrained=old_module('_prior_constrained_reference',OUT/'constrained_reference-before.py')
    prior_constrained._run_reduced_ssw=prior_shared._run_reduced_ssw
    a=fcc111('Cu',size=(3,3,3),a=3.6,vacuum=8.)
    fixed=np.flatnonzero(a.get_tags()>1)
    add_adsorbate(a,'Cu',height=2.,position='fcc')
    a.set_constraint(FixAtoms(indices=fixed))
    active=np.setdiff1d(np.arange(len(a)),fixed)
    relaxed_support=np.setdiff1d(np.arange(len(a)-1),fixed)
    config=ConstrainedSSWConfig(width=.6,rotation_bias=100.,temperature_K=100.)
    plan=dict(input='ASE Cu111 3x3x3 a3.6 vacuum8, FCC Cu adatom height2; bottom two layers fixed',
              atoms=serial(a),physical_fixed=fixed.tolist(),masked_direction_fixed=list(range(len(a)-1)),
              config=asdict(config),seed=29,steps=1,per_search_cap=800,per_search_seconds=60,
              fresh_cap=2,interpretation='single-input feasibility; EMT metallic surface model, no DFT or efficiency generalization')
    (OUT/'emt-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (OUT/'emt-script.py').write_text(Path(__file__).read_text())
    rows={};traces={}
    for arm in ('before','default','adatom_mode'):
        trace=[];start=time.monotonic()
        class Counted(ASESurface):
            def evaluate(self,atoms):
                if self.requests>=800 or time.monotonic()-start>60:
                    raise RuntimeError('declared feasibility budget')
                e,f=super().evaluate(atoms)
                trace.append(dict(positions=atoms.positions.copy(),cell=atoms.cell.array.copy(),energy=e,forces=f.copy()))
                return e,f
        surface=Counted(EMT())
        runner=prior_constrained.run_constrained_ssw if arm=='before' else run_constrained_ssw
        settings=prior_constrained.ConstrainedSSWConfig(**asdict(config)) if arm=='before' else config
        kwargs=dict(direction_fixed_indices=list(range(len(a)-1))) if arm=='adatom_mode' else {}
        result=runner(a,surface,steps=1,config=settings,rng=np.random.default_rng(29),**kwargs)
        elapsed=time.monotonic()-start
        fresh=[]
        for minimum in result.minima:
            raw=minimum.atoms.copy();raw.set_constraint();raw.calc=EMT()
            energy=raw.get_potential_energy();forces=raw.get_forces()
            fresh.append(dict(energy=energy,energy_error=energy-minimum.energy,
                              active_fmax=float(np.linalg.norm(forces[active],axis=1).max()),
                              raw_fmax=float(np.linalg.norm(forces,axis=1).max()),
                              fixed_exact=bool(np.array_equal(raw.positions[fixed],a.positions[fixed])),
                              cell_exact=bool(np.array_equal(raw.cell.array,a.cell.array))))
        rows[arm]=dict(result=serial(result),fresh=fresh,search_requests=surface.requests,
                       fresh_requests=len(fresh),search_seconds=elapsed)
        traces[arm]=trace
        with (OUT/(arm+'-emt-calls.jsonl')).open('x') as handle:
            for item in trace:handle.write(json.dumps(serial(item))+'\n')
        assert all(item['fixed_exact'] and item['cell_exact'] and item['active_fmax']<=config.fmax for item in fresh)
    assert len(traces['before'])==len(traces['default'])
    for old,new in zip(traces['before'],traces['default']):
        for key in old:assert np.array_equal(old[key],new[key]),key
    masked=rows['adatom_mode']['result']
    mode_rows=masked['records'][1]['climb']
    # All top-layer support directions are zero; their relaxed coordinates need not be.
    for stage in mode_rows:
        if 'mode' in stage:
            direction=np.array(stage['mode']['direction']).reshape(-1,3)
            assert np.array_equal(direction[:-1],np.zeros_like(direction[:-1]))
    rows['verification']=dict(default_trace_exact=True,
                              masked_top_layer_support_atoms=relaxed_support.tolist(),
                              all_search_requests=sum(row['search_requests'] for row in rows.values()),
                              all_fresh_requests=sum(row['fresh_requests'] for row in rows.values()))
    with (OUT/'emt-result.json').open('x') as handle:json.dump(rows,handle,indent=2);handle.write('\n')
    for arm in ('before','default','adatom_mode'):
        row=rows[arm];print(arm,row['result']['status'],row['search_requests'],len(row['fresh']),row['search_seconds'])


if __name__=='__main__':main()
