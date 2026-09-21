"""Bounded real XXXII qualification of the primitive ASE/replicated engine map."""
import json, shutil, time, traceback
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import read
from ase.stress import voigt_6_to_full_3x3_stress
from research.ga_ssw.xxxii_replicated_calculator import XXXIIReplicatedCalculator
from pamssw.standalone.rc_topology import read_rigid_topology
from pamssw.standalone.rc_periodic_input import unwrap_rigid_molecules
from pamssw.standalone.rc_optimization_domain import PrincipalRigidForestCellChart

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'research/ga_ssw/evidence/xxxii-replicated-qualification'
SRC = ROOT / 'research/ga_ssw/evidence/xxxii-rc-vc-central-ritz-completion'
MODEL = ROOT / 'research/ga_ssw/evidence/xxxii-stock-charmm-converted'
TOPO = Path('/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_examples_run/global_exploration/input-templates/TYPE2-XXXII/mc')
OUT.mkdir(exist_ok=False)
shutil.copy2(__file__, OUT / 'runner-executed.py')
shutil.copy2(Path(__file__).with_name('xxxii_replicated_calculator.py'), OUT)
top = read_rigid_topology(TOPO / 'rigidbody', TOPO / 'blist', natoms=172)
initial = unwrap_rigid_molecules(read(ROOT / 'tests/standalone/fixtures/type2_xxxii.extxyz'), top.bonds).atoms
points = [('initial', initial)]
points += [(p['label'], Atoms(**p['atoms'])) for p in map(json.loads, (SRC / 'line-search-gradient/calls.jsonl').read_text().splitlines()) if p['label'] in ('center', 'lbfgs+1e-06')]
plan = dict(max_EFS=33, representations=[[1,1,2],[1,1,3],[2,2,2]], primitive_atoms=172,
            derivative_h=[1e-4, 2.5e-5], seed=17, purpose='Real molecule/crystal API, representation and gradient checks; no search effectiveness claim')
(OUT / 'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
rows, derivatives, engines = [], [], []
start = time.monotonic()
def evaluate(c, a, label):
    assert len(rows) < 33
    row = dict(label=label, repetitions=c.repetitions, status='attempted')
    rows.append(row)
    try:
        c.calculate(a)
        e, f, s = c.results['energy'], c.results['forces'], voigt_6_to_full_3x3_stress(c.results['stress'])
        assert f.shape == (172,3)
        row.update(status='completed', energy=e, forces=f.tolist(), stress=s.tolist())
        return e, f, s
    except Exception as exc:
        row.update(status='failed', error=repr(exc)); raise
    finally:
        (OUT/'calls.json').write_text(json.dumps(rows,indent=2)+'\n')
report = {}
try:
    for rep in ((1,1,2), (1,1,3), (2,2,2)):
        c = XXXIIReplicatedCalculator(data_path=MODEL/'lmp.data', input_path=MODEL/'in.simple', model_manifest=MODEL/'manifest.json', reference_atoms=initial, repetitions=rep)
        engines.append(c)
        for label, a in points:
            base = evaluate(c,a,label)
            if rep != (1,1,2) or label == 'lbfgs+1e-06': continue
            chart = PrincipalRigidForestCellChart(a,top.components,anchor=0,rotation_length=1.,torsion_length=1.,strain_length=5.)
            q = np.zeros(chart.dimension)
            gradient = chart.evaluate(q,lambda _:base).gradient
            rng = np.random.default_rng(17)
            directions = {}
            d = np.zeros_like(q);d[-6]=1.;directions['cell0']=d
            d = rng.normal(size=q.shape);d/=np.linalg.norm(d);directions['RC']=d
            d = np.zeros_like(q);d[0]=1.;directions['coordinate0']=d
            for name,d in directions.items():
                for h in (1e-4,2.5e-5):
                    ep=evaluate(c,chart.unpack(q+h*d),f'{label}:{name}+{h}')[0]
                    em=evaluate(c,chart.unpack(q-h*d),f'{label}:{name}-{h}')[0]
                    derivatives.append(dict(point=label,direction=name,h=h,analytic=float(gradient@d),fd=(ep-em)/(2*h),error=(ep-em)/(2*h)-float(gradient@d)))
    report['status']='completed'
except Exception as exc:
    report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
finally:
    for c in engines:c.close()
    report.update(derivatives=derivatives,api_attempts=sum(c.api_calls for c in engines),engine_calls=sum(c.engine_calls for c in engines),internal_atoms_evaluated=sum(c.atoms_evaluated for c in engines),wall_seconds=time.monotonic()-start)
    (OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
