"""Continuous versus disk-resumed real-system SSW/LS; no backend substitution.

Run only after implementation tests. Frozen code, paid geometries and fresh
minimum certificates are retained. Electronic-state restart is not assumed.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def enc(x):
    from ase import Atoms
    if isinstance(x, Atoms):
        return dict(numbers=x.numbers.tolist(), positions=x.positions.tolist(),
                    cell=x.cell.array.tolist(), pbc=x.pbc.tolist())
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, dict): return {str(k): enc(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [enc(v) for v in x]
    if hasattr(x, '__dict__'): return enc(vars(x))
    return x


def dump(path, x):
    path.write_text(json.dumps(enc(x), indent=2, allow_nan=False)+'\n')


def ledger_audit(continuous, split, first_count=None):
    """Report exact and tolerance-based replay agreement without hiding drift."""
    n = min(len(continuous), len(split)); max_pos = max_force = max_energy = 0.0
    first_count = n if first_count is None else first_count
    exact_prefix = continuous[:first_count] == split[:first_count]
    compared = zip(continuous, split)
    for a, b in compared:
        aa, bb = np.asarray(a['atoms']['positions']), np.asarray(b['atoms']['positions'])
        max_pos = max(max_pos, float(np.max(np.abs(aa-bb))))
        if 'energy' in a and 'energy' in b: max_energy = max(max_energy, abs(a['energy']-b['energy']))
        if 'forces' in a and 'forces' in b:
            max_force = max(max_force, float(np.max(np.abs(np.asarray(a['forces'])-np.asarray(b['forces'])))))
    return dict(ledger_exact=continuous == split, prefix_exact=exact_prefix,
                resume_exact=(continuous[first_count:] == split[first_count:]),
                length_equal=len(continuous)==len(split), max_position_difference=max_pos,
                max_force_difference=max_force, max_energy_difference=max_energy,
                numerical_replay=(len(continuous)==len(split) and max_pos<=1e-8 and
                    max_force<=1e-7 and max_energy<=1e-8))


class LedgerSurface:
    def __init__(self, calc, cap=6000):
        from pamssw.standalone import ASESurface
        self.surface = ASESurface(calc)
        self.cap = cap
        self.rows = []
    @property
    def requests(self): return self.surface.requests
    def evaluate(self, atoms):
        if self.requests >= self.cap: raise RuntimeError('fixed request budget exhausted')
        try:
            e, f = self.surface.evaluate(atoms)
        except Exception as exc:
            self.rows.append(dict(atoms=enc(atoms), error=repr(exc)))
            raise
        self.rows.append(dict(atoms=enc(atoms), energy=e, forces=f.tolist()))
        return e, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--cases', nargs='*', default=None)
    args = ap.parse_args(); out = args.output.resolve(); out.mkdir(exist_ok=False)
    shutil.copytree(ROOT/'pamssw', out/'source'/'pamssw', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(__file__, out/'runner.py'); sys.path.insert(0, str(out/'source'))
    import pamssw
    assert Path(pamssw.__file__).resolve().is_relative_to(out/'source')
    from pamssw.standalone.paper_reference import SSWConfig, LSSettings, run_ssw, load_ssw_checkpoint
    from pamssw.standalone.ls_native_reference import NativeLSSettings
    from pamssw.standalone.native_ls import HC_BOND_ENERGIES, HC_BOND_LENGTHS
    from ase import Atoms
    from ase.collections import g2
    sources = ROOT/'research/ga_ssw/evidence/verified-ritz-multicase-20260912'
    inputs = {}
    provenance = {}
    for case in ('cu13', 'cu31_fixed', 'bicyclobutane'):
        p = sources/f'{case}-verified-seed11/result.json'
        inputs[case] = json.loads(p.read_text())['initial']['atoms']
        provenance[case] = dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    inputs['butadiene'] = enc(g2['butadiene'])
    provenance['butadiene'] = "ASE G2 butadiene, same source as preceding public LS campaign"
    base = SSWConfig(width=.1, rotation_bias=None, pre_rotation_hvp=5,
        max_gaussians=25, temperature_K=150., fmax=.01, bias_fmax=.1,
        relax_steps=400, fd_step=1e-4, rotation_hvp=100, rotation_tol=.02,
        direction_sampling='global', rotation_solver='ritz', cluster_frame='direction_only',
        quench_optimizer='safe-lbfgs-total', lbfgs_memory=10)
    arms = [('cu13','ssw',None), ('cu31_fixed','ssw',None), ('bicyclobutane','ssw',None),
        ('butadiene','paper_ls',LSSettings(HC_BOND_ENERGIES,
            {k:v+.1 for k,v in HC_BOND_LENGTHS.items()}, target_per_atom=.7)),
        ('butadiene','native_ls',NativeLSSettings(HC_BOND_ENERGIES,HC_BOND_LENGTHS,target_mev_per_atom=700.))]
    dump(out/'inputs.json',inputs); dump(out/'input-provenance.json',provenance)
    dump(out/'plan.json',dict(seed=11, steps=2, split=[1,1], config=asdict(base),
        arms=arms, per_execution_cap=6000, calculator='EMT or GFN2-xTB accuracy .001',
        scope='persistence contract; fresh calculator on resume; no performance or full electronic-state parity claim'))
    dump(out/'source-manifest.json',{str(p.relative_to(out/'source')):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (out/'source').rglob('*.py')})
    if not args.execute: return
    from ase.calculators.emt import EMT
    from tblite.ase import TBLite
    rows=[]
    selected = set(args.cases) if args.cases else None
    for case, arm, ls in arms:
        if selected is not None and f'{case}-{arm}' not in selected and case not in selected:
            continue
        folder=out/f'{case}-{arm}'; folder.mkdir(); payload=inputs[case]
        atoms=Atoms(**payload); cfg=replace(base,cluster_frame='translation_only') if atoms.pbc.all() else base
        def calc(): return EMT() if case.startswith('cu') else TBLite(method='GFN2-xTB',accuracy=.001,verbosity=0)
        started=time.monotonic()
        continuous=LedgerSurface(calc()); first=LedgerSurface(calc()); second=LedgerSurface(calc())
        result=run_ssw(atoms,continuous,steps=2,config=cfg,rng=np.random.default_rng(11),ls=ls)
        split=run_ssw(atoms,first,steps=1,config=cfg,rng=np.random.default_rng(11),ls=ls,checkpoint_path=folder/'boundary.pkl')
        checkpoint=load_ssw_checkpoint(folder/'boundary.pkl')
        resumed=run_ssw(checkpoint.current,second,steps=1,config=cfg,rng=np.random.default_rng(999),ls=ls,checkpoint=checkpoint,checkpoint_path=folder/'resumed.pkl')
        joined=first.rows+second.rows
        dump(folder/'continuous-ledger.json',continuous.rows); dump(folder/'split-ledger.json',joined)
        dump(folder/'continuous-result.json',result); dump(folder/'resumed-result.json',resumed)
        assert resumed.evaluation_requests==first.requests+second.requests
        assert resumed.evaluation_requests==resumed.initial.evaluation_requests+sum(r.evaluation_requests for r in resumed.records)
        assert [r.index for r in resumed.records]==[0,1]
        assert len(checkpoint.records)==1 and checkpoint.next_index==1
        assert result.status==resumed.status=='completed'
        fresh=LedgerSurface(calc()); checks=[]
        for label,run in [('continuous',result),('resumed',resumed)]:
            for i,q in enumerate(run.minima):
                fresh.surface.calculator=calc(); e,f=fresh.evaluate(q.atoms)
                checks.append(dict(run=label,index=i,energy_error=e-q.energy,fmax=float(np.linalg.norm(f,axis=1).max()),
                    fixed_cell=bool(np.array_equal(atoms.cell.array,q.atoms.cell.array))))
        dump(folder/'fresh.json',checks)
        assert all(c['fmax']<=cfg.fmax and abs(c['energy_error'])<=1e-8 and c['fixed_cell'] for c in checks)
        audit=ledger_audit(continuous.rows, joined, first.requests)
        row=dict(case=case,arm=arm,status='completed',continuous_requests=continuous.requests,
            first_requests=first.requests,resumed_requests=second.requests,fresh_requests=fresh.requests,
            **audit, final_position_max_difference=float(np.max(np.abs(result.current.positions-resumed.current.positions))),
            accepted_continuous=[r.accepted for r in result.records],accepted_resumed=[r.accepted for r in resumed.records],
            elapsed=time.monotonic()-started)
        rows.append(row);dump(out/'summary.json',rows);print(json.dumps(row),flush=True)

if __name__=='__main__': main()
