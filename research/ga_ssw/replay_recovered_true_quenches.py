"""Matched original-start bare-PES quench diagnostic; no SSW reruns or tuning.

Inputs are SSWStep.last_atoms, before true quench (quench copies its argument).
Each optimizer starts with empty history and the same request/force limits.
"""
import argparse
import json
import sys
import time
from pathlib import Path


def execute(out):
    plan=json.loads((out/'plan.json').read_text())
    source=Path(plan['frozen_source'])
    sys.path.insert(0,str(source)); sys.path.insert(0,str(source.parent))
    from ledger_helpers import CountedSurface,atoms_from,dump
    import pamssw
    if not Path(pamssw.__file__).resolve().is_relative_to(source):
        raise RuntimeError('frozen source import required')
    from pamssw.standalone.surface import quench,ASESurface
    from mace.calculators import MACECalculator
    import numpy as np
    import hashlib
    if hashlib.sha256(Path(plan['model']).read_bytes()).hexdigest()!=plan['model_sha256']:
        raise ValueError('model changed')
    kwargs=dict(model_paths=plan['model'],device='cuda',default_dtype='float64',enable_cueq=False,enable_oeq=False)
    started=time.monotonic()
    calculator=MACECalculator(**kwargs); validator=ASESurface(MACECalculator(**kwargs))
    dump(out/'initialization.json',{'seconds':time.monotonic()-started})
    rows=[]
    for case in plan['starts']:
        atoms=atoms_from(case['atoms'])
        for method in plan['methods']:
            name=case['name']+'-'+method['name']; folder=out/name; folder.mkdir()
            surface=CountedSurface(calculator,folder/'requests.jsonl',cap=plan['request_cap'],wall=120)
            row=dict(name=name,source=case['source'],record=case['record'],method=method,
                     status='started',fresh_requests=0)
            fresh_before=validator.requests
            try:
                result=quench(atoms,surface,fmax=plan['fmax'],steps=plan['steps'],
                              optimizer=method['optimizer'],lbfgs_memory=method['memory'])
                dump(folder/'result.json',result)
                before=validator.requests
                energy,forces=validator.evaluate(result.atoms)
                fmax=float(np.linalg.norm(forces,axis=1).max())
                row.update(status='force_qualified' if result.converged and fmax<=plan['fmax'] else 'not_converged',
                           max_force=fmax,energy=energy,steps=result.optimizer_steps,
                           telemetry=result.optimizer_telemetry,fresh_requests=validator.requests-before)
            except Exception as error:
                row.update(status='censored' if surface.boundary else 'error',error=repr(error))
            row.update(fresh_requests=validator.requests-fresh_before,requests=surface.requests,boundary=surface.boundary,seconds=time.monotonic()-surface.started)
            dump(folder/'summary.json',row); rows.append(row); dump(out/'summary.json',rows)
            print(name,row['status'],row['requests'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    execute(parser.parse_args().output.resolve())
