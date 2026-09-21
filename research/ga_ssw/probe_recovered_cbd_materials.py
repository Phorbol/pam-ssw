"""Bounded OMAT-small CBD callback check; not an end-to-end SSW benchmark."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    plan = json.loads((out/'plan.json').read_text())
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    import torch
    from ase.io import read
    from mace.calculators import MACECalculator
    from pamssw.standalone.recovered_cbd import recovered_cbd_direction
    import pamssw
    assert Path(pamssw.__file__).resolve().is_relative_to(out/'source')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = Path(plan['model'])
    assert hashlib.sha256(model.read_bytes()).hexdigest() == plan['model_sha256']
    calc = MACECalculator(model_paths=str(model), device='cuda',
                          default_dtype='float64', enable_cueq=False, enable_oeq=False)
    report = dict(scope=plan['scope'], cases=[], device=torch.cuda.get_device_name())
    started = time.monotonic()
    for name in plan['cases']:
        atoms = read(out/f'{name}.extxyz')
        rng = np.random.default_rng(plan['seed'])
        anchor = rng.normal(size=atoms.positions.shape)
        # Fixed-cell unconstrained probe: remove net translation only. This
        # is an explicit input direction, not the restored native generator.
        anchor -= anchor.mean(axis=0)
        anchor /= np.linalg.norm(anchor)
        requests = []

        def evaluate(candidate):
            row = dict(positions=candidate.positions.tolist())
            requests.append(row)
            try:
                calc.calculate(candidate, properties=['energy', 'forces'])
                energy = float(calc.results['energy'])
                force = np.array(calc.results['forces'], copy=True)
                row.update(energy=energy, forces=force.tolist())
                return energy, force
            except Exception as error:
                row['error'] = repr(error)
                raise

        case = dict(case=name, anchor=anchor.tolist())
        try:
            result = recovered_cbd_direction(atoms, anchor, evaluate=evaluate, **plan['solver'])
            case.update(result=asdict(result), request_count=len(requests),
                        accounting_passed=result.force_calls == len(requests))
            endpoint = atoms.positions + plan['solver']['fd_step']*result.direction
            case['returned_endpoint_was_evaluated'] = any(
                np.allclose(endpoint, row['positions'], rtol=0, atol=1e-12) for row in requests)
        except Exception as error:
            case.update(error=repr(error), traceback=traceback.format_exc(), request_count=len(requests))
        case['evaluations'] = requests
        report['cases'].append(case)
        report['seconds'] = time.monotonic()-started
        (out/'result.json').write_text(json.dumps(report, indent=2,
            default=lambda x: x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
        print(name, {k:v for k,v in case.items() if k not in ('evaluations','result','anchor')}, flush=True)
    if any('error' in case or not case['accounting_passed'] or
           not case['returned_endpoint_was_evaluated'] for case in report['cases']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
