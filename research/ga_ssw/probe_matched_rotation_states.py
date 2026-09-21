"""Matched rotation diagnostics on saved relaxed starts, never on the login node."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', type=Path, required=True)
    args = ap.parse_args()
    out = args.outdir.resolve()
    plan = json.loads((out/'plan.json').read_text())
    import torch
    from ase.io import read
    from mace.calculators import MACECalculator
    from pamssw.standalone import ASESurface
    from pamssw.standalone.paper_reference import sample_initial_direction
    from pamssw.standalone.periodic_geometry import FixedCellTranslationFrame
    from pamssw.standalone.dimer import paper_dimer_direction
    from pamssw.standalone.direction import paper_biased_direction
    from pamssw.standalone.broyden_direction import paper_broyden_direction
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    calc = MACECalculator(model_paths=plan['model'], device=plan['device'],
        default_dtype='float64', enable_cueq=False, enable_oeq=False)
    surface = ASESurface(calc)
    report = dict(scope='frozen initial minima; independent solvers, not native CBD',
                  tolerances=[.02, 2.], rows=[], requests=0)
    started = time.monotonic()
    for case in plan['cases']:
        # Every solver/seed originally starts at the same deterministic quench.
        source = out/f'{case}-ritz-seed{plan["seeds"][0]}'/'minima.extxyz'
        if not source.exists():
            report['rows'].append(dict(case=case,error='no saved relaxed start'))
            continue
        atoms = read(source, index=0)
        frame = FixedCellTranslationFrame(atoms)
        def evaluate(candidate):
            return frame.evaluate(candidate, surface.evaluate)
        for seed in plan['seeds']:
            anchor = frame.project(sample_initial_direction(atoms,
                np.random.default_rng(seed), mode='global'))
            anchor /= np.linalg.norm(anchor)
            pre = paper_dimer_direction(atoms, anchor, rotation_bias=0.,
                fd_step=.001,max_hvp=5,tol=.02,evaluate=evaluate)
            bias = max(pre.curvature, 0.)
            remaining = 100-pre.force_calls
            for tol in report['tolerances']:
                for solver in plan['solvers']:
                    before = surface.requests
                    row = dict(case=case,seed=seed,solver=solver,tol=tol,
                               shared_pre=asdict(pre),bias=bias)
                    try:
                        solve = {'ritz':paper_biased_direction,'dimer':paper_dimer_direction,
                                 'broyden-euclidean':paper_broyden_direction}[solver]
                        result = solve(atoms, pre.direction, rotation_bias=bias,
                            fd_step=.001,max_hvp=remaining,tol=tol,evaluate=evaluate)
                        row['result'] = asdict(result)
                        row['solver_requests'] = surface.requests-before
                        row['accounting_passed'] = result.force_calls == surface.requests-before
                        # Common independent force certificate of returned endpoint.
                        _, f0 = evaluate(atoms)
                        endpoint = atoms.copy()
                        endpoint.positions += .001*result.direction
                        _, f1 = evaluate(endpoint)
                        real_hn = (f0-f1)/.001
                        biased_hn = real_hn - bias * np.sum(
                            pre.direction*result.direction)*pre.direction
                        curv = float(np.sum(result.direction*biased_hn))
                        residual = float(np.linalg.norm(biased_hn-curv*result.direction))
                        row.update(certified_residual=residual,
                            certified_curvature=curv,
                            real_curvature=float(np.sum(result.direction*real_hn)),
                            certificate_error=abs(residual-result.residual_norm),
                            overlap_anchor=float(np.sum(result.direction*pre.direction)))
                    except Exception as error:
                        row['error'] = repr(error)
                    row['total_requests'] = surface.requests-before
                    report['rows'].append(row)
                    report.update(requests=surface.requests, seconds=time.monotonic()-started)
                    (out/'frozen-rotation.json').write_text(json.dumps(report,indent=2,
                        default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
                    print({k:v for k,v in row.items() if k not in ('result','shared_pre')},flush=True)


if __name__ == '__main__':
    main()
