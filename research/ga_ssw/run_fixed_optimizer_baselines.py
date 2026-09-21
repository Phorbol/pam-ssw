"""Bounded fixed-cell optimizer integration audit, including a non-Ih C60 cage."""
from pathlib import Path
import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict, replace

import numpy as np
from ase.build import bulk
from ase.cluster import Icosahedron
from ase.collections import g2
from ase.io import read, write
from ase.calculators.emt import EMT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree('pamssw', out / 'source' / 'pamssw',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(__file__, out / 'runner.py')
    # Import the immutable copy before any pamssw modules enter this process.
    sys.path.insert(0, str(out / 'source'))
    from pamssw.standalone import ASESurface, SSWConfig, run_ssw
    from research.ga_ssw.compare_vc_arms import serial

    def dump(path, value):
        path.write_text(json.dumps(serial(value), indent=2, allow_nan=False) + '\n')

    cu31 = bulk('Cu', 'fcc', a=3.6, cubic=True).repeat((2, 2, 2))
    del cu31[0]
    cage_source = Path('research/ga_ssw/evidence/ccd-c60-gfn2-qualification/ccd1809asym-after.extxyz')
    cases = {'cu13': Icosahedron('Cu', 2), 'cu31_fixed': cu31,
             'cyclobutene': g2['cyclobutene'].copy(), 'c60_nonih': read(cage_source)}
    base = SSWConfig(width=.1, rotation_bias=100., max_gaussians=3,
                     temperature_K=150., fmax=.01, bias_fmax=.1,
                     relax_steps=400, fd_step=1e-4, rotation_hvp=100,
                     rotation_tol=.02, rotation_solver='dimer',
                     cluster_frame='direction_only', direction_sampling='global',
                     quench_optimizer='safe-lbfgs-total')
    optimizers = ['safe-lbfgs-total', 'ase-lbfgs-linesearch', 'scipy-lbfgsb']
    plan = dict(cases=list(cases), optimizers=optimizers, seed=11, outer_steps=1,
                search_cap_per_arm=2000, fresh_reserve_per_arm=2,
                wall_seconds_per_arm=600, config=asdict(base),
                c60_source=str(cage_source), c60_overrides=dict(width=.6, max_gaussians=12),
                purpose='Public optimizer integration across domains; one seed is not a statistical efficiency ranking.',
                sources='Cu/ASE EMT and ASE G2 cyclobutene/GFN2; CCD1809asym previously GFN2-qualified non-Ih C60 cage. C60 width .6/NG12 from existing paper-SI runner; other numerical values from fixed-cell lifecycle audit. Only optimizer differs within each case.',
                scope='Ordinary fixed-cell SSW only. No LS changes, no tuning, no retries. Physical force and molecular graph checks are separate from global-minimum or Hessian certification.')
    dump(out / 'plan.json', plan)
    for name, atoms in cases.items():
        write(out / f'{name}.extxyz', atoms)
    if not args.execute:
        return
    from tblite.ase import TBLite
    import ase, scipy
    dump(out / 'environment.json', dict(python=sys.version, ase=ase.__version__,
         scipy=scipy.__version__, source=str(out / 'source'), tblite=TBLite.__module__))
    summary = []
    for name, atoms in cases.items():
        molecular = name in ('cyclobutene', 'c60_nonih')

        def calculator():
            return TBLite(method='GFN2-xTB', accuracy=.001, verbosity=0) if molecular else EMT()

        for optimizer in optimizers:
            directory = out / f'{name}-{optimizer}'
            directory.mkdir()
            cfg = replace(base, quench_optimizer=optimizer,
                          cluster_frame='translation_only' if atoms.pbc.all() else 'direction_only')
            if name == 'c60_nonih':
                cfg = replace(cfg, width=.6, max_gaussians=12)
            dump(directory / 'config.json', cfg)
            started = time.monotonic()

            class Counted(ASESurface):
                denied = 0

                def evaluate(self, candidate):
                    if self.requests >= 2000 or time.monotonic() - started > 600:
                        self.denied += 1
                        raise RuntimeError('fixed optimizer audit request/wall cap')
                    try:
                        energy, forces = super().evaluate(candidate)
                    except Exception as error:
                        with (directory / 'evaluations.jsonl').open('a') as stream:
                            stream.write(json.dumps(serial(dict(request=self.requests, error=repr(error), atoms=candidate))) + '\n')
                        raise
                    with (directory / 'evaluations.jsonl').open('a') as stream:
                        stream.write(json.dumps(serial(dict(request=self.requests, atoms=candidate,
                                                            energy=energy, forces=forces))) + '\n')
                    return energy, forces

            surface = Counted(calculator())
            row = dict(case=name, optimizer=optimizer, seed=11, checks=[])
            result = None
            try:
                result = run_ssw(atoms, surface, steps=1, config=cfg, rng=np.random.default_rng(11))
                dump(directory / 'result.json', result)
                row.update(status=result.status, steps=[r.status for r in result.records],
                           accepted=[r.accepted for r in result.records],
                           best_delta=min(q.energy for q in result.minima) - result.initial.energy,
                           accounted=result.evaluation_requests == surface.requests)
                minima = list(result.minima)
            except Exception as error:
                row.update(status='exception', error=repr(error))
                # InitialQuenchError carries a numerical result worth retaining.
                failed = getattr(error, 'result', None)
                if failed is not None:
                    dump(directory / 'initial-failure.json', failed)
                minima = []
            fresh = ASESurface(calculator())
            for minimum in minima[:2]:
                try:
                    energy, forces = fresh.evaluate(minimum.atoms)
                    row['checks'].append(dict(energy_error=energy-minimum.energy,
                        fmax=float(np.linalg.norm(forces, axis=1).max()),
                        certified=bool(np.linalg.norm(forces, axis=1).max() <= cfg.fmax),
                        cell_unchanged=bool(np.array_equal(atoms.cell.array, minimum.atoms.cell.array))))
                except Exception as error:
                    row['checks'].append(dict(error=repr(error)))
            row.update(search_requests=surface.requests, fresh_requests=fresh.requests,
                       denied=surface.denied, wall_seconds=time.monotonic()-started)
            dump(directory / 'summary.json', row)
            summary.append(row)
            dump(out / 'summary.json', summary)
            print(name, optimizer, row, flush=True)


if __name__ == '__main__':
    main()
