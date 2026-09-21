"""Mass-scaled versus isotropic Cartesian proposals on molecular PESs."""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.cluster import Icosahedron
from ase.collections import g2
from ase.data.s22 import create_s22_system
from ase.calculators.emt import EMT
from ase.io import write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree('pamssw', out/'source'/'pamssw', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))

    shutil.copy2(__file__, out/'runner.py')
    sys.path.insert(0, str(out/'source'))
    from pamssw.standalone import ASESurface, SSWConfig, run_ssw
    from research.ga_ssw.compare_vc_arms import serial

    def dump(path, value):
        path.write_text(json.dumps(serial(value), indent=2, allow_nan=False)+'\n')

    cases = dict(bicyclobutane=g2['bicyclobutane'].copy(),
                 formic_acid_dimer=create_s22_system('Formic_acid_dimer'))
    base = SSWConfig(width=.1, rotation_bias=None, pre_rotation_hvp=5, max_gaussians=25,
        temperature_K=150., fmax=.01, bias_fmax=.1, relax_steps=400,
        fd_step=1e-4, rotation_hvp=100, rotation_tol=.02,
        direction_sampling='global', rotation_solver='ritz',
        cluster_frame='direction_only', quench_optimizer='safe-lbfgs-total')
    plan = dict(cases=list(cases), seeds=[11, 29], arms=['global', 'isotropic'],
        steps=100, search_cap=1200, fresh_reserve=101, wall_seconds=120,
        pre_rotation_hvp=5, config=asdict(base),
        baseline_provenance='Bicyclobutane global seed11/29 reused from public-staged-direction-replay-20260912-v3. Formic acid global arms are evaluated here. All isotope-neutral arms are new.',
        hypothesis='Isolate mass scaling in the initial random proposal: global scales by 1/sqrt(mass), isotropic does not. Shared rotation projection, staged Ritz and forward-force Gaussian.',
        implementation='Direct public direction_sampling, no monkeypatch, adaptive Gaussian, LS or GA. Inspired by native VMB2 unit denominator; not full native RNG/mix parity.',
        parameter_source='Five pre-HVP calls permit two plane updates in this independent solver; '
        'explicit development budget, not native five-iteration parity or a universal default. '
        'Both center evaluations share the original total1+100 request ceiling. '
        'Other settings inherited from prior fixed-cell/LJ protocols. No parameter sweep.',
        boundary='GFN2 bicyclobutane and formic-acid dimer molecular PESs. '
        'Fresh certificates use a new calculator per geometry (no previous minimum electronic guess). '
        'Force-qualified minima are not automatically distinct basins or physically intact molecules. '
        'No GM rate for unknown targets, no optimizer/LS retuning or default promotion.')
    dump(out/'plan.json', plan)
    for name, atoms in cases.items():
        write(out/f'{name}.extxyz', atoms)
    if not args.execute:
        return
    from tblite.ase import TBLite
    summaries = []
    for name, atoms in cases.items():
        cfg = replace(base, cluster_frame='translation_only') if atoms.pbc.all() else base
        def calculator():
            return TBLite(method='GFN2-xTB', accuracy=.001, verbosity=0)
        for seed in plan['seeds']:
            for arm in plan['arms']:
                if name == 'bicyclobutane' and arm == 'global':
                    continue
                folder = out/f'{name}-{arm}-{seed}'
                folder.mkdir()
                started = time.monotonic()
                class Counted(ASESurface):
                    boundary = None
                    def evaluate(self, candidate):
                        if self.requests >= 1200 or time.monotonic()-started >= 120:
                            self.boundary = 'request_cap' if self.requests >= 1200 else 'wall_cap'
                            raise RuntimeError(self.boundary)
                        try:
                            e, f = super().evaluate(candidate)
                            item = dict(request=self.requests, energy=e, fmax=float(np.linalg.norm(f, axis=1).max()))
                        except Exception as error:
                            item = dict(request=self.requests, error=repr(error), atoms=candidate)
                            with (folder/'evaluations.jsonl').open('a') as stream:
                                stream.write(json.dumps(serial(item))+'\n')
                            raise
                        with (folder/'evaluations.jsonl').open('a') as stream:
                            stream.write(json.dumps(item)+'\n')
                        return e, f
                surface = Counted(calculator())
                row = dict(case=name, arm=arm, seed=seed)
                fresh = None
                try:
                    result = run_ssw(atoms, surface, steps=100, config=replace(cfg, direction_sampling=arm),
                                     rng=np.random.default_rng(seed))
                    dump(folder/'result.json', result)
                    fresh = ASESurface(calculator())
                    checks = []
                    for index, q in enumerate(result.minima):
                        try:
                            fresh.calculator = calculator()
                            e, f = fresh.evaluate(q.atoms)
                            check = dict(index=index, energy=e, energy_error=e-q.energy,
                                fmax=float(np.linalg.norm(f, axis=1).max()),
                                cell_unchanged=bool(np.array_equal(atoms.cell.array, q.atoms.cell.array)))
                        except Exception as error:
                            check = dict(index=index, error=repr(error))
                        checks.append(check)
                        dump(folder/'fresh-checks.json', checks)
                    write(folder/'minima.extxyz', [q.atoms for q in result.minima])
                    row.update(status=result.status, steps=len(result.records),
                        statuses=dict(Counter(r.status for r in result.records)),
                        accepted=sum(r.accepted for r in result.records), checks=checks,
                        best_delta=min(q.energy for q in result.minima)-result.initial.energy,
                        accounted=result.evaluation_requests==surface.requests, fresh=fresh.requests)
                except Exception as error:
                    row.update(status='exception', error=repr(error))
                finally:
                    if fresh is not None:
                        row['fresh'] = fresh.requests
                row.update(search=surface.requests, boundary=surface.boundary,
                           seconds=time.monotonic()-started)
                dump(folder/'summary.json', row)
                summaries.append(row)
                dump(out/'summary.json', summaries)
                print({k:v for k,v in row.items() if k != 'checks'}, flush=True)


if __name__ == '__main__':
    main()
