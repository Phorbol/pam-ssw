"""Paired research-only direction substitution across four fixed-cell cases."""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from ase import Atoms, units
from ase.build import bulk
from ase.cluster import Icosahedron
from ase.collections import g2
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
    for name in ('full_pair_lj.py', 'two_stage_dimer.py'):
        shutil.copy2(Path(__file__).parent/name, out/'source'/name)
    shutil.copy2(__file__, out/'runner.py')
    sys.path.insert(0, str(out/'source'))
    from pamssw.standalone import ASESurface, SSWConfig, run_ssw
    import pamssw.standalone.dimer as dimer_module
    from two_stage_dimer import two_stage_dimer_direction
    from full_pair_lj import FullPairLJ
    from research.ga_ssw.compare_vc_arms import serial
    original_dimer = dimer_module.paper_dimer_direction

    def dump(path, value):
        path.write_text(json.dumps(serial(value), indent=2, allow_nan=False)+'\n')

    cu31 = bulk('Cu', 'fcc', a=3.6, cubic=True).repeat((2, 2, 2))
    del cu31[0]
    lj_source = Path('research/ga_ssw/evidence/lj38-source-20260912/optim-finish')
    cases = dict(cu13=Icosahedron('Cu', 2), cu31_fixed=cu31,
                 bicyclobutane=g2['bicyclobutane'].copy(),
                 lj38_slm=Atoms('Ar38', positions=np.loadtxt(lj_source)*2.7))
    base = SSWConfig(width=.1, rotation_bias=100., max_gaussians=25,
        temperature_K=150., fmax=.01, bias_fmax=.1, relax_steps=400,
        fd_step=1e-4, rotation_hvp=100, rotation_tol=.02,
        direction_sampling='global', rotation_solver='dimer',
        cluster_frame='direction_only', quench_optimizer='safe-lbfgs-total')
    plan = dict(cases=list(cases), seeds=[11, 29], arms=['fixed100', 'two_stage'],
        steps=100, search_cap=6000, fresh_reserve=101, wall_seconds=120,
        pre_rotation_hvp=5, config=asdict(base), lj_source=str(lj_source),
        lj_overrides=dict(width=.6, max_gaussians=14, temperature_K=.8/units.kB),
        hypothesis='Replacing fixed100/original anchor by a short unbiased presweep followed by '
        'a=max(precurvature,0) and its evaluated anchor may improve qualified low-energy '
        'landings at a common force-request cap. Negative precurvature continues unbiased '
        'plane dimer; this is independent SSW, not CBD TS search or native parity.',
        implementation='Research-only temporary replacement of dimer_module.paper_dimer_direction; '
        'no production source modification. Fixed100 config value is explicitly superseded by '
        'the helper only in two_stage arm. Height policy is None in both arms, avoiding '
        'any native-height curvature reconstruction that assumes fixed100.',
        parameter_source='Five pre-HVP calls permit two plane updates in this independent solver; '
        'explicit development budget, not native five-iteration parity or a universal default. '
        'Both center evaluations share the original total1+100 request ceiling. '
        'Other settings inherited from prior fixed-cell/LJ protocols. No parameter sweep.',
        boundary='EMT metal clusters/periodic vacancy, GFN2 molecular isomer, LJ model supplement. '
        'Fresh certificates use a new calculator per geometry (no previous minimum electronic guess). '
        'Force-qualified minima are not automatically distinct basins or physically intact molecules. '
        'No GM rate for unknown targets, no Gaussian/optimizer/LS retuning.')
    dump(out/'plan.json', plan)
    for name, atoms in cases.items():
        write(out/f'{name}.extxyz', atoms)
    if not args.execute:
        return
    from tblite.ase import TBLite
    summaries = []
    for name, atoms in cases.items():
        cfg = replace(base, cluster_frame='translation_only') if atoms.pbc.all() else base
        if name == 'lj38_slm':
            cfg = replace(cfg, **plan['lj_overrides'])
        def calculator():
            if name == 'bicyclobutane':
                return TBLite(method='GFN2-xTB', accuracy=.001, verbosity=0)
            return FullPairLJ() if name == 'lj38_slm' else EMT()
        for seed in plan['seeds']:
            for arm in plan['arms']:
                folder = out/f'{name}-{arm}-{seed}'
                folder.mkdir()
                started = time.monotonic()
                traces = []
                def research_direction(work, anchor, **kwargs):
                    supplied_bias = kwargs.pop('rotation_bias')
                    mode = two_stage_dimer_direction(work, anchor, pre_rotation_hvp=5, **kwargs)
                    traces.append(dict(superseded_fixed_bias=supplied_bias, result=mode))
                    return mode
                dimer_module.paper_dimer_direction = original_dimer if arm == 'fixed100' else research_direction
                class Counted(ASESurface):
                    boundary = None
                    def evaluate(self, candidate):
                        if self.requests >= 6000 or time.monotonic()-started >= 120:
                            self.boundary = 'request_cap' if self.requests >= 6000 else 'wall_cap'
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
                    result = run_ssw(atoms, surface, steps=100, config=cfg,
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
                    dimer_module.paper_dimer_direction = original_dimer
                    dump(folder/'rotation-traces.json', traces)
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
