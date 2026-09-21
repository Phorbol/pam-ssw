"""Frozen, capped comparison of two existing SSW height rules; no tuning."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from ase import Atoms, units
from ase.io import write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree('pamssw', out / 'source' / 'pamssw', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(__file__, out / 'runner.py')
    shutil.copy2('research/ga_ssw/full_pair_lj.py', out / 'source' / 'full_pair_lj.py')
    sys.path.insert(0, str(out / 'source'))
    from full_pair_lj import FullPairLJ
    from pamssw.standalone import ASESurface, SSWConfig, run_ssw
    from pamssw.standalone.minimal_angle_height import MinimalAngleHeightPolicy
    from research.ga_ssw.compare_vc_arms import serial

    def dump(path, value):
        path.write_text(json.dumps(serial(value), indent=2, allow_nan=False) + '\n')

    source = Path('research/ga_ssw/evidence/lj38-source-20260912')
    start = Atoms('Ar38', positions=np.loadtxt(source / 'optim-finish') * 2.7)
    # Target is held out from all proposals, directions, and MC decisions.
    target = Atoms('Ar38', positions=np.loadtxt(source / 'gm-points') * 2.7)
    config = SSWConfig(width=.6, rotation_bias=100., max_gaussians=14,
        temperature_K=.8 / units.kB, fmax=.01, bias_fmax=.1, relax_steps=400,
        fd_step=1e-4, rotation_hvp=100, rotation_tol=.02,
        direction_sampling='global', rotation_solver='dimer',
        cluster_frame='direction_only', quench_optimizer='safe-lbfgs-total')
    plan = dict(config=asdict(config), seeds=[11, 29], steps=100,
        search_cap=12000, fresh_reserve=101, wall_seconds=300,
        arms=['forward_force', 'minimal_angle87'], epsilon_eV=1., sigma_A=2.7,
        source=str(source), target_use='post hoc only; never a search seed',
        protocol='Development comparison, not published LJ38 success-rate reproduction. '
        'SSW2013 LJ75 section supplies epsilon1/sigma2.7, width.6 and kBT.8; '
        'NG14 from general SSW discussion; numerical settings inherited from fixed-cell audit. '
        'Global direction and Safe-total are explicit independent variants. '
        'Minimal angle isolates recovered87-degree condition; it omits native history growth/ceilings. '
        'Same starting structure, seeds, optimizer and total search cap. '
        'Report all failures, force certificates and energy/structural diagnostics separately; '
        'pair-distance spectrum is necessary, not sufficient, for target identity. '
        'LJ is a model supplement, not replacement for real-system validation.')
    dump(out / 'plan.json', plan)
    write(out / 'start.extxyz', start)
    write(out / 'held-out-target.extxyz', target)
    if not args.execute:
        return
    summary = []
    for seed in plan['seeds']:
        for arm in plan['arms']:
            folder = out / f'{arm}-{seed}'
            folder.mkdir()
            started = time.monotonic()

            class Counted(ASESurface):
                denied = 0
                boundary = None

                def evaluate(self, atoms):
                    if self.requests >= plan['search_cap'] or time.monotonic() - started >= plan['wall_seconds']:
                        self.denied += 1
                        self.boundary = 'request_cap' if self.requests >= plan['search_cap'] else 'wall_cap'
                        raise RuntimeError(self.boundary)
                    energy, forces = super().evaluate(atoms)
                    with (folder / 'evaluations.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(request=self.requests, energy=energy,
                            fmax=float(np.linalg.norm(forces, axis=1).max()))) + '\n')
                    return energy, forces

            surface = Counted(FullPairLJ())
            result = run_ssw(start, surface, steps=plan['steps'], config=config,
                rng=np.random.default_rng(seed),
                height_policy=None if arm == 'forward_force' else MinimalAngleHeightPolicy())
            dump(folder / 'result.json', result)
            fresh = ASESurface(FullPairLJ())
            checks = []
            tri = np.triu_indices(len(start), 1)
            target_distances = np.sort(target.get_all_distances()[tri])
            for idx, minimum in enumerate(result.minima):
                energy, force = fresh.evaluate(minimum.atoms)
                checks.append(dict(index=idx, energy=energy, energy_error=energy-minimum.energy,
                    fmax=float(np.linalg.norm(force, axis=1).max()),
                    target_pair_distance_rms=float(np.sqrt(np.mean((np.sort(minimum.atoms.get_all_distances()[tri])-target_distances)**2)))))
            write(folder / 'minima.extxyz', [m.atoms for m in result.minima])
            row = dict(seed=seed, arm=arm, status=result.status, boundary=surface.boundary,
                requests=surface.requests, fresh_requests=fresh.requests,
                accounting_matches=result.evaluation_requests == surface.requests,
                steps=len(result.records), statuses=dict(Counter(r.status for r in result.records)),
                accepted=sum(r.accepted for r in result.records), checks=checks,
                best_energy=min(c['energy'] for c in checks),
                wall_seconds=time.monotonic()-started)
            dump(folder / 'summary.json', row)
            summary.append(row)
            dump(out / 'summary.json', summary)
            print({k:v for k,v in row.items() if k != 'checks'}, flush=True)


if __name__ == '__main__':
    main()
