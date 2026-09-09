"""Single-step independent C60/GFN2-xTB LS-SSW workflow, not efficacy proof.

Run with one OpenMP/BLAS thread and an external 300-second timeout.
The LS bond energy 3.61 eV and target 0.02 eV/atom come from LS-SSW paper/SI.
The pair cutoff is chosen in the observed C60 first/second-shell gap; it is
not a new universal elemental cutoff. All numerical settings are recorded.
"""
from dataclasses import asdict
import json
from pathlib import Path
import signal
import time

import numpy as np
from ase.build import molecule
from ase.io import write
from tblite.ase import TBLite

from pamssw.standalone import ASESurface, SSWConfig, LSSettings, run_ls_ssw
from pamssw.standalone.surface import quench
from .run_independent_water_ga import serial


def main():
    output = Path('research/ga_ssw/evidence/independent-c60-gfn2-ls')
    output.mkdir(parents=True, exist_ok=False)
    (output/'run_script.py').write_text(Path(__file__).read_text())
    started = time.monotonic()
    log = (output/'evaluations.jsonl').open('w')
    class TracedSurface(ASESurface):
        def evaluate(self, atoms):
            e, f = super().evaluate(atoms)
            log.write(json.dumps(dict(request=self.requests, energy=e,
                max_force=float(np.linalg.norm(f, axis=1).max()),
                positions=atoms.positions.tolist()))+'\n')
            log.flush()
            if self.requests % 100 == 0:
                print(f'requests={self.requests}, elapsed={time.monotonic()-started:.1f}s', flush=True)
            return e, f
    surface = TracedSurface(TBLite(method='GFN2-xTB', verbosity=0, accuracy=.001))
    def terminate(signum, frame):
        raise KeyboardInterrupt('external bounded-run timeout')
    signal.signal(signal.SIGTERM, terminate)
    try:
        atoms = molecule('C60')
        write(output/'input.extxyz', atoms)
        initial = quench(atoms, surface, fmax=.01, steps=400)
        (output/'initial-quench.json').write_text(json.dumps(serial(initial), indent=2)+'\n')
        if not initial.converged:
            raise RuntimeError('explicit starting-structure quench failed')
        distances = np.sort(initial.atoms.get_all_distances(), axis=1)
        first_max, second_min = float(distances[:, 3].max()), float(distances[:, 4].min())
        if first_max >= second_min:
            raise RuntimeError('no unambiguous three-neighbor C60 shell gap')
        cutoff = .5 * (first_max + second_min)
        ls = LSSettings(bond_energies={(6, 6): 3.61}, bond_lengths={(6, 6): cutoff},
                        target_per_atom=.02)
        config = SSWConfig(width=.2, rotation_bias=100., max_gaussians=2,
            temperature_K=300., fmax=.01, relax_steps=400, fd_step=.0001,
            rotation_hvp=100, rotation_tol=.02, direction_sampling='paper')
        (output/'config.json').write_text(json.dumps(dict(
            backend='tblite 0.7.0 GFN2-xTB', accuracy=.001, ssw=asdict(config),
            ls=serial(ls), seed=20260909, first_shell_max=first_max,
            second_shell_min=second_min, cutoff_definition='midpoint of observed shell gap',
            source='LS DOI 10.1021/acs.jctc.4c01081 and SI: C-C 3.61 eV, C60 target 0.02 eV/atom',
            budget='one CPU; 300-second external limit; one outer LS step',
            interpretation='workflow validation, not comparison of search efficiency'), indent=2)+'\n')
        result = run_ls_ssw(initial.atoms, surface, steps=1, config=config,
                           rng=np.random.default_rng(20260909), ls=ls)
        (output/'result.json').write_text(json.dumps(serial(result), indent=2)+'\n')
        checks = []
        for minimum in result.minima:
            fresh = ASESurface(TBLite(method='GFN2-xTB', verbosity=0, accuracy=.001))
            e, f = fresh.evaluate(minimum.atoms)
            checks.append(dict(energy=e, max_force=float(np.linalg.norm(f, axis=1).max()),
                force_pass=bool(np.linalg.norm(f, axis=1).max() <= .01)))
        summary = dict(status=result.status, stages=[r.status for r in result.records],
            ls_energy_response=[r.energy_response for r in result.records],
            true_landings=len(result.minima), checks=checks,
            prefix_quench_requests=initial.evaluation_requests,
            search_requests=result.evaluation_requests, total_requests=surface.requests+len(checks),
            wall_seconds=time.monotonic()-started, physical_stability_certified=False,
            different_basins_certified=False)
        (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps(summary, indent=2))
    except BaseException as error:
        (output/'failed.json').write_text(json.dumps(dict(error=repr(error),
            requests=surface.requests, wall_seconds=time.monotonic()-started), indent=2)+'\n')
        raise
    finally:
        log.close()


if __name__ == '__main__':
    main()
