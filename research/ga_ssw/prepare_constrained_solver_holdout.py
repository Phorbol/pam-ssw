"""Prepare or execute the fixed-substrate direction-solver diagnostic."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


SOLVERS = ('generalized-dimer', 'ritz', 'dimer', 'broyden-euclidean')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
                        default=Path('research/ga_ssw/evidence/constrained-direction-diagnostic-20260912'))
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    shutil.copy2(Path(__file__).resolve(), out / 'runner.py')
    snapshot = out / 'source'
    shutil.copytree('pamssw', snapshot / 'pamssw', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    if not args.execute:
        (out / 'PREPARED-NO-PES.md').write_text(
            'Source snapshot prepared. Re-run with --execute only after adapter review.\n')
        return
    env = os.environ.copy()
    env['PYTHONPATH'] = str(snapshot) + os.pathsep + str(Path.cwd().resolve())
    env['PAM_CONSTRAINED_DIAGNOSTIC_CHILD'] = '1'
    env['PAM_CONSTRAINED_OUTPUT'] = str(out)
    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--output', str(out), '--execute'],
                   env=env, check=True)


def _child(out):
    import json
    import time
    import numpy as np
    from ase import Atoms
    from ase.build import add_adsorbate, fcc111
    from ase.calculators.emt import EMT
    from ase.constraints import FixAtoms
    from ase.data import reference_states
    import pamssw
    from pamssw.standalone import ASESurface
    from pamssw.standalone.constrained_reference import ConstrainedSSWConfig, run_constrained_ssw
    from research.ga_ssw.compare_vc_arms import serial
    assert str(Path(pamssw.__file__).resolve()).startswith(str((out / 'source').resolve()))
    if 'rotation_solver' not in ConstrainedSSWConfig.__dataclass_fields__:
        raise RuntimeError('rotation_solver adapter unavailable; no PES run')
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    (out / 'plan.json').write_text(json.dumps(dict(cases=['Cu', 'Al'], seeds=[3], solvers=SOLVERS,
        steps=2, request_cap=6000, wall_seconds=60, temperature_K=300.,
        gradient_tol=.1, gradient_tol_definition='whole active Cartesian gradient L2 norm',
        fmax=.01, relax_steps=200, rotation_hvp=100, max_gaussians=14, width=.2,
        rotation_bias=100., model='ASE EMT', source='ASE fcc111; Al reference_states[13][a]=4.05',
        interpretation='fixed-substrate direction diagnostic'), indent=2) + '\n')
    for element, lattice in (('Cu', 3.6), ('Al', float(reference_states[13]['a']))):
        atoms = fcc111(element, size=(2, 2, 3), a=lattice, vacuum=8.)
        fixed = np.flatnonzero(atoms.get_tags() > 1)
        add_adsorbate(atoms, element, height=2., position='fcc')
        atoms.set_constraint(FixAtoms(indices=fixed))
        base_positions = atoms.positions.copy(); base_cell = atoms.cell.array.copy()
        active = np.setdiff1d(np.arange(len(atoms)), fixed)
        top_support = np.flatnonzero(atoms.get_tags() == 1)
        for seed in (3,):
            for solver in SOLVERS:
                ledger = []
                started = time.monotonic()
                class Counted(ASESurface):
                    def evaluate(self, candidate):
                        if self.requests >= 6000:
                            ledger.append({'event': 'request_cap_denial', 'charged': False})
                            raise RuntimeError('diagnostic request cap reached')
                        if time.monotonic() - started >= 60:
                            ledger.append({'event': 'wall_cap_denial', 'charged': False})
                            raise RuntimeError('diagnostic wall cap reached')
                        if not np.array_equal(candidate.positions[fixed], base_positions[fixed]):
                            ledger.append({'event': 'fixed_coordinate_violation', 'charged': False})
                            raise RuntimeError('fixed substrate changed before paid request')
                        if not np.array_equal(candidate.cell.array, base_cell):
                            ledger.append({'event': 'cell_violation', 'charged': False})
                            raise RuntimeError('cell changed before paid request')
                        try:
                            energy, forces = super().evaluate(candidate)
                            ledger.append(dict(ok=True, positions=candidate.positions.tolist(),
                                cell=candidate.cell.array.tolist(), energy=float(energy),
                                forces=np.asarray(forces).tolist()))
                            return energy, forces
                        except Exception as error:
                            ledger.append(dict(ok=False, positions=candidate.positions.tolist(),
                                cell=candidate.cell.array.tolist(), error=f'{type(error).__name__}: {error}'))
                            raise
                surface = Counted(EMT())
                config = ConstrainedSSWConfig(width=.2, rotation_bias=100., temperature_K=300.,
                    gradient_tol=.1, fmax=.01, relax_steps=200, rotation_hvp=100,
                    max_gaussians=14, rotation_solver=solver)
                try:
                    result = run_constrained_ssw(atoms, surface, steps=2, config=config,
                                                 rng=np.random.default_rng(seed))
                    status, error = getattr(result, 'status', 'completed'), None
                except Exception as exc:
                    result, status, error = None, 'failed', f'{type(exc).__name__}: {exc}'
                fresh = []
                if result is not None:
                    for minimum in result.minima:
                        try:
                            raw = minimum.atoms.copy(); raw.set_constraint(); raw.calc = EMT()
                            energy = float(raw.get_potential_energy()); forces = raw.get_forces()
                            fresh.append(dict(ok=True, energy=energy, energy_error=energy - float(minimum.energy),
                                active_fmax=float(np.linalg.norm(forces[active], axis=1).max()),
                                full_raw_fmax=float(np.linalg.norm(forces, axis=1).max()),
                                fixed_exact=bool(np.array_equal(raw.positions[fixed], base_positions[fixed])),
                                cell_exact=bool(np.array_equal(raw.cell.array, base_cell)),
                                adatom_height=float(raw.positions[-1, 2] - raw.positions[top_support, 2].max()),
                                min_distance=float(raw.get_all_distances(mic=True)[np.triu_indices(len(raw), 1)].min())))
                        except Exception as exc:
                            fresh.append(dict(ok=False, error=f'{type(exc).__name__}: {exc}'))
                key = f'{element}-seed{seed}-{solver}'
                (out / f'{key}-ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')
                (out / f'{key}-result.json').write_text(json.dumps(
                     {'status': status, 'error': error, 'requests': surface.requests,
                     'config': asdict(config),
                     'initial': serial(atoms), 'fixed_indices': fixed.tolist(),
                     'active_indices': active.tolist(), 'result': serial(result),
                     'fresh': fresh}, indent=2) + '\n')
                rows.append(dict(key=key, status=status, error=error,
                                 requests=surface.requests, fresh_count=len(fresh)))
    (out / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    if os.environ.get('PAM_CONSTRAINED_DIAGNOSTIC_CHILD') == '1':
        _child(Path(os.environ.get('PAM_CONSTRAINED_OUTPUT', '.')))
    else:
        main()
