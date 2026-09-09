"""Bounded native-free Cu13/EMT surface and direction integration probe.

Run from the worktree: python -m research.ga_ssw.run_independent_surface_probe
This is not a complete SSW trajectory or a search-efficiency benchmark.
"""
import json
import time
from pathlib import Path

import ase
import numpy as np
from ase.calculators.emt import EMT
from ase.cluster.icosahedron import Icosahedron
from ase.io import write

from pamssw.standalone.surface import ASESurface, quench
from pamssw.standalone.direction import reference_soft_mode


def main():
    output = Path('research/ga_ssw/evidence/independent-cu13-surface')
    output.mkdir(parents=True, exist_ok=True)
    atoms = Icosahedron('Cu', 2)
    atoms.positions += np.random.default_rng(20260909).normal(scale=.03, size=(13, 3))
    write(output / 'input.extxyz', atoms)
    start = time.monotonic()
    surface = ASESurface(EMT())
    result = quench(atoms, surface, fmax=.001, steps=100)
    write(output / 'quenched.extxyz', result.atoms)
    mode = reference_soft_mode(result.atoms,
        np.random.default_rng(3).normal(size=(13, 3)), fd_step=.001,
        max_hvp=25, residual_tol=.001, evaluate=surface.evaluate)
    independent = ASESurface(EMT())
    checked_energy, checked_forces = independent.evaluate(result.atoms)
    report = dict(
        status='component_integration_only_not_complete_ssw',
        backend='ASE EMT', ase_version=ase.__version__, numpy_version=np.__version__,
        external_native_programs_used=False, system='Cu13',
        input_seed=20260909, perturbation_sigma_angstrom=.03, direction_seed=3,
        parameters_are_diagnostic_not_universal_defaults=True,
        force_tolerance_eV_per_angstrom=.001, optimizer_step_limit=100,
        quench=dict(energy_eV=result.energy, max_force_eV_per_angstrom=result.max_force,
                    force_converged=result.converged, optimizer_steps=result.optimizer_steps,
                    evaluation_requests=result.evaluation_requests),
        soft_mode=dict(curvature_eV_per_angstrom2=mode.curvature,
                       residual_eV_per_angstrom2=mode.residual_norm,
                       fd_step_angstrom=.001, hvp_budget=25,
                       residual_tolerance=.001, hvp_calls=mode.hvp_calls,
                       force_requests=mode.force_calls, residual_converged=mode.converged,
                       global_rigid_modes_removed=False),
        fresh_calculator_check=dict(energy_eV=checked_energy,
             max_force_eV_per_angstrom=float(np.linalg.norm(checked_forces, axis=1).max()),
             evaluation_requests=independent.requests),
        total_evaluation_requests=surface.requests + independent.requests,
        wall_seconds=time.monotonic()-start,
        limitations=['No SSW escape, LS or GA trajectory in this probe.',
                     'No Hessian-positive or physical stability certificate.',
                     'Soft direction can contain global rigid motion.',
                     'API request counts are not native or SCF cost counts.'])
    (output / 'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
