"""Small analytic LJ validation; no production or global-search quality claim.

Run from the repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m runs.20260908-certified-cell-quench-validation.run_cpu_smoke
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

import ase
from ase import units
from ase.build import bulk
from ase.calculators.lj import LennardJones
import numpy as np

from pamssw import LSSSWConfig, state_from_atoms, write_state
from pamssw.calculators import ASECalculator
from pamssw.walker import SurfaceWalker


def main():
    output = Path(__file__).resolve().parent
    cases = []
    for mode, pressure in [('fixed', 0.), ('volume_only', 0.), ('shape', 0.), ('volume_only', 2.)]:
        state = state_from_atoms(bulk('Ar', 'fcc', a=1.8, cubic=True))
        config = LSSSWConfig(
            max_trials=2, max_steps_per_walk=2, oracle_candidates=2,
            proposal_relax_steps=8, quench_maxiter=200,
            quench_optimizer='ase-fire', quench_fallback_optimizer='ase-lbfgs',
            quench_cell_mode=mode, external_pressure_gpa=pressure,
            quench_fmax=0.005, quench_stress_tol=0.001,
            max_force_evals=3000, rng_seed=8,
        )
        walker = SurfaceWalker(ASECalculator(LennardJones(rc=2.7)), config, True)
        result = walker.run(state)
        # Independent calculator verifies the delivered endpoint. This is one
        # additional QA call, explicitly outside the search's FE budget.
        check = ASECalculator(LennardJones(rc=2.7)).evaluate(result.best_state)
        volume = float(np.linalg.det(result.best_state.cell))
        fmax = float(np.max(np.linalg.norm(check.gradient, axis=1)))
        residual = check.stress + pressure * units.GPa * np.eye(3)
        stress_norm = float(abs(np.trace(residual)/3) if mode == 'volume_only'
                            else np.max(np.abs(residual)))
        objective = check.energy + pressure * units.GPa * volume
        assert abs(objective - result.best_energy) < 1e-8
        assert fmax <= config.quench_fmax
        assert check.energy < 0, "LJ endpoint escaped to the cutoff zero-energy plateau"
        if mode != 'fixed':
            assert stress_norm <= config.quench_stress_tol
        counts = walker.calculator.snapshot()
        assert counts.total == result.stats['force_evaluations']
        assert counts.as_dict()['unattributed'] == 0
        label = f'{mode}-pressure{pressure:g}'
        filename = label + '.extxyz'
        write_state(output / filename, result.best_state)
        cases.append(dict(
            case=label, config=asdict(config), initial_cell=state.cell.tolist(),
            final_cell=result.best_state.cell.tolist(), potential_energy=check.energy,
            objective_energy=result.best_energy, volume=volume, force_max=fmax,
            stress_residual=stress_norm, stress_certified=(mode != 'fixed' and stress_norm <= config.quench_stress_tol),
            stats=result.stats, purpose_counts=counts.as_dict(), independent_validation_calls=1,
            structure=filename,
        ))
    payload = dict(
        claim='CPU analytic LJ quench/integration validation only; no global optimum or MACE validation',
        base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(Path('pamssw').rglob('*.py'))},
        versions=dict(ase=ase.__version__, numpy=np.__version__), cases=cases,
    )
    (output / 'smoke.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
