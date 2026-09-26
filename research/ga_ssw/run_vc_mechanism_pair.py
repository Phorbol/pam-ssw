"""Paired cell-escape versus posterior-cell-quench development runner.

Research-only orchestration. Both arms use the same atomic_climb kernel,
per-outer-step atomic RNG stream, MC draw, pressure, all-DOF true quench and
calculator budget. The cell-on arm inserts five CBD-style cell cycles before
each atomic climb. This is a clean cell-cycle ablation, not the paper's lambda=2
interleaving schedule, native CBD parity, or a production search claim.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np
from ase import units
from ase.calculators.emt import EMT
from ase.io import read, write

from pamssw.standalone.atomic_climb import atomic_climb
from pamssw.standalone.block_ssw import FixedCellSurface, _cell_displacement
from pamssw.standalone.cbd_cell import CellChart, cell_direction
from pamssw.standalone.cell_relax import cell_quench
from pamssw.standalone.generalized_numerics import safe_lbfgs
from pamssw.standalone.paper_reference import SSWConfig
from pamssw.standalone.vc_geometry import ASEStressSurface


FMAX = 0.03  # eV/A; chosen inside the user's requested 0.01--0.05 range.
STRESS_TOL = 0.001  # eV/A^3; retain the existing physical stress certificate.
PRESSURE = 0.0  # eV/A^3.
STRAIN_LENGTH = 5.0  # A; numerical metric inherited from the qualified rutile pilot.
MAX_STEP = 0.2  # A in the respective optimizer coordinates.


class BudgetExhausted(RuntimeError):
    """The declared per-run call or wall budget censored the trajectory."""


def serial(value):
    from ase import Atoms
    if isinstance(value, Atoms):
        return {
            'numbers': value.numbers.tolist(),
            'positions': value.positions.tolist(),
            'cell': value.cell.array.tolist(),
            'pbc': value.pbc.tolist(),
        }
    if is_dataclass(value):
        return {item.name: serial(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): serial(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


class CountedSurface(ASEStressSurface):
    def __init__(self, calculator, *, request_cap, seconds):
        super().__init__(calculator)
        self.request_cap = int(request_cap)
        self.deadline = time.monotonic() + float(seconds)
        self.exhausted = False
        self.censor_reason = None
        self.stage = 'unset'
        self.ledger = []

    def evaluate(self, atoms):
        if time.monotonic() >= self.deadline:
            self.exhausted = True
            self.censor_reason = 'wall_time_cap'
            raise BudgetExhausted(self.censor_reason)
        if self.requests >= self.request_cap:
            self.exhausted = True
            self.censor_reason = 'request_cap'
            raise BudgetExhausted(self.censor_reason)
        before = self.requests
        try:
            energy, forces, stress = super().evaluate(atoms)
        except Exception as error:
            self.ledger.append({
                'request': self.requests,
                'stage': self.stage,
                'charged': self.requests > before,
                'error': repr(error),
            })
            raise
        self.ledger.append({
            'request': self.requests,
            'stage': self.stage,
            'energy_eV': energy,
            'fmax_eV_per_angstrom': float(np.linalg.norm(forces, axis=1).max()),
            'max_abs_stress_eV_per_angstrom3': float(np.abs(stress).max()),
        })
        return energy, forces, stress


def atomic_config(*, temperature_K=300.0, atomic_gaussians=10,
                  atomic_rotation_hvp=100, fmax=FMAX):
    return SSWConfig(
        width=0.6,
        rotation_bias=100.0,
        max_gaussians=atomic_gaussians,
        temperature_K=temperature_K,
        fmax=fmax,
        relax_steps=400,
        fd_step=0.001,
        rotation_hvp=atomic_rotation_hvp,
        rotation_tol=0.02,
        forward_force=0.1,
        direction_sampling='global',
        rotation_solver='dimer',
        cluster_frame='translation_only',
        quench_optimizer='safe-lbfgs-total',
        lbfgs_memory=500,
    )


def _full_quench(atoms, surface, *, maxiter):
    surface.stage = 'full_cell_true_quench'
    return cell_quench(
        atoms, surface, strain_length=STRAIN_LENGTH, pressure=PRESSURE,
        fmax=FMAX, stress_tol=STRESS_TOL, max_step=MAX_STEP,
        maxiter=maxiter, lbfgs_memory=500,
    )


def _atomic_stage(atoms, surface, *, config, rng, reference_energy):
    surface.stage = 'fixed_cell_atomic_climb'
    return atomic_climb(
        atoms, FixedCellSurface(surface), reference_energy=reference_energy,
        config=config, rng=rng,
    )


def _cell_cycle(atoms, surface, *, rng, cycle_index,
                fd_step=0.005, rotation_requests=6,
                rotation_force_tol=0.1, displacement_fraction=0.15,
                partial_steps=25):
    start = surface.requests
    surface.stage = f'cell_cycle_{cycle_index}_rotation'
    chart = CellChart(atoms)
    q0 = chart.pack(atoms)
    mode = cell_direction(
        chart, q0, rng.normal(size=9), evaluate=surface.evaluate,
        pressure=PRESSURE, fd_step=fd_step,
        max_hvp=rotation_requests - 1,
        rotation_force_tol=rotation_force_tol,
    )
    delta = _cell_displacement(
        atoms.cell.array, mode.direction, displacement_fraction,
        'lattice_frobenius',
    )
    displaced = chart.unpack(q0 + delta)
    trial = displaced.copy()
    fixed = FixedCellSurface(surface)

    def evaluate(position):
        trial.positions = np.asarray(position, dtype=float).reshape(-1, 3)
        energy, forces = fixed.evaluate(trial)
        return energy, -forces.reshape(-1)

    def force_norm(position):
        return float(np.linalg.norm(position.reshape(-1, 3), axis=1).max())

    surface.stage = f'cell_cycle_{cycle_index}_partial_fixed_cell_relax'
    result = safe_lbfgs(
        displaced.positions.reshape(-1), evaluate,
        gradient_norm=lambda gradient: force_norm(-gradient),
        step_norm=force_norm, gtol=FMAX, max_step=MAX_STEP,
        maxiter=partial_steps, lbfgs_memory=500,
    )
    work = displaced.copy()
    work.positions = result.q.reshape(-1, 3)
    affine = np.linalg.solve(chart.reference.cell.array,
                             displaced.cell.array - chart.reference.cell.array)
    record = {
        'cycle': cycle_index,
        'status': 'completed',
        'requests': surface.requests - start,
        'cell_mode': mode,
        'cell_mode_converged': bool(mode.converged),
        'cell_mode_force_calls': int(mode.force_calls),
        'delta_lattice_angstrom': delta,
        'deformation_rms': float(np.linalg.norm(affine) / np.sqrt(3.0)),
        'principal_stretches': np.linalg.svd(np.eye(3) + affine,
                                             compute_uv=False),
        'partial_relax_status': result.status,
        'partial_relax_steps': result.steps,
        'partial_relax_error': result.error,
        'work': work,
    }
    if result.status not in ('converged', 'maxiter'):
        record['status'] = 'partial_relax_failed'
    return work, record


def _metropolis(current, candidate, *, temperature_K, draw):
    delta = float(candidate.objective - current.objective)
    if delta <= 0:
        return True, delta, draw
    if temperature_K <= 0:
        return False, delta, draw
    return bool(draw < math.exp(-delta / (units.kB * temperature_K))), delta, draw


def _rngs(seed, step):
    root = np.random.SeedSequence([int(seed), int(step)])
    cell_seed, atomic_seed, mc_seed = root.spawn(3)
    return (np.random.default_rng(cell_seed),
            np.random.default_rng(atomic_seed),
            float(np.random.default_rng(mc_seed).random()))


def _finalize_arm(report, surface, calculator, *, started):
    report['requests'] = surface.requests
    report['censor_reason'] = surface.censor_reason
    report['wall_seconds'] = float(time.monotonic() - started)
    report['ledger_rows'] = len(surface.ledger)
    checks = []
    check_surface = ASEStressSurface(calculator)
    for landing in report.get('landings', []):
        calculator.reset()
        try:
            e, f, s = check_surface.evaluate(landing.atoms)
            fmax = float(np.linalg.norm(f, axis=1).max())
            stress_max = float(np.abs(s + PRESSURE * np.eye(3)).max())
            checks.append({
                'energy_eV': e,
                'energy_delta_eV': float(e - landing.energy),
                'fmax_eV_per_angstrom': fmax,
                'max_abs_stress_eV_per_angstrom3': stress_max,
                'certified': fmax <= FMAX and stress_max <= STRESS_TOL,
            })
        except Exception as error:
            checks.append({'certified': False, 'error': repr(error)})
    report['fresh_endpoint_checks'] = checks
    report['fresh_check_requests'] = check_surface.requests
    return report


def run_one_arm(atoms, calculator, *, arm, seed, steps, cell_cycles,
                request_cap, seconds, config, atomic_gaussians,
                atomic_rotation_hvp, quench_steps=400):
    if arm not in ('cell_on', 'cell_off'):
        raise ValueError("arm must be 'cell_on' or 'cell_off'")
    if steps < 1 or cell_cycles < 1 or request_cap < 1 or seconds <= 0:
        raise ValueError('steps, cell_cycles, request_cap, and seconds must be positive')
    surface = CountedSurface(calculator, request_cap=request_cap, seconds=seconds)
    atom_cfg = atomic_config(
        temperature_K=config['temperature_K'],
        atomic_gaussians=atomic_gaussians,
        atomic_rotation_hvp=atomic_rotation_hvp,
    )
    started = time.monotonic()
    report = {
        'arm': arm, 'seed': int(seed), 'requested_steps': steps,
        'status': 'running', 'censor_reason': None,
        'request_cap': request_cap, 'wall_cap_seconds': seconds,
        'initial_reference_quench_is_separate': True,
        'config': config | {'atomic': atom_cfg},
        'steps': [], 'landings': [], 'request_ledger': surface.ledger,
    }
    try:
        # Arms begin at the one separately qualified common reference geometry.
        # This initial per-arm refresh establishes a fresh same-PES E/F/stress
        # record and is charged to the arm budget.
        surface.stage = 'arm_start_fresh_certificate'
        energy, forces, stress = surface.evaluate(atoms)
        force_max = float(np.linalg.norm(forces, axis=1).max())
        stress_max = float(np.abs(stress + PRESSURE * np.eye(3)).max())
        if force_max > FMAX or stress_max > STRESS_TOL:
            report.update(status='invalid_start', start_fmax=force_max,
                          start_stress_max=stress_max)
        else:
            from pamssw.standalone.vc_geometry import SymmetricLogStrainChart
            chart = SymmetricLogStrainChart(atoms, strain_length=STRAIN_LENGTH)
            current = chart.evaluate(chart.pack(atoms), lambda a: (energy, forces, stress),
                                     pressure=PRESSURE)
            report['start'] = current.atoms.copy()
            report['start_objective_eV'] = float(current.objective)
            report['start_certificate'] = {
                'fmax_eV_per_angstrom': force_max,
                'max_abs_stress_eV_per_angstrom3': stress_max,
                'certified': True,
            }
            for step in range(steps):
                before = surface.requests
                cell_rng, atom_rng, mc_draw = _rngs(seed, step)
                event = {
                    'step': step,
                    'status': 'running',
                    'requests': 0,
                    'cell_rng_initial_state': cell_rng.bit_generator.state,
                    'atomic_rng_initial_state': atom_rng.bit_generator.state,
                    'mc_draw': mc_draw,
                    'cell_cycles': [],
                    'atomic': None,
                    'true_quench': None,
                    'cell_block_complete': arm == 'cell_off',
                    'atomic_escape_complete': False,
                    'true_quench_complete': False,
                    'full_escape_complete': False,
                    'accepted': False,
                    'input': current.atoms.copy(),
                }
                report['steps'].append(event)
                work = current.atoms.copy()
                try:
                    if arm == 'cell_on':
                        for cycle in range(cell_cycles):
                            cycle_before = surface.requests
                            try:
                                work, record = _cell_cycle(
                                    work, surface, rng=cell_rng, cycle_index=cycle,
                                    fd_step=config['cell_fd_step'],
                                    rotation_requests=config['cell_rotation_requests'],
                                    rotation_force_tol=config['cell_rotation_force_tol'],
                                    displacement_fraction=config['cell_step_fraction'],
                                    partial_steps=config['partial_atom_steps'],
                                )
                            except (BudgetExhausted, ValueError, RuntimeError,
                                    FloatingPointError, np.linalg.LinAlgError) as error:
                                event['cell_cycles'].append({
                                    'cycle': cycle,
                                    'status': 'censored' if surface.exhausted else 'failed',
                                    'failed_stage': surface.stage,
                                    'requests': surface.requests - cycle_before,
                                    'error': repr(error),
                                    'last_work': work.copy(),
                                })
                                raise
                            event['cell_cycles'].append(record)
                            if record['status'] != 'completed':
                                event['status'] = 'partial_cell_relax_failed'
                                break
                        else:
                            event['cell_block_complete'] = True
                    if event['status'] == 'running':
                        atom = _atomic_stage(
                            work, surface, config=atom_cfg, rng=atom_rng,
                            reference_energy=current.objective - PRESSURE * work.get_volume(),
                        )
                        event['atomic'] = atom
                        work = atom.atoms.copy()
                        if atom.status not in ('gaussian_limit', 'lower_true_energy'):
                            event['status'] = 'atomic_' + str(atom.status)
                        else:
                            event['atomic_escape_complete'] = True
                    if event['status'] == 'running':
                        landing = _full_quench(work, surface, maxiter=quench_steps)
                        event['true_quench'] = landing
                        if not landing.converged:
                            event['status'] = 'true_quench_failed'
                        else:
                            event['true_quench_complete'] = True
                            event['full_escape_complete'] = bool(
                                event['cell_block_complete'] and
                                event['atomic_escape_complete'] and
                                event['true_quench_complete'])
                            candidate = landing.evaluation
                            accepted, delta, draw = _metropolis(
                                current, candidate,
                                temperature_K=config['temperature_K'], draw=mc_draw,
                            )
                            event.update(
                                status='valid_landing', accepted=accepted,
                                delta_enthalpy_eV=delta, mc_draw=draw,
                                landing=candidate, work=work.copy(),
                            )
                            report['landings'].append(candidate)
                            if accepted:
                                current = candidate
                    event['work'] = work.copy()
                except (BudgetExhausted, ValueError, RuntimeError,
                        FloatingPointError, np.linalg.LinAlgError) as error:
                    event.update(
                        status='censored' if surface.exhausted else 'failed',
                        failed_stage=surface.stage,
                        error=repr(error), last_work=work.copy(),
                    )
                event['requests'] = surface.requests - before
                event['current_after'] = current.atoms.copy()
                event['current_objective_after_eV'] = float(current.objective)
                if event['status'] in ('failed', 'censored', 'partial_cell_relax_failed',
                                       'true_quench_failed') and surface.exhausted:
                    event['status'] = 'censored'
                if event['status'] not in ('valid_landing',):
                    report['status'] = 'censored' if surface.exhausted else 'completed_with_failed_attempt'
                if surface.exhausted:
                    break
            if surface.exhausted:
                report['status'] = 'censored'
            elif report['status'] == 'running':
                report['status'] = 'completed'
            report['current'] = current.atoms.copy()
            report['current_objective_eV'] = float(current.objective)
    except (BudgetExhausted, ValueError, RuntimeError,
            FloatingPointError, np.linalg.LinAlgError) as error:
        report.update(status='censored' if surface.exhausted else 'failed', error=repr(error))
    return _finalize_arm(report, surface, calculator, started=started)


def run_paired(atoms, calculator, *, seed, steps, cell_cycles, request_cap,
               seconds, fmax=FMAX, atomic_gaussians=10,
               rotation_hvp=100, cell_rotation_requests=6):
    if not math.isclose(float(fmax), FMAX, abs_tol=0.0):
        # Keep the requested fmax visible at the call site without allowing
        # divergent arm thresholds in this preregistered protocol.
        raise ValueError(f'protocol fmax is frozen at {FMAX} eV/A')
    config = {
        'temperature_K': 300.0,
        'cell_fd_step': 0.005,
        'cell_rotation_requests': int(cell_rotation_requests),
        'cell_rotation_force_tol': 0.1,
        'cell_step_fraction': 0.15,
        'partial_atom_steps': 25,
        'partial_atom_fmax_eV_per_angstrom': FMAX,
        'cell_direction_lifetime': 'fresh random anchor each outer step; Python dimer, approximate when budget ended',
        'cell_metric': 'lattice Frobenius; delta_L/L_F=0.15',
        'atomic': atomic_config(atomic_gaussians=atomic_gaussians,
                                atomic_rotation_hvp=rotation_hvp),
        'true_quench': {
            'method': 'existing cell_quench / Safe-total',
            'strain_length_angstrom': STRAIN_LENGTH,
            'fmax_eV_per_angstrom': FMAX,
            'max_abs_stress_plus_pI_eV_per_angstrom3': STRESS_TOL,
            'maxiter': 400,
            'lbfgs_memory': 500,
        },
        'mc': 'same Metropolis objective E+pV and temperature; per-step draw comes from an independent, identical seed stream in both arms',
        'matching': 'paired per-step atomic RNG initial state and MC draw; physical atomic directions may differ after geometry changes',
        'paper_schedule_boundary': 'cell_on includes cell cycles and an atomic climb on every outer step for a matched atomic stage; this is not paper lambda=2 interleaving',
    }
    return {
        arm: run_one_arm(
            atoms.copy(), calculator, arm=arm, seed=seed, steps=steps,
            cell_cycles=cell_cycles, request_cap=request_cap, seconds=seconds,
            config=config, atomic_gaussians=atomic_gaussians,
            atomic_rotation_hvp=rotation_hvp,
        )
        for arm in ('cell_on', 'cell_off')
    } | {
        'pairing': config['matching'],
        'protocol': config,
    }


def _metadata(args, calculator):
    try:
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--short'], text=True).strip()
    except Exception:
        head, dirty = None, None
    model_hash = None
    if args.model:
        digest = hashlib.sha256()
        with Path(args.model).open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        model_hash = digest.hexdigest()
    packages = {}
    for name in ('ase', 'mace-torch', 'torch', 'numpy', 'scipy'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        'arguments': vars(args), 'git_head': head, 'git_dirty': dirty,
        'input_path': str(Path(args.input).resolve()),
        'input_sha256': hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        'model_path': str(Path(args.model).resolve()) if args.model else None,
        'model_sha256': model_hash, 'packages': packages,
        'python': platform.python_version(), 'platform': platform.platform(),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'calculator_class': type(calculator).__qualname__,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--backend', choices=('mace', 'emt'), default='mace')
    parser.add_argument('--model')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--cell-cycles', type=int, default=5)
    parser.add_argument('--request-cap-per-arm', type=int, default=6000)
    parser.add_argument('--seconds-per-arm', type=float, default=600.0)
    parser.add_argument('--gate-request-cap', type=int, default=300)
    parser.add_argument('--gate-seconds', type=float, default=300.0)
    parser.add_argument('--atomic-gaussians', type=int, default=10)
    parser.add_argument('--rotation-hvp', type=int, default=100)
    parser.add_argument('--cell-rotation-requests', type=int, default=6)
    args = parser.parse_args(argv)
    if args.steps < 2:
        parser.error('--steps must be at least 2 for this repeated cell-on/off screen; it deliberately runs the same atomic stage on every step and is not the paper lambda=2 schedule')
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    atoms = read(args.input, index=0)
    write(output / 'input.extxyz', atoms)
    if args.backend == 'emt':
        calculator = EMT()
    else:
        if not args.model:
            parser.error('--model is required for --backend mace')
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        from mace.calculators import MACECalculator
        calculator = MACECalculator(
            model_paths=args.model, head='omat_pbe', device=args.device,
            default_dtype='float64', enable_cueq=False, enable_oeq=False,
        )
    metadata = _metadata(args, calculator)
    (output / 'provenance.json').write_text(json.dumps(metadata, indent=2) + '\n')
    source_snapshot = output / 'source'
    source_snapshot.mkdir()
    shutil.copy2(__file__, source_snapshot / 'run_vc_mechanism_pair.py')
    shutil.copytree('pamssw', source_snapshot / 'pamssw',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    gate = CountedSurface(calculator, request_cap=args.gate_request_cap,
                          seconds=args.gate_seconds)
    start = time.monotonic()
    reference = None
    gate_error = None
    try:
        reference = _full_quench(atoms, gate, maxiter=400)
    except Exception as error:
        gate_error = repr(error)
    gate_result = {
        'status': 'qualified' if reference is not None and reference.converged else
                  ('censored' if gate.exhausted else 'failed'),
        'requests': gate.requests, 'wall_seconds': time.monotonic() - start,
        'censor_reason': gate.censor_reason, 'error': gate_error,
        'quench': serial(reference), 'ledger': gate.ledger,
    }
    if reference is None or not reference.converged:
        (output / 'reference-gate.json').write_text(json.dumps(gate_result, indent=2) + '\n')
        print(json.dumps({'status': gate_result['status'], 'gate_requests': gate.requests}))
        return 2
    # One new independent calculator request qualifies the common reference.
    gate.stage = 'reference_fresh_certificate'
    calculator.reset()
    try:
        energy, forces, stress = gate.evaluate(reference.evaluation.atoms)
        fmax = float(np.linalg.norm(forces, axis=1).max())
        stress_max = float(np.abs(stress + PRESSURE * np.eye(3)).max())
        fresh_error = None
    except Exception as error:
        energy = fmax = stress_max = None
        fresh_error = repr(error)
    gate_result['fresh_certificate'] = {
        'energy_eV': energy, 'fmax_eV_per_angstrom': fmax,
        'max_abs_stress_eV_per_angstrom3': stress_max,
        'certified': (fresh_error is None and fmax <= FMAX and stress_max <= STRESS_TOL),
        'error': fresh_error,
    }
    gate_result['requests'] = gate.requests
    if not gate_result['fresh_certificate']['certified']:
        gate_result['status'] = 'fresh_certificate_failed'
        (output / 'reference-gate.json').write_text(json.dumps(gate_result, indent=2) + '\n')
        return 2
    write(output / 'qualified-start.extxyz', reference.evaluation.atoms)
    (output / 'reference-gate.json').write_text(json.dumps(gate_result, indent=2) + '\n')
    pair = run_paired(
        reference.evaluation.atoms, calculator, seed=args.seed,
        steps=args.steps, cell_cycles=args.cell_cycles,
        request_cap=args.request_cap_per_arm, seconds=args.seconds_per_arm,
        atomic_gaussians=args.atomic_gaussians, rotation_hvp=args.rotation_hvp,
        cell_rotation_requests=args.cell_rotation_requests,
    )
    payload = {
        'status': 'completed',
        'scientific_status': 'bounded_development_mechanism_screen_not_validation',
        'reference_gate': {
            'requests_separate_from_arm_caps': gate.requests,
            'fresh_certificate': gate_result['fresh_certificate'],
        },
        'pairing': pair['pairing'], 'protocol': serial(pair['protocol']),
        'cell_on': serial(pair['cell_on']), 'cell_off': serial(pair['cell_off']),
    }
    (output / 'result.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    for name in ('cell_on', 'cell_off'):
        arm = pair[name]
        if arm.get('landings'):
            write(output / f'{name}-landings.extxyz',
                  [item.atoms for item in arm['landings']])
        if 'current' in arm:
            write(output / f'{name}-current.extxyz', arm['current'])
    print(json.dumps({
        'status': payload['status'],
        'gate_requests_including_fresh': gate.requests,
        'arms': {name: {'status': pair[name]['status'],
                        'requests': pair[name].get('requests'),
                        'completed_steps': len(pair[name].get('steps', [])),
                        'valid_landings': len(pair[name].get('landings', [])),
                        'censor_reason': pair[name].get('censor_reason')}
                 for name in ('cell_on', 'cell_off')},
    }), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
