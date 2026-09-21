"""Bounded Fe7C3-80 filtered versus unfiltered independent LS comparison.

This is an execution wrapper around ``compare_material_arms.run_material_arm``.
It does not alter either search kernel.  The input directory must contain
``input.json`` (the frozen, qualified 80-atom ASE JSON) and ``plan.json``.
The plan supplies the MACE model/hash and the qualification ``strain_length_A``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from ase import Atoms

from pamssw.standalone.block_ssw import BlockSSWConfig
from pamssw.standalone.paper_reference import SSWConfig, LSSettings
from pamssw.standalone.vc_reference import run_vc_ssw
from pamssw.standalone.vc_geometry import ASEStressSurface
from pamssw.standalone.vc_reference import VCSSWConfig
from research.ga_ssw.compare_material_arms import (
    BudgetExhausted, run_material_arm, serial, validate_configs,
)


class RecordedBudgetSurface(ASEStressSurface):
    """Count and archive every attempted E/F/stress request, including errors."""

    def __init__(self, calculator, *, cap, deadline, output):
        super().__init__(calculator)
        self.cap = int(cap)
        self.limit = int(cap)
        self.deadline = float(deadline)
        self.output = Path(output)
        self.stage = "search"
        self.exhausted = False

    def evaluate(self, atoms):
        if self.requests >= self.limit:
            self.exhausted = True
            self._write(dict(request=self.requests, stage=self.stage, charged=False,
                             error=f"BudgetExhausted: request limit exhausted ({self.limit})",
                             atoms=atoms.copy()))
            raise BudgetExhausted(f"request limit exhausted ({self.limit})")
        if time.monotonic() >= self.deadline:
            self.exhausted = True
            self._write(dict(request=self.requests, stage=self.stage, charged=False,
                             error="BudgetExhausted: 480 second comparison wall cap reached",
                             atoms=atoms.copy()))
            raise BudgetExhausted("480 second comparison wall cap reached")
        before = self.requests
        try:
            energy, forces, stress = super().evaluate(atoms)
        except Exception as error:
            self._write(dict(request=self.requests, stage=self.stage,
                             charged=self.requests > before, error=repr(error),
                             atoms=atoms.copy()))
            raise
        self._write(dict(request=self.requests, stage=self.stage, charged=True,
                         energy=energy, forces=forces, stress=stress,
                         atoms=atoms.copy()))
        return energy, forces, stress

    def _write(self, row):
        with (self.output / "evaluations.jsonl").open("a") as stream:
            stream.write(json.dumps(serial(row), allow_nan=False) + "\n")

    def replace_calculator(self, calculator):
        self.calculator = calculator


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atoms(data):
    if not isinstance(data, dict):
        raise ValueError("input.json must be an ASE JSON object")
    if "numbers" in data:
        return Atoms(numbers=data["numbers"], positions=data["positions"],
                     cell=data["cell"], pbc=data["pbc"])
    return Atoms(symbols=data["symbols"], positions=data["positions"],
                 cell=data["cell"], pbc=data["pbc"])


def _model_path(plan):
    value = plan.get("model", plan.get("model_path"))
    if isinstance(value, dict):
        value = value.get("path")
    if not value:
        raise ValueError("plan.json must provide model or model_path")
    return Path(value)


def _strain_length(plan):
    value = plan.get("strain_length_A")
    if value is None:
        value = plan.get("qualification", {}).get("strain_length_A")
    if value is None:
        raise ValueError("plan.json must provide qualified strain_length_A")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("strain_length_A must be finite and positive")
    return value


def _configs(plan):
    """Build fully materialized matched configs; plan overrides are explicit."""
    L = _strain_length(plan)
    atomic = dict(width=.6, rotation_bias=100., max_gaussians=14,
                  temperature_K=300., fmax=.001, relax_steps=300,
                  fd_step=1e-4, rotation_hvp=100, rotation_tol=.02,
                  forward_force=.1, direction_sampling="global",
                  rotation_solver="dimer", cluster_frame="translation_only",
                  quench_optimizer="safe-lbfgs-total", lbfgs_memory=10)
    atomic.update(plan.get("atomic", {}))
    block = dict(quench_length=L, cell_cycles=5, atomic_period=2,
                 cell_step_fraction=.15, cell_fd_step=.005,
                 cell_rotation_requests=6, cell_rotation_force_tol=.1,
                 partial_atom_steps=25, pressure=0., stress_tol=.0001,
                 max_step=.2)
    block.update(plan.get("block", {}))
    joint = dict(strain_length=L, width=.6, rotation_bias=100., pressure=0.,
                 temperature_K=300., forward_force=.1, max_gaussians=14,
                 gradient_tol=.001, fmax=.001, stress_tol=.0001, max_step=.2,
                 relax_steps=300, fd_step=1e-4, rotation_hvp=100,
                 rotation_tol=.02, lbfgs_memory=10)
    joint.update(plan.get("joint", {}))
    atomic_config = SSWConfig(**atomic)
    block_config = BlockSSWConfig(atomic=atomic_config, **block)
    joint_config = VCSSWConfig(**joint)
    validate_configs(atomic_config, block_config, joint_config)
    return atomic_config, block_config, joint_config


def _runtime(model):
    import torch
    packages = {}
    for name in ("ase", "numpy", "scipy", "torch", "mace-torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(python=__import__("sys").version, packages=packages,
                torch=torch.__version__, torch_num_threads=torch.get_num_threads(),
                torch_num_interop_threads=torch.get_num_interop_threads(),
                cuda_available=bool(torch.cuda.is_available()),
                cuda_device_count=int(torch.cuda.device_count()),
                cuda_device=(torch.cuda.get_device_name(0)
                             if torch.cuda.is_available() else None),
                model=str(model), model_sha256=_sha256(model))


def _fresh_checks(report, surface, pressure, fmax, stress_tol, reserve):
    landings = report.get("landings") or []
    checks = []
    for index, landing in enumerate(landings[:reserve]):
        atoms = landing["atoms"]
        try:
            energy, forces, stress = surface.evaluate(atoms)
            residual = np.asarray(stress) + pressure * np.eye(3)
            checks.append(dict(index=index, status="checked", energy=float(energy),
                              objective=float(energy + pressure * atoms.get_volume()),
                              energy_error=float(energy-landing["energy"]),
                              objective_error=float(energy+pressure*atoms.get_volume()-landing["objective"]),
                              force_error=float(np.abs(forces-landing["forces"]).max()),
                              stress_error=float(np.abs(stress-landing["stress"]).max()),
                              fmax=float(np.linalg.norm(forces, axis=1).max()),
                              stress_max=float(np.abs(residual).max()),
                              certified=bool(np.linalg.norm(forces, axis=1).max() <= fmax
                                             and np.abs(residual).max() <= stress_tol),
                              atoms=atoms.copy()))
        except Exception as error:
            checks.append(dict(index=index, status="failed", error=repr(error),
                               atoms=atoms.copy()))
    return dict(status="no_landings" if not landings else "checked",
                requested=len(landings), checked=len(checks), reserved=reserve,
                omitted=max(0, len(landings) - len(checks)), checks=checks)


def ls_settings(arm):
    # Frozen release raw lookup, with recovered +0.1 Angstrom neighbor tolerance.
    # This is an explicitly version-derived experiment, not a chemical default.
    return LSSettings(
        bond_energies={(6,6):3.4468400478363037,(6,26):13.779999732971191,(26,26):3.6298000812530518},
        bond_lengths={(6,6):1.5399999618530273+.1,(6,26):1.9199999570846558+.1,(26,26):2.630000114440918+.1},
        target_per_atom=.01,energy_filter={(26,26):0.} if arm=='ls_filter' else None)


def run_ls_arm(atoms, surface, *, arm, joint, seed):
    settings=ls_settings(arm)
    result=run_vc_ssw(atoms,surface,steps=2,config=joint,rng=np.random.default_rng(seed),ls=settings)
    landings=[]
    if result.minima:
        ev=result.minima[0]
        landings.append(dict(atoms=ev.atoms,energy=ev.energy,objective=ev.objective,
            forces=ev.forces,stress=ev.stress,index=-1,accepted=True))
    for event in result.records[1:]:
        ev=event.get('landing');cert=event.get('certificate')
        if ev is not None and cert is not None and cert['certified'] and event['status'] in ('gaussian_limit','lower_true_enthalpy'):
            landings.append(dict(atoms=ev.atoms,energy=ev.energy,objective=ev.objective,
                forces=ev.forces,stress=ev.stress,index=event['index'],accepted=event['accepted']))
    assert result.requests==surface.requests==sum(r['requests'] for r in result.records)
    return dict(arm=arm,seed=seed,steps_requested=2,ls=settings,joint_config=joint,
        records=result.records,landings=landings,current=result.current,best=result.best,
        status=result.status,requests=result.requests,valid_proposals=max(0,len(landings)-1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root
    plan = json.loads((root / "plan.json").read_text())
    input_data = json.loads((root / "input.json").read_text())
    atoms = _atoms(input_data)
    if len(atoms) != 80 or not atoms.pbc.all():
        raise ValueError("input.json must be the qualified fully periodic 80-atom cell")
    model = _model_path(plan)
    if not model.is_file():
        raise FileNotFoundError(model)
    expected_sha = plan.get("model_sha256")
    if expected_sha is None and isinstance(plan.get("model"), dict):
        expected_sha = plan["model"].get("sha256")
    actual_sha = _sha256(model)
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(f"model SHA256 mismatch: expected {expected_sha}, got {actual_sha}")
    atomic, block, joint = _configs(plan)
    out = root / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    (out / "input.json").write_text(json.dumps(serial(atoms), indent=2) + "\n")
    (out / "plan-used.json").write_text(json.dumps(dict(plan=plan,
        atomic=asdict(atomic), block=asdict(block), joint=asdict(joint),
        model_sha256=actual_sha), indent=2, allow_nan=False) + "\n")
    script_copy = out / "run_fe7c3_ls_comparison.py"
    shutil.copy2(Path(__file__), script_copy)

    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from mace.calculators import MACECalculator
    if not torch.cuda.is_available():raise RuntimeError("CUDA required, no CPU fallback")
    runtime = _runtime(model)
    (out / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    started = time.monotonic()
    for seed in (7, 101):
        for arm in ("ls_all", "ls_filter"):
            arm_out = out / f"{arm}-seed{seed}"
            arm_out.mkdir(parents=True, exist_ok=False)
            arm_started = time.monotonic()
            deadline = arm_started + 480.
            calc = MACECalculator(model_paths=str(model), device="cuda",
                                  default_dtype="float64", enable_cueq=False)
            surface = RecordedBudgetSurface(calc, cap=2000, deadline=deadline,
                                            output=arm_out)
            surface.limit = 1997  # reserve at most three calls for final fresh checks
            surface.stage = "search"
            try:
                report = run_ls_arm(atoms.copy(), surface, arm=arm, joint=joint, seed=seed)
            except Exception as error:
                report = dict(arm=arm, seed=seed, steps_requested=2,
                              status="exception", error=repr(error), records=[],
                              landings=[], requests=surface.requests)
            report["runtime"] = runtime
            report["search_requests"] = surface.requests
            report["search_status"] = report.get("status")
            (arm_out / "search-result.json").write_text(
                json.dumps(serial(report), indent=2, allow_nan=False) + "\n")
            del calc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            fresh_calc = MACECalculator(model_paths=str(model), device="cuda",
                                        default_dtype="float64", enable_cueq=False)
            surface.replace_calculator(fresh_calc)
            surface.limit = 2000
            surface.stage = "fresh"
            report["fresh"] = _fresh_checks(
                report, surface, joint.pressure, joint.fmax, joint.stress_tol, 3)
            report["requests"] = surface.requests
            report["wall_seconds"] = time.monotonic() - arm_started
            report["campaign_seconds"] = time.monotonic() - started
            report["status"] = "censored" if surface.exhausted else report.get("status")
            (arm_out / "result.json").write_text(
                json.dumps(serial(report), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
