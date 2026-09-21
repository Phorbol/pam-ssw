"""Matched forward generalized-dimer control on the corrected XXXII representation."""

import argparse
import hashlib
import json
import os
import shutil
import signal
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/ga_ssw/evidence/xxxii-rc-vc-replicated-forward-control"
SRC = ROOT / "research/ga_ssw/evidence/xxxii-rc-vc-central-ritz-completion"
FIX = ROOT / "tests/standalone/fixtures/type2_xxxii.extxyz"
TOPO = Path("/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_examples_run/global_exploration/input-templates/TYPE2-XXXII/mc")
MODEL = SRC
REPETITIONS = (1, 1, 2)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_plan():
    from ase.io import read
    from pamssw.standalone.rc_optimization_domain import PrincipalRigidForestCellChart
    from pamssw.standalone.rc_periodic_input import unwrap_rigid_molecules
    from pamssw.standalone.rc_topology import read_rigid_topology

    atoms = read(FIX)
    topology = read_rigid_topology(TOPO / "rigidbody", TOPO / "blist", natoms=len(atoms))
    lifted = unwrap_rigid_molecules(atoms, topology.bonds).atoms
    chart = PrincipalRigidForestCellChart(
        lifted, topology.components, anchor=0, rotation_length=1.0,
        torsion_length=1.0, strain_length=5.0,
    )
    return {
        "status": "prepared",
        "pes_calls": 0,
        "input": str(FIX),
        "input_sha256": sha(FIX),
        "public_atoms": 172,
        "public_input": "original 172 atoms with unwrapped molecular positions; internal explicit repeat only",
        "repetitions": REPETITIONS,
        "internal_atoms_per_evaluation": 344,
        "topology": {"rigidbody": str(TOPO / "rigidbody"), "blist": str(TOPO / "blist"), "components": len(topology.components)},
        "model_files": {name: sha(MODEL / name) for name in ("lmp.data", "in.simple", "manifest.json")},
        "config": {
            "rotation_length": 1.0, "torsion_length": 1.0, "strain_length": 5.0,
            "width": 0.6, "rotation_bias": 100.0, "max_gaussians": 12,
            "temperature_K": 300.0, "forward_force": 0.1, "gradient_tol": 0.005,
            "fmax": 0.01, "stress_tol": 0.001, "max_step": 0.2,
            "relax_steps": 4998, "fd_step": 1e-4, "rotation_hvp": 100,
            "rotation_tol": 0.02, "pressure": 0.0, "lbfgs_memory": 400,
        },
        "fixed_G": 0.47570069, "pair_table": 0, "ewald_accuracy": 1e-12,
        "seed": 3, "steps": 1,
        "control": "existing forward generalized dimer versus central Ritz; same initial/config/5000 API/120 s and fixed1x1x2 backend; only rotation operator differs",
        "budget": {"max_EFS": 5000, "search_cap": 4998, "seconds": 120, "fresh_reserved": 2},
        "chart_dimension": chart.dimension,
        "scope": "experimental periodic representation qualification; internal 344-atom engine per E/F; not a primitive-cost comparison",
    }


def freeze_sources(plan):
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    shutil.copy2(__file__, OUT / "runner-prepared.py")
    shutil.copy2(ROOT / "research/ga_ssw/xxxii_replicated_calculator.py", OUT / "xxxii_replicated_calculator.py")
    shutil.copy2(ROOT / "research/ga_ssw/generalized_ritz_probe.py", OUT / "generalized_ritz_probe.py")
    shutil.copytree(ROOT / "pamssw", OUT / "source/pamssw", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for source, target in ((FIX, OUT / FIX.name), (TOPO / "rigidbody", OUT / "rigidbody"),
                           (TOPO / "blist", OUT / "blist"), (MODEL / "lmp.data", OUT / "lmp.data"),
                           (MODEL / "in.simple", OUT / "in.simple"), (MODEL / "manifest.json", OUT / "manifest.json")):
        shutil.copy2(source, target)


def execute():
    from ase.io import read, write
    from pamssw.standalone.rc_periodic_input import unwrap_rigid_molecules
    from pamssw.standalone.rc_topology import read_rigid_topology
    from pamssw.standalone.rc_vc_reference import RCVCSSWConfig, run_rc_vc_ssw
    from pamssw.standalone.vc_geometry import ASEStressSurface
    import pamssw.standalone.rc_vc_reference as rc_module
    from research.ga_ssw.compare_vc_arms import serial
    from research.ga_ssw.xxxii_replicated_calculator import XXXIIReplicatedCalculator
    from research.ga_ssw.generalized_ritz_probe import solve

    def central_ritz(q0, anchor, *, rotation_bias, fd_step, max_hvp, tol, evaluate):
        return solve(q0, anchor, evaluate, rotation_bias, fd_step, tol, max_hvp, "central")

    # Preserve the existing forward generalized_dimer baseline.
    if (OUT / "result.json").exists() or (OUT / "ledger.jsonl").exists():
        raise RuntimeError("refuse overwrite previously executed experiment")
    plan = make_plan()
    if OUT.exists():
        if json.loads((OUT / 'plan.json').read_text()) != json.loads(json.dumps(plan)):
            raise RuntimeError('prepared plan changed')
        for name in ('xxxii_replicated_calculator.py', 'generalized_ritz_probe.py'):
            if sha(OUT / name) != sha(ROOT / 'research/ga_ssw' / name):
                raise RuntimeError('prepared implementation changed')
        for frozen in (OUT / 'source/pamssw').rglob('*.py'):
            if sha(frozen) != sha(ROOT / 'pamssw' / frozen.relative_to(OUT / 'source/pamssw')):
                raise RuntimeError('prepared core implementation changed')
    else:
        freeze_sources(plan)
    shutil.copy2(__file__, OUT / 'runner-executed.py')
    import ase, lammps, sys
    (OUT / 'environment.json').write_text(json.dumps(dict(python=sys.executable, ase=ase.__version__, lammps=lammps.__version__, lammps_file=lammps.__file__, variables={k:os.environ.get(k) for k in ('LD_LIBRARY_PATH','PYTHONPATH','OMP_NUM_THREADS','CUDA_VISIBLE_DEVICES')}), indent=2)+'\n')
    atoms = read(OUT / FIX.name)
    topology = read_rigid_topology(OUT / "rigidbody", OUT / "blist", natoms=len(atoms))
    atoms = unwrap_rigid_molecules(atoms, topology.bonds).atoms
    engines = []

    def calculator():
        calc = XXXIIReplicatedCalculator(
            data_path=OUT / "lmp.data", input_path=OUT / "in.simple",
            model_manifest=OUT / "manifest.json", reference_atoms=atoms,
            repetitions=REPETITIONS,
        )
        engines.append(calc)
        return calc

    cfg = RCVCSSWConfig(**plan["config"])
    start = time.monotonic()
    counts = {"search": 0, "fresh": 0}
    attempts = []

    class Counted(ASEStressSurface):
        def __init__(self, calc, role):
            super().__init__(calc)
            self.role = role

        def evaluate(self, candidate):
            row = {"index": len(attempts), "role": self.role, "status": "started",
                   "api_calls": 0,
                   "positions": candidate.positions.tolist(), "cell": candidate.cell.array.tolist(),
                   "numbers": candidate.numbers.tolist(), "pbc": candidate.pbc.tolist()}
            attempts.append(row)
            engine_before = self.calculator.engine_calls
            try:
                if sum(counts.values()) >= 5000 or counts[self.role] >= (4998 if self.role == "search" else 2):
                    raise RuntimeError("declared request budget exhausted")
                if time.monotonic() - start >= 120:
                    raise RuntimeError("declared wall budget exhausted")
                counts[self.role] += 1
                row['api_calls'] = 1
                energy, force, stress = super().evaluate(candidate)
                row.update(status="completed", energy=energy, forces=force.tolist(), stress=stress.tolist())
                return energy, force, stress
            except BaseException as error:
                row.update(status="failed", error=repr(error))
                raise
            finally:
                row.update(elapsed_seconds=time.monotonic() - start,
                           engine_calls=self.calculator.engine_calls - engine_before,
                           calculator_api_calls=self.calculator.api_calls,
                           atoms_evaluated=self.calculator.atoms_evaluated)
                with (OUT / "ledger.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")

    def timeout(*_):
        raise RuntimeError("declared 120 second wall limit")

    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 120)
    output = {"status": "running", "result": None, "fresh": []}
    try:
        surface = Counted(calculator(), "search")
        result = run_rc_vc_ssw(atoms, surface, trees=topology.components, anchor=0,
                                steps=1, config=cfg, rng=np.random.default_rng(3))
        output.update(status=result.status, result=serial(result))
        for index, minimum in enumerate(result.minima[:2]):
            write(OUT / f"minimum-{index}.extxyz", minimum.atoms)
            fresh = Counted(calculator(), "fresh")
            energy, force, stress = fresh.evaluate(minimum.atoms)
            output["fresh"].append({"index": index, "energy": energy,
                "energy_error": energy - minimum.energy, "forces": force.tolist(),
                "stress": stress.tolist(), "fmax": float(np.linalg.norm(force, axis=1).max()),
                "stress_max": float(abs(stress + cfg.pressure * np.eye(3)).max()),
                "volume": minimum.atoms.get_volume()})
    except BaseException as error:
        output.update(status="failed", error=repr(error), traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for engine in engines:
            engine.close()
        output.update(api_requests=counts, total_EFS=sum(counts.values()),
                      completed_EFS=sum(row["status"] == "completed" for row in attempts),
                      engine_calls=sum(c.engine_calls for c in engines),
                      atoms_evaluated=sum(c.atoms_evaluated for c in engines),
                      replicas=REPETITIONS, wall_seconds=time.monotonic() - start)
        (OUT / "result.json").write_text(json.dumps(output, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = make_plan()
    if args.execute:
        execute()
    else:
        if OUT.exists():
            raise RuntimeError("output directory already exists")
        freeze_sources(plan)
        print(json.dumps({"status": "prepared", "pes_calls": 0,
                          "repetitions": REPETITIONS, "chart_dimension": plan["chart_dimension"]}))


if __name__ == "__main__":
    main()
