"""Research-only finite-cell Hessian qualification for the XXXII replica.

This file prepares (and, only when explicitly requested, runs) a numerical
qualification of the two endpoint structures from the replicated endpoint
quench.  It is deliberately not connected to an SSW walker.  The Hessian is
the central finite difference of the gradient of ``E + pV`` in the
``SymmetricLogStrainChart`` coordinates, projected only to remove the three
uniform atomic translations.  The spectra are diagnostic: ERFC
non-conservativity means that they do not establish a DFT/phonon or
all-wave-vector stability result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import time
from pathlib import Path

import numpy as np
from ase import Atoms


HERE = Path(__file__).resolve().parent
ENDPOINT_RESULT = (HERE / "evidence" / "xxxii-replicated-endpoint-quench" /
                   "result.json")
DEFAULT_OUT = (HERE / "evidence" / "xxxii-replicated-hessian-qualification")
H_STEPS = (1.0e-4, 5.0e-5)
NATOMS = 172
DIMENSION = 3 * NATOMS + 6 - 3
PER_ENDPOINT_HESSIAN_CALLS = DIMENSION * len(H_STEPS) * 2
TOTAL_HESSIAN_CALLS = 2 * PER_ENDPOINT_HESSIAN_CALLS
TOTAL_WITH_FRESH = TOTAL_HESSIAN_CALLS + 2


class WallLimit(TimeoutError):
    pass


def _alarm_handler(signum, frame):  # pragma: no cover - exercised by run only
    raise WallLimit("endpoint 90-second wall limit")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atoms_from_final(row: dict) -> Atoms:
    final = row["final"]
    return Atoms(numbers=np.asarray(final["numbers"], dtype=int),
                 positions=np.asarray(final["positions"], dtype=float),
                 cell=np.asarray(final["cell"], dtype=float),
                 pbc=np.asarray(final["pbc"], dtype=bool))


def load_endpoints(path: Path = ENDPOINT_RESULT) -> list[Atoms]:
    data = json.loads(path.read_text())
    endpoints = sorted(data["endpoints"], key=lambda x: int(x["endpoint"]))
    if [e["endpoint"] for e in endpoints] != [0, 1]:
        raise ValueError("endpoint result must contain exactly endpoints 0 and 1")
    atoms = [_atoms_from_final(e) for e in endpoints]
    if any(len(a) != NATOMS or not a.pbc.all() for a in atoms):
        raise ValueError("XXXII endpoints must be 172-atom fully periodic structures")
    return atoms


def translation_free_basis(ndof: int, natoms: int = NATOMS) -> np.ndarray:
    """Return the SVD basis used by ``qualify_material_gate.py``."""
    translations = np.zeros((ndof, 3))
    for axis in range(3):
        translations[axis:3 * natoms:3, axis] = 1.0 / np.sqrt(natoms)
    u, _, _ = np.linalg.svd(translations, full_matrices=True)
    return u[:, 3:]


def _atom_record(atoms: Atoms) -> dict:
    return dict(numbers=atoms.numbers.tolist(), positions=atoms.positions.tolist(),
                cell=atoms.cell.array.tolist(), pbc=atoms.pbc.tolist())


def plan() -> dict:
    return {
        "status": "prepared_only",
        "endpoint_result": str(ENDPOINT_RESULT),
        "endpoint_result_sha256": _sha(ENDPOINT_RESULT),
        "backend": "audited XXXII replicated representation, repetitions=(1,1,2)",
        "chart": "SymmetricLogStrainChart(strain_length=5, pressure=0)",
        "domain": "3N+6 coordinates minus the three uniform atomic translations",
        "dimension": DIMENSION,
        "h_steps": list(H_STEPS),
        "per_endpoint_hessian_EF": PER_ENDPOINT_HESSIAN_CALLS,
        "total_hessian_EF": TOTAL_HESSIAN_CALLS,
        "fresh_center_checks": 2,
        "declared_total_EF": TOTAL_WITH_FRESH,
        "per_endpoint_api_cap": 2200,
        "per_endpoint_wall_seconds": 90,
        "pressure_eV_A3": 0.0,
        "source_note": "Frozen endpoint-quench source and explicit replica adapter; no production-code changes.",
        "interpretation_limits": [
            "finite-cell E+pV numerical Hessian only",
            "ERFC non-conservativity is known in this backend",
            "not a DFT, phonon, or all-q stability certificate",
        ],
    }


def prepare(out: Path) -> None:
    """Create a frozen, reviewable run package without evaluating a PES."""
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {out}")
    endpoints = load_endpoints()
    out.mkdir(parents=True)
    (out / "source").mkdir()
    (out / "endpoint-inputs").mkdir()
    (out / "matrices").mkdir()
    (out / "progress").mkdir()
    (out / "plan.json").write_text(json.dumps(plan(), indent=2) + "\n")
    (out / "endpoint-source-fullprecision.json").write_text(
        json.dumps(json.loads(ENDPOINT_RESULT.read_text()), indent=2) + "\n")
    for i, atoms in enumerate(endpoints):
        (out / "endpoint-inputs" / f"endpoint-{i}.json").write_text(
            json.dumps(_atom_record(atoms), indent=2) + "\n")
    # These are the actual imported algorithm files and their converted input;
    # snapshotting them keeps a future run independent of the mutable checkout.
    repo = HERE.parents[1]
    source_files = [
        HERE / "xxxii_replicated_calculator.py",
        HERE / "xxxii_lammps_calculator.py",
        HERE / "convert_xxxii_amber.py",
        repo / "pamssw" / "standalone" / "vc_geometry.py",
        repo / "pamssw" / "standalone" / "__init__.py",
    ]
    for src in source_files:
        if not src.exists():
            raise FileNotFoundError(src)
        if src.parent.name == "standalone":
            target = out / "source" / "pamssw" / "standalone" / src.name
        else:
            target = out / "source" / "research" / "ga_ssw" / src.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    shutil.copy2(repo / "pamssw" / "__init__.py", out / "source" / "pamssw" / "__init__.py")
    endpoint_dir = ENDPOINT_RESULT.parent
    for name in ("lmp.data", "in.simple", "manifest.json"):
        shutil.copy2(endpoint_dir / name, out / "source" / "research" / "ga_ssw" / name)
    digest = {str(p.relative_to(out / "source")): _sha(p)
              for p in (out / "source").rglob("*") if p.is_file()}
    (out / "source-digests.sha256").write_text(
        "\n".join(f"{value}  {name}" for name, value in sorted(digest.items())) + "\n")
    (out / "preflight.json").write_text(json.dumps({
        "PES_evaluations": 0,
        "endpoint_count": len(endpoints),
        "dimension": int(translation_free_basis(3 * NATOMS + 6).shape[1]),
        "declared_total_EF": TOTAL_WITH_FRESH,
        "source_snapshot": str(out / "source"),
        "full_precision_inputs": True,
    }, indent=2) + "\n")
    shutil.copy2(__file__, out / "runner-prepared.py")


class LoggedSurface:
    """Count and persist every allocated API request before calling ASE."""

    def __init__(self, surface, progress: Path, endpoint: int, cap: int):
        self.surface, self.progress, self.endpoint, self.cap = surface, progress, endpoint, cap
        self.calls = 0

    def evaluate(self, atoms):
        if self.calls >= self.cap:
            raise RuntimeError("endpoint API budget exhausted")
        self.calls += 1
        row = {"endpoint": self.endpoint, "api_call": self.calls,
               "status": "started", "time": time.time()}
        with self.progress.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        try:
            result = self.surface.evaluate(atoms)
            row.update(status="success", energy=float(result[0]),
                       gradient_norm=float(np.linalg.norm(result[1])))
            return result
        except BaseException as exc:
            row.update(status="failed", error=repr(exc))
            raise
        finally:
            # A compact completion row preserves the charged request without
            # duplicating the potentially large forces/stress arrays.
            with self.progress.open("a") as fh:
                fh.write(json.dumps(row) + "\n")


def execute(out: Path) -> None:  # pragma: no cover - real PES is withheld in this turn
    """Run a prepared package; never called by ``--prepare``."""
    if not (out / "plan.json").exists():
        raise ValueError("run --prepare first")
    if (out / 'result.json').exists():
        raise RuntimeError('refuse overwrite executed Hessian qualification')
    # The prepared package contained selected modules only. Freeze the complete
    # package so package-level imports cannot silently resolve against live code.
    shutil.copytree(HERE.parents[1] / 'pamssw', out / 'source/pamssw',
                    dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(__file__, out / 'runner-executed.py')
    (out / 'execution-source-digests.json').write_text(json.dumps({str(p.relative_to(out/'source')):_sha(p) for p in (out/'source').rglob('*.py')}, indent=2)+'\n')
    import sys
    sys.path.insert(0, str(out / "source"))
    from pamssw.standalone.vc_geometry import ASEStressSurface, SymmetricLogStrainChart
    from research.ga_ssw.xxxii_replicated_calculator import XXXIIReplicatedCalculator

    source = out / "source" / "research" / "ga_ssw"
    endpoint_data = [json.loads((out / "endpoint-inputs" / f"endpoint-{i}.json").read_text())
                     for i in (0, 1)]
    source_result = json.loads((out / "endpoint-source-fullprecision.json").read_text())
    summary = {"status": "running", "endpoints": [], "declared_total_EF": TOTAL_WITH_FRESH}
    (out / "result.json").write_text(json.dumps(summary, indent=2) + "\n")

    def atoms_from_record(record):
        return Atoms(numbers=record["numbers"], positions=record["positions"],
                     cell=record["cell"], pbc=record["pbc"])

    def save():
        (out / "result.json").write_text(json.dumps(summary, indent=2) + "\n")

    for endpoint in (0, 1):
        begun = time.monotonic()
        record = endpoint_data[endpoint]
        atoms = atoms_from_record(record)
        row = {"endpoint": endpoint, "status": "running", "h_steps": list(H_STEPS),
               "dimension": DIMENSION, "charged_api_calls": 0, "engine_calls": 0,
               "atoms_evaluated": 0, "matrices": {}}
        summary["endpoints"].append(row); save()
        progress = out / "progress" / f"endpoint-{endpoint}.jsonl"
        calculators = []
        surfaces = []
        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, 90.0)
        try:
            kwargs = dict(data_path=source / "lmp.data", input_path=source / "in.simple",
                          model_manifest=source / "manifest.json", reference_atoms=atoms,
                          repetitions=(1, 1, 2))
            # The center check is an independently constructed calculator and
            # consumes the declared one fresh E/F per endpoint.
            fresh_calc = XXXIIReplicatedCalculator(**kwargs); calculators.append(fresh_calc)
            fresh = LoggedSurface(ASEStressSurface(fresh_calc), progress, endpoint, 1)
            surfaces.append(fresh)
            energy, forces, stress = fresh.evaluate(atoms)
            row["fresh"] = {"energy": float(energy),
                            "source_energy": source_result["endpoints"][endpoint]["final"]["energy"],
                            "energy_delta": float(energy) - source_result["endpoints"][endpoint]["final"]["energy"],
                            "fmax": float(np.linalg.norm(forces, axis=1).max()),
                            "stress_max": float(np.max(np.abs(stress)))}
            fresh_calc.close()
            calc = XXXIIReplicatedCalculator(**kwargs); calculators.append(calc)
            surface = LoggedSurface(ASEStressSurface(calc), progress, endpoint, PER_ENDPOINT_HESSIAN_CALLS)
            surfaces.append(surface)
            chart = SymmetricLogStrainChart(atoms, strain_length=5.0)
            center = chart.pack(atoms)
            basis = translation_free_basis(len(center))
            np.save(out/'matrices'/f'endpoint-{endpoint}-basis.npy', basis)
            np.save(out/'matrices'/f'endpoint-{endpoint}-center.npy', center)
            row["basis_dimension"] = int(basis.shape[1])
            for h in H_STEPS:
                path = out / "matrices" / f"endpoint-{endpoint}-h-{h:.0e}.npy"
                matrix = np.lib.format.open_memmap(path, mode='w+', dtype=np.float64,
                    shape=(DIMENSION, DIMENSION), fortran_order=True)
                matrix[:] = np.nan
                for column, direction in enumerate(basis.T):
                    plus = chart.evaluate(center + h * direction, surface.evaluate, pressure=0.0)
                    minus = chart.evaluate(center - h * direction, surface.evaluate, pressure=0.0)
                    matrix[:, column] = basis.T @ (plus.gradient - minus.gradient) / (2.0 * h)
                    matrix.flush()
                    row["charged_api_calls"] = fresh.calls + surface.calls
                    row["engine_calls"] = fresh_calc.engine_calls + calc.engine_calls
                    row["atoms_evaluated"] = fresh_calc.atoms_evaluated + calc.atoms_evaluated
                    (out / "progress" / f"endpoint-{endpoint}.json").write_text(
                        json.dumps({"endpoint": endpoint, "h": h, "column": column + 1,
                                    "dimension": DIMENSION, "charged_api_calls": row["charged_api_calls"],
                                    "engine_calls": row["engine_calls"],
                                    "atoms_evaluated": row["atoms_evaluated"]}, indent=2) + "\n")
                symmetric = (matrix + matrix.T) / 2.0
                eig = np.linalg.eigvalsh(symmetric)
                row["matrices"][f"{h:.0e}"] = {
                    "path": str(path), "columns_completed": int(np.isfinite(matrix).all(axis=0).sum()),
                    "skew_frobenius": float(np.linalg.norm(matrix - matrix.T)),
                    "min_eigenvalue": float(eig[0]), "max_eigenvalue": float(eig[-1]),
                    "eigenvalues_path": str(path.with_name(path.stem + "-eigenvalues.npy")),
                }
                np.save(path.with_name(path.stem + "-eigenvalues.npy"), eig)
            row["status"] = "completed"
        except BaseException as exc:
            row.update(status="failed", error=repr(exc))
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, old_handler)
            for calc in calculators:
                try: calc.close()
                except Exception: pass
            row['charged_api_calls'] = sum(s.calls for s in surfaces)
            row['engine_calls'] = sum(c.engine_calls for c in calculators)
            row['atoms_evaluated'] = sum(c.atoms_evaluated for c in calculators)
            row["wall_seconds"] = time.monotonic() - begun
            save()
    summary["status"] = "completed" if all(r["status"] == "completed" for r in summary["endpoints"]) else "completed_with_failures"
    summary.update(total_API=sum(r['charged_api_calls'] for r in summary['endpoints']),
        engine_calls=sum(r['engine_calls'] for r in summary['endpoints']),
        atoms_evaluated=sum(r['atoms_evaluated'] for r in summary['endpoints']))
    save()
    print({k:v for k,v in summary.items() if k!='endpoints'})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        execute(args.output)
    elif args.prepare:
        prepare(args.output)
    else:
        parser.error("choose --prepare; real PES execution is withheld")


if __name__ == "__main__":
    main()
