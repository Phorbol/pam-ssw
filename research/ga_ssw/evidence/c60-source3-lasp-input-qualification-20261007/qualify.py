#!/usr/bin/env python3
"""Bounded source-3 LASP external-input qualification; no search."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from ase import Atoms
from ase.io import read, write

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
INPUT = ROOT / "research/ga_ssw/evidence/c60-ls-source3-qualification-20261007/qualification-1664282/final-candidate.extxyz"
REFERENCE = ROOT / "research/ga_ssw/evidence/c60-ls-source3-qualification-20261007/qualification-1664282/source/historical-ih-reference.extxyz"
REFERENCE_RESULT = ROOT / "research/ga_ssw/evidence/c60-ls-source3-qualification-20261007/qualification-1664282/reference-cold.json"
MODEL = Path("/home/gengjianrui/.cache/mace/mace-mh-1.model")
BINARY = Path("/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp")
OLD = Path("/home/gengjianrui/bin/pam-ssw-worktrees/c60-vacuum-geometry/research/ga_ssw/evidence/c60-native-long-20260920/seed17093")
HELPER = ROOT / "research/ga_ssw/lasp_external_ase.py"
EXPECTED_INPUT_SHA = "dcc81d45c4e7193b306a0c1957b49c19e44751231fa95fef75201183280f0758"
EXPECTED_MODEL_SHA = "a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47"
EXPECTED_BINARY_SHA = "bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704"
CELL = np.eye(3) * 50.0


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")


def load_lasp_helper(path: Path):
    spec = importlib.util.spec_from_file_location("qualified_lasp_external_ase", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_atoms(atoms: Atoms) -> None:
    if atoms.get_chemical_symbols() != ["C"] * 60:
        raise ValueError("candidate must be exactly C60 in archived atom order")
    if not np.array_equal(atoms.cell.array, CELL) or not np.array_equal(atoms.pbc, [False] * 3):
        raise ValueError("candidate cell/PBC differs from the archived centered 50 A nonperiodic input")
    if atoms.constraints or not np.isfinite(atoms.positions).all():
        raise ValueError("constraints or nonfinite coordinates are not allowed")
    if np.any(atoms.positions < 0.0) or np.any(atoms.positions >= 50.0):
        raise ValueError("candidate coordinates must remain within its archived storage cell")


def nearest_interimage(atoms: Atoms) -> float:
    # For cubic 50 A cell, neighboring translations suffice for this compact cage.
    best = float("inf")
    pos = atoms.positions
    for ix in (-1, 0, 1):
        for iy in (-1, 0, 1):
            for iz in (-1, 0, 1):
                if (ix, iy, iz) == (0, 0, 0):
                    continue
                shift = np.array([ix, iy, iz], dtype=float) * 50.0
                d = pos[:, None, :] - (pos[None, :, :] + shift)
                best = min(best, float(np.sqrt(np.sum(d * d, axis=-1)).min()))
    return best


def arc_text(atoms: Atoms) -> str:
    rows = ["!BIOSYM archive 2", "PBC=ON", "Energy 0 0.0 0.0", "!DATE",
            "PBC 50 50 50 90 90 90"]
    for i, (sym, xyz) in enumerate(zip(atoms.get_chemical_symbols(), atoms.positions), 1):
        rows.append(f"{sym} {xyz[0]:.14f} {xyz[1]:.14f} {xyz[2]:.14f} CORE {i} {sym} {sym} 0.0 {i}")
    rows.extend(["end", "end", ""])
    return "\n".join(rows)


class Counted:
    def __init__(self, calc, max_paid_ef=3):
        self.calc = calc
        self.actual_calculations = 0
        self.paid_ef_slots = 0
        self.max_paid_ef = int(max_paid_ef)
        self._charged_atoms = {}
        self._charged_refs = []
        original = calc.calculate
        def counted(*args, **kwargs):
            self.actual_calculations += 1
            return original(*args, **kwargs)
        calc.calculate = counted

    def charge_slot(self, atoms):
        key = id(atoms)
        if key in self._charged_atoms:
            return
        if self.paid_ef_slots >= self.max_paid_ef:
            raise RuntimeError("paid E/F cap reached before calculator invocation")
        self.paid_ef_slots += 1
        self._charged_atoms[key] = True
        self._charged_refs.append(atoms)

    def get_potential_energy(self, atoms=None, force_consistent=False):
        self.charge_slot(atoms)
        return self.calc.get_potential_energy(atoms, force_consistent=force_consistent)

    def get_forces(self, atoms=None):
        return self.calc.get_forces(atoms)

    def __getattr__(self, name):
        return getattr(self.calc, name)


def model_calculator():
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from mace.calculators import MACECalculator
    calc = MACECalculator(model_paths=str(MODEL), head="omol", device="cuda",
                          default_dtype="float64", enable_cueq=False, enable_oeq=False)
    wrapper = Counted(calc)
    wrapper.mace_calculator_source = str(Path(sys.modules[MACECalculator.__module__].__file__).resolve())
    return wrapper


def write_slot_inputs(out: Path, atoms: Atoms) -> None:
    write(out / "input.extxyz", atoms, format="extxyz")
    (out / "input.arc").write_text(arc_text(atoms))
    (out / "lasp.in").write_text(
        "potential external\nexplore_type ssw\nEwaldflag 0\nRun_type 5\n"
        "SSW.SSWsteps 1\nSSW.ftol 0.0173205080756888\nSSW.MaxOptstep 1\n"
        "SSW.NG 12\nSSW.Temp 150\nSSW.ds_atom .6\nSSW.internal_LJ F\n"
        "SSW.globalcompress .0001\nSSW.vapor_cri 1.7\nSSW.output T\nSSW.printevery T\n")
    for name in ("client.py", "bounded_process.py"):
        shutil.copy2(OLD / name, out / name)
    launcher = ("#!/bin/bash\nset -euo pipefail\n"
                "exec /home/gengjianrui/.conda/envs/mace_env/bin/python "
                '"$(dirname "$0")/client.py"\n')
    (out / "lasp.external.sh").write_text(launcher)
    (out / "lasp.external.sh").chmod(0o755)


def bounded_command(out: Path, command: list[str], timeout: float) -> list[str]:
    return [sys.executable, str(out / "bounded_process.py"), "--cwd", str(out),
            "--timeout", str(timeout), "--log", str(out / "child.log"),
            "--status", str(out / "process.json"), "--", *command]


def preflight() -> None:
    """Exercise the real run_lasp service with dummy E/F and a socket client only."""
    # ASE Calculator protocol, with a deterministic synthetic response.
    from ase.calculators.calculator import Calculator, all_changes
    class DummyCalc(Calculator):
        implemented_properties = ["energy", "forces"]
        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {"energy": -60.0, "forces": np.zeros((len(atoms), 3))}

    atoms = read(INPUT)
    validate_atoms(atoms)
    periodic = atoms.copy(); periodic.pbc = True
    old_tmpdir = os.environ.get("TMPDIR")
    old_tempfile_tmpdir = tempfile.tempdir
    with tempfile.TemporaryDirectory(prefix=".c60-pf-", dir=Path.home()) as parent:
        os.environ["TMPDIR"] = parent
        tempfile.tempdir = parent
        if len(os.fsencode(parent)) + len(os.fsencode("/lasp-ase-XXXXXXXX/eval.sock")) >= 108:
            raise RuntimeError("preflight Unix socket path could exceed Linux limit")
        with tempfile.TemporaryDirectory(prefix="c60pf-") as td:
            out = Path(td)
            write_slot_inputs(out, periodic)
            shutil.copy2(HELPER, out / "lasp_external_ase.py")
            helper = load_lasp_helper(out / "lasp_external_ase.py")
            # Parse the just-written ARC, then produce the native external.coord
            # row shape. This checks the actual serialization path, not a fresh
            # coordinate construction from the in-memory Atoms object.
            arc_rows = (out / "input.arc").read_text().splitlines()[5:65]
            if len(arc_rows) != 60:
                raise RuntimeError("preflight ARC does not contain exactly 60 atom records")
            parsed = [(row.split()[0], np.asarray(row.split()[1:4], float)) for row in arc_rows]
            if any(sym != "C" for sym, _ in parsed) or not np.allclose(
                    np.asarray([xyz for _, xyz in parsed]), periodic.positions, rtol=0., atol=1e-12):
                raise RuntimeError("preflight ARC did not preserve symbols/coordinates")
            coord = "\n".join([*(" ".join(f"{v:.14f}" for v in row) for row in CELL),
                                *(f"{sym} {xyz[0]:.14f} {xyz[1]:.14f} {xyz[2]:.14f} {i}"
                                  for i, (sym, xyz) in enumerate(parsed, 1))]) + "\n"
            child = ("import json,os,socket; s=socket.socket(socket.AF_UNIX); s.connect(os.environ['LASP_MACE_SOCKET']); "
                     f"s.sendall((json.dumps({{'coord':{coord!r}}})+'\\n').encode()); "
                     "r=s.makefile().readline(); print(r); assert json.loads(r)['ok']")
            launch = bounded_command(out, [sys.executable, "-c", child], 60)
            counted_dummy = Counted(DummyCalc())
            counted_dummy.charge_slot(atoms.copy())
            counted_dummy.charge_slot(atoms.copy())
            result = helper.run_lasp(launch, cwd=out, atoms=periodic, calculator=counted_dummy,
                                     pbc=[True] * 3, max_requests=1, env=os.environ.copy())
            if result["returncode"] != 0 or len(result["requests"]) != 1 or result["errors"]:
                raise RuntimeError(f"dummy callback preflight failed: {result}")
            got = result["requests"][0]
            if not np.array_equal(np.asarray(got["positions"]), periodic.positions):
                raise RuntimeError("dummy callback preflight coordinate serialization mismatch")
            if got["response"]["energy"] != -60.0:
                raise RuntimeError("dummy callback preflight result mismatch")
            # Regression for the observed helper behavior: a failed callback is
            # absent from its success log, so max_requests alone would retry it.
            # Charge two synthetic direct slots, let one dummy calculation fail,
            # then prove a second request is rejected before another calculation.
            class FailingCalc(Calculator):
                implemented_properties = ["energy", "forces"]
                def __init__(self):
                    super().__init__(); self.calls = 0
                def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
                    self.calls += 1
                    raise RuntimeError("synthetic calculator failure")

            fail_out = out / "failure-case"
            fail_out.mkdir()
            write_slot_inputs(fail_out, periodic)
            shutil.copy2(HELPER, fail_out / "lasp_external_ase.py")
            fail_helper = load_lasp_helper(fail_out / "lasp_external_ase.py")
            failing_calc = FailingCalc()
            guarded = Counted(failing_calc)
            guarded.charge_slot(atoms.copy()); guarded.charge_slot(atoms.copy())
            repeated_child = ("import json,os,socket; coord=" + repr(coord) + "; "
                "[(lambda s: (s.connect(os.environ['LASP_MACE_SOCKET']), "
                "s.sendall((json.dumps({'coord':coord})+'\\n').encode()), "
                "print(s.makefile().readline())))(socket.socket(socket.AF_UNIX)) for _ in range(2)]")
            failed = fail_helper.run_lasp(
                bounded_command(fail_out, [sys.executable, "-c", repeated_child], 60),
                cwd=fail_out, atoms=periodic, calculator=guarded, pbc=[True] * 3,
                max_requests=1, env=os.environ.copy())
            if (len(failed["errors"]) != 2 or failing_calc.calls != 1
                    or guarded.paid_ef_slots != 3):
                raise RuntimeError("failed-callback budget guard regression failed")
            print(json.dumps({"preflight": "passed", "physical_pes": 0, "mace_calls": 0,
                              "dummy_requests": len(result["requests"]),
                              "dummy_paid_ef_slots": counted_dummy.paid_ef_slots,
                              "failure_guard_requests": 2, "failure_guard_calculations": failing_calc.calls,
                              "failure_guard_paid_slots": guarded.paid_ef_slots,
                              "temporary_only": True}))
    if old_tmpdir is None:
        os.environ.pop("TMPDIR", None)
    else:
        os.environ["TMPDIR"] = old_tmpdir
    tempfile.tempdir = old_tempfile_tmpdir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight()
        return
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        raise SystemExit("real qualification requires Slurm; use --preflight for login-side synthetic check")
    socket_tmp = os.environ.get("TMPDIR")
    if not socket_tmp or not Path(socket_tmp).is_dir():
        raise RuntimeError("Slurm wrapper must provide its private short TMPDIR before any model work")
    tempfile.tempdir = socket_tmp
    if len(os.fsencode(socket_tmp)) + len(os.fsencode("/lasp-ase-XXXXXXXX/eval.sock")) >= 108:
        raise RuntimeError("LASP_MACE_SOCKET could exceed Linux AF_UNIX path limit")
    out = HERE / f"run-{job}"
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    for name in ("qualify.py", "gpu.sbatch", "protocol.md"):
        shutil.copy2(HERE / name, out / name)
    source_hashes = {name: sha(out / name) for name in ("qualify.py", "gpu.sbatch", "protocol.md")}
    source_head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                 check=False, capture_output=True, text=True).stdout.strip()
    atoms = read(INPUT)
    validate_atoms(atoms)
    if sha(INPUT) != EXPECTED_INPUT_SHA or sha(MODEL) != EXPECTED_MODEL_SHA or sha(BINARY) != EXPECTED_BINARY_SHA:
        raise RuntimeError("archived input/model/native binary checksum differs from protocol")
    if not REFERENCE.is_file() or not REFERENCE_RESULT.is_file():
        raise FileNotFoundError("archived Ih reference inputs/results missing")
    shutil.copy2(INPUT, out / "source-input.extxyz")
    shutil.copy2(REFERENCE, out / "historical-ih-reference.extxyz")
    shutil.copy2(REFERENCE_RESULT, out / "reference-cold.json")
    periodic = atoms.copy(); periodic.pbc = True
    write_slot_inputs(out, periodic)
    arc = arc_text(periodic)
    (out / "serialized.arc").write_text(arc)
    provenance = {"input": str(INPUT), "input_sha256": sha(INPUT), "reference": str(REFERENCE),
                  "reference_sha256": sha(REFERENCE), "reference_result": str(REFERENCE_RESULT),
                  "model": str(MODEL), "model_sha256": sha(MODEL), "head": "omol", "dtype": "float64",
                  "device": "cuda", "cueq": False, "oeq": False, "binary": str(BINARY),
                  "binary_sha256": sha(BINARY), "helper_path": str(out / "lasp_external_ase.py"),
                  "helper_sha256": sha(HELPER), "used_helper_imported_path": str(out / "lasp_external_ase.py"),
                  "actual_source_snapshot_hashes": source_hashes, "git_head": source_head,
                  "worker_archive": str(OLD), "python": sys.executable, "python_version": sys.version,
                  "cell_A": CELL.tolist(), "pbc_comparison": [False, True], "request_cap": 1,
                  "paid_ef_cap_total": 3, "binary_timeout_seconds": 60, "task_deadline_seconds": 300,
                  "nearest_interimage_A": nearest_interimage(atoms), "candidate_atoms": len(atoms)}
    provenance["used_worker_sources"] = {
        name: {"path": str(OLD / name), "sha256": sha(OLD / name)}
        for name in ("client.py", "lasp.external.sh", "bounded_process.py", "lasp.in")
    }
    provenance["used_local_launcher"] = {"path": str(out / "lasp.external.sh"),
                                          "sha256": sha(out / "lasp.external.sh"),
                                          "client_path": str(out / "client.py")}
    provenance["native_command"] = ["/lib64/ld-linux-x86-64.so.2", str(BINARY)]
    provenance["mpi_library_path"] = "/opt/devtools/intel/oneapi/mpi/2021.13/lib"
    shutil.copy2(HELPER, out / "lasp_external_ase.py")
    (out / "actual_input.arc").write_text(arc)
    dump(out / "provenance.json", provenance)
    calc = model_calculator()
    provenance.update({"torch_version": __import__("torch").__version__,
                       "cuda_device_name": __import__("torch").cuda.get_device_name(0),
                       "mace_calculator_source": calc.mace_calculator_source,
                       "model_r_max_A": float(calc.r_max),
                       "slurm_job_id": job, "slurm_job_node": os.environ.get("SLURMD_NODENAME"),
                       "slurm_tmpdir": os.environ.get("TMPDIR")})
    dump(out / "provenance.json", provenance)
    direct = {}
    for label, pbc in (("isolated", [False] * 3), ("periodic", [True] * 3)):
        case = atoms.copy(); case.pbc = pbc; case.calc = calc
        before = calc.actual_calculations
        calc.charge_slot(case)
        try:
            energy = float(case.get_potential_energy())
            forces = np.asarray(case.get_forces(), dtype=float)
            if not np.isfinite(energy) or not np.isfinite(forces).all():
                raise ValueError("nonfinite E/F")
            direct[label] = {"ok": True, "energy_eV": energy, "forces_eV_A": forces.tolist(),
                             "actual_calculations": calc.actual_calculations - before,
                             "pbc": list(pbc), "positions_A": case.positions.tolist(), "cell_A": CELL.tolist()}
        except Exception as exc:
            direct[label] = {"ok": False, "error": repr(exc), "issued": 1, "paid": 1,
                             "actual_calculations": calc.actual_calculations - before,
                             "pbc": list(pbc), "positions_A": case.positions.tolist(), "cell_A": CELL.tolist()}
            dump(out / "direct-results.json", direct)
            continue
        dump(out / "direct-results.json", direct)
    direct_paid = len(direct)  # each started direct E/F slot is charged, even on failure
    direct_actual = sum(int(row.get("actual_calculations", 0)) for row in direct.values())
    if not all(direct.get(key, {}).get("ok") for key in ("isolated", "periodic")):
        dump(out / "cost-ledger.json", {"direct": {"issued": len(direct), "paid": direct_paid,
              "actual_calculations": direct_actual}, "callback": {"issued": 0, "paid": 0,
              "actual_calculations": 0, "denials": 0}, "aggregate": {"issued": len(direct),
              "paid": direct_paid, "actual_calculations": direct_actual, "denials": 0}})
        dump(out / "summary.json", {"status": "direct_evaluation_failure", "direct": direct,
              "direct_issued": len(direct), "direct_paid": direct_paid,
              "direct_actual_calculations": direct_actual, "callback_issued": 0, "callback_paid": 0,
              "callback_actual_calculations": 0, "callback_denials": 0,
              "total_issued": len(direct), "total_paid": direct_paid,
              "total_actual_calculations": direct_actual, "total_denials": 0,
              "elapsed_seconds": time.monotonic() - started, "qualification_pass": False})
        raise SystemExit("direct evaluation failure retained; native callback not attempted")
    de = abs(direct["isolated"]["energy_eV"] - direct["periodic"]["energy_eV"])
    df = float(np.max(np.abs(np.asarray(direct["isolated"]["forces_eV_A"]) - np.asarray(direct["periodic"]["forces_eV_A"]))))
    cutoff = float(calc.r_max)
    distance = nearest_interimage(atoms)
    model_gate = distance > cutoff and de <= 1e-4 and df <= 1e-4
    if not model_gate:
        dump(out / "cost-ledger.json", {"direct": {"issued": len(direct), "paid": direct_paid,
              "actual_calculations": direct_actual}, "callback": {"issued": 0, "paid": 0,
              "actual_calculations": 0, "denials": 0}, "aggregate": {"issued": len(direct),
              "paid": direct_paid, "actual_calculations": direct_actual, "denials": 0}})
        dump(out / "summary.json", {"status": "model_boundary_gate_failed", "model_boundary_gate": False,
              "energy_abs_difference_eV": de, "max_force_component_difference_eV_A": df,
              "interimage_distance_A": distance, "model_cutoff_A": cutoff,
              "direct_issued": len(direct), "direct_paid": direct_paid,
              "direct_actual_calculations": direct_actual, "callback_issued": 0,
              "callback_paid": 0, "callback_actual_calculations": 0, "callback_denials": 0,
              "total_issued": len(direct), "total_paid": direct_paid,
              "total_actual_calculations": direct_actual, "total_denials": 0,
              "elapsed_seconds": time.monotonic() - started, "qualification_pass": False})
        raise SystemExit("model/PBC qualification failed; no native callback attempted")
    calc.calc.reset()
    provenance["actual_input_artifacts"] = {
        name: {"path": str(out / name), "sha256": sha(out / name)}
        for name in ("input.extxyz", "input.arc", "lasp.in", "client.py", "lasp.external.sh",
                     "bounded_process.py", "lasp_external_ase.py")
    }
    dump(out / "provenance.json", provenance)
    mpi_lib = "/opt/devtools/intel/oneapi/mpi/2021.13/lib"
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = mpi_lib + ((":" + env["LD_LIBRARY_PATH"]) if env.get("LD_LIBRARY_PATH") else "")
    env.update(I_MPI_FABRICS="shm", I_MPI_PIN="0")
    command = bounded_command(out, ["/lib64/ld-linux-x86-64.so.2", str(BINARY)], 60)
    helper = load_lasp_helper(out / "lasp_external_ase.py")
    result = helper.run_lasp(command, cwd=out, atoms=periodic, calculator=calc,
                                         pbc=[True] * 3, max_requests=1, env=env)
    dump(out / "lasp-callback.json", result)
    proc = json.loads((out / "process.json").read_text()) if (out / "process.json").exists() else None
    raw = result.get("requests", [{}])[0].get("raw_coord") if result.get("requests") else None
    (out / "external.coord.raw").write_text(raw or "")
    for name in ("external.ene", "external.coord", "lasp.out"):
        p = out / name
        if p.exists(): shutil.copy2(p, out / (name.replace(".", "-") + ".captured"))
    req = result.get("requests", [])
    callback = req[0] if req else None
    coord_delta = None
    ef_delta = None
    callback_geometry_match = False
    callback_ef_match = False
    if callback:
        pos = np.asarray(callback["positions"], float)
        coord_delta = float(np.max(np.abs(pos - atoms.positions)))
        callback_geometry_match = (np.allclose(callback["cell"], CELL, rtol=0., atol=1e-8)
                                   and coord_delta <= 1e-4)
        ef_delta = {"energy_abs_eV": abs(float(callback["response"]["energy"]) - direct["periodic"]["energy_eV"]),
                    "max_force_component_eV_A": float(np.max(np.abs(np.asarray(callback["response"]["forces"]) - np.asarray(direct["periodic"]["forces_eV_A"]))))}
        callback_ef_match = ef_delta["energy_abs_eV"] <= 1e-4 and ef_delta["max_force_component_eV_A"] <= 1e-4
    issued = len(req) + len(result.get("errors", []))
    callback_actual = calc.actual_calculations - direct_actual
    callback_denials = sum(any(token in str(item.get("error", "")) for token in
                               ("external request cap reached", "paid E/F cap reached"))
                           for item in result.get("errors", []))
    callback_errors = len(result.get("errors", []))
    callback_paid = max(0, calc.paid_ef_slots - direct_paid)
    summary = {"status": "completed", "model_boundary_gate": bool(model_gate), "energy_abs_difference_eV": de,
               "max_force_component_difference_eV_A": df, "interimage_distance_A": distance,
               "model_cutoff_A": cutoff, "direct_issued": len(direct), "direct_paid": direct_paid,
               "direct_actual_calculations": direct_actual, "callback_issued": issued, "callback_paid": callback_paid,
               "callback_actual_calculations": callback_actual,
               "total_issued": len(direct) + issued, "total_paid": direct_paid + callback_paid,
               "total_actual_calculations": direct_actual + callback_actual,
               "callback_denials": callback_denials, "callback_errors": callback_errors,
               "total_denials": callback_denials,
               "callback_geometry_max_abs_A": coord_delta, "callback_geometry_match": callback_geometry_match,
               "callback_ef_difference": ef_delta, "callback_ef_match": callback_ef_match,
               "callback_process": proc, "callback_state": result.get("state"),
               "callback_supervisor_returncode": result.get("returncode"),
               "native_returncode": None if proc is None else proc.get("returncode"),
               "native_log_contains_ssw_done": False,
               "elapsed_seconds": time.monotonic() - started,
               "qualification_pass": bool(model_gate and callback and callback_geometry_match and callback_ef_match
                                           and result.get("returncode") == 0 and proc
                                           and proc.get("state") == "completed"
                                           and not proc.get("cleanup_survivors"))}
    if (out / "lasp.out").exists():
        summary["native_log_contains_ssw_done"] = "SSW all done" in (out / "lasp.out").read_text(errors="replace")
    dump(out / "summary.json", summary)
    dump(out / "cost-ledger.json", {"direct": {"issued": len(direct), "paid": direct_paid,
          "actual_calculations": direct_actual}, "callback": {"issued": issued, "paid": callback_paid,
          "actual_calculations": callback_actual, "denials": callback_denials,
          "errors": callback_errors},
          "aggregate": {"issued": summary["total_issued"], "paid": summary["total_paid"],
          "actual_calculations": summary["total_actual_calculations"],
          "denials": summary["total_denials"]}})
    if not summary["qualification_pass"]:
        raise SystemExit("qualification failed; see preserved run artifacts")


if __name__ == "__main__":
    main()
