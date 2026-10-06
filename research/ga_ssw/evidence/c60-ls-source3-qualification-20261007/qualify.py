#!/usr/bin/env python3
"""Qualify one archived C60 isomer on MH-1; this program never searches."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARCHIVE_ROOT = Path("/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/literature")
ZIP_PATH = ARCHIVE_ROOT / "c60-defect-input-20260925/41524_2024_1410_MOESM3_ESM.zip"
CSV_PATH = ARCHIVE_ROOT / "c60-defect-input-20260925/41524_2024_1410_MOESM2_ESM.csv"
ZIP_MEMBER = "c60/c60-iso-3_opt.xyz"
REFERENCE = ROOT / "research/ga_ssw/evidence/c60-local-defect-20260925/qualification/isomer-1/final.extxyz"
EXPECTED_MODEL_SHA = "a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47"
MODEL = Path("/home/gengjianrui/.cache/mace/mace-mh-1.model")
HELPER_PATH = ROOT / "research/ga_ssw/evidence/c60-direction-transfer-20261007/prepare.py"
VALIDATOR_PATH = ROOT / "research/ga_ssw/evidence/c60-long-budget-20260924/ssw-17101/validator.py"
INPUT_CAP, INPUT_SECONDS, PROCESS_SECONDS = 3000, 120.0, 240.0
REFERENCE_ENERGY = -62215.393370790625
CUTOFFS = (1.64, 1.70, 1.80)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_helper(path: Path):
    spec = importlib.util.spec_from_file_location("c60_direction_prepare_helper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import counted-surface helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_ledgers(out: Path):
    """CountedSurface requires the ledger parent before its first request."""
    ledgers = out / "ledgers"
    ledgers.mkdir(exist_ok=False)
    return ledgers


def source_records():
    if not ZIP_PATH.is_file() or not CSV_PATH.is_file():
        raise FileNotFoundError(f"C60 author archive or CSV is missing under {ARCHIVE_ROOT}")
    with zipfile.ZipFile(ZIP_PATH) as archive:
        info = archive.getinfo(ZIP_MEMBER)
        raw_member = archive.read(ZIP_MEMBER)
    csv_bytes = CSV_PATH.read_bytes()
    text = csv_bytes.decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text, newline="")))
    matches = [row for row in rows if row.get("Cn", "").strip().lower() == "c60"
               and row.get("#iso", "").strip() == "3"]
    if len(matches) != 1:
        raise ValueError(f"expected one C60 #3 CSV row, found {len(matches)}")
    physical_lines = text.splitlines(keepends=True)
    row_lines = [line for line in physical_lines if line.lower().startswith("c60,60,3,")]
    if len(row_lines) != 1:
        raise ValueError(f"could not retain exactly one raw CSV line for C60 #3: {len(row_lines)}")
    csv_header = physical_lines[0]
    return {
        "raw_member": raw_member,
        "zip_info": {"archive_path": str(ZIP_PATH), "archive_sha256": sha256(ZIP_PATH),
                     "member": ZIP_MEMBER, "member_sha256": sha256_bytes(raw_member),
                     "member_bytes": len(raw_member), "member_crc32": f"{info.CRC:08x}"},
        "csv_bytes": csv_bytes,
        "csv_row": matches[0],
        "csv_raw": (csv_header + row_lines[0]).encode("utf-8"),
        "csv_info": {"path": str(CSV_PATH), "sha256": sha256(CSV_PATH)},
    }


def build_input(raw_member: bytes):
    import numpy as np
    from ase.io import read

    atoms = read(io.StringIO(raw_member.decode("utf-8").rstrip() + "\n"), format="xyz")
    if len(atoms) != 60 or not np.all(atoms.numbers == 6) or not np.isfinite(atoms.positions).all():
        raise ValueError("author C60 #3 source is not finite 60-carbon geometry")
    source = atoms.copy()
    atoms.positions += np.array([25., 25., 25.]) - atoms.get_center_of_mass()
    atoms.set_cell([50., 50., 50.])
    atoms.set_pbc(False)
    return source, atoms


def graph_for(numbers, positions, cutoff):
    import networkx as nx
    import numpy as np

    xyz = np.asarray(positions, dtype=float)
    distance = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(numbers)))
    graph.add_edges_from((int(i), int(j)) for i, j in zip(*np.where(np.triu((distance < cutoff) & (distance > 0), 1))))
    return graph


def graph_diagnostics(atoms, validator, ih_atoms):
    import networkx as nx

    rows = {}
    for cutoff in CUTOFFS:
        graph = graph_for(atoms.numbers, atoms.positions, cutoff)
        ih_graph = graph_for(ih_atoms.numbers, ih_atoms.positions, cutoff)
        row = validator.graph_row(atoms.numbers, atoms.positions, cutoff, ih_graph)
        row.update({
            "fullerene_cage": bool(row["graph_cage_candidate"]),
            "isomorphic_to_ih": bool(nx.is_isomorphic(graph, ih_graph)),
            "wl_hash": nx.weisfeiler_lehman_graph_hash(graph),
        })
        rows[str(cutoff)] = row
    return rows


def source_csv_and_input_preflight():
    import numpy as np
    from ase.io import read, write

    records = source_records()
    source, atoms = build_input(records["raw_member"])
    if atoms.pbc.any() or not np.allclose(atoms.cell.array, np.diag([50., 50., 50.])):
        raise RuntimeError("translated input does not meet isolated 50-A storage-cell contract")
    if not np.allclose(atoms.get_center_of_mass(), [25., 25., 25.], atol=1e-10, rtol=0):
        raise RuntimeError("translated input COM differs from (25,25,25)")
    with tempfile.TemporaryDirectory(prefix="c60-source3-preflight-") as temp:
        input_path = Path(temp) / "input.extxyz"
        write(input_path, atoms, format="extxyz")
        serialized = read(input_path, format="extxyz")
        if (len(serialized) != 60 or not np.array_equal(serialized.numbers, atoms.numbers)
                or not np.array_equal(serialized.pbc, atoms.pbc)
                or not np.allclose(serialized.cell.array, atoms.cell.array, atol=0, rtol=0)
                or not np.allclose(serialized.positions, atoms.positions, atol=1e-8, rtol=0)):
            raise RuntimeError("serialized qualification input differs from frozen geometry")
        (Path(temp) / "raw-source.xyz").write_bytes(records["raw_member"])
        (Path(temp) / "isomer-3-row.csv").write_bytes(records["csv_raw"])
        reread = read(input_path, format="extxyz")
        if (not np.allclose(reread.positions, serialized.positions, atol=1e-8, rtol=0)
                or sha256(Path(temp) / "raw-source.xyz") != records["zip_info"]["member_sha256"]):
            raise RuntimeError("input/source serialization preflight failed")

    helper = load_helper(HELPER_PATH)
    sys.path.insert(0, str(ROOT))
    validator = load_helper(VALIDATOR_PATH)
    ih_atoms = read(REFERENCE)
    source_graphs = graph_diagnostics(source, validator, ih_atoms)
    input_graphs = graph_diagnostics(serialized, validator, ih_atoms)
    if not all(input_graphs[key]["fullerene_cage"] and not input_graphs[key]["isomorphic_to_ih"]
               and input_graphs[key]["isomorphic_to_ih"] == source_graphs[key]["isomorphic_to_ih"]
               for key in ("1.64", "1.7", "1.8")):
        raise RuntimeError("source #3 graph screening failed during preflight")
    import numpy as np
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes

    class Dummy(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self):
            super().__init__()
            self.calls = 0

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.calls += 1
            self.results = {"energy": 0.0, "forces": np.zeros((len(self.atoms), 3))}

    with tempfile.TemporaryDirectory(prefix="c60-source3-fake-qualification-") as temp:
        ledgers = create_ledgers(Path(temp))
        calculator = Dummy()
        call_counter = helper.instrument_calculate(calculator)
        calculator._c60_call_counter = call_counter
        surface = helper.CountedSurface(calculator, ledgers / "initial-quench.jsonl", 8,
            time.monotonic() + 60, time.monotonic() + 60)
        from ase import Atoms
        from pamssw.standalone import NativeMCSettings, SSWConfig, run_ssw
        result = run_ssw(Atoms("C2", positions=[[0., 0., 0.], [1.4, 0., 0.]]), surface,
            steps=0, config=SSWConfig(**helper.protocol_config()),
            rng=np.random.default_rng(3), mc=NativeMCSettings(.1, 99999))
        if not (result.status == "completed" and result.initial.converged
                and len(result.records) == 0 and surface.requests == 1
                and result.evaluation_requests == 1 and call_counter["count"] == 1):
            raise RuntimeError("fake run_ssw qualification integration failed")

    print(json.dumps({"status": "preflight_passed", "real_pes_requests": 0,
        "real_model_initialized": False, "archive_member": ZIP_MEMBER,
        "archive_member_sha256": records["zip_info"]["member_sha256"],
        "csv_row": records["csv_row"], "serialized_input_write_read": "passed",
        "fake_run_ssw_steps": 0, "fake_requests": surface.requests,
        "fake_calculator_calls": call_counter["count"], "fake_outer_records": 0}, indent=2))


def snapshot_sources(out: Path):
    frozen = out / "frozen"
    copied = []
    for live_path in sorted((ROOT / "pamssw").rglob("*.py")):
        rel = live_path.relative_to(ROOT)
        target = frozen / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(live_path, target)
        copied.append({"path": str(rel), "sha256": sha256(target), "live_path": str(live_path.resolve())})
    helper_snapshot = out / "used-helpers" / "c60-direction-prepare.py"
    helper_snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HELPER_PATH, helper_snapshot)
    validator_snapshot = out / "used-helpers" / "validator.py"
    shutil.copy2(VALIDATOR_PATH, validator_snapshot)
    script_snapshot = out / "used-helpers" / "qualify.py"
    shutil.copy2(Path(__file__).resolve(), script_snapshot)
    shutil.copy2(HERE / "protocol.md", out / "protocol.md")
    git = lambda *args: subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()
    return {
        "checkout": str(ROOT), "git_head": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "git_status_short_core": git("status", "--short", "--", "pamssw", "research/ga_ssw/c60_long_budget.py").splitlines(),
        "core_snapshot_root": "frozen/pamssw", "core_files": copied,
        "qualification_script": {"live_path": str(Path(__file__).resolve()),
            "snapshot_path": "used-helpers/qualify.py", "sha256": sha256(script_snapshot)},
        "counted_surface_helper": {"live_path": str(HELPER_PATH.resolve()),
            "snapshot_path": "used-helpers/c60-direction-prepare.py", "sha256": sha256(helper_snapshot)},
        "graph_validator": {"live_path": str(VALIDATOR_PATH.resolve()),
            "snapshot_path": "used-helpers/validator.py", "sha256": sha256(validator_snapshot)},
        "protocol_sha256": sha256(out / "protocol.md"),
    }


def execute(out: Path):
    import numpy as np
    from ase.io import read, write

    process_started = time.monotonic()
    process_deadline = process_started + PROCESS_SECONDS
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    create_ledgers(out)
    dump(out / "status.json", {"status": "preparing", "started_unix": time.time(),
                                "process_cap_seconds": PROCESS_SECONDS})
    try:
        records = source_records()
        source_atoms, input_atoms = build_input(records["raw_member"])
        (out / "source").mkdir()
        (out / "source" / "author-isomer-3.xyz").write_bytes(records["raw_member"])
        (out / "source" / "author-isomer-3-row.csv").write_bytes(records["csv_raw"])
        (out / "inputs").mkdir()
        input_path = out / "inputs" / "input.extxyz"
        write(input_path, input_atoms, format="extxyz")
        serialized = read(input_path, format="extxyz")
        if (len(serialized) != 60 or not np.array_equal(serialized.numbers, input_atoms.numbers)
                or not np.array_equal(serialized.pbc, input_atoms.pbc)
                or not np.allclose(serialized.cell.array, input_atoms.cell.array, atol=0, rtol=0)
                or not np.allclose(serialized.positions, input_atoms.positions, atol=1e-8, rtol=0)):
            raise RuntimeError("serialized input changed source-3 geometry")
        dump(out / "source" / "source-record.json", {"archive": records["zip_info"],
            "csv": records["csv_info"], "csv_row": records["csv_row"],
            "csv_row_snapshot_sha256": sha256(out / "source" / "author-isomer-3-row.csv"),
            "input_path": str(input_path.relative_to(out)), "input_sha256": sha256(input_path),
            "input_transformation": "translate COM to (25,25,25); no rotation or internal-coordinate change",
            "cell_A": [50., 50., 50.], "pbc": [False, False, False],
            "source_author_isomer": 3})
        ih_atoms = read(REFERENCE)
        if (len(ih_atoms) != 60 or not np.all(ih_atoms.numbers == 6)
                or ih_atoms.pbc.any() or not np.isfinite(ih_atoms.positions).all()):
            raise ValueError("historical Ih reference is not finite nonperiodic C60")
        shutil.copy2(REFERENCE, out / "source" / "historical-ih-reference.extxyz")
        source_manifest = snapshot_sources(out)
        dump(out / "source-manifest.json", source_manifest)

        helper = load_helper(HELPER_PATH)
        validator_spec = importlib.util.spec_from_file_location("c60_source3_validator", out / "used-helpers" / "validator.py")
        validator = importlib.util.module_from_spec(validator_spec)
        validator_spec.loader.exec_module(validator)
        source_graphs = graph_diagnostics(source_atoms, validator, ih_atoms)
        input_graphs = graph_diagnostics(serialized, validator, ih_atoms)
        graph_source_check = {cutoff: {
            "isomorphic_to_source": bool(__import__("networkx").is_isomorphic(
                graph_for(source_atoms.numbers, source_atoms.positions, float(cutoff)),
                graph_for(serialized.numbers, serialized.positions, float(cutoff)))),
            "isomorphic_to_ih": input_graphs[cutoff]["isomorphic_to_ih"],
            "fullerene_cage": input_graphs[cutoff]["fullerene_cage"]}
            for cutoff in ("1.64", "1.7", "1.8")}
        if not all(row["isomorphic_to_source"] and row["fullerene_cage"] and not row["isomorphic_to_ih"]
                   for row in graph_source_check.values()):
            raise ValueError("author #3 serialized input fails all-cutoff source/non-Ih cage gate")

        model_sha = sha256(MODEL)
        if model_sha != EXPECTED_MODEL_SHA:
            raise ValueError(f"MH-1 model SHA differs from protocol: {model_sha}")
        config_dict = helper.protocol_config()
        config_dict.update({"cluster_frame": "direction_only", "relax_steps": 1000,
                            "fmax": .03, "quench_optimizer": "safe-lbfgs-total",
                            "lbfgs_memory": 500})
        effective_config = {"mode": "ordinary run_ssw(steps=0)", "ssw_config": config_dict,
            "ls": None, "recovered_rotation": None, "recovered_direction": None,
            "outer_steps": 0, "model": str(MODEL), "model_sha256": model_sha,
            "head": "omol", "device": "cuda", "dtype": "float64",
            "enable_cueq": False, "enable_oeq": False,
            "runtime": {"torch_manual_seed": 0, "torch_deterministic_algorithms": True,
                "torch_num_threads": 1, "tf32": False,
                "CUBLAS_WORKSPACE_CONFIG": os.getenv("CUBLAS_WORKSPACE_CONFIG")}}
        dump(out / "effective-config.json", effective_config)

        frozen_root = out / "frozen"
        sys.path.insert(0, str(frozen_root))
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("qualification requires its authorized single-GPU CUDA allocation")
        from mace.calculators import MACECalculator
        from pamssw.standalone import NativeMCSettings, SSWConfig, run_ssw
        from pamssw.standalone.paper_reference import SSWConfig as FrozenSSWConfig
        imports_before = {name: str(Path(module.__file__).resolve()) for name, module in {
            "paper_reference": sys.modules["pamssw.standalone.paper_reference"],
            "surface": sys.modules["pamssw.standalone.surface"],
            "native_mc": sys.modules["pamssw.standalone.native_mc"]}.items()}
        for import_path in imports_before.values():
            if not Path(import_path).is_relative_to(frozen_root.resolve()):
                raise RuntimeError(f"qualification imported live core source: {import_path}")
        from importlib.metadata import version
        runtime = {"python": sys.version, "python_executable": sys.executable,
            "numpy": np.__version__, "ase": version("ase"), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "mace_torch": version("mace-torch"),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "slurm_job_id": os.getenv("SLURM_JOB_ID"), "core_imports": imports_before,
            "source_manifest_sha256": sha256(out / "source-manifest.json"),
            "helper_live_path": str(HELPER_PATH.resolve()),
            "helper_sha256": sha256(HELPER_PATH), "validator_sha256": sha256(out / "used-helpers" / "validator.py")}
        dump(out / "runtime.json", runtime)
        if time.monotonic() >= process_deadline:
            raise RuntimeError("process deadline reached during startup/model preparation")

        model_kwargs = dict(model_paths=str(MODEL), head="omol", device="cuda",
                            default_dtype="float64", enable_cueq=False, enable_oeq=False)
        search_calc = MACECalculator(**model_kwargs)
        search_calls = helper.instrument_calculate(search_calc)
        search_calc._c60_call_counter = search_calls
        cold_calc = MACECalculator(**model_kwargs)
        cold_calls = helper.instrument_calculate(cold_calc)
        cold_calc._c60_call_counter = cold_calls

        # The helper's counted wrapper keeps request, actual calculate, denial,
        # and failure accounting in the same schema as the qualified C60 panel.
        reference_surface = helper.CountedSurface(cold_calc,
            out / "ledgers" / "reference-cold.jsonl", 1, process_deadline, process_deadline)
        ref_before = cold_calls["count"]
        reference_row = {"source_path": str(REFERENCE.resolve()), "energy_source_historical_only": True}
        try:
            ref_energy, ref_forces = reference_surface.evaluate(ih_atoms)
            ref_fmax = float(np.linalg.norm(ref_forces, axis=1).max())
            ref_graphs = graph_diagnostics(ih_atoms, validator, ih_atoms)
            ref_finite = bool(np.isfinite(ref_energy) and np.isfinite(ref_forces).all() and np.isfinite(ref_fmax))
            reference_row.update({"status": "completed", "energy_eV": ref_energy,
                "historical_energy_eV": REFERENCE_ENERGY,
                "energy_error_eV": float(ref_energy - REFERENCE_ENERGY), "fmax_eV_A": ref_fmax,
                "requests": reference_surface.requests, "calculator_calls": cold_calls["count"] - ref_before,
                "denials": reference_surface.denials, "finite": ref_finite,
                "graphs": ref_graphs, "qualified": bool(ref_finite and reference_surface.requests == 1
                    and abs(ref_energy - REFERENCE_ENERGY) <= 1e-6 and ref_fmax <= .03
                    and all(row["fullerene_cage"] and row["isomorphic_to_ih"] for row in ref_graphs.values()))})
        except Exception as error:
            reference_row.update({"status": "failed", "error": repr(error),
                "requests": reference_surface.requests, "denials": reference_surface.denials,
                "calculator_calls": cold_calls["count"] - ref_before, "qualified": False})
        dump(out / "reference-cold.json", reference_row)

        input_started = time.monotonic()
        input_deadline = min(input_started + INPUT_SECONDS, process_deadline)
        search_surface = helper.CountedSurface(search_calc, out / "ledgers" / "initial-quench.jsonl",
            INPUT_CAP, input_deadline, process_deadline)
        config = FrozenSSWConfig(**config_dict)
        mc = NativeMCSettings(.1, 99999)
        initial_before = search_calls["count"]
        result = None
        initial = None
        run_error = None
        try:
            result = run_ssw(serialized.copy(), search_surface, steps=0, config=config,
                rng=np.random.default_rng(2026100703), mc=mc)
            initial = result.initial
            if result.records or result.evaluation_requests != search_surface.requests:
                raise RuntimeError("steps=0 qualification unexpectedly emitted outer records or lost accounting")
        except Exception as error:
            run_error = repr(error)
            result = getattr(error, "result", None)
            initial = getattr(result, "initial", None) if result is not None else None
            if initial is None:
                initial = getattr(error, "initial", None)
        initial_row = {"status": "failed" if run_error else "completed",
            "error": run_error, "search_status": getattr(result, "status", None),
            "outer_records": len(getattr(result, "records", ())) if result is not None else None,
            "requests": search_surface.requests, "denials": search_surface.denials,
            "calculator_calls": search_calls["count"] - initial_before,
            "elapsed_seconds": time.monotonic() - input_started,
            "input_deadline_seconds": INPUT_SECONDS, "request_cap": INPUT_CAP,
            "evaluation_requests_reported": getattr(result, "evaluation_requests", None)}
        endpoint_ok = False
        if initial is not None:
            endpoint_path = out / "initial-endpoint.extxyz"
            write(endpoint_path, initial.atoms, format="extxyz")
            endpoint_graphs = graph_diagnostics(initial.atoms, validator, ih_atoms)
            energy = float(initial.energy) if initial.energy is not None else float("nan")
            initial_fmax = float(initial.max_force)
            finite = bool(np.isfinite(energy) and np.isfinite(initial.atoms.positions).all()
                           and np.isfinite(initial_fmax))
            endpoint_retain = {cutoff: bool(__import__("networkx").is_isomorphic(
                graph_for(serialized.numbers, serialized.positions, float(cutoff)),
                graph_for(initial.atoms.numbers, initial.atoms.positions, float(cutoff))))
                for cutoff in ("1.64", "1.7", "1.8")}
            initial_row.update({"endpoint_path": str(endpoint_path.relative_to(out)),
                "endpoint_sha256": sha256(endpoint_path), "converged": bool(initial.converged),
                "energy_eV": energy if np.isfinite(energy) else None,
                "fmax_eV_A": initial_fmax if np.isfinite(initial_fmax) else None,
                "finite": finite, "endpoint_graphs": endpoint_graphs,
                "endpoint_isomorphic_to_source": endpoint_retain,
                "endpoint_isomorphic_to_ih": {key: row["isomorphic_to_ih"] for key, row in endpoint_graphs.items()},
                "endpoint_fullerene_at_all_cutoffs": all(row["fullerene_cage"] for row in endpoint_graphs.values())})
            source_retained = all(endpoint_retain.values())
            ih_distinct = all(not row["isomorphic_to_ih"] for row in endpoint_graphs.values())
            endpoint_cage = all(row["fullerene_cage"] for row in endpoint_graphs.values())
            endpoint_ok = bool(finite and initial.converged and initial_fmax <= .03
                               and source_retained and ih_distinct and endpoint_cage
                               and getattr(result, "status", None) == "completed"
                               and search_surface.denials == 0
                               and time.monotonic() < process_deadline)
        dump(out / "initial-quench.json", initial_row)

        candidate_cold = {"status": "not_run_initial_not_certified", "qualified": False}
        if endpoint_ok and reference_row.get("qualified") and time.monotonic() < process_deadline:
            cold_calc.reset()
            candidate_before = cold_calls["count"]
            candidate_surface = helper.CountedSurface(cold_calc,
                out / "ledgers" / "candidate-cold.jsonl", 1, process_deadline, process_deadline)
            try:
                cold_energy, cold_forces = candidate_surface.evaluate(initial.atoms)
                cold_fmax = float(np.linalg.norm(cold_forces, axis=1).max())
                cold_graphs = graph_diagnostics(initial.atoms, validator, ih_atoms)
                initial_energy = float(initial.energy)
                energy_repeat = abs(cold_energy - initial_energy) <= 1e-6
                relative_energy = float(cold_energy - reference_row["energy_eV"])
                source_retained = all(row["isomorphic_to_ih"] is False
                    and __import__("networkx").is_isomorphic(
                        graph_for(serialized.numbers, serialized.positions, float(cutoff)),
                        graph_for(initial.atoms.numbers, initial.atoms.positions, float(cutoff)))
                    for cutoff, row in cold_graphs.items())
                cage = all(row["fullerene_cage"] for row in cold_graphs.values())
                candidate_qualified = bool(np.isfinite(cold_energy) and np.isfinite(cold_forces).all()
                    and np.isfinite(cold_fmax) and cold_fmax <= .03 and energy_repeat
                    and source_retained and cage and candidate_surface.requests == 1
                    and candidate_surface.denials == 0)
                candidate_cold = {"status": "completed", "energy_eV": cold_energy,
                    "energy_repeat_error_eV": float(cold_energy - initial_energy),
                    "energy_repeat": bool(energy_repeat), "relative_to_ih_eV": relative_energy,
                    "fmax_eV_A": cold_fmax, "requests": candidate_surface.requests,
                    "calculator_calls": cold_calls["count"] - candidate_before,
                    "denials": candidate_surface.denials, "graphs": cold_graphs,
                    "qualified": candidate_qualified,
                    "within_ih_plus_0.01_eV": bool(relative_energy <= .01)}
                if candidate_qualified:
                    shutil.copy2(out / "initial-endpoint.extxyz", out / "final-candidate.extxyz")
            except Exception as error:
                candidate_cold = {"status": "failed", "error": repr(error),
                    "requests": candidate_surface.requests, "denials": candidate_surface.denials,
                    "calculator_calls": cold_calls["count"] - candidate_before, "qualified": False}
        dump(out / "candidate-cold.json", candidate_cold)

        source_rows = graph_diagnostics(serialized, validator, ih_atoms)
        initial_pass = bool(endpoint_ok and reference_row.get("qualified") and candidate_cold.get("qualified"))
        delta = candidate_cold.get("relative_to_ih_eV")
        local_energy_case = bool(initial_pass and delta is not None and delta > .01)
        reasons = []
        if not reference_row.get("qualified"):
            reasons.append("reference_cold_qualification_failed")
        if not endpoint_ok:
            reasons.append("initial_quench_or_source_graph_gate_failed")
        if endpoint_ok and candidate_cold.get("qualified") and delta is not None and delta <= .01:
            reasons.append("already_within_ih_plus_0.01_eV_not_an_energy_repair_case")
        if endpoint_ok and reference_row.get("qualified") and not candidate_cold.get("qualified"):
            reasons.append("candidate_cold_qualification_failed")
        if time.monotonic() >= process_deadline:
            reasons.append("process_deadline_reached")
        eligible = bool(local_energy_case and not reasons and time.monotonic() < process_deadline)
        actual_core_imports = {}
        for name, module in sorted(sys.modules.items()):
            module_path = getattr(module, "__file__", None)
            if name == "pamssw" or name.startswith("pamssw."):
                if module_path is None:
                    continue
                resolved = Path(module_path).resolve()
                if not resolved.is_relative_to(frozen_root.resolve()):
                    raise RuntimeError(f"runtime imported core module outside frozen snapshot: {name}={resolved}")
                actual_core_imports[name] = {"path": str(resolved), "sha256": sha256(resolved)}
        runtime["core_imports"] = actual_core_imports
        dump(out / "runtime.json", runtime)
        qualification = {"status": "eligible_for_later_local_ls_protocol" if eligible else "not_eligible",
            "scientific_scope": "input qualification only; no LS, outer search, optimizer comparison, or global-minimum claim",
            "search_allowed": False, "eligibility_gate_passed": eligible,
            "block_reasons": reasons, "source_geometry": records["zip_info"],
            "source_csv": records["csv_info"], "source_csv_row": records["csv_row"],
            "input_path": str(input_path.relative_to(out)), "input_sha256": sha256(input_path),
            "source_graphs": source_rows, "reference": reference_row,
            "initial_quench": initial_row, "candidate_cold": candidate_cold,
            "local_energy_repair_case": local_energy_case,
            "reference_energy_window_eV": .01,
            "budgets": {"maximum_initial_requests": INPUT_CAP,
                "maximum_initial_seconds": INPUT_SECONDS, "maximum_cold_checks": 2,
                "maximum_total_requests": INPUT_CAP + 2,
                "process_cap_seconds": PROCESS_SECONDS},
            "elapsed_seconds": time.monotonic() - process_started,
            "actual_calculator_calls": {"initial": search_calls["count"], "cold": cold_calls["count"]},
            "source_manifest_sha256": sha256(out / "source-manifest.json")}
        dump(out / "qualification.json", qualification)
        dump(out / "status.json", {"status": qualification["status"],
            "search_allowed": False, "elapsed_seconds": qualification["elapsed_seconds"]})
        print(json.dumps(qualification, indent=2, allow_nan=False))
        return 0 if eligible else 2
    except Exception as error:
        dump(out / "status.json", {"status": "failed", "search_allowed": False,
            "error": repr(error), "elapsed_seconds": time.monotonic() - process_started})
        raise


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.preflight:
        if args.output is not None:
            parser.error("--preflight does not write an output directory")
        source_csv_and_input_preflight()
        return
    if not args.output:
        parser.error("--execute requires --output NEW_DIRECTORY")
    if args.output.exists():
        parser.error("--execute output must be a new directory; retries/overwrite are prohibited")
    raise SystemExit(execute(args.output))


if __name__ == "__main__":
    main()
