#!/usr/bin/env python3
"""Freeze a four-arm, fixed-start C60 source-3 SSW/paper-LS transfer panel."""
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PROTOCOL = HERE / "protocol.md"
QUAL_DIR = ROOT / "research/ga_ssw/evidence/c60-ls-source3-qualification-20261007/qualification-1664282"
QUAL_JSON = QUAL_DIR / "qualification.json"
RUNNER = ROOT / "research/ga_ssw/c60_long_budget.py"
VALIDATOR = ROOT / "research/ga_ssw/evidence/c60-long-budget-20260924/ssw-17101/validator.py"
DIRECTION_HELPER = ROOT / "research/ga_ssw/evidence/c60-direction-transfer-20261007/prepare.py"
MODEL = Path("/home/gengjianrui/.cache/mace/mace-mh-1.model")
MODEL_SHA = "a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47"
SEEDS = (26100791, 26100792)
ARMS = ("ssw", "paper_ls")
SEARCH_CAP, FRESH_CAP, WALL_SECONDS, OUTER_STEPS = 20000, 3, 1200, 100
CUTOFFS = (1.64, 1.70, 1.80)


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


def load_json(path):
    return json.loads(Path(path).read_text())


def qualified_sources():
    if not QUAL_JSON.is_file():
        raise FileNotFoundError(f"required source-3 qualification is absent: {QUAL_JSON}")
    q = load_json(QUAL_JSON)
    if (q.get("status") != "eligible_for_later_local_ls_protocol"
            or q.get("eligibility_gate_passed") is not True
            or q.get("local_energy_repair_case") is not True
            or q.get("search_allowed") is not False
            or q.get("candidate_cold", {}).get("qualified") is not True
            or q.get("reference", {}).get("qualified") is not True):
        raise ValueError("source-3 qualification did not pass the frozen protocol gate")
    if q.get("source_geometry", {}).get("member") != "c60/c60-iso-3_opt.xyz":
        raise ValueError("qualification is not the authorized author isomer #3")
    candidate = QUAL_DIR / "final-candidate.extxyz"
    reference = QUAL_DIR / "source" / "historical-ih-reference.extxyz"
    source = QUAL_DIR / "source" / "author-isomer-3.xyz"
    if not candidate.is_file() or not reference.is_file() or not source.is_file():
        raise FileNotFoundError("qualified candidate or historical Ih reference is missing")
    source_csv = QUAL_DIR / "source" / "author-isomer-3-row.csv"
    source_record = load_json(QUAL_DIR / "source" / "source-record.json")
    if (not source_csv.is_file()
            or sha256(source_csv) != source_record.get("csv_row_snapshot_sha256")
            or source_record.get("csv", {}).get("sha256") != q.get("source_csv", {}).get("sha256")):
        raise ValueError("qualification source CSV row does not match its frozen source record")
    initial, cold = q.get("initial_quench", {}), q["candidate_cold"]
    if (initial.get("status") != "completed" or initial.get("converged") is not True
            or initial.get("finite") is not True or initial.get("fmax_eV_A", float("inf")) > .03
            or not all(initial.get("endpoint_isomorphic_to_source", {}).get(k) is True
                       for k in ("1.64", "1.7", "1.8"))
            or cold.get("fmax_eV_A", float("inf")) > .03
            or cold.get("within_ih_plus_0.01_eV") is not False
            or cold.get("relative_to_ih_eV", float("-inf")) <= .01):
        raise ValueError("qualification endpoint/cold-check does not satisfy the frozen #3 gate")
    if sha256(candidate) != q.get("initial_quench", {}).get("endpoint_sha256"):
        raise ValueError("qualified final candidate differs from cold-qualified endpoint")
    if sha256(source) != q["source_geometry"]["member_sha256"]:
        raise ValueError("qualification source member hash does not match its source record")
    return q, candidate, reference, source


def load_validator(path=VALIDATOR):
    spec = importlib.util.spec_from_file_location("c60_source3_graph_validator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load C60 graph validator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_geometry(input_path: Path, reference_path: Path, validator, source_path: Path | None = None):
    import networkx as nx
    import numpy as np
    from ase.io import read

    atoms, ref = read(input_path), read(reference_path)
    if (len(atoms) != 60 or not np.all(atoms.numbers == 6)
            or atoms.pbc.any() or not np.isfinite(atoms.positions).all()
            or not np.allclose(atoms.cell.array, np.diag([50., 50., 50.]), atol=1e-8, rtol=0)):
        raise ValueError("frozen source-3 candidate violates isolated C60 geometry contract")
    if (len(ref) != 60 or not np.all(ref.numbers == 6) or ref.pbc.any()
            or not np.isfinite(ref.positions).all()):
        raise ValueError("frozen Ih reference is not finite nonperiodic C60")
    rows = {}
    source = read(source_path) if source_path is not None else None
    if source is not None and (len(source) != 60 or not np.all(source.numbers == 6)
                               or not np.isfinite(source.positions).all()):
        raise ValueError("archived source #3 is not a finite 60-carbon geometry")
    for cutoff in CUTOFFS:
        row = validator.graph_row(atoms.numbers, atoms.positions, cutoff,
            _graph(ref.numbers, ref.positions, cutoff))
        graph = _graph(atoms.numbers, atoms.positions, cutoff)
        row["isomorphic_to_ih"] = bool(nx.is_isomorphic(graph,
            _graph(ref.numbers, ref.positions, cutoff)))
        row["isomorphic_to_source"] = (None if source is None else bool(nx.is_isomorphic(
            graph, _graph(source.numbers, source.positions, cutoff))))
        rows[str(cutoff)] = row
    if not all(row["graph_cage_candidate"] and not row["isomorphic_to_ih"]
               and (source is None or row["isomorphic_to_source"]) for row in rows.values()):
        raise ValueError("candidate must retain its connected source #3 cage and remain non-Ih at every cutoff")
    return {"atoms": atoms, "reference": ref, "graphs": rows}


def _graph(numbers, positions, cutoff):
    import networkx as nx
    import numpy as np

    distances = np.linalg.norm(np.asarray(positions)[:, None] - np.asarray(positions)[None, :], axis=2)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(numbers)))
    graph.add_edges_from((int(i), int(j)) for i, j in zip(*np.where(
        np.triu((distances < cutoff) & (distances > 0), 1))))
    return graph


def direction_settings(path=DIRECTION_HELPER):
    helper = importlib.util.spec_from_file_location("c60_direction_transfer_prepare", path)
    if helper is None or helper.loader is None:
        raise RuntimeError("cannot import the approved C60 direction settings helper")
    module = importlib.util.module_from_spec(helper)
    helper.loader.exec_module(module)
    return module.protocol_config(), module.direction_settings()


def copy_qualification(out: Path, q, candidate: Path, reference: Path):
    dest = out / "qualification"
    (dest / "source").mkdir(parents=True, exist_ok=False)
    shutil.copy2(candidate, dest / "final-candidate.extxyz")
    shutil.copy2(reference, dest / "reference-source.extxyz")
    shutil.copy2(QUAL_DIR / "initial-endpoint.extxyz", dest / "initial-endpoint.extxyz")
    shutil.copy2(QUAL_DIR / "inputs" / "input.extxyz", dest / "source-input.extxyz")
    shutil.copy2(QUAL_DIR / "source" / "author-isomer-3.xyz", dest / "source" / "author-isomer-3.xyz")
    shutil.copy2(QUAL_DIR / "source" / "author-isomer-3-row.csv", dest / "source" / "author-isomer-3-row.csv")
    shutil.copy2(QUAL_DIR / "source" / "source-record.json", dest / "source" / "source-record.json")
    shutil.copy2(QUAL_DIR / "source" / "historical-ih-reference.extxyz",
                 dest / "source" / "historical-ih-reference.extxyz")
    dump(dest / "qualification-summary.json", q)
    return dest


def snapshot(out: Path):
    source = out / "source" / "pamssw"
    copied = []
    for live in sorted((ROOT / "pamssw").rglob("*.py")):
        relative = live.relative_to(ROOT / "pamssw")
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(live, target)
        copied.append({"path": f"pamssw/{relative}", "sha256": sha256(target),
                       "live_path": str(live.resolve())})
    helper_copy = out / "source" / "direction-transfer-prepare.py"
    shutil.copy2(DIRECTION_HELPER, helper_copy)
    shutil.copy2(RUNNER, out / "runner.py")
    shutil.copy2(VALIDATOR, out / "validator.py")
    shutil.copy2(PROTOCOL, out / "protocol.md")
    shutil.copy2(Path(__file__).resolve(), out / "prepare.py")
    git = lambda *args: subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()
    manifest = {
        "checkout": str(ROOT), "git_head": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "git_status_short_core": git("status", "--short", "--", "pamssw", "research/ga_ssw/c60_long_budget.py").splitlines(),
        "snapshot_scope": "runtime core package plus frozen research runner, graph validator, protocol, preparation script, and used direction settings helper; no repository clone",
        "core_root": "source/pamssw", "core_files": copied,
        "runner": {"path": "runner.py", "live_path": str(RUNNER.resolve()), "sha256": sha256(out / "runner.py")},
        "validator": {"path": "validator.py", "live_path": str(VALIDATOR.resolve()), "sha256": sha256(out / "validator.py")},
        "protocol": {"path": "protocol.md", "sha256": sha256(out / "protocol.md")},
        "prepare": {"path": "prepare.py", "live_path": str(Path(__file__).resolve()), "sha256": sha256(out / "prepare.py")},
        "direction_settings_helper": {"path": "source/direction-transfer-prepare.py",
            "live_path": str(DIRECTION_HELPER.resolve()), "sha256": sha256(helper_copy)},
        "qualification_summary_sha256": sha256(out / "qualification" / "qualification-summary.json"),
        "qualification_candidate_sha256": sha256(out / "qualification" / "final-candidate.extxyz"),
        "qualification_source_member_sha256": sha256(out / "qualification" / "source" / "author-isomer-3.xyz"),
        "qualification_source_csv_sha256": sha256(out / "qualification" / "source" / "author-isomer-3-row.csv"),
        "qualification_source_record_sha256": sha256(out / "qualification" / "source" / "source-record.json"),
        "reference_sha256": sha256(out / "qualification" / "reference-source.extxyz"),
    }
    dump(out / "source-manifest.json", manifest)
    return manifest


def paper_ls_spec():
    return {
        "bond_energies": {"6,6": 3.61}, "bond_lengths": {"6,6": 1.64},
        "target_per_atom": .02, "initial_fraction": .03, "xi": .2,
        "learning_rate": 1.8, "energy_filter": None,
        "prequench": {"fmax": .05, "steps": 1000, "exit_policy": "force"},
    }


def common_plan(out: Path, seed: int, arm: str, config, direction, q):
    case = "author-isomer-3"
    run_dir = out / f"seed-{seed}" / arm
    input_path = run_dir / "input.extxyz"
    shutil.copy2(out / "qualification" / "final-candidate.extxyz", input_path)
    reference_path = out / "qualification" / "reference-source.extxyz"
    summary_path = out / "qualification" / "qualification-summary.json"
    plan = {
        "model": str(MODEL), "model_sha256": MODEL_SHA, "head": "omol",
        "device": "cuda", "dtype": "float64",
        "runtime": {"torch_manual_seed": 0, "torch_deterministic_algorithms": True,
                    "torch_num_threads": 1, "tf32": False,
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
        "ssw_config": config, "recovered_direction": direction,
        "native_mc": {"energy_tol_eV": .1, "maxtrap": 99999},
        "seed": seed, "input": "input.extxyz", "input_sha256": sha256(input_path),
        "outer_steps": OUTER_STEPS, "search_cap": SEARCH_CAP,
        "fresh_cap": FRESH_CAP, "wall_seconds": WALL_SECONDS,
        "reference_energy_eV": float(q["reference"]["energy_eV"]),
        "reference_input": "../../qualification/reference-source.extxyz",
        "reference_sha256": sha256(reference_path),
        "qualification_summary": "../../qualification/qualification-summary.json",
        "frozen_files": {
            "input.extxyz": sha256(input_path), "../../runner.py": sha256(out / "runner.py"),
            "../../validator.py": sha256(out / "validator.py"),
            "validator.py": sha256(out / "validator.py"),
            "../../source-manifest.json": sha256(out / "source-manifest.json"),
            "../../protocol.md": sha256(out / "protocol.md"),
            "../../qualification/qualification-summary.json": sha256(summary_path),
            "../../qualification/reference-source.extxyz": sha256(reference_path),
        },
        "provenance": {"prepared_git_head": load_json(out / "source-manifest.json")["git_head"],
            "source_manifest": "../../source-manifest.json", "experiment": HERE.name,
            "case": case, "seed": seed, "arm": arm,
            "qualification_source": "../../qualification/qualification-summary.json",
            "qualification_status": q["status"]},
        "case": case, "arm": arm,
    }
    if arm == "paper_ls":
        plan["paper_ls"] = paper_ls_spec()
    return plan


def build_plans(out: Path, q):
    config, direction = direction_settings(out / "source" / "direction-transfer-prepare.py")
    plan_rows, arm_map = [], []
    for seed_index, seed in enumerate(SEEDS):
        for arm_index, arm in enumerate(ARMS):
            run_dir = out / f"seed-{seed}" / arm
            run_dir.mkdir(parents=True, exist_ok=False)
            shutil.copy2(out / "validator.py", run_dir / "validator.py")
            plan = common_plan(out, seed, arm, config, direction, q)
            path = run_dir / "plan.json"
            dump(path, plan)
            dump(run_dir / "qualification-link.json", {
                "qualification_summary": "../../qualification/qualification-summary.json",
                "candidate": "../../qualification/final-candidate.extxyz",
                "reference": "../../qualification/reference-source.extxyz"})
            relative = path.relative_to(out)
            plan_rows.append({"path": str(relative), "sha256": sha256(path)})
            arm_map.append({"array_task_id": seed_index * len(ARMS) + arm_index,
                "case": plan["case"], "seed": seed, "arm": arm,
                "run_dir": str(run_dir.relative_to(out)), "plan": str(relative),
                "input": str((run_dir / "input.extxyz").relative_to(out)),
                "input_sha256": plan["input_sha256"]})
    dump(out / "search-plan-manifest.json", {
        "plans": plan_rows, "arm_map": arm_map,
        "qualification_summary": "qualification/qualification-summary.json",
        "qualification_summary_sha256": sha256(out / "qualification" / "qualification-summary.json"),
        "source_snapshot_paths": {"core_root": "source/pamssw", "source_manifest": "source-manifest.json",
            "runner": "runner.py", "validator": "validator.py", "prepare": "prepare.py",
            "protocol": "protocol.md", "direction_settings_helper": "source/direction-transfer-prepare.py"},
        "source_manifest_sha256": sha256(out / "source-manifest.json")})


def verify_snapshot(out: Path):
    manifest = load_json(out / "source-manifest.json")
    for item in manifest["core_files"]:
        path = out / "source" / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"frozen core file changed or missing: {path}")
    for name in ("runner", "validator", "protocol", "prepare", "direction_settings_helper"):
        item = manifest[name]
        path = out / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"frozen {name} changed or missing: {path}")
    if sha256(out / "qualification" / "qualification-summary.json") != manifest["qualification_summary_sha256"]:
        raise ValueError("qualification summary changed")
    if sha256(out / "qualification" / "final-candidate.extxyz") != manifest["qualification_candidate_sha256"]:
        raise ValueError("qualified candidate changed")
    if sha256(out / "qualification" / "source" / "author-isomer-3.xyz") != manifest["qualification_source_member_sha256"]:
        raise ValueError("qualified source member changed")
    if sha256(out / "qualification" / "source" / "author-isomer-3-row.csv") != manifest["qualification_source_csv_sha256"]:
        raise ValueError("qualified source CSV changed")
    if sha256(out / "qualification" / "source" / "source-record.json") != manifest["qualification_source_record_sha256"]:
        raise ValueError("qualified source record changed")
    if sha256(out / "qualification" / "reference-source.extxyz") != manifest["reference_sha256"]:
        raise ValueError("qualified reference changed")
    return manifest


def verify_output(out: Path):
    import copy
    q, candidate, reference, source = qualified_sources()
    manifest = verify_snapshot(out)
    if (sha256(candidate) != sha256(out / "qualification" / "final-candidate.extxyz")
            or sha256(reference) != sha256(out / "qualification" / "reference-source.extxyz")
            or sha256(source) != sha256(out / "qualification" / "source" / "author-isomer-3.xyz")
            or sha256(QUAL_DIR / "source" / "author-isomer-3-row.csv") != sha256(
                out / "qualification" / "source" / "author-isomer-3-row.csv")):
        raise ValueError("prepared qualification inputs do not match the authorized source")
    if not MODEL.is_file() or sha256(MODEL) != MODEL_SHA:
        raise ValueError("cached MH-1 model SHA differs from protocol")
    validator = load_validator(out / "validator.py")
    check_geometry(out / "qualification" / "final-candidate.extxyz",
                   out / "qualification" / "reference-source.extxyz", validator,
                   out / "qualification" / "source" / "author-isomer-3.xyz")
    config, direction = direction_settings(out / "source" / "direction-transfer-prepare.py")
    map_path = out / "search-plan-manifest.json"
    plan_manifest = load_json(map_path)
    if sha256(out / "source-manifest.json") != plan_manifest["source_manifest_sha256"]:
        raise ValueError("plan manifest points at a different source snapshot")
    if sha256(out / "qualification" / "qualification-summary.json") != plan_manifest["qualification_summary_sha256"]:
        raise ValueError("plan manifest points at a different qualification record")
    if len(plan_manifest["plans"]) != 4 or len(plan_manifest["arm_map"]) != 4:
        raise ValueError("expected exactly four frozen arm plans")
    seen_ids = set()
    for mapping in plan_manifest["arm_map"]:
        if mapping["array_task_id"] in seen_ids:
            raise ValueError("duplicate array task id")
        seen_ids.add(mapping["array_task_id"])
        plan_path = out / mapping["plan"]
        if sha256(plan_path) != next(row["sha256"] for row in plan_manifest["plans"]
                                    if row["path"] == mapping["plan"]):
            raise ValueError(f"plan changed: {plan_path}")
        plan = load_json(plan_path)
        run_dir = plan_path.parent
        if (plan["seed"] != mapping["seed"] or plan["arm"] != mapping["arm"]
                or plan["input_sha256"] != mapping["input_sha256"]
                or sha256(run_dir / plan["input"]) != plan["input_sha256"]):
            raise ValueError("arm mapping, plan, and actual input differ")
        if (plan["outer_steps"] != OUTER_STEPS or plan["search_cap"] != SEARCH_CAP
                or plan["fresh_cap"] != FRESH_CAP or plan["wall_seconds"] != WALL_SECONDS
                or plan["model_sha256"] != MODEL_SHA or plan["ssw_config"] != config
                or plan["recovered_direction"] != direction
                or plan["native_mc"] != {"energy_tol_eV": .1, "maxtrap": 99999}):
            raise ValueError("plan differs from frozen common protocol settings")
        for filename, expected in plan["frozen_files"].items():
            if sha256(run_dir / filename) != expected:
                raise ValueError(f"run-time frozen file changed: {filename}")
        if mapping["arm"] == "ssw" and plan.get("paper_ls") is not None:
            raise ValueError("plain SSW arm unexpectedly has paper_ls")
        if mapping["arm"] == "paper_ls" and plan.get("paper_ls") != paper_ls_spec():
            raise ValueError("paper-LS arm has unexpected settings")
    if seen_ids != set(range(4)):
        raise ValueError("array task IDs must be exactly 0..3")
    by_pair = {}
    for item in plan_manifest["arm_map"]:
        by_pair.setdefault(item["seed"], {})[item["arm"]] = load_json(out / item["plan"])
    for seed, pair in by_pair.items():
        if set(pair) != set(ARMS):
            raise ValueError(f"seed {seed} lacks one matched arm")
        ssw, ls = (copy.deepcopy(pair[name]) for name in ARMS)
        ssw.pop("arm"); ls.pop("arm"); ls.pop("paper_ls")
        ssw["provenance"].pop("arm"); ls["provenance"].pop("arm")
        if ssw != ls:
            raise ValueError(f"arms differ outside paper_ls for seed {seed}")
    return {"status": "verified", "plans": 4, "arm_map": plan_manifest["arm_map"],
            "source_manifest_sha256": sha256(out / "source-manifest.json"),
            "qualification_summary_sha256": sha256(out / "qualification" / "qualification-summary.json"),
            "qualification_status": q["status"], "git_head": manifest["git_head"]}


def import_frozen_runner(out: Path):
    source_root = (out / "source").resolve()
    sys.path.insert(0, str(source_root))
    import pamssw
    if not Path(pamssw.__file__).resolve().is_relative_to(source_root):
        raise RuntimeError(f"live pamssw imported instead of source snapshot: {pamssw.__file__}")
    spec = importlib.util.spec_from_file_location("c60_long_budget_frozen", out / "runner.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen C60 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, source_root


def preflight():
    """Build and validate real four-plan scaffolding in a temporary directory; no PES/model."""
    compile(Path(__file__).read_text(), str(__file__), "exec")
    compile(RUNNER.read_text(), str(RUNNER), "exec")
    compile(VALIDATOR.read_text(), str(VALIDATOR), "exec")
    for item in (ROOT / "pamssw").rglob("*.py"):
        compile(item.read_text(), str(item), "exec")
    q, candidate, reference, source = qualified_sources()
    check_geometry(candidate, reference, load_validator(), source)
    with tempfile.TemporaryDirectory(prefix="c60-source3-paper-ls-preflight-") as temp:
        out = Path(temp) / "prepared"
        out.mkdir()
        copy_qualification(out, q, candidate, reference)
        snapshot(out)
        build_plans(out, q)
        result = verify_output(out)
        runner, source_root = import_frozen_runner(out)
        from pamssw import standalone
        from pamssw.standalone import LSSettings
        received = []
        class DummyCalculator:
            def calculate(self, *args, **kwargs):
                raise AssertionError("preflight stub must never evaluate a PES")
        old_calculator, old_run_ssw = runner.calculator, standalone.run_ssw
        runner.calculator = lambda plan: DummyCalculator()
        def stub_run_ssw(*args, **kwargs):
            received.append(kwargs)
            return type("StubResult", (), {"status": "completed", "checkpoint": None})()
        standalone.run_ssw = stub_run_ssw
        try:
            for mapping in result["arm_map"]:
                run_dir = out / mapping["run_dir"]
                runner.run_segment(run_dir, 5)
        finally:
            runner.calculator, standalone.run_ssw = old_calculator, old_run_ssw
        if len(received) != 4 or any(row.get("steps") != OUTER_STEPS for row in received):
            raise RuntimeError("stub run_segment did not forward the frozen outer-step cap")
        parsed = [row.get("ls") for row in received]
        if sum(isinstance(value, LSSettings) for value in parsed) != 2 or sum(value is None for value in parsed) != 2:
            raise RuntimeError("frozen runner did not parse the two plain/two paper-LS plans")
        core_imports = {name: str(Path(module.__file__).resolve())
                        for name, module in sys.modules.items()
                        if (name == "pamssw" or name.startswith("pamssw.")) and getattr(module, "__file__", None)}
        if not all(Path(path).is_relative_to(source_root) for path in core_imports.values()):
            raise RuntimeError("preflight imported a live core module")
    print(json.dumps({"status": "preflight_passed", "qualification_status": q["status"],
        "real_pes_requests": 0, "real_model_initialized": False,
        "temporary_four_plan_build_and_verify": "passed", "stubbed_run_segment_calls": len(received),
        "stubbed_steps": [row["steps"] for row in received],
        "parsed_paper_ls": sum(isinstance(value, LSSettings) for value in parsed),
        "parsed_plain_ssw": sum(value is None for value in parsed),
        "frozen_core_module_count": len(core_imports),
        "key_frozen_imports": {name: core_imports[name] for name in
            ("pamssw", "pamssw.standalone", "pamssw.standalone.paper_reference",
             "pamssw.standalone.softening", "pamssw.standalone.recovered_direction")
            if name in core_imports}}, indent=2))


def prepare(out: Path):
    out = out.resolve()
    if out.exists():
        raise FileExistsError(f"--out must be a new directory: {out}")
    q, candidate, reference, source = qualified_sources()
    check_geometry(candidate, reference, load_validator(), source)
    if sha256(MODEL) != MODEL_SHA:
        raise ValueError("cached MH-1 model SHA differs from protocol")
    out.mkdir(parents=True, exist_ok=False)
    copy_qualification(out, q, candidate, reference)
    snapshot(out)
    build_plans(out, q)
    result = verify_output(out)
    dump(out / "preparation-status.json", {"status": "prepared_not_submitted", **result,
        "model_sha256_checked": MODEL_SHA, "pes_requests": 0})
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--verify-output", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.prepare:
        if args.out is None:
            parser.error("--prepare requires --out NEW_DIRECTORY")
        prepare(args.out)
    elif args.preflight:
        if args.out is not None:
            parser.error("--preflight does not accept --out")
        preflight()
    else:
        if args.out is not None:
            parser.error("use --verify-output PATH rather than combining --out")
        print(json.dumps(verify_output(args.verify_output), indent=2))


if __name__ == "__main__":
    main()
