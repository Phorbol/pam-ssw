#!/usr/bin/env python3
"""Stage and verify the frozen native C60 source-3 reference inputs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path("/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification")
HERE = Path(__file__).resolve().parent
INPUT = ROOT / "research/ga_ssw/evidence/c60-ls-source3-qualification-20261007/qualification-1664282/final-candidate.extxyz"
QUAL = ROOT / "research/ga_ssw/evidence/c60-source3-lasp-input-qualification-20261007/run-1664642"
SEED_PROBE = ROOT / "research/ga_ssw/evidence/lasp-explicit-seed-qualification-20261007/run-1664773"
REFERENCE = Path("/home/gengjianrui/bin/pam-ssw-worktrees/c60-vacuum-geometry/research/ga_ssw/evidence/c60-native-long-20260920/seed17093")
MODEL = Path("/home/gengjianrui/.cache/mace/mace-mh-1.model")
BINARY = Path("/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp")
SEEDS = (26100791, 26100792)
INPUT_SHA = "dcc81d45c4e7193b306a0c1957b49c19e44751231fa95fef75201183280f0758"
MODEL_SHA = "a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47"
BINARY_SHA = "bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704"
MPI_LIB = "/opt/devtools/intel/oneapi/mpi/2021.13/lib"
VACUUM_GATE = Path("/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/c60-mh1-vacuum-equivalence-20260920/result.json")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_inputs() -> None:
    if sha(INPUT) != INPUT_SHA:
        raise ValueError(f"qualified input changed: {INPUT}")
    if sha(MODEL) != MODEL_SHA or sha(BINARY) != BINARY_SHA:
        raise ValueError("qualified MH-1 model or LASP binary changed")
    summary = json.loads((QUAL / "summary.json").read_text())
    prov = json.loads((QUAL / "provenance.json").read_text())
    if summary.get("qualification_pass") is not True or summary.get("callback_ef_match") is not True:
        raise ValueError("source3 native callback qualification is not passing")
    if prov.get("model_sha256") != MODEL_SHA or prov.get("binary_sha256") != BINARY_SHA:
        raise ValueError("qualification provenance does not match frozen model/binary")
    seed = json.loads((SEED_PROBE / "summary.json").read_text())
    cases = seed.get("cases", [])
    for requested in SEEDS:
        matching = [row for row in cases if row.get("input_ranseed") == requested]
        if len(matching) < (2 if requested == SEEDS[0] else 1) or any(row.get("requested_key_present_in_allkeys") is not True for row in matching):
            raise ValueError(f"explicit native ranseed key {requested} is missing or not retained by allkeys")
    repeat = [row for row in cases if row.get("input_ranseed") == SEEDS[0]]
    other = [row for row in cases if row.get("input_ranseed") == SEEDS[1]]
    if repeat[0].get("requested_coordinates") != repeat[1].get("requested_coordinates"):
        raise ValueError("same-seed native request prefixes were not reproducible")
    if repeat[0].get("requested_coordinates") == other[0].get("requested_coordinates"):
        raise ValueError("different native seeds did not produce different request prefixes")
    input_artifact = prov["actual_input_artifacts"]["input.arc"]
    if input_artifact["path"] != str((QUAL / "input.arc").resolve()) or sha(QUAL / "input.arc") != input_artifact["sha256"]:
        raise ValueError("qualified native ARC does not match its recorded actual input artifact")
    if not (REFERENCE / "runner.py").is_file():
        raise FileNotFoundError(REFERENCE / "runner.py")
    vacuum = json.loads(VACUUM_GATE.read_text())
    if vacuum.get("all_rows_pass") is not True:
        raise ValueError("frozen native vacuum-equivalence gate is not all_rows_pass")


def run_plan(seed: int, run: Path) -> dict:
    return {
        "status": "prepared_not_executed", "case": f"source3-seed-{seed}", "seed": seed,
        "input": "input.extxyz", "model": str(MODEL), "model_sha256": MODEL_SHA,
        "head": "omol", "dtype": "float64", "device": "cuda",
        "binary": str(BINARY), "binary_sha256": BINARY_SHA, "mpi_lib": MPI_LIB,
        "cell_A": [[50.0, 0, 0], [0, 50.0, 0], [0, 0, 50.0]], "pbc": [True, True, True],
        "request_cap": 16000, "wall_seconds": 900,
        "fresh_budget": {"search_excluded": True, "frames_max": 3, "representations": 1, "ef_max": 3},
        "run_dir": str(run), "input_sha256": INPUT_SHA,
        "vacuum_equivalence_result": "vacuum-equivalence-result.json",
        "frozen_source": {},
        "provenance": {
            "qualified_input_source": str(INPUT), "qualified_input_sha256": INPUT_SHA,
            "qualification_run": str(QUAL), "qualification_provenance_sha256": sha(QUAL / "provenance.json"),
            "seed_qualification_run": str(SEED_PROBE), "model_sha256": MODEL_SHA, "binary_sha256": BINARY_SHA,
            "upstream_runner": str(REFERENCE / "runner.py"), "upstream_runner_sha256": sha(REFERENCE / "runner.py"),
        },
    }


def prepare(output: Path) -> None:
    check_inputs()
    output.mkdir(parents=True, exist_ok=False)
    for name in ("analyze.py", "readout.sbatch", "qualify.py", "qualify.sbatch", "prepare.py", "gpu.sbatch", "protocol.md"):
        source = HERE / name
        shutil.copy2(source, output / name)
    for seed in SEEDS:
        run = output / f"seed-{seed}"
        run.mkdir()
        for name in ("vacuum_geometry.py", "bounded_process.py", "client.py"):
            shutil.copy2(REFERENCE / name, run / name)
        shutil.copy2(QUAL / "lasp_external_ase.py", run / "lasp_external_ase.py")
        shutil.copy2(INPUT, run / "input.extxyz")
        # The qualified 50 A ARC is the exact native serialization of this source.
        shutil.copy2(QUAL / "input.arc", run / "input.arc")
        shutil.copy2(QUAL / "historical-ih-reference.extxyz", run / "reference.extxyz")
        shutil.copy2(QUAL / "reference-cold.json", run / "reference-cold.json")
        shutil.copy2(QUAL / "summary.json", run / "source3-input-qualification-summary.json")
        shutil.copy2(QUAL / "provenance.json", run / "source3-input-qualification-provenance.json")
        shutil.copy2(QUAL / "lasp-callback.json", run / "source3-input-qualification-callback.json")
        shutil.copy2(VACUUM_GATE, run / "vacuum-equivalence-result.json")
        seed_probe_case = SEED_PROBE / "case-0-seed-26100791" if seed == SEEDS[0] else SEED_PROBE / "case-2-seed-26100792"
        shutil.copy2(seed_probe_case / "allkeys.log", run / "seed-probe-allkeys.log")
        shutil.copy2(seed_probe_case / "lasp.in", run / "seed-probe.lasp.in")
        validator_source = ROOT / "research/ga_ssw/evidence/c60-source3-paper-ls-transfer-20261007/prepared-20261007-a/validator.py"
        shutil.copy2(validator_source, run / "graph_helper.py")
        # This task runner adds request accounting to the frozen socket implementation.
        shutil.copy2(HERE / "runner.py", run / "runner.py")
        (run / "lasp.external.sh").write_text(
            '#!/bin/bash\nset -euo pipefail\n'
            '/home/gengjianrui/.conda/envs/mace_env/bin/python "$(dirname "$0")/client.py"\n')
        (run / "lasp.external.sh").chmod(0o755)
        (run / "lasp.in").write_text(
            "potential external\nexplore_type ssw\nEwaldflag 0\nRun_type 5\n"
            "SSW.SSWsteps 1001\nSSW.ftol 0.0173205080756888\nSSW.MaxOptstep 1000\n"
            "SSW.NG 12\nSSW.Temp 150\nSSW.ds_atom .6\nSSW.internal_LJ F\n"
            "SSW.globalcompress .0001\nSSW.vapor_cri 1.7\nSSW.output T\nSSW.printevery T\n"
            f"ranseed {seed}\n")
        plan = run_plan(seed, run)
        ref_cold = json.loads((run / "reference-cold.json").read_text())
        plan["reference_energy_eV"] = float(ref_cold["energy_eV"])
        plan["provenance"]["reference_sha256"] = sha(run / "reference.extxyz")
        plan["provenance"]["seed_probe_allkeys_sha256"] = sha(run / "seed-probe-allkeys.log")
        plan["vacuum_equivalence_result"] = "vacuum-equivalence-result.json"
        plan["source3_input_qualification_summary"] = "source3-input-qualification-summary.json"
        plan["source3_input_qualification_summary_sha256"] = sha(run / "source3-input-qualification-summary.json")
        # Hash the actual run inputs and staged sources; the runner verifies these before launch.
        plan["frozen_source"] = {name: sha(run / name) for name in (
            "runner.py", "vacuum_geometry.py", "bounded_process.py", "client.py", "lasp.external.sh",
            "lasp_external_ase.py", "graph_helper.py", "source3-input-qualification-summary.json",
            "source3-input-qualification-provenance.json", "source3-input-qualification-callback.json",
            "vacuum-equivalence-result.json", "seed-probe-allkeys.log")}
        plan["frozen_source"].update({"input.arc": sha(run / "input.arc"), "input.extxyz": sha(run / "input.extxyz"), "lasp.in": sha(run / "lasp.in"), "reference.extxyz": sha(run / "reference.extxyz")})
        (run / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps({"status": "prepared_not_executed", "cases": [
        {"seed": seed, "run_dir": f"seed-{seed}"} for seed in SEEDS
    ], "request_cap_each": 16000, "wall_seconds_each": 900, "source3_input_sha256": INPUT_SHA}, indent=2) + "\n")
    staged_sources = {name: {"source": str((HERE / name).resolve()), "sha256": sha(output / name)}
                      for name in ("analyze.py", "readout.sbatch", "qualify.py", "qualify.sbatch", "prepare.py", "gpu.sbatch", "protocol.md")}
    (output / "source-manifest.json").write_text(json.dumps({"sources": staged_sources,
        "checkout": str(ROOT), "git_head": subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        "seed_runs": {f"seed-{seed}": json.loads((output / f"seed-{seed}" / "plan.json").read_text())["frozen_source"] for seed in SEEDS}}, indent=2) + "\n")
    verify(output)


def verify(output: Path) -> None:
    manifest = json.loads((output / "manifest.json").read_text())
    if [row["seed"] for row in manifest["cases"]] != list(SEEDS):
        raise ValueError("prepared manifest does not contain the two frozen native seeds")
    source_manifest = json.loads((output / "source-manifest.json").read_text())
    for name, row in source_manifest["sources"].items():
        if sha(output / name) != row["sha256"]:
            raise ValueError(f"staged readout source changed: {name}")
    for seed in SEEDS:
        run = output / f"seed-{seed}"
        plan = json.loads((run / "plan.json").read_text())
        if plan["seed"] != seed or plan["request_cap"] != 16000 or plan["wall_seconds"] != 900:
            raise ValueError(f"wrong budget or seed in {run}")
        for name, expected in plan["frozen_source"].items():
            path = run / name
            if sha(path) != expected:
                raise ValueError(f"staged artifact changed: {path}")
        if sha(run / "input.extxyz") != INPUT_SHA:
            raise ValueError("staged input hash mismatch")
        if sha(run / "reference.extxyz") != plan["provenance"]["reference_sha256"]:
            raise ValueError("staged Ih reference hash mismatch")
        allkeys = (run / "seed-probe-allkeys.log").read_text()
        if not re.search(rf"^ranseed\s+{seed}\s*$", allkeys, re.M) or not re.search(r"^SSW\.NG\s+12\s*$", allkeys, re.M):
            raise ValueError("native seed/NG12 are not confirmed in explicit-seed allkeys evidence")
        if f"ranseed {seed}\n" not in (run / "lasp.in").read_text():
            raise ValueError("native seed missing from LASP input")
        if '"$(dirname "$0")/client.py"' not in (run / "lasp.external.sh").read_text():
            raise ValueError("native callback launcher does not use the staged client")
    print(json.dumps({"status": "verified", "output": str(output.resolve()), "manifest": str((output / 'manifest.json').resolve()), "seeds": list(SEEDS), "request_cap_each": 16000, "wall_seconds_each": 900}))


def preflight(output: Path) -> None:
    """Verify inputs and finite-cutoff geometry without loading MACE or evaluating PES."""
    verify(output)
    seed_run = output / f"seed-{SEEDS[0]}"
    spec = importlib.util.spec_from_file_location("source3_vacuum_geometry_preflight", seed_run / "vacuum_geometry.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load staged geometry helper")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    from ase.io import read
    atoms = read(seed_run / "input.extxyz")
    atoms.set_cell([[50.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 50.0]])
    atoms.set_pbc(True)
    summary = json.loads((seed_run / "source3-input-qualification-summary.json").read_text())
    cutoff = float(summary["model_cutoff_A"])
    canonical, diagnostics = helper.inspect_vacuum(atoms, cutoff)
    if canonical is None or diagnostics.get("eligible") is not True:
        raise RuntimeError(f"source3 callback geometry preflight failed: {diagnostics}")
    prov = json.loads((seed_run / "source3-input-qualification-provenance.json").read_text())
    if sha(Path(prov["model"])) != MODEL_SHA or sha(Path(prov["binary"])) != BINARY_SHA:
        raise ValueError("live model/binary differs from qualified exact hashes")
    print(json.dumps({"status": "preflight_passed_no_pes", "output": str(output.resolve()),
                      "input_sha256": INPUT_SHA, "model_sha256": MODEL_SHA,
                      "binary_sha256": BINARY_SHA, "model_cutoff_A": cutoff,
                      "geometry": diagnostics, "mace_loaded": False, "pes_calculations": 0}))


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", type=Path)
    group.add_argument("--verify-output", type=Path)
    group.add_argument("--preflight", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare.resolve())
    elif args.preflight:
        preflight(args.preflight.resolve())
    else:
        verify(args.verify_output.resolve())


if __name__ == "__main__":
    main()
