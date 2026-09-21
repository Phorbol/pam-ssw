"""Prepare the phase-139 joint-VC central-Ritz comparison; no PES on import."""
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "research/ga_ssw/evidence/tio2-phase139-joint-memory-compare/memory400"
INPUT = ROOT / "literature/benchmark-sources/coordinates/phase-139.extxyz"
MODEL = Path("/home/gengjianrui/.cache/mace/mace-omat-0-small.model")
DEFAULT_OUT = ROOT / "research/ga_ssw/evidence/tio2-phase139-joint-central-ritz-memory400"

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prepare(out):
    if out.exists(): raise FileExistsError(out)
    baseline_plan = json.loads((BASE / "plan.json").read_text())
    cfg = json.loads((ROOT / "research/ga_ssw/evidence/tio2-phase139-joint-memory-compare/memory400-config.json").read_text())
    if sha(INPUT) != "38b00df3777900257a1be1e6e10c680fb5d91a441c78e6594c5dba7dee11a8c6": raise ValueError("phase-139 input changed")
    if sha(MODEL) != "0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5": raise ValueError("MACE model changed")
    out.mkdir(parents=True); (out / "source").mkdir(); (out / "input").mkdir()
    shutil.copytree(BASE / "source/pamssw", out / "source/pamssw")
    rsrc = out / "source/research/ga_ssw"; rsrc.mkdir(parents=True)
    # Baseline source plus the already reviewed optional solver plumbing.
    shutil.copy2(ROOT / "research/ga_ssw" / "compare_material_arms.py", rsrc / "compare_material_arms.py")
    shutil.copy2(ROOT / "research/ga_ssw" / "compare_vc_arms.py", rsrc / "compare_vc_arms.py")
    shutil.copy2(ROOT / "pamssw/standalone/generalized_ritz.py", out / "source/pamssw/standalone/generalized_ritz.py")
    for name in ("generalized_numerics.py", "vc_reference.py"):
        shutil.copy2(ROOT / "pamssw/standalone" / name, out / "source/pamssw/standalone" / name)
    shutil.copy2(INPUT, out / "input/phase-139.extxyz")
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    plan = {
        "status": "prepared_only", "baseline": str(BASE), "baseline_plan_sha256": sha(BASE / "plan.json"),
        "input": str(INPUT), "input_sha256": sha(INPUT), "model": str(MODEL), "model_sha256": sha(MODEL),
        "seed": 3, "steps": 1, "arm": "joint", "budget_EF": 1500, "wall_seconds": 900,
        "threads": 1, "memory": 400, "solver": "pamssw.standalone.generalized_numerics.generalized_central_ritz",
        "direction_solver": "central finite-difference Ritz", "rotation_force_calls": 100,
        "only_intervention": "joint run_vc_ssw direction_solver + rotation_force_calls; all input/config/initial quench/MC retained",
        "scope": "one seed, one material, falsification control; no claim that Ritz wins or generalizes",
        "config": cfg, "baseline_arguments": baseline_plan.get("arguments", {}),
    }
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    shutil.copy2(__file__, out / "runner-prepared.py")
    # Clone the completed memory400 CLI and apply only the reviewed solver
    # injection.  Assertions make accidental drift fail during preparation.
    baseline_runner = (BASE / "script.py").read_text()
    baseline_runner = baseline_runner.replace(
        "from pamssw.standalone.generalized_numerics import safe_lbfgs",
        "from pamssw.standalone.generalized_numerics import safe_lbfgs, generalized_central_ritz")
    baseline_runner = baseline_runner.replace(
        "joint_config, steps, seed):",
        "joint_config, steps, seed, direction_solver=None, rotation_force_calls=None):", 1)
    baseline_runner = baseline_runner.replace(
        "else run_vc_ssw(atoms,s,steps=steps,config=joint_config,rng=rng))",
        "else run_vc_ssw(atoms,s,steps=steps,config=joint_config,rng=rng,"
        "direction_solver=direction_solver,rotation_force_calls=rotation_force_calls))", 1)
    baseline_runner = baseline_runner.replace(
        "joint_config=joint,steps=args.steps,seed=args.seed)",
        "joint_config=joint,steps=args.steps,seed=args.seed,"
        "direction_solver=generalized_central_ritz,rotation_force_calls=100)", 1)
    if "generalized_central_ritz" not in baseline_runner or "rotation_force_calls=100" not in baseline_runner:
        raise RuntimeError("central Ritz runner patch did not apply")
    (out / "runner-executable.py").write_text(baseline_runner)
    try:
        import ase, numpy
        versions = {"python": sys.version, "python_executable": sys.executable,
                    "ase": ase.__version__, "numpy": numpy.__version__}
    except Exception as exc:
        versions = {"python": sys.version, "python_executable": sys.executable,
                    "error": repr(exc)}
    (out / "environment-preflight.json").write_text(json.dumps(versions, indent=2) + "\n")
    (out / "runner-digests.json").write_text(json.dumps({
        "baseline_runner_sha256": sha(BASE / "script.py"),
        "prepared_runner_sha256": sha(out / "runner-executable.py"),
        "config_sha256": sha(out / "config.json"),
    }, indent=2) + "\n")
    # Preserve an auditable source-level intervention record.
    diffs=[]
    pairs=[(BASE/"source/pamssw/standalone/generalized_numerics.py",ROOT/"pamssw/standalone/generalized_numerics.py"),
           (BASE/"source/pamssw/standalone/vc_reference.py",ROOT/"pamssw/standalone/vc_reference.py"),
           (BASE/"source/research/ga_ssw/compare_material_arms.py",ROOT/"research/ga_ssw/compare_material_arms.py")]
    for old,new in pairs:
        if not old.exists():
            diffs.append(f"### {old.name} baseline -> current\n(base snapshot has no file; current helper is the reviewed optional wiring)\n")
            continue
        p=subprocess.run(["diff","-u",str(old),str(new)],text=True,capture_output=True).stdout
        diffs.append(f"### {old.name} baseline -> current\n{p or '(identical)'}")
    runner_diff = subprocess.run(["diff", "-u", str(BASE / "script.py"), str(out / "runner-executable.py")], text=True, capture_output=True).stdout
    (out / "source-diff.patch").write_text("\n".join(diffs) + "\n### runner intervention\n" + runner_diff)
    (out / "preflight.json").write_text(json.dumps({"PES_evaluations": 0, "input_atoms": 48,
        "input_sha256": sha(INPUT), "model_sha256": sha(MODEL), "solver_import": "deferred",
        "declared_budget_EF": 1500, "declared_wall_seconds": 900, "source": "baseline pamssw snapshot + reviewed central Ritz plumbing"}, indent=2) + "\n")

def execute(out):
    raise RuntimeError("execution withheld pending root review; prepared package only")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--prepare",action="store_true"); p.add_argument("--execute",action="store_true"); p.add_argument("--output",type=Path,default=DEFAULT_OUT); a=p.parse_args()
    if a.prepare: prepare(a.output)
    elif a.execute: execute(a.output)
    else: p.error("choose --prepare; no PES is run by default")
if __name__ == '__main__': main()
