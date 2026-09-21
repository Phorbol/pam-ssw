"""Prepare the frozen Cu55 EMT height-policy comparison."""
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "research/ga_ssw/evidence/emt-cu55-qualified-rotation-20260920"
OUT = ROOT / "research/ga_ssw/evidence/emt-cu55-qualified-height-20260920"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    if OUT.exists():
        raise FileExistsError(OUT)
    source = OUT / "source"
    shutil.copytree(FROZEN / "source", source,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(FROZEN / "ledger_helpers.py", OUT / "ledger_helpers.py")
    (OUT / "inputs").mkdir(parents=True)

    first = json.loads((FROZEN / "candidate1/plan.json").read_text())
    second = json.loads((FROZEN / "candidate2/plan.json").read_text())
    if first["config"] != second["config"] or first["native_mc"] != second["native_mc"]:
        raise ValueError("candidate baseline scientific settings differ")
    cases = []
    for index, baseline in ((1, first), (2, second)):
        source_input = FROZEN / f"candidate{index}" / baseline["input_path"]
        target = OUT / "inputs" / f"candidate{index}-cu55.extxyz"
        shutil.copy2(source_input, target)
        actual = sha256(target)
        if actual != baseline["input_sha256"]:
            raise ValueError(f"candidate{index} input hash changed")
        cases.append({"name": f"candidate{index}", "seed": 11,
                      "path": str(target.relative_to(OUT)),
                      "sha256": actual,
                      "source": str(source_input),
                      "candidate_index": index})

    source_manifest = {}
    for path in sorted(source.rglob("*.py")):
        source_manifest[str(path.relative_to(source))] = sha256(path)
    (OUT / "source-manifest.json").write_text(json.dumps(source_manifest, indent=2) + "\n")
    plan = {
        "scope": ("Two prequalified Cu55 EMT states, seed 11, baseline rotation "
                   "with only ConservativeNativeHeightPolicy enabled; development "
                   "cross-system evidence, not an independent success-rate study."),
        "parent": str(FROZEN),
        "baseline_reference": str(FROZEN),
        "backend": {"name": "ase-emt", "description": "ASE EMT, nonperiodic Cu55"},
        "config": first["config"], "native_mc": first["native_mc"],
        "cases": cases, "arms": ["native_height"], "steps": 100,
        "search_cap_per_arm": 6000, "fresh_cap_per_arm": 101,
        "total_search_cap": 12000, "total_fresh_cap": 202,
        "wall_seconds_per_arm": 300,
        "height_policy": {
            "initial_weight": 0.5, "negative_weight": 0.2, "level": 1,
            "max_weight": 10.0, "growth_step": 1.0, "growth_scale": 2.0,
            "height_update_budget": 1000,
            "parameter_source": "Same ConservativeNativeHeightPolicy values as MH1 height comparison.",
        },
        "parameter_source": (
            "Only height_policy differs from the completed baseline. Source, EMT "
            "configuration, inputs, seed, native MC, ordinary baseline rotation, "
            "steps, and budgets are inherited from emt-cu55-qualified-rotation-20260920."),
        "qualification": "fresh EMT force qualification at config.fmax; retain all failures and partial costs",
        "provenance": {"frozen_source": str(FROZEN / "source"),
                       "frozen_ledger": str(FROZEN / "ledger_helpers.py"),
                       "no_geometry_reselection": True, "no_current_core_copy": True},
    }
    (OUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")

    (OUT / "runner.py").write_text('''"""Execute the bounded Cu55 EMT native-height comparison."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(out):
    plan = json.loads((out / "plan.json").read_text())
    if (out / "summary.json").exists() or (out / "execution-started.json").exists():
        raise FileExistsError("preserve prior attempt; prepare a new run")
    (out / "execution-started.json").write_text(json.dumps({
        "job_id": os.getenv("SLURM_JOB_ID"), "account": os.getenv("SLURM_JOB_ACCOUNT")
    }) + "\\n")
    sys.path[:0] = [str(out / "source"), str(out)]
    import numpy as np
    from ase.calculators.emt import EMT
    from ase.io import read
    from ledger_helpers import CountedSurface, dump
    from pamssw.standalone import (ASESurface, ConservativeNativeHeightPolicy,
                                   NativeMCSettings, SSWConfig, run_ssw)

    config = SSWConfig(**plan["config"])
    mc = NativeMCSettings(plan["native_mc"]["energy_tol_eV"], plan["native_mc"]["maxtrap"])
    spec = plan["height_policy"]
    height_policy = ConservativeNativeHeightPolicy(**{
        key: value for key, value in spec.items()
        if key not in ("parameter_source", "height_update_budget")})
    rows = []
    for case in plan["cases"]:
        input_path = out / case["path"]
        if digest(input_path) != case["sha256"]:
            raise ValueError(f"frozen input changed: {case['name']}")
        folder = out / f"{case['name']}-seed{case['seed']}-native_height"
        folder.mkdir()
        atoms = read(input_path)
        calc = EMT()
        surface = CountedSurface(calc, folder / "requests.jsonl",
                                 cap=plan["search_cap_per_arm"],
                                 wall=plan["wall_seconds_per_arm"])
        fresh = None
        checks = []
        row = {"case": case["name"], "seed": case["seed"],
               "arm": "native_height", "status": "started"}
        try:
            result = run_ssw(
                atoms, surface, steps=plan["steps"], config=config,
                rng=np.random.default_rng(case["seed"]), mc=mc,
                height_policy=height_policy,
                height_update_budget=spec["height_update_budget"],
                recovered_rotation=None, checkpoint_path=folder / "checkpoint.pkl")
            dump(folder / "result.json", result)
            fresh = ASESurface(calc)
            for i, minimum in enumerate(result.minima[:plan["fresh_cap_per_arm"]]):
                calc.reset()
                try:
                    energy, forces = fresh.evaluate(minimum.atoms)
                    fmax = float(np.linalg.norm(forces, axis=1).max())
                    checks.append({"index": i, "energy_eV": energy, "fmax_eV_A": fmax,
                                   "qualified": bool(fmax <= config.fmax),
                                   "delta_energy_from_initial_emt": energy - result.initial.energy})
                except Exception as exc:
                    checks.append({"index": i, "qualified": False, "error": repr(exc)})
            row.update(status=result.status, records=len(result.records),
                       gaussian_counts=[len(r.climb) for r in result.records],
                       record_statuses=[r.status for r in result.records],
                       search_calls=surface.requests, fresh_calls=fresh.requests,
                       initial_energy_eV=result.initial.energy, fresh_checks=checks)
        except Exception as exc:
            row.update(status="exception", error=repr(exc))
        row.update(search_calls=surface.requests,
                   fresh_calls=0 if fresh is None else fresh.requests,
                   fresh_checks=checks, boundary=surface.boundary,
                   denials=surface.denials)
        rows.append(row)
        dump(out / "summary.json", rows)
        print(row, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true", required=True)
    args = parser.parse_args()
    execute(args.output.resolve())
''')
    (OUT / "job.sh").write_text('''#!/bin/bash
# Prepared only; submit explicitly after review.
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/emt-cu55-qualified-height-20260920/slurm-%j.out
set -euo pipefail
cd /home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/emt-cu55-qualified-height-20260920
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1
/home/gengjianrui/.conda/envs/mace_env/bin/python runner.py --execute --output .
''')
    (OUT / "job.sh").chmod(0o755)
    (OUT / "README.md").write_text('''# Cu55 EMT native-height comparison

This is a two-state cross-system development control. It reuses the two
prequalified candidate input byte streams, seed 11, ordinary baseline rotation,
EMT backend, native MC, and 100-step/6000-search/101-fresh/300-second protocol
from `emt-cu55-qualified-rotation-20260920`. The only enabled change is the
same stage-frozen `ConservativeNativeHeightPolicy` used in the MH1 comparison:
`.5/.2/level1/max10/growth_step1/growth_scale2`, with height-update budget 1000.

The prior baseline remains the comparison reference; no baseline rerun is done.
This does not claim native LASP force or addgaussian parity, and it is not an
independent success-rate estimate. Failed and partial terminal records remain.

Prepared command (not submitted):

```bash
sbatch job.sh
```
''')


if __name__ == "__main__":
    prepare()
