"""Build a symlink-only run-0 view for the existing VC panel analyzer."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

METHODS = ("safe_total", "ase", "scipy")
SEEDS = (71, 83)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", required=True, type=Path,
                        help="arms-<GPU-array-job-id> directory")
    parser.add_argument("--out", required=True, type=Path,
                        help="new analysis-view-<jobid> directory")
    args = parser.parse_args()
    arms = args.arms.resolve()
    out = args.out.resolve()
    if not arms.is_dir():
        parser.error(f"missing arms directory: {arms}")
    if out.exists():
        parser.error(f"refusing to overwrite: {out}")
    arm_dirs = sorted(arms.glob("arm-*"), key=lambda p: int(p.name.split("-")[-1]))
    plans = [p / "plan.json" for p in arm_dirs if (p / "plan.json").is_file()]
    if not plans:
        parser.error("no worker plan found; preserve the panel as missing/incomplete")
    plan_bytes = plans[0].read_bytes()
    plan = json.loads(plan_bytes)
    if len(plan.get("cases", [])) != 1 or plan["budget"].get("search_requests_per_arm") != 6000:
        parser.error("input plan is not the frozen one-case, 6000-request panel")
    for path in plans[1:]:
        if path.read_bytes() != plan_bytes:
            parser.error(f"plan mismatch across arms: {path}")

    run_view = out / "run-0"
    run_view.mkdir(parents=True)
    (run_view / "plan.json").write_bytes(plan_bytes)
    by_task = {p.name: p for p in arm_dirs}
    rows = []
    expected = [f"case0-{method}-seed{seed}" for seed in SEEDS for method in METHODS]
    task_by_name = {
        f"case0-{METHODS[index // 2]}-seed{SEEDS[index % 2]}": index
        for index in range(6)
    }
    for name in expected:
        arm_root = by_task.get(f"arm-{task_by_name[name]}")
        rc = "missing"
        if arm_root is not None:
            exits = arm_root / "worker-exits.tsv"
            if exits.is_file():
                fields = exits.read_text().strip().split("\t")
                if len(fields) == 2 and fields[0] == name:
                    rc = fields[1]
            target = arm_root / name
            if target.exists():
                os.symlink(os.path.relpath(target, run_view), run_view / name,
                           target_is_directory=True)
        rows.append(f"{name}\t{rc}")
    (run_view / "worker-exits.tsv").write_text("\n".join(rows) + "\n")
    print(json.dumps({"analysis_run_dir": str(run_view),
                      "expected_arms": expected,
                      "linked_arm_dirs": sorted(p.name for p in run_view.iterdir()
                          if p.is_symlink()),
                      "calculator_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
