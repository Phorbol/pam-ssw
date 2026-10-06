#!/bin/bash
#SBATCH --job-name=c60-src3-native
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=4V100
#SBATCH --qos=rush-1o2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --array=0-1%2
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/c60-source3-native-reference-20261007/gpu-%A_%a.out
#SBATCH --error=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/c60-source3-native-reference-20261007/gpu-%A_%a.err
set -euo pipefail

ROOT=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
TASK_DIR="$ROOT/research/ga_ssw/evidence/c60-source3-native-reference-20261007"
PYTHON=/home/gengjianrui/.conda/envs/mace_env/bin/python
PREPARED="${1:?usage: sbatch gpu.sbatch /absolute/path/to/prepared-directory}"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

cd "$ROOT"
"$PYTHON" "$PREPARED/prepare.py" --verify-output "$PREPARED"
"$PYTHON" - "$PREPARED" "$SLURM_ARRAY_TASK_ID" <<'PY'
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

prepared = Path(sys.argv[1]).resolve()
array_id = int(sys.argv[2])
manifest = json.loads((prepared / "manifest.json").read_text())
cases = manifest["cases"]
if array_id < 0 or array_id >= len(cases):
    raise RuntimeError(f"array index {array_id} has no prepared case")
case = cases[array_id]
run_dir = (prepared / case["run_dir"]).resolve()
if run_dir.parent != prepared:
    raise RuntimeError(f"case path escapes prepared directory: {run_dir}")
sys.path.insert(0, str(run_dir))
import vacuum_geometry
if Path(vacuum_geometry.__file__).resolve() != (run_dir / "vacuum_geometry.py").resolve():
    raise RuntimeError(f"unexpected vacuum geometry import: {vacuum_geometry.__file__}")
spec = importlib.util.spec_from_file_location("source3_native_frozen_runner", run_dir / "runner.py")
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load staged native runner")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)
plan = json.loads((run_dir / "plan.json").read_text())
if plan["seed"] != case["seed"] or plan["request_cap"] != 16000 or plan["wall_seconds"] != 900:
    raise RuntimeError("case seed or hard per-run cap differs from frozen manifest")
for name, expected in plan["frozen_source"].items():
    digest = hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
    if digest != expected:
        raise RuntimeError(f"staged source changed: {name}")
if any((run_dir / name).exists() for name in ("request.jsonl", "summary.json", "process.json")):
    raise FileExistsError(f"refusing to resume or overwrite existing run: {run_dir}")
runner.main(run_dir)
PY
