#!/bin/bash
#SBATCH --job-name=c60-src3-ls
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=4V100
#SBATCH --qos=rush-1o2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --array=0-3%2
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/c60-source3-paper-ls-transfer-20261007/gpu-%A_%a.out
#SBATCH --error=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/c60-source3-paper-ls-transfer-20261007/gpu-%A_%a.err
set -euo pipefail
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
ROOT=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
TASK_DIR="$ROOT/research/ga_ssw/evidence/c60-source3-paper-ls-transfer-20261007"
PREPARED="${1:?usage: sbatch gpu.sbatch /path/to/prepared-OUTPUT}"
cd "$ROOT"
/home/gengjianrui/.conda/envs/mace_env/bin/python "$TASK_DIR/prepare.py" --verify-output "$PREPARED"
exec /home/gengjianrui/.conda/envs/mace_env/bin/python - "$PREPARED" "$SLURM_ARRAY_TASK_ID" <<'PY'
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

prepared = Path(sys.argv[1]).resolve()
array_id = int(sys.argv[2])
manifest_path = prepared / "search-plan-manifest.json"
manifest = json.loads(manifest_path.read_text())
matches = [row for row in manifest["arm_map"] if row["array_task_id"] == array_id]
if len(matches) != 1:
    raise RuntimeError(f"expected exactly one arm for array task {array_id}, got {len(matches)}")
mapping = matches[0]
run_dir = prepared / mapping["run_dir"]
source_root = (prepared / "source").resolve()
sys.path.insert(0, str(source_root))
import pamssw
if not Path(pamssw.__file__).resolve().is_relative_to(source_root):
    raise RuntimeError(f"worker imported live core source: {pamssw.__file__}")
spec = importlib.util.spec_from_file_location("c60_long_budget_frozen", prepared / "runner.py")
if spec is None or spec.loader is None:
    raise RuntimeError("cannot import frozen C60 research runner")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)

def core_imports():
    rows = {}
    for name, module in sorted(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if (name == "pamssw" or name.startswith("pamssw.")) and path:
            resolved = Path(path).resolve()
            if not resolved.is_relative_to(source_root):
                raise RuntimeError(f"worker imported live core source: {name}={resolved}")
            rows[name] = {"path": str(resolved), "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest()}
    return rows

def validator_import():
    module = sys.modules.get("validator")
    if module is None or not getattr(module, "__file__", None):
        return None
    path = Path(module.__file__).resolve()
    expected = (run_dir / "validator.py").resolve()
    if path != expected:
        raise RuntimeError(f"worker imported unexpected graph validator: {path}")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

runtime_dir = prepared / "runtime-source"
runtime_dir.mkdir(exist_ok=True)
record = runtime_dir / f"array-{array_id}-job-{os.environ['SLURM_JOB_ID']}.json"
started = time.monotonic()
runtime = {"prepared": str(prepared), "run_dir": str(run_dir), "array_id": array_id,
    "seed": mapping["seed"], "arm": mapping["arm"], "python": sys.executable,
    "job_id": os.getenv("SLURM_JOB_ID"), "array_job_id": os.getenv("SLURM_ARRAY_JOB_ID"),
    "plan_sha256": hashlib.sha256((run_dir / "plan.json").read_bytes()).hexdigest(),
    "source_manifest_sha256": hashlib.sha256((prepared / "source-manifest.json").read_bytes()).hexdigest(),
    "runner": {"path": str((prepared / "runner.py").resolve()),
               "sha256": hashlib.sha256((prepared / "runner.py").read_bytes()).hexdigest()},
    "validator_snapshot": {"path": str((prepared / "validator.py").resolve()),
                  "sha256": hashlib.sha256((prepared / "validator.py").read_bytes()).hexdigest()},
    "validator_import_before": validator_import(),
    "core_imports_before": core_imports(), "status": "running"}
record.write_text(json.dumps(runtime, indent=2) + "\n")
try:
    state = runner.run_segment(run_dir, 1200)
    if state["status"] in ("search_exhausted", "wall_exhausted", "completed", "failed"):
        runner.finalize(run_dir)
    state = json.loads((run_dir / "budget.json").read_text())
    runtime.update(status="finished", run_status=state["status"],
        search_requests=state.get("search"), fresh_requests=state.get("fresh"),
        elapsed_seconds=time.monotonic() - started, core_imports_after=core_imports(),
        validator_import_after=validator_import())
    record.write_text(json.dumps(runtime, indent=2) + "\n")
except Exception as error:
    runtime.update(status="failed", error=repr(error), elapsed_seconds=time.monotonic() - started,
        core_imports_after=core_imports(), validator_import_after=validator_import())
    record.write_text(json.dumps(runtime, indent=2) + "\n")
    raise
PY
