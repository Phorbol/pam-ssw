#!/bin/bash
#SBATCH --job-name=c60-dir-transfer
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=4V100
#SBATCH --qos=rush-1o2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:20:00
#SBATCH --array=0-3%2
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/c60-direction-transfer-20261007/gpu-%A_%a.out
#SBATCH --error=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/c60-direction-transfer-20261007/gpu-%A_%a.err
set -euo pipefail
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

ROOT=/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification
TASK_DIR="$ROOT/research/ga_ssw/evidence/c60-direction-transfer-20261007"
PYTHON=/home/gengjianrui/.conda/envs/mace_env/bin/python
PREPARED="${1:?pass the reviewed prepared-JOBID directory as the first argument}"
PREPARED="$(realpath "$PREPARED")"
case "$SLURM_ARRAY_TASK_ID" in
  0) RUN_REL=seed-26100781/rotation ;;
  1) RUN_REL=seed-26100781/direction ;;
  2) RUN_REL=seed-26100782/rotation ;;
  3) RUN_REL=seed-26100782/direction ;;
  *) echo "unexpected array index: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac
RUN_DIR="$PREPARED/$RUN_REL"
test -d "$RUN_DIR"
"$PYTHON" "$TASK_DIR/prepare.py" --verify-output "$PREPARED"

# Run source qualification and the frozen worker in one Python process. Since
# cwd has no live pamssw package and the snapshot is first on PYTHONPATH, these
# recorded module paths are the same modules used by c60_long_budget.py below.
cd "$PREPARED"
export PYTHONPATH="$PREPARED/source"
exec "$PYTHON" - "$PREPARED" "$RUN_DIR" "$SLURM_ARRAY_TASK_ID" <<'PY'
import hashlib
import importlib
import json
from pathlib import Path
import runpy
import sys

prepared = Path(sys.argv[1]).resolve()
run_dir = Path(sys.argv[2]).resolve()
array_id = int(sys.argv[3])
source_root = (prepared / "source").resolve()
modules = [importlib.import_module(name) for name in (
    "pamssw.standalone.paper_reference",
    "pamssw.standalone.surface",
    "pamssw.standalone.recovered_rotation",
    "pamssw.standalone.recovered_direction",
)]
imports = {}
for module in modules:
    path = Path(module.__file__).resolve()
    if not path.is_relative_to(source_root):
        raise RuntimeError(f"worker imported live core source: {path}")
    imports[module.__name__] = {"path": str(path),
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
validator_path = prepared / "validator.py"
runtime_dir = prepared / "runtime-source"
runtime_dir.mkdir(exist_ok=True)
(runtime_dir / f"array-{array_id}-job-{__import__('os').environ['SLURM_JOB_ID']}.json").write_text(
    json.dumps({"prepared": str(prepared), "run_dir": str(run_dir),
                "array_id": array_id, "python": sys.executable,
                "sys_path": sys.path, "core_imports": imports,
                "runner": {"path": str(prepared / "runner.py"),
                           "sha256": hashlib.sha256((prepared / "runner.py").read_bytes()).hexdigest()},
                "validator": {"path": str(validator_path),
                              "sha256": hashlib.sha256(validator_path.read_bytes()).hexdigest()}}, indent=2) + "\n")
sys.argv = [str(prepared / "runner.py"), "--run-dir", str(run_dir),
            "--segment-seconds", "4800", "--execute"]
runpy.run_path(str(prepared / "runner.py"), run_name="__main__")
PY
