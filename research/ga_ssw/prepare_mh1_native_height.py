"""Prepare the frozen MH1 native-height equal-budget evidence directory."""
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "research/ga_ssw/evidence/mh1-equal-budget-rotation-20260920"
OUT = ROOT / "research/ga_ssw/evidence/mh1-native-height-equal-budget-20260920"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    if OUT.exists():
        raise FileExistsError(OUT)
    parent = json.loads((FROZEN / "plan.json").read_text())
    OUT.mkdir(parents=True)
    shutil.copytree(FROZEN / "source", OUT / "source")
    shutil.copy2(FROZEN / "ledger_helpers.py", OUT / "ledger_helpers.py")
    shutil.copy2(FROZEN / "source-manifest.json", OUT / "source-manifest.json")
    (OUT / "inputs").mkdir()
    inputs = {}
    for name, metadata in parent["inputs"].items():
        source = FROZEN / metadata["path"]
        target = OUT / metadata["path"]
        shutil.copy2(source, target)
        inputs[name] = dict(metadata, path=str(target.relative_to(OUT)),
                            sha256=sha256(target), source=str(source))

    plan = dict(
        scope=("Two MH1 C60 first-state inputs; native-derived height policy versus "
               "the completed equal-budget baseline; development evidence only, "
               "not an independent success-rate study."),
        parent=str(FROZEN),
        baseline_reference=str(FROZEN),
        backend=parent["backend"], runtime=parent["runtime"], config=parent["config"],
        native_mc=parent["native_mc"], inputs=inputs, arms=["native_height"], steps=100,
        search_cap_per_arm=12000, fresh_cap_per_arm=101, total_cap_per_arm=12101,
        wall_seconds_per_arm=900, budget_slices=[3000, 6000, 12000],
        native_height={
            "initial_weight": 0.5, "negative_weight": 0.2, "level": 1,
            "max_weight": 10.0, "growth_step": 1.0, "growth_scale": 2.0,
            "height_update_budget": 1000,
            "parameter_source": (
                "run_c60_fixed_budget.py native-height variant; values from "
                "current native allkeys settings, with stage-frozen ConservativeNativeHeightPolicy; "
                "not complete native addgaussian/force implementation parity"
            ),
        },
        parameter_source=(
            "Single factor is enabling ConservativeNativeHeightPolicy. Baseline, "
            "rotation, native MC, inputs, seeds, frozen source, and budgets are "
            "inherited from mh1-equal-budget-rotation-20260920."
        ),
        success_rule=(
            "Report fresh force qualification, MH1 reference energy, cage graph "
            "criteria, labelled graph counts, costs, denials, and failures separately; "
            "do not infer basin identity or native-force parity."
        ),
        total_search_cap=24000, total_fresh_cap=202,
        provenance=dict(
            frozen_source=str(FROZEN / "source"),
            frozen_runner=str(FROZEN / "runner.py"),
            frozen_ledger_helper=str(FROZEN / "ledger_helpers.py"),
            no_current_core_copy=True,
            baseline_result=str(FROZEN),
        ),
    )
    (OUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")

    runner = (FROZEN / "runner.py").read_text()
    runner = runner.replace(
        "from pamssw.standalone import ASESurface, NativeMCSettings, SSWConfig, RecoveredRotationSettings, run_ssw",
        "from pamssw.standalone import (ASESurface, ConservativeNativeHeightPolicy, "
        "NativeMCSettings, SSWConfig, RecoveredRotationSettings, run_ssw)")
    runner = runner.replace(
        "config=SSWConfig(**plan['config']);mc=NativeMCSettings(plan['native_mc']['energy_tol_eV'],plan['native_mc']['maxtrap'])",
        "config=SSWConfig(**plan['config']);mc=NativeMCSettings(plan['native_mc']['energy_tol_eV'],plan['native_mc']['maxtrap'])\n"
        "    height_policy=ConservativeNativeHeightPolicy(**{k:v for k,v in plan['native_height'].items() if k not in ('parameter_source','height_update_budget')})")
    runner = runner.replace("for arm in plan['arms']:", "for arm in plan['arms']:")
    runner = runner.replace(
        "recovered_rotation=(RecoveredRotationSettings(**plan['recovered_rotation']) if arm=='recovered_cbd' else None),",
        "recovered_rotation=None, height_policy=height_policy,\n"
        "                    height_update_budget=plan['native_height']['height_update_budget'],")
    (OUT / "runner.py").write_text(runner)
    (OUT / "job.sh").write_text("""#!/bin/bash
# Prepared only; submit explicitly after review.
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=4V100
#SBATCH --qos=rush-1o2gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=00:35:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/mh1-native-height-equal-budget-20260920/slurm-%j.out
set -euo pipefail
cd /home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/mh1-native-height-equal-budget-20260920
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
/home/gengjianrui/.conda/envs/mace_env/bin/python runner.py --execute --output .
""")
    (OUT / "job.sh").chmod(0o755)
    (OUT / "README.md").write_text("""# MH1 native-height equal-budget comparison

This is a development ablation with two saved C60 first-state inputs. It enables
only `ConservativeNativeHeightPolicy` in the frozen MH1 rotation protocol. The
completed baseline is retained in `baseline_reference`; no baseline rerun is
included here.

The policy values are `initial_weight=.5`, `negative_weight=.2`, `level=1`,
`max_weight=10`, `growth_step=1`, `growth_scale=2`, with the explicit numerical
height-update ceiling `1000`. They come from the existing `native-height` variant
in `research/ga_ssw/run_c60_fixed_budget.py`. This is a stage-frozen repair and
cannot claim exact native LASP force or addgaussian parity.

Each arm has 100 outer steps, 12000 search calls, 101 fresh checks, and a 900 s
wall limit. Budget slices are 3000, 6000, and 12000 search calls. Failed and
partial terminal records remain part of the evidence. The result is not an
independent success-rate estimate.

Prepared command (not submitted):

```bash
sbatch job.sh
```
""")


if __name__ == "__main__":
    prepare()
