"""Prepare the frozen MH1 native-ls equal-budget evidence directory."""
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "research/ga_ssw/evidence/mh1-equal-budget-rotation-20260920"
OUT = ROOT / "research/ga_ssw/evidence/mh1-native-ls-equal-budget-20260920"


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
        scope=("Two MH1 C60 first-state inputs; native-derived LS versus "
               "the completed equal-budget baseline; development evidence only, "
               "not an independent success-rate study."),
        parent=str(FROZEN),
        baseline_reference=str(FROZEN),
        backend=parent["backend"], runtime=parent["runtime"], config=parent["config"],
        native_mc=parent["native_mc"], inputs=inputs, arms=["native_ls"], steps=100,
        search_cap_per_arm=12000, fresh_cap_per_arm=101, total_cap_per_arm=12101,
        wall_seconds_per_arm=900, budget_slices=[3000, 6000, 12000],
        native_ls={
            "bond_energies": {"6,6": 3.4468400478363037},
            "bond_lengths": {"6,6": 1.5399999618530273},
            "scale": 5.0, "amp_c": 2.0, "length_tolerance": 0.1,
            "target_mev_per_atom": 20.0, "eta": 0.005, "max_change": 0.01,
            "frequency": 10, "presteps": 100, "cycle": 100,
            "ratio": 1.100000023841858, "lselfadapt": True,
            "bond_geometry": "native-mic",
            "prequench": {"fmax": 0.1, "steps": 50,
                          "exit_policy": "force_or_step_limit"},
            "parameter_source": (
                "run_c60_fixed_budget.py native-ls variant; C-C release lookup "
                "and recovered-materials .1/50 prequench protocol; not C60 fitting "
                "or exact native LASP LS/stage-exit parity"
            ),
        },
        parameter_source=(
            "Single factor is enabling NativeLSSettings. Baseline, "
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
        "from pamssw.standalone import (ASESurface, LSPrequenchSettings, NativeLSSettings, "
        "NativeMCSettings, SSWConfig, RecoveredRotationSettings, run_ssw)")
    runner = runner.replace(
        "config=SSWConfig(**plan['config']);mc=NativeMCSettings(plan['native_mc']['energy_tol_eV'],plan['native_mc']['maxtrap'])",
        "config=SSWConfig(**plan['config']);mc=NativeMCSettings(plan['native_mc']['energy_tol_eV'],plan['native_mc']['maxtrap'])\n"
        "    ls_spec=plan['native_ls']\n"
        "    pair_table=lambda values: {tuple(int(part) for part in key.split(',')): value for key, value in values.items()}\n"
        "    ls=NativeLSSettings(bond_energies=pair_table(ls_spec['bond_energies']), bond_lengths=pair_table(ls_spec['bond_lengths']),\n"
        "                        scale=ls_spec['scale'], amp_c=ls_spec['amp_c'], length_tolerance=ls_spec['length_tolerance'],\n"
        "                        target_mev_per_atom=ls_spec['target_mev_per_atom'], eta=ls_spec['eta'], max_change=ls_spec['max_change'],\n"
        "                        frequency=ls_spec['frequency'], presteps=ls_spec['presteps'], cycle=ls_spec['cycle'],\n"
        "                        ratio=ls_spec['ratio'], lselfadapt=ls_spec['lselfadapt'], bond_geometry=ls_spec['bond_geometry'],\n"
        "                        prequench=LSPrequenchSettings(**ls_spec['prequench']))")
    runner = runner.replace("for arm in plan['arms']:", "for arm in plan['arms']:")
    runner = runner.replace(
        "recovered_rotation=(RecoveredRotationSettings(**plan['recovered_rotation']) if arm=='recovered_cbd' else None),",
        "recovered_rotation=None, ls=ls, height_policy=None,")
    (OUT / "runner.py").write_text(runner)
    (OUT / "job.sh").write_text("""#!/bin/bash
# Prepared only; submit explicitly after review.
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --partition=4V100
#SBATCH --qos=rush-1o2gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=00:35:00
#SBATCH --output=/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/mh1-native-ls-equal-budget-20260920/slurm-%j.out
set -euo pipefail
cd /home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/mh1-native-ls-equal-budget-20260920
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
/home/gengjianrui/.conda/envs/mace_env/bin/python runner.py --execute --output .
""")
    (OUT / "job.sh").chmod(0o755)
    (OUT / "README.md").write_text("""# MH1 native-ls equal-budget comparison

This is a development ablation with two saved C60 first-state inputs. It enables
only `NativeLSSettings` in the frozen MH1 rotation protocol. The
completed baseline is retained in `baseline_reference`; no baseline rerun is
included here.

The LS values are the C-C release lookup (`B=3.4468400478363037`,
`r=1.5399999618530273`), scale 5, `amp_c=2`, length tolerance .1,
20 meV/atom target, eta .005, max change .01, frequency 10, presteps/cycle
100, ratio 1.100000023841858, native-mic geometry, and a `.1/50`
`force_or_step_limit` prequench. They come from the existing `native-ls` variant
in `research/ga_ssw/run_c60_fixed_budget.py`; this is not a C60 fit and cannot
claim exact native LASP LS or stage-exit parity.

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
