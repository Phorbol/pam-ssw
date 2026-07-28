# Block-Krylov fixed-state direction audit

This is a direction-only, preregistered Stage-1 audit.  It does not run an
SSW campaign, change the starter selector/walker protocol, or rank arms by
terminal energy.

The four and only four arms are `variational_breadth` (6x1),
`shallow_refinement` (3x2), `balanced_refinement` (2x3), and
`deep_refinement` (1x6).  Every central-FD HVP uses `h=1e-3`; every row must
close `force_evaluations == 2 * krylov_hvp_count` with at most 12 HVPs.

Run the CPU-only analytic audit first:

```bash
cd /tmp/SSW-worktrees/posterior-terminal-outcome-validation
python runs/20260728-block-krylov-direction-audit/run_fixed_state_audit.py \
  --analytic-only \
  --output /tmp/block-krylov-analytic.json
python runs/20260728-block-krylov-direction-audit/analyze_fixed_state_audit.py \
  --input /tmp/block-krylov-analytic.json \
  --output /tmp/block-krylov-analytic-evidence.json
```

The full fixed-state audit needs the local MACE CUDA contract.  It uses
`/root/.cache/mace/mace-omat-0-small.model`, a clean tracked worktree, and the
existing strict 200-production artifacts.  It reconstructs the missing
starter-quench state with the frozen production `load_state`, `build_config`,
MACE calculator, and `STARTER_TRUE_QUENCH`; failure to reconstruct it is a
hard error, never a state substitution.  The intermediate and plateau states
are locked path/checksum registry entries.  No structure is copied into Git.

```bash
cd /tmp/SSW-worktrees/posterior-terminal-outcome-validation
source /root/miniforge3/etc/profile.d/conda.sh
conda activate mace_les
python runs/20260728-block-krylov-direction-audit/run_fixed_state_audit.py \
  --output runs/20260728-block-krylov-direction-audit/raw.json
python runs/20260728-block-krylov-direction-audit/analyze_fixed_state_audit.py \
  --input runs/20260728-block-krylov-direction-audit/raw.json \
  --output runs/20260728-block-krylov-direction-audit/evidence.json
```

The evidence is limited to direction algebra, central-FD operator diagnostics,
and purpose-resolved force accounting.  It is not evidence that any allocation
improves terminal energy, basin discovery, chemistry, an optimizer, or a
production search.
