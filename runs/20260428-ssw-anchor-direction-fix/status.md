Current phase: implemented and verified.

Completed:
- Implemented mixed random + bond anchor direction for standard PAM-SSW.
- Added anchor penalty to soft-mode candidate scoring.
- Added dynamic non-neighbor bond candidates while preserving the fixed HVP candidate budget.
- Corrected bound-constrained proposal-relax convergence reporting to use projected gradient.
- Relaxed default proposal `fmax` to 0.02 for biased intermediate walks; true quench remains strict.

Verification:
- `pytest -q`: 73 passed in 4.21 s.
- SSW-only LJ13/LJ38 smoke confirms bond directions are selected and candidate pool size stays near the configured budget.

Next:
- Run corrected full ASE quick gate after this commit.
- Tune trust controller and step target to reduce proposal-relax unconverged fraction without simply raising local-relax iteration caps.
