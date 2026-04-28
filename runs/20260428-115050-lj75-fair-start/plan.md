Task: fix ASE-GA trace fairness and run a larger LJ75 comparison.

Scope:
- Make ASE-GA include the same initial structure used by SSW and ASE-BH as the first relaxed population member.
- Regenerate fair-start LJ13/LJ38 budget-20 curves.
- Run LJ75 seed 0 budget 60 with SSW, ASE-BH, and ASE-GA.
- Export all minima XYZ files for LJ75.

Rationale:
- The previous ASE-GA trace did not start from the same geometry as SSW/BH because ASE-GA initialized candidates with StartGenerator.
- For a fair energy-vs-step plot, the first relaxed minimum must be common across algorithms.

Limits:
- This is still a short-budget comparison against ASE-family baselines, not a reproduction of Wales-Doye/Deaven-Ho/Wolf-Landman long campaign searches.
